"""
Object Extractor
================
Extract objects from COCO datasets as transparent PNG images.

Supports:
- Direct extraction using polygon/RLE segmentation masks
- SAM3-based segmentation for bbox-only annotations
- Individual COCO JSON generation per object
"""

import os
import json
import logging
import asyncio
import base64
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from PIL import Image

from app.models.extraction_schemas import (
    AnnotationType,
    ExtractionMethod,
    CategoryInfo,
    ExtractedObjectInfo,
    ExtractionSummary,
)

logger = logging.getLogger(__name__)


class ObjectExtractor:
    """
    Extract objects from a COCO dataset as PNG images with transparency.

    Uses existing segmentation masks when available, or SAM3 for bbox-only annotations.
    """

    def __init__(
        self,
        sam3_model=None,
        sam3_processor=None,
        device: str = "cpu"
    ):
        """
        Initialize the ObjectExtractor.

        Args:
            sam3_model: Pre-loaded SAM3 model instance (from service state)
            sam3_processor: Pre-loaded SAM3 processor instance (from service state)
            device: Device to use for SAM3 inference ("cuda" or "cpu")
        """
        self.sam3_model = sam3_model
        self.sam3_processor = sam3_processor
        self.device = device
        self.sam3_available = sam3_model is not None and sam3_processor is not None

        # Thread pool for CPU-bound operations to avoid blocking the event loop
        # Using max_workers=2 to limit concurrent heavy operations
        self._executor = ThreadPoolExecutor(max_workers=2)

        logger.info(f"ObjectExtractor initialized (SAM3 available: {self.sam3_available})")

    # =========================================================================
    # ANNOTATION TYPE DETECTION
    # =========================================================================

    @staticmethod
    def detect_annotation_type(annotation: Dict[str, Any]) -> AnnotationType:
        """
        Detect the type of annotation (polygon, RLE, or bbox_only).

        Args:
            annotation: COCO annotation dictionary

        Returns:
            AnnotationType enum value
        """
        segmentation = annotation.get("segmentation")

        if segmentation is None:
            return AnnotationType.BBOX_ONLY

        # Empty list or empty segmentation
        if isinstance(segmentation, list) and len(segmentation) == 0:
            return AnnotationType.BBOX_ONLY

        # Polygon format: list of lists with coordinates
        if isinstance(segmentation, list) and len(segmentation) > 0:
            first_seg = segmentation[0]
            if isinstance(first_seg, list) and len(first_seg) >= 6:
                return AnnotationType.POLYGON

        # RLE format: dictionary with 'counts' and 'size'
        if isinstance(segmentation, dict):
            if "counts" in segmentation and "size" in segmentation:
                return AnnotationType.RLE

        return AnnotationType.BBOX_ONLY

    # =========================================================================
    # MASK CONVERSION
    # =========================================================================

    @staticmethod
    def polygon_to_mask(
        segmentation: List[List[float]],
        height: int,
        width: int
    ) -> np.ndarray:
        """
        Convert COCO polygon segmentation to binary mask.

        Args:
            segmentation: List of polygons, each polygon is [x1,y1,x2,y2,...]
            height: Image height
            width: Image width

        Returns:
            Binary mask as uint8 numpy array (0 or 255)
        """
        mask = np.zeros((height, width), dtype=np.uint8)

        for polygon in segmentation:
            if len(polygon) < 6:
                continue

            # Reshape flat list to array of points
            pts = np.array(polygon).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)

        return mask

    @staticmethod
    def rle_to_mask(
        rle: Dict[str, Any],
        height: int,
        width: int
    ) -> np.ndarray:
        """
        Convert COCO RLE segmentation to binary mask.

        Args:
            rle: RLE dictionary with 'counts' and 'size'
            height: Image height
            width: Image width

        Returns:
            Binary mask as uint8 numpy array (0 or 255)
        """
        try:
            from pycocotools import mask as mask_util

            # Ensure proper format
            if isinstance(rle["counts"], list):
                # Uncompressed RLE
                rle_obj = mask_util.frPyObjects([rle], height, width)[0]
            else:
                # Compressed RLE
                rle_obj = rle

            mask = mask_util.decode(rle_obj)
            return (mask * 255).astype(np.uint8)

        except ImportError:
            logger.warning("pycocotools not available, using fallback RLE decoder")
            return ObjectExtractor._decode_rle_fallback(rle, height, width)

    @staticmethod
    def _decode_rle_fallback(
        rle: Dict[str, Any],
        height: int,
        width: int
    ) -> np.ndarray:
        """Fallback RLE decoder when pycocotools is not available."""
        counts = rle.get("counts", [])
        size = rle.get("size", [height, width])

        if isinstance(counts, str):
            # Compressed format - simplified decoding
            logger.warning("Compressed RLE requires pycocotools for accurate decoding")
            return np.zeros((height, width), dtype=np.uint8)

        # Uncompressed RLE
        mask = np.zeros(size[0] * size[1], dtype=np.uint8)
        pos = 0
        val = 0

        for count in counts:
            mask[pos:pos + count] = val * 255
            pos += count
            val = 1 - val

        return mask.reshape(size).T

    @staticmethod
    def mask_to_polygon(
        mask: np.ndarray,
        simplify: bool = True,
        tolerance: float = 2.0
    ) -> List[List[float]]:
        """
        Convert binary mask to COCO polygon format.

        Args:
            mask: Binary mask (0 or 255)
            simplify: Whether to simplify the polygon
            tolerance: Simplification tolerance

        Returns:
            List of polygons in COCO format
        """
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        polygons = []
        for contour in contours:
            if len(contour) < 3:
                continue

            if simplify and tolerance > 0:
                epsilon = tolerance
                contour = cv2.approxPolyDP(contour, epsilon, True)

            if len(contour) < 3:
                continue

            # Flatten to COCO format [x1,y1,x2,y2,...]
            polygon = contour.flatten().tolist()
            if len(polygon) >= 6:
                polygons.append(polygon)

        return polygons

    # =========================================================================
    # SAM3 SEGMENTATION
    # =========================================================================

    def _segment_with_sam3_sync(
        self,
        image: np.ndarray,
        bbox: List[float]
    ) -> Optional[np.ndarray]:
        """
        Synchronous SAM3 segmentation (runs in thread pool).

        Args:
            image: Image as numpy array (BGR or RGB)
            bbox: Bounding box [x, y, width, height]

        Returns:
            Binary mask as numpy array, or None if failed
        """
        if not self.sam3_available:
            logger.warning("SAM3 not available for segmentation")
            return None

        # Validate image
        if image is None or image.size == 0:
            logger.error("segment_with_sam3: received empty or None image")
            return None

        import torch

        try:
            h, w = image.shape[:2]
            x, y, bw, bh = [int(v) for v in bbox]

            # Validate bbox dimensions
            if bw <= 0 or bh <= 0:
                logger.warning(f"segment_with_sam3: invalid bbox dimensions: {bbox}")
                return None

            # Compute intersection of bbox with image bounds and clamp
            x1_valid = max(0, x)
            y1_valid = max(0, y)
            x2_valid = min(w, x + bw)
            y2_valid = min(h, y + bh)

            # Check if there's a valid intersection
            if x1_valid >= x2_valid or y1_valid >= y2_valid:
                logger.info(f"segment_with_sam3: bbox has no overlap with image - bbox=({x},{y},{bw},{bh}) image=({w},{h}). Skipping.")
                return None

            # Use clamped coordinates for SAM3
            x, y = x1_valid, y1_valid
            bw, bh = x2_valid - x1_valid, y2_valid - y1_valid

            # Convert to RGB PIL image
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                logger.error(f"Unexpected image shape: {image.shape}")
                return None

            pil_image = Image.fromarray(image_rgb)

            # SAM3 uses box format [x1, y1, x2, y2] with labels (1=positive, 0=negative)
            input_box = [x, y, x + bw, y + bh]
            input_boxes = [[input_box]]  # [batch, num_boxes, 4]
            input_boxes_labels = [[1]]    # 1 = positive box

            # Process with SAM3
            inputs = self.sam3_processor(
                images=pil_image,
                input_boxes=input_boxes,
                input_boxes_labels=input_boxes_labels,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.sam3_model(**inputs)

            # Post-process using SAM3's instance segmentation method
            results = self.sam3_processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist()
            )

            if len(results) > 0 and len(results[0].get("masks", [])) > 0:
                # Get best mask (highest score)
                masks = results[0]["masks"]
                scores = results[0].get("scores", [])

                if len(scores) > 0:
                    best_idx = scores.index(max(scores))
                    mask = masks[best_idx]
                else:
                    mask = masks[0]

                # Convert to numpy - mask is already a tensor of shape (H, W)
                if hasattr(mask, 'cpu'):
                    mask_np = mask.cpu().numpy().astype(np.uint8) * 255
                else:
                    mask_np = np.array(mask).astype(np.uint8) * 255

                # Ensure correct size
                if mask_np.shape != (h, w):
                    mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

                return mask_np

            logger.warning("SAM3 returned no masks")
            return None

        except Exception as e:
            logger.error(f"SAM3 segmentation failed: {e}")
            return None

    async def segment_with_sam3(
        self,
        image: np.ndarray,
        bbox: List[float]
    ) -> Optional[np.ndarray]:
        """
        Use SAM3 to segment an object within a bounding box.
        Runs in thread pool to avoid blocking the event loop.

        Args:
            image: Image as numpy array (BGR or RGB)
            bbox: Bounding box [x, y, width, height]

        Returns:
            Binary mask as numpy array, or None if failed
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._segment_with_sam3_sync,
            image,
            bbox
        )

    def _segment_with_sam3_text_prompt_sync(
        self,
        image: np.ndarray,
        class_name: str,
        min_area: int = 100,
        return_all_instances: bool = False
    ) -> Optional[np.ndarray]:
        """
        Segment object using SAM3 with text prompt (class name).
        Synchronous implementation for thread pool execution.

        Uses SAM3's native Promptable Concept Segmentation (PCS) to detect
        and segment ALL instances of an object class using only a text description.

        Args:
            image: Input image (HxWx3, BGR format from OpenCV)
            class_name: Name of object class (e.g., "fish", "bottle", "person")
            min_area: Minimum mask area to accept
            return_all_instances: If True, return List[SAM3Instance] instead of single mask
                                 (NEW - for deduplication and multi-instance matching)

        Returns:
            If return_all_instances=False (default):
                Binary mask (HxW, uint8, 0-255) of the largest detected instance,
                or None if segmentation fails
            If return_all_instances=True:
                List[SAM3Instance] with all detected instances,
                or None if segmentation fails
        """
        if not self.sam3_available:
            logger.warning("SAM3 not available for text prompt segmentation")
            return None

        try:
            import torch
            from PIL import Image as PILImage

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image
            pil_image = PILImage.fromarray(image_rgb)

            # Preprocess class name for better SAM3 recognition
            # Convert technical names like "Plastic_Debris" to natural language "plastic debris"
            processed_class_name = class_name.replace("_", " ").lower().strip()

            # Try with original name first, then with first word only as fallback
            text_prompts_to_try = [processed_class_name]

            # Add simpler fallback (first word only) if name has multiple words
            words = processed_class_name.split()
            if len(words) > 1:
                text_prompts_to_try.append(words[0])  # e.g., "plastic debris" → "plastic"

            logger.debug(f"SAM3 text prompt: '{class_name}' → trying prompts: {text_prompts_to_try}")

            # Try each text prompt until one succeeds
            results = None
            successful_prompt = None

            for prompt in text_prompts_to_try:
                # Prepare inputs with text prompt
                inputs = self.sam3_processor(
                    images=pil_image,
                    text=prompt,
                    return_tensors="pt"
                ).to(self.device)

                # Run inference
                with torch.no_grad():
                    outputs = self.sam3_model(**inputs)

                # Post-process results to get instance masks
                temp_results = self.sam3_processor.post_process_instance_segmentation(
                    outputs,
                    threshold=0.4,  # Lower confidence threshold for more detections
                    mask_threshold=0.4,  # Lower mask threshold for more permissive segmentation
                    target_sizes=inputs.get("original_sizes").tolist()
                )[0]  # Get first (and only) image results

                # Check if any instances were found
                if "masks" in temp_results and len(temp_results["masks"]) > 0:
                    results = temp_results
                    successful_prompt = prompt
                    logger.debug(f"SAM3 text prompt: '{prompt}' found {len(results['masks'])} instance(s)")
                    break
                else:
                    logger.debug(f"SAM3 text prompt: '{prompt}' found no instances, trying next...")

            # If no prompt succeeded
            if results is None or "masks" not in results or len(results["masks"]) == 0:
                logger.warning(
                    f"SAM3 text prompt: No instances found with any prompt {text_prompts_to_try}. "
                    f"Original class: '{class_name}'. "
                    f"Suggestions: (1) Use bbox-guided SAM3 instead, (2) Check if object is visible in image, "
                    f"(3) Try more generic class names in your dataset."
                )
                return None

            # Get masks, boxes, and scores
            masks = results["masks"]  # List of binary masks (H, W)
            boxes = results.get("boxes", [])  # List of boxes (xyxy format)
            scores = results.get("scores", [])  # List of confidence scores

            logger.info(
                f"SAM3 text prompt: Found {len(masks)} instance(s) using prompt '{successful_prompt}' "
                f"(original class: '{class_name}') with scores {scores}"
            )

            # NEW: Return all instances mode (for deduplication)
            if return_all_instances:
                from app.utils import SAM3Instance

                instances = []
                for idx, mask in enumerate(masks):
                    # Convert mask to numpy if it's a tensor
                    if hasattr(mask, 'cpu'):
                        mask_np = mask.cpu().numpy()
                    else:
                        mask_np = np.array(mask)

                    # Ensure binary (0 or 255)
                    mask_np = (mask_np > 0.5).astype(np.uint8) * 255

                    # Calculate area
                    area = np.sum(mask_np > 0)

                    # Skip if too small
                    if area < min_area:
                        logger.debug(f"Mask {idx} too small: {area} < {min_area}")
                        continue

                    # Get bbox from mask
                    bbox = self.get_mask_bbox(mask_np)
                    if bbox is None:
                        logger.debug(f"Mask {idx} has no valid bbox")
                        continue

                    # Get score
                    score = scores[idx] if idx < len(scores) else 0.5

                    # Create SAM3Instance
                    instances.append(SAM3Instance(
                        mask=mask_np,
                        bbox=bbox,
                        score=score,
                        assigned=False
                    ))

                if not instances:
                    logger.warning(
                        f"SAM3 text prompt: All {len(masks)} mask(s) found with prompt '{successful_prompt}' "
                        f"(original class: '{class_name}') are below minimum area {min_area} pixels."
                    )
                    return None

                logger.info(
                    f"✓ SAM3 text prompt: Returning {len(instances)} instances for '{class_name}' "
                    f"(prompt: '{successful_prompt}')"
                )
                return instances

            # EXISTING: Return best single mask (backward compatible)
            best_mask = None
            best_area = 0
            best_score = 0

            for idx, mask in enumerate(masks):
                # Convert mask to numpy if it's a tensor
                if hasattr(mask, 'cpu'):
                    mask_np = mask.cpu().numpy()
                else:
                    mask_np = np.array(mask)

                # Ensure binary (0 or 255)
                mask_np = (mask_np > 0.5).astype(np.uint8) * 255

                # Calculate area
                area = np.sum(mask_np > 0)

                # Skip if too small
                if area < min_area:
                    logger.debug(f"Mask {idx} too small: {area} < {min_area}")
                    continue

                # Select mask with best combination of area and score
                score = scores[idx] if idx < len(scores) else 0.5

                # Prioritize by score, then area
                if score > best_score or (score == best_score and area > best_area):
                    best_mask = mask_np
                    best_area = area
                    best_score = score

            if best_mask is None:
                logger.warning(
                    f"SAM3 text prompt: All {len(masks)} mask(s) found with prompt '{successful_prompt}' "
                    f"(original class: '{class_name}') are below minimum area {min_area} pixels. "
                    f"Largest mask area: {max([np.sum(m > 0) if hasattr(m, '__iter__') else 0 for m in masks], default=0)}px. "
                    f"Consider: (1) Reducing min_area (currently {min_area}), (2) Using bbox-guided SAM3, "
                    f"(3) Checking if object is too small/distant in image."
                )
                return None

            logger.info(
                f"✓ SAM3 text prompt: Successfully segmented using prompt '{successful_prompt}' "
                f"(original class: '{class_name}', area={best_area}px, score={best_score:.3f})"
            )
            return best_mask

        except Exception as e:
            logger.error(f"SAM3 text prompt segmentation failed for '{class_name}': {e}", exc_info=True)
            return None

    async def segment_with_sam3_text_prompt(
        self,
        image: np.ndarray,
        class_name: str,
        min_area: int = 100
    ) -> Optional[np.ndarray]:
        """
        Segment object using SAM3 with text prompt (class name).
        Async wrapper that runs in thread pool to avoid blocking.

        Args:
            image: Input image (HxWx3, BGR or RGB)
            class_name: Name of object class (e.g., "fish", "bottle")
            min_area: Minimum mask area in pixels to accept

        Returns:
            Binary mask as numpy array (HxW, uint8, 0-255), or None if failed

        Example:
            >>> mask = await extractor.segment_with_sam3_text_prompt(
            ...     image=img,
            ...     class_name="fish",
            ...     min_area=100
            ... )
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._segment_with_sam3_text_prompt_sync,
            image,
            class_name,
            min_area
        )

    # =========================================================================
    # IMAGE CROPPING
    # =========================================================================

    @staticmethod
    def crop_with_mask(
        image: np.ndarray,
        mask: np.ndarray,
        bbox: List[float],
        padding: int = 5
    ) -> Tuple[Optional[np.ndarray], List[int]]:
        """
        Crop image using mask and add alpha channel.

        Args:
            image: Source image (BGR)
            mask: Binary mask (0 or 255)
            bbox: Bounding box [x, y, width, height]
            padding: Pixels of padding around object

        Returns:
            Tuple of (RGBA cropped image, new bbox [x, y, w, h]) or (None, []) on error
        """
        if image is None or image.size == 0:
            logger.error("crop_with_mask: received empty or None image")
            return None, []

        h, w = image.shape[:2]
        x, y, bw, bh = [int(v) for v in bbox]

        # Compute intersection of bbox with image bounds
        # bbox region: [x, y] to [x+bw, y+bh]
        # image region: [0, 0] to [w, h]
        x1_bbox, y1_bbox = x, y
        x2_bbox, y2_bbox = x + bw, y + bh

        # Intersection
        x1_valid = max(0, x1_bbox)
        y1_valid = max(0, y1_bbox)
        x2_valid = min(w, x2_bbox)
        y2_valid = min(h, y2_bbox)

        # Check if there's a valid intersection (bbox overlaps with image)
        if x1_valid >= x2_valid or y1_valid >= y2_valid:
            logger.info(f"crop_with_mask: bbox has no overlap with image - bbox=({x},{y},{bw},{bh}) image=({w},{h}). Skipping annotation (data quality issue).")
            return None, []

        # Use the valid (clamped) bbox region
        valid_bw = x2_valid - x1_valid
        valid_bh = y2_valid - y1_valid

        # Apply padding around the valid region, clamped to image bounds
        x1 = max(0, x1_valid - padding)
        y1 = max(0, y1_valid - padding)
        x2 = min(w, x2_valid + padding)
        y2 = min(h, y2_valid + padding)

        # Final validation (should always pass after above logic)
        if x1 >= x2 or y1 >= y2:
            logger.error(f"crop_with_mask: invalid crop after padding x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            return None, []

        # Crop image and mask
        cropped = image[y1:y2, x1:x2].copy()
        cropped_mask = mask[y1:y2, x1:x2].copy()

        # Validate cropped image
        if cropped is None or cropped.size == 0:
            logger.error(f"crop_with_mask: cropped image is empty for bbox={bbox}")
            return None, []

        # Convert to RGBA
        try:
            if len(cropped.shape) == 2:
                # Grayscale
                rgba = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGRA)
            elif cropped.shape[2] == 3:
                # BGR
                rgba = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
            else:
                # Already has alpha
                rgba = cropped.copy()
        except cv2.error as e:
            logger.error(f"crop_with_mask: cvtColor failed: {e}, cropped shape={cropped.shape}")
            return None, []

        # Apply mask to alpha channel
        rgba[:, :, 3] = cropped_mask

        return rgba, [x1, y1, x2 - x1, y2 - y1]

    @staticmethod
    def crop_bbox_only(
        image: np.ndarray,
        bbox: List[float],
        padding: int = 5
    ) -> Tuple[Optional[np.ndarray], List[int]]:
        """
        Crop image using only bounding box (no transparency).

        Args:
            image: Source image
            bbox: Bounding box [x, y, width, height]
            padding: Pixels of padding

        Returns:
            Tuple of (cropped image, new bbox) or (None, []) on error
        """
        if image is None or image.size == 0:
            logger.error("crop_bbox_only: received empty or None image")
            return None, []

        h, w = image.shape[:2]
        x, y, bw, bh = [int(v) for v in bbox]

        # Compute intersection of bbox with image bounds
        # bbox region: [x, y] to [x+bw, y+bh]
        # image region: [0, 0] to [w, h]
        x1_bbox, y1_bbox = x, y
        x2_bbox, y2_bbox = x + bw, y + bh

        # Intersection
        x1_valid = max(0, x1_bbox)
        y1_valid = max(0, y1_bbox)
        x2_valid = min(w, x2_bbox)
        y2_valid = min(h, y2_bbox)

        # Check if there's a valid intersection (bbox overlaps with image)
        if x1_valid >= x2_valid or y1_valid >= y2_valid:
            logger.info(f"crop_bbox_only: bbox has no overlap with image - bbox=({x},{y},{bw},{bh}) image=({w},{h}). Skipping annotation (data quality issue).")
            return None, []

        # Use the valid (clamped) bbox region
        valid_bw = x2_valid - x1_valid
        valid_bh = y2_valid - y1_valid

        # Apply padding around the valid region, clamped to image bounds
        x1 = max(0, x1_valid - padding)
        y1 = max(0, y1_valid - padding)
        x2 = min(w, x2_valid + padding)
        y2 = min(h, y2_valid + padding)

        # Final validation (should always pass after above logic)
        if x1 >= x2 or y1 >= y2:
            logger.error(f"crop_bbox_only: invalid crop after padding x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            return None, []

        cropped = image[y1:y2, x1:x2].copy()

        # Validate cropped image
        if cropped is None or cropped.size == 0:
            logger.error(f"crop_bbox_only: cropped image is empty for bbox={bbox}")
            return None, []

        # Convert to RGBA with full opacity
        try:
            if len(cropped.shape) == 2:
                rgba = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGRA)
            elif cropped.shape[2] == 3:
                rgba = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
            else:
                rgba = cropped.copy()
        except cv2.error as e:
            logger.error(f"crop_bbox_only: cvtColor failed: {e}, cropped shape={cropped.shape}")
            return None, []

        return rgba, [x1, y1, x2 - x1, y2 - y1]

    # =========================================================================
    # COCO JSON GENERATION
    # =========================================================================

    @staticmethod
    def create_individual_coco(
        annotation: Dict[str, Any],
        category: Dict[str, Any],
        image_shape: Tuple[int, int],
        original_image_filename: str
    ) -> Dict[str, Any]:
        """
        Create individual COCO JSON for an extracted object.

        Args:
            annotation: Original COCO annotation
            category: Category information
            image_shape: Shape of extracted image (height, width)
            original_image_filename: Original source image filename

        Returns:
            COCO format dictionary for the individual object
        """
        h, w = image_shape[:2]

        return {
            "info": {
                "description": "Extracted object from COCO dataset",
                "version": "1.0",
                "extracted_from": original_image_filename,
                "original_annotation_id": annotation.get("id"),
                "extraction_date": datetime.now().isoformat()
            },
            "licenses": [],
            "images": [{
                "id": 1,
                "file_name": f"{annotation['id']}.png",
                "width": w,
                "height": h
            }],
            "annotations": [{
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": [[0, 0, w, 0, w, h, 0, h]]  # Full image polygon
            }],
            "categories": [{
                "id": 1,
                "name": category.get("name", "object"),
                "supercategory": category.get("supercategory", "none")
            }]
        }

    # =========================================================================
    # DATASET ANALYSIS
    # =========================================================================

    def analyze_dataset(
        self,
        coco_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze a COCO dataset to determine annotation types.

        Args:
            coco_data: COCO format dataset dictionary

        Returns:
            Analysis results dictionary
        """
        categories = {c["id"]: c for c in coco_data.get("categories", [])}
        annotations = coco_data.get("annotations", [])
        images = coco_data.get("images", [])

        # Count by category and type
        category_stats = {}
        for cat_id, cat in categories.items():
            category_stats[cat_id] = {
                "id": cat_id,
                "name": cat["name"],
                "count": 0,
                "with_segmentation": 0,
                "bbox_only": 0
            }

        total_with_seg = 0
        total_bbox_only = 0
        sample_annotation = None

        for ann in annotations:
            cat_id = ann.get("category_id")
            if cat_id not in category_stats:
                continue

            ann_type = self.detect_annotation_type(ann)
            category_stats[cat_id]["count"] += 1

            if ann_type in [AnnotationType.POLYGON, AnnotationType.RLE]:
                category_stats[cat_id]["with_segmentation"] += 1
                total_with_seg += 1
            else:
                category_stats[cat_id]["bbox_only"] += 1
                total_bbox_only += 1

            if sample_annotation is None:
                sample_annotation = ann

        # Determine recommendation
        if total_bbox_only == 0:
            recommendation = "use_masks"
        elif total_with_seg == 0:
            recommendation = "use_sam3"
        else:
            recommendation = "mixed"

        return {
            "total_images": len(images),
            "total_annotations": len(annotations),
            "annotations_with_segmentation": total_with_seg,
            "annotations_bbox_only": total_bbox_only,
            "categories": list(category_stats.values()),
            "recommendation": recommendation,
            "sample_annotation": sample_annotation
        }

    # =========================================================================
    # PATH RESOLUTION
    # =========================================================================

    @staticmethod
    def resolve_image_path(images_base_dir: str, coco_image: Dict[str, Any]) -> Path:
        """
        Resolve full path to an image.

        Args:
            images_base_dir: Base directory for images
            coco_image: COCO image entry with file_name

        Returns:
            Full path to image file
        """
        file_name = coco_image.get("file_name", "")

        # Handle empty file_name
        if not file_name:
            logger.warning(f"Empty file_name in COCO image entry: {coco_image}")
            return Path("")

        # Normalize path separators (handle Windows vs Unix)
        file_name = file_name.replace("\\", "/")

        # If absolute path, use directly
        if os.path.isabs(file_name):
            return Path(file_name)

        # Combine with base directory
        base_path = Path(images_base_dir)
        full_path = base_path / file_name

        # Log for debugging
        if not full_path.exists():
            logger.warning(f"Image path does not exist: {full_path} (base: {images_base_dir}, file_name: {file_name})")

        return full_path

    # =========================================================================
    # SINGLE OBJECT EXTRACTION
    # =========================================================================

    @staticmethod
    def get_mask_bbox(mask: np.ndarray) -> Optional[List[int]]:
        """
        Get the bounding box of non-zero pixels in a mask.

        Args:
            mask: Binary mask (0 or 255)

        Returns:
            Bounding box [x, y, width, height] or None if mask is empty
        """
        # Find non-zero pixels
        coords = np.where(mask > 0)
        if len(coords[0]) == 0 or len(coords[1]) == 0:
            return None

        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()

        return [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]

    def _extract_single_object_sync(
        self,
        image: np.ndarray,
        annotation: Dict[str, Any],
        category_name: str,
        use_sam3: bool = False,
        padding: int = 5,
        force_bbox_only: bool = False,
        force_sam3_resegmentation: bool = False,
        force_sam3_text_prompt: bool = False
    ) -> Dict[str, Any]:
        """
        Synchronous version of extract_single_object (runs in thread pool).

        Args:
            image: Source image as numpy array
            annotation: COCO annotation for the object
            category_name: Name of the category
            use_sam3: Whether to use SAM3 for bbox-only annotations
            padding: Padding around extracted object
            force_bbox_only: Force extraction using only bbox, ignore existing masks
            force_sam3_resegmentation: Force SAM3 to regenerate masks even if polygon/RLE exist
            force_sam3_text_prompt: Use only class label with SAM3, ignore bbox and masks

        Returns:
            Dictionary with extraction results
        """
        # Validate image
        if image is None:
            return {"success": False, "error": "Image is None"}
        if not hasattr(image, 'shape') or len(image.shape) < 2:
            return {"success": False, "error": f"Invalid image shape: {getattr(image, 'shape', 'no shape')}"}
        if image.size == 0:
            return {"success": False, "error": "Image is empty (size=0)"}

        h, w = image.shape[:2]
        bbox = annotation.get("bbox", [0, 0, w, h])

        # Force SAM3 text prompt mode - ignore everything, use only class label
        if force_sam3_text_prompt and use_sam3 and self.sam3_available:
            logger.debug(f"Force text prompt mode: using only class label '{category_name}' with SAM3")

            # Use SAM3 with class name as text prompt on full image
            # Lower min_area (50px) to be more permissive with small objects
            mask = self._segment_with_sam3_text_prompt_sync(
                image=image,
                class_name=category_name,
                min_area=50
            )

            if mask is not None:
                # Calculate bbox from generated mask
                y_indices, x_indices = np.where(mask > 0)
                if len(x_indices) > 0 and len(y_indices) > 0:
                    x = int(x_indices.min())
                    y = int(y_indices.min())
                    bw = int(x_indices.max() - x + 1)
                    bh = int(y_indices.max() - y + 1)
                    bbox = [x, y, bw, bh]

                    # Crop with mask
                    cropped, new_bbox = self.crop_with_mask(image, mask, bbox, padding)

                    if cropped is None:
                        return {"success": False, "error": f"Failed to crop with text-prompted mask"}

                    # Calculate mask coverage
                    bbox_area = bw * bh
                    mask_area = np.sum(mask[y:y+bh, x:x+bw] > 0) if bbox_area > 0 else 0
                    mask_coverage = mask_area / bbox_area if bbox_area > 0 else 0.0

                    # Encode image
                    success, encoded = cv2.imencode('.png', cropped)
                    if not success:
                        return {"success": False, "error": "Failed to encode image"}
                    cropped_base64 = base64.b64encode(encoded.tobytes()).decode('utf-8')

                    # Encode mask
                    x1, y1, mw, mh = new_bbox
                    cropped_mask = mask[y1:y1+mh, x1:x1+mw]
                    success_mask, encoded_mask = cv2.imencode('.png', cropped_mask)
                    mask_base64 = base64.b64encode(encoded_mask.tobytes()).decode('utf-8') if success_mask else None

                    return {
                        "success": True,
                        "cropped_image_base64": cropped_base64,
                        "mask_base64": mask_base64,
                        "annotation_type": AnnotationType.BBOX_ONLY.value,
                        "method_used": ExtractionMethod.SAM3_TEXT_PROMPT.value,
                        "original_bbox": bbox,
                        "extracted_size": [cropped.shape[1], cropped.shape[0]],
                        "mask_coverage": mask_coverage
                    }

            # If SAM3 text prompt failed
            logger.warning(f"SAM3 text prompt failed for '{category_name}' - cannot extract")
            return {"success": False, "error": "Text-prompted segmentation failed"}

        # Validate bbox
        if not bbox or len(bbox) < 4:
            return {"success": False, "error": f"Invalid bbox: {bbox}"}

        # Ensure bbox values are valid
        bx, by, bw, bh = bbox[:4]
        if bw <= 0 or bh <= 0:
            return {"success": False, "error": f"Invalid bbox dimensions: {bbox}"}

        # Check if annotation lacks both bbox and segmentation (classification-only)
        # In this case, try SAM3 with text prompt to segment the object
        has_real_bbox = annotation.get("bbox") is not None and annotation.get("bbox") != [0, 0, w, h]
        has_segmentation = annotation.get("segmentation") is not None and len(annotation.get("segmentation", [])) > 0

        if not has_real_bbox and not has_segmentation:
            # Annotation only has category_id - try text-prompted segmentation
            if use_sam3 and self.sam3_available:
                logger.debug(f"No bbox/segmentation found - trying SAM3 text prompt for '{category_name}'")

                # Attempt segmentation using class name as text prompt
                # Lower min_area (50px) to be more permissive with small objects
                mask = self._segment_with_sam3_text_prompt_sync(
                    image=image,
                    class_name=category_name,
                    min_area=50
                )

                if mask is not None:
                    # Calculate bbox from generated mask
                    y_indices, x_indices = np.where(mask > 0)
                    if len(x_indices) > 0 and len(y_indices) > 0:
                        x = int(x_indices.min())
                        y = int(y_indices.min())
                        bw = int(x_indices.max() - x + 1)
                        bh = int(y_indices.max() - y + 1)
                        bbox = [x, y, bw, bh]

                        # Crop with mask
                        cropped, new_bbox = self.crop_with_mask(image, mask, bbox, padding)

                        if cropped is None:
                            return {"success": False, "error": f"Failed to crop with text-prompted mask"}

                        # Calculate mask coverage
                        bbox_area = bw * bh
                        mask_area = np.sum(mask[y:y+bh, x:x+bw] > 0) if bbox_area > 0 else 0
                        mask_coverage = mask_area / bbox_area if bbox_area > 0 else 0.0

                        # Encode image
                        success, encoded = cv2.imencode('.png', cropped)
                        if not success:
                            return {"success": False, "error": "Failed to encode image"}
                        cropped_base64 = base64.b64encode(encoded.tobytes()).decode('utf-8')

                        # Encode mask
                        x1, y1, mw, mh = new_bbox
                        cropped_mask = mask[y1:y1+mh, x1:x1+mw]
                        success_mask, encoded_mask = cv2.imencode('.png', cropped_mask)
                        mask_base64 = base64.b64encode(encoded_mask.tobytes()).decode('utf-8') if success_mask else None

                        return {
                            "success": True,
                            "cropped_image_base64": cropped_base64,
                            "mask_base64": mask_base64,
                            "annotation_type": AnnotationType.BBOX_ONLY.value,
                            "method_used": ExtractionMethod.SAM3_TEXT_PROMPT.value,
                            "original_bbox": bbox,
                            "extracted_size": [cropped.shape[1], cropped.shape[0]],
                            "mask_coverage": mask_coverage
                        }

                # If SAM3 text prompt failed or returned no mask
                logger.warning(f"SAM3 text prompt segmentation failed for '{category_name}' - skipping object")
                return {"success": False, "error": "Text-prompted segmentation not available or failed"}
            else:
                # SAM3 not available and no bbox/segmentation
                logger.warning(f"Cannot extract '{category_name}' without bbox/mask and SAM3 unavailable")
                return {"success": False, "error": "No bbox/segmentation and SAM3 not available"}

        # Force bbox-only extraction if requested (ignore masks)
        if force_bbox_only:
            logger.debug(f"Force bbox-only mode: skipping mask extraction for {category_name}")
            cropped, new_bbox = self.crop_bbox_only(image, bbox, padding)

            if cropped is None:
                return {"success": False, "error": f"Failed to crop image for bbox={bbox}"}

            # Encode to base64
            success, encoded = cv2.imencode('.png', cropped)
            if not success:
                return {"success": False, "error": "Failed to encode image"}

            cropped_base64 = base64.b64encode(encoded.tobytes()).decode('utf-8')

            return {
                "success": True,
                "cropped_image_base64": cropped_base64,
                "mask_base64": None,
                "annotation_type": AnnotationType.BBOX_ONLY.value,
                "method_used": ExtractionMethod.BBOX_CROP.value,
                "original_bbox": bbox,
                "extracted_size": [cropped.shape[1], cropped.shape[0]],
                "mask_coverage": 1.0
            }

        # Force SAM3 resegmentation if requested (regenerate masks even if they exist)
        if force_sam3_resegmentation and use_sam3 and self.sam3_available:
            # Check if annotation has existing segmentation (polygon or RLE)
            has_segmentation = annotation.get("segmentation") is not None and len(annotation.get("segmentation", [])) > 0

            if has_segmentation:
                logger.debug(f"Force SAM3 resegmentation: ignoring existing mask for {category_name}")

                # Use SAM3 with bbox to regenerate mask
                mask = self._segment_with_sam3_sync(image, bbox)

                if mask is not None:
                    # Crop with new SAM3-generated mask
                    cropped, new_bbox = self.crop_with_mask(image, mask, bbox, padding)

                    if cropped is None:
                        return {"success": False, "error": f"Failed to crop with SAM3-regenerated mask"}

                    # Calculate mask coverage
                    x, y, bw, bh = [int(v) for v in bbox]
                    bbox_area = bw * bh
                    mask_area = np.sum(mask[y:y+bh, x:x+bw] > 0) if bbox_area > 0 else 0
                    mask_coverage = mask_area / bbox_area if bbox_area > 0 else 0.0

                    # Encode image
                    success, encoded = cv2.imencode('.png', cropped)
                    if not success:
                        return {"success": False, "error": "Failed to encode image"}
                    cropped_base64 = base64.b64encode(encoded.tobytes()).decode('utf-8')

                    # Encode mask
                    x1, y1, mw, mh = new_bbox
                    cropped_mask = mask[y1:y1+mh, x1:x1+mw]
                    success_mask, encoded_mask = cv2.imencode('.png', cropped_mask)
                    mask_base64 = base64.b64encode(encoded_mask.tobytes()).decode('utf-8') if success_mask else None

                    return {
                        "success": True,
                        "cropped_image_base64": cropped_base64,
                        "mask_base64": mask_base64,
                        "annotation_type": AnnotationType.BBOX_ONLY.value,
                        "method_used": ExtractionMethod.SAM3_FROM_BBOX.value,
                        "original_bbox": bbox,
                        "extracted_size": [cropped.shape[1], cropped.shape[0]],
                        "mask_coverage": mask_coverage
                    }
                else:
                    logger.warning(f"SAM3 resegmentation failed for {category_name} - falling back to original mask")
                    # Continue with normal flow to use original mask

        ann_type = self.detect_annotation_type(annotation)

        mask = None
        method = ExtractionMethod.BBOX_CROP

        # Get or generate mask
        if ann_type == AnnotationType.POLYGON:
            mask = self.polygon_to_mask(annotation["segmentation"], h, w)
            method = ExtractionMethod.POLYGON_MASK

        elif ann_type == AnnotationType.RLE:
            mask = self.rle_to_mask(annotation["segmentation"], h, w)
            method = ExtractionMethod.RLE_MASK

        elif use_sam3 and self.sam3_available:
            # Call sync version directly since we're already in a thread
            mask = self._segment_with_sam3_sync(image, bbox)
            if mask is not None:
                method = ExtractionMethod.SAM3_FROM_BBOX

        # Crop image
        if mask is not None:
            # IMPORTANT: When we have a mask from polygon/RLE, use the actual mask bbox
            # instead of the annotation bbox, to ensure correct alignment
            # This fixes issues where annotation bbox doesn't match the segmentation
            if method in [ExtractionMethod.POLYGON_MASK, ExtractionMethod.RLE_MASK]:
                mask_bbox = self.get_mask_bbox(mask)
                if mask_bbox is not None:
                    # Use mask's actual bounding box for cropping
                    bbox = mask_bbox
                    logger.debug(f"Using mask-derived bbox {mask_bbox} instead of annotation bbox {annotation.get('bbox')}")

            cropped, new_bbox = self.crop_with_mask(image, mask, bbox, padding)

            if cropped is None:
                return {"success": False, "error": f"Failed to crop image with mask for bbox={bbox}"}

            # Calculate mask coverage
            x, y, bw, bh = [int(v) for v in bbox]
            bbox_area = bw * bh
            mask_area = np.sum(mask[y:y+bh, x:x+bw] > 0) if bbox_area > 0 else 0
            mask_coverage = mask_area / bbox_area if bbox_area > 0 else 0.0
        else:
            cropped, new_bbox = self.crop_bbox_only(image, bbox, padding)

            if cropped is None:
                return {"success": False, "error": f"Failed to crop image for bbox={bbox}"}

            mask_coverage = 1.0

        # Encode to base64
        success, encoded = cv2.imencode('.png', cropped)
        if not success:
            return {"success": False, "error": "Failed to encode image"}

        cropped_base64 = base64.b64encode(encoded.tobytes()).decode('utf-8')

        # Encode mask if available
        mask_base64 = None
        if mask is not None:
            x1, y1, mw, mh = new_bbox
            cropped_mask = mask[y1:y1+mh, x1:x1+mw]
            success, encoded_mask = cv2.imencode('.png', cropped_mask)
            if success:
                mask_base64 = base64.b64encode(encoded_mask.tobytes()).decode('utf-8')

        return {
            "success": True,
            "cropped_image_base64": cropped_base64,
            "mask_base64": mask_base64,
            "annotation_type": ann_type.value,
            "method_used": method.value,
            "original_bbox": bbox,
            "extracted_size": [cropped.shape[1], cropped.shape[0]],
            "mask_coverage": mask_coverage
        }

    async def extract_single_object(
        self,
        image: np.ndarray,
        annotation: Dict[str, Any],
        category_name: str,
        use_sam3: bool = False,
        padding: int = 5,
        force_bbox_only: bool = False,
        force_sam3_resegmentation: bool = False,
        force_sam3_text_prompt: bool = False
    ) -> Dict[str, Any]:
        """
        Extract a single object from an image.
        Runs in thread pool to avoid blocking the event loop.

        Args:
            image: Source image as numpy array
            annotation: COCO annotation for the object
            category_name: Name of the category
            use_sam3: Whether to use SAM3 for bbox-only annotations
            padding: Padding around extracted object
            force_bbox_only: Force extraction using only bbox, ignore existing masks
            force_sam3_resegmentation: Force SAM3 to regenerate masks even if polygon/RLE exist
            force_sam3_text_prompt: Use only class label with SAM3, ignore bbox and masks

        Returns:
            Dictionary with extraction results
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(
                self._extract_single_object_sync,
                image=image,
                annotation=annotation,
                category_name=category_name,
                use_sam3=use_sam3,
                padding=padding,
                force_bbox_only=force_bbox_only,
                force_sam3_resegmentation=force_sam3_resegmentation,
                force_sam3_text_prompt=force_sam3_text_prompt
            )
        )

    # =========================================================================
    # BATCH EXTRACTION
    # =========================================================================

    def _extract_from_dataset_sync(
        self,
        coco_data: Dict[str, Any],
        images_dir: str,
        output_dir: str,
        categories_to_extract: List[str] = None,
        use_sam3_for_bbox: bool = True,
        force_bbox_only: bool = False,
        force_sam3_resegmentation: bool = False,
        force_sam3_text_prompt: bool = False,
        padding: int = 5,
        min_object_area: int = 100,
        save_individual_coco: bool = True,
        progress_callback: Callable[[Dict[str, Any]], None] = None,
        deduplication_config: Optional['DeduplicationConfig'] = None
    ) -> Dict[str, Any]:
        """
        Extract all objects from a COCO dataset with deduplication support.

        Args:
            coco_data: COCO format dataset
            images_dir: Directory containing source images
            output_dir: Output directory for extracted objects
            categories_to_extract: List of category names to extract (None = all)
            use_sam3_for_bbox: Use SAM3 for bbox-only annotations
            force_bbox_only: Force extraction using only bbox, ignore existing masks
            force_sam3_resegmentation: Force SAM3 to regenerate masks even if polygon/RLE exist
            force_sam3_text_prompt: Use only class label with SAM3, ignore bbox and masks
            padding: Padding around objects
            min_object_area: Minimum area to extract
            save_individual_coco: Save JSON for each object
            progress_callback: Callback function for progress updates
            deduplication_config: Configuration for deduplication (None = default enabled)

        Returns:
            Extraction results dictionary with deduplication stats
        """
        start_time = datetime.now()

        # Build lookups
        categories = {c["id"]: c for c in coco_data.get("categories", [])}
        images = {i["id"]: i for i in coco_data.get("images", [])}
        annotations = coco_data.get("annotations", [])

        # Filter categories
        if categories_to_extract:
            category_ids = {
                cid for cid, cat in categories.items()
                if cat["name"] in categories_to_extract
            }
        else:
            category_ids = set(categories.keys())

        # Create output directories
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for cat in categories.values():
            if cat["id"] in category_ids:
                (output_path / cat["name"]).mkdir(exist_ok=True)

        # Process annotations
        results = {
            "extracted": 0,
            "failed": 0,
            "by_category": {categories[cid]["name"]: 0 for cid in category_ids},
            "by_method": {},
            "errors": [],
            "extracted_files": [],
            "deduplication_stats": {
                "duplicates_prevented": 0,
                "enabled": deduplication_config.enabled if deduplication_config else True
            }
        }

        # Group annotations differently based on mode
        if force_sam3_text_prompt:
            # Text prompt mode: group by (image_id, category_name) for batch processing
            annotations_grouped = {}
            for ann in annotations:
                if ann.get("category_id") not in category_ids:
                    continue

                # Check minimum area
                bbox = ann.get("bbox", [0, 0, 0, 0])
                if len(bbox) >= 4 and bbox[2] * bbox[3] < min_object_area:
                    continue

                img_id = ann.get("image_id")
                cat_name = categories[ann["category_id"]]["name"]
                key = (img_id, cat_name)

                if key not in annotations_grouped:
                    annotations_grouped[key] = []
                annotations_grouped[key].append(ann)
        else:
            # Bbox/mask mode: group by image_id only
            annotations_grouped = {}
            for ann in annotations:
                if ann.get("category_id") not in category_ids:
                    continue

                # Check minimum area
                bbox = ann.get("bbox", [0, 0, 0, 0])
                if len(bbox) >= 4 and bbox[2] * bbox[3] < min_object_area:
                    continue

                img_id = ann.get("image_id")
                if img_id not in annotations_grouped:
                    annotations_grouped[img_id] = []
                annotations_grouped[img_id].append(ann)

        # Calculate total objects
        total_objects = sum(
            len(anns) if isinstance(anns, list) else 0
            for anns in annotations_grouped.values()
        )

        # Process each image (FINALLY USE current_image_id and current_image!)
        current_image = None
        current_image_id = None
        registry = None

        # Get unique image IDs
        if force_sam3_text_prompt:
            unique_img_ids = sorted(set(key[0] for key in annotations_grouped.keys()))
        else:
            unique_img_ids = sorted(annotations_grouped.keys())

        for img_id in unique_img_ids:
            # Load image once per image_id (reuse current_image)
            if current_image_id != img_id:
                current_image_id = img_id

                if img_id not in images:
                    # Fail all annotations for this image
                    if force_sam3_text_prompt:
                        failed_anns = [
                            ann for key, anns in annotations_grouped.items()
                            if key[0] == img_id
                            for ann in anns
                        ]
                    else:
                        failed_anns = annotations_grouped.get(img_id, [])

                    for ann in failed_anns:
                        results["failed"] += 1
                        results["errors"].append(f"Image ID {img_id} not found in dataset")
                    continue

                img_info = images[img_id]
                img_path = self.resolve_image_path(images_dir, img_info)

                # Load image
                if not img_path.exists():
                    error_msg = f"Image not found: {img_path} (file_name in JSON: {img_info.get('file_name', '?')}, images_dir: {images_dir})"
                    logger.error(error_msg)

                    if force_sam3_text_prompt:
                        failed_anns = [
                            ann for key, anns in annotations_grouped.items()
                            if key[0] == img_id
                            for ann in anns
                        ]
                    else:
                        failed_anns = annotations_grouped.get(img_id, [])

                    for ann in failed_anns:
                        results["failed"] += 1
                        results["errors"].append(error_msg)
                    continue

                try:
                    current_image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                except Exception as e:
                    error_msg = f"Exception loading image {img_path}: {e}"
                    logger.error(error_msg)

                    if force_sam3_text_prompt:
                        failed_anns = [
                            ann for key, anns in annotations_grouped.items()
                            if key[0] == img_id
                            for ann in anns
                        ]
                    else:
                        failed_anns = annotations_grouped.get(img_id, [])

                    for ann in failed_anns:
                        results["failed"] += 1
                        results["errors"].append(error_msg)
                    continue

                if current_image is None:
                    error_msg = f"cv2.imread returned None for: {img_path}"
                    logger.error(error_msg)

                    if force_sam3_text_prompt:
                        failed_anns = [
                            ann for key, anns in annotations_grouped.items()
                            if key[0] == img_id
                            for ann in anns
                        ]
                    else:
                        failed_anns = annotations_grouped.get(img_id, [])

                    for ann in failed_anns:
                        results["failed"] += 1
                        results["errors"].append(error_msg)
                    continue

                logger.debug(f"Loaded image {img_path}: shape={current_image.shape}, dtype={current_image.dtype}")

                # Create registry for this image
                if deduplication_config is None:
                    # Default: enabled with IoU=0.7
                    from app.models.extraction_schemas import DeduplicationConfig
                    deduplication_config = DeduplicationConfig(enabled=True, iou_threshold=0.7)

                if deduplication_config.enabled:
                    from app.utils import ExtractionRegistry
                    registry = ExtractionRegistry(
                        iou_threshold=deduplication_config.iou_threshold,
                        cross_category_dedup=deduplication_config.cross_category_dedup,
                        matching_strategy=deduplication_config.matching_strategy
                    )
                    logger.info(
                        f"Created ExtractionRegistry for image {img_id} "
                        f"(iou_threshold={deduplication_config.iou_threshold}, "
                        f"cross_category={deduplication_config.cross_category_dedup})"
                    )
                else:
                    registry = None

            # Process based on mode
            if force_sam3_text_prompt:
                self._process_text_prompt_mode(
                    image=current_image,
                    img_id=img_id,
                    img_info=images[img_id],
                    annotations_grouped=annotations_grouped,
                    registry=registry,
                    results=results,
                    categories=categories,
                    output_path=output_path,
                    padding=padding,
                    min_object_area=min_object_area,
                    save_individual_coco=save_individual_coco,
                    progress_callback=progress_callback
                )
            else:
                self._process_bbox_mask_mode(
                    image=current_image,
                    img_id=img_id,
                    img_info=images[img_id],
                    annotations_grouped=annotations_grouped,
                    registry=registry,
                    results=results,
                    categories=categories,
                    output_path=output_path,
                    use_sam3_for_bbox=use_sam3_for_bbox,
                    force_bbox_only=force_bbox_only,
                    force_sam3_resegmentation=force_sam3_resegmentation,
                    padding=padding,
                    save_individual_coco=save_individual_coco,
                    progress_callback=progress_callback
                )

            # Collect dedup stats from registry
            if registry:
                stats = registry.get_stats()
                results["deduplication_stats"]["duplicates_prevented"] += stats["duplicates_prevented"]

        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Save extraction summary
        summary = ExtractionSummary(
            extraction_date=start_time.isoformat(),
            source_dataset=coco_data.get("info", {}).get("description", "Unknown"),
            images_dir=images_dir,
            output_dir=output_dir,
            total_objects_extracted=results["extracted"],
            categories={
                name: {
                    "count": count,
                    "method": "mixed" if len(results["by_method"]) > 1 else list(results["by_method"].keys())[0] if results["by_method"] else "none"
                }
                for name, count in results["by_category"].items()
                if count > 0
            },
            failed_extractions=results["failed"],
            errors=results["errors"][:100],  # Limit errors in summary
            processing_time_seconds=processing_time
        )

        summary_path = output_path / "extraction_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary.model_dump(), f, indent=2)

        results["summary_path"] = str(summary_path)
        results["processing_time_seconds"] = processing_time

        logger.info(
            f"Extraction complete: {results['extracted']} extracted, "
            f"{results['failed']} failed, "
            f"{results['deduplication_stats']['duplicates_prevented']} duplicates prevented"
        )

        return results

    def _process_text_prompt_mode(
        self,
        image: np.ndarray,
        img_id: int,
        img_info: Dict,
        annotations_grouped: Dict,
        registry: Optional['ExtractionRegistry'],
        results: Dict,
        categories: Dict,
        output_path: 'Path',
        padding: int,
        min_object_area: int,
        save_individual_coco: bool,
        progress_callback: Callable
    ) -> None:
        """
        Process annotations in text prompt mode with instance matching.

        Algorithm:
        1. Group annotations by category for this image
        2. For each category:
           - Check registry cache for SAM3 results
           - If not cached: Run SAM3 once → get N instances → cache
           - Match N instances to M annotations (one-to-one)
           - For each matched pair: check dedup, extract, register
           - For unmatched: FAIL with clear message

        Args:
            image: Loaded image
            img_id: Image ID
            img_info: Image metadata from COCO JSON
            annotations_grouped: Dict with (img_id, cat_name) keys
            registry: ExtractionRegistry or None
            results: Results dict to update
            categories: Category lookup dict
            output_path: Output directory Path
            padding: Padding pixels
            min_object_area: Minimum area threshold
            save_individual_coco: Whether to save individual JSON
            progress_callback: Progress callback function
        """
        from app.utils import match_instances_to_annotations, MatchingStrategy

        # Get all annotations for this image, grouped by category
        anns_by_category = {}
        for key, anns in annotations_grouped.items():
            if isinstance(key, tuple) and key[0] == img_id:
                cat_name = key[1]
                anns_by_category[cat_name] = anns

        # Process each category
        for cat_name, annotations in anns_by_category.items():
            logger.info(f"Processing category '{cat_name}': {len(annotations)} annotations (text prompt mode)")

            # Check cache
            instances = registry.get_sam3_instances(cat_name) if registry else None

            if instances is None:
                # Run SAM3 ONCE for this category
                instances = self._segment_with_sam3_text_prompt_sync(
                    image=image,
                    class_name=cat_name,
                    min_area=min_object_area,
                    return_all_instances=True  # Get ALL instances
                )

                if instances is None or len(instances) == 0:
                    # SAM3 failed - fail all annotations
                    logger.warning(f"SAM3 text prompt failed for '{cat_name}' - marking {len(annotations)} annotations as failed")
                    for ann in annotations:
                        results["failed"] += 1
                        results["errors"].append(
                            f"SAM3 text prompt returned no instances for '{cat_name}' (ann_id={ann['id']})"
                        )
                    continue

                # Cache results
                if registry:
                    registry.cache_sam3_results(cat_name, instances)

                logger.info(f"SAM3 found {len(instances)} instances for '{cat_name}' ({len(annotations)} annotations)")
            else:
                logger.debug(f"Using cached SAM3 results for '{cat_name}': {len(instances)} instances")

            # Match instances to annotations
            matching_strategy = (registry.matching_strategy
                               if registry
                               else MatchingStrategy.BBOX_IOU)

            assignment_map = match_instances_to_annotations(
                instances=instances,
                annotations=annotations,
                strategy=matching_strategy
            )

            matched_count = sum(1 for v in assignment_map.values() if v is not None)
            logger.info(f"Matched {matched_count}/{len(annotations)} annotations to instances")

            # Extract each matched annotation
            for ann in annotations:
                ann_id = ann["id"]
                cat_id = ann["category_id"]
                instance_idx = assignment_map.get(ann_id)

                if instance_idx is None:
                    # No matching instance - FAIL
                    results["failed"] += 1
                    results["errors"].append(
                        f"No SAM3 instance matched annotation {ann_id} "
                        f"(class='{cat_name}', {len(instances)} instances available)"
                    )
                    logger.debug(f"Annotation {ann_id} unmatched (bbox={ann.get('bbox')})")
                    continue

                instance = instances[instance_idx]

                # Check for duplicates
                if registry:
                    is_dup, dup_id, max_iou = registry.is_duplicate(
                        mask=instance.mask,
                        bbox=instance.bbox,
                        category_id=cat_id,
                        category_name=cat_name
                    )

                    if is_dup:
                        logger.info(
                            f"Skipping duplicate: ann_id={ann_id} (IoU={max_iou:.3f} "
                            f"with ann_id={dup_id}, class='{cat_name}')"
                        )
                        results["failed"] += 1
                        results["errors"].append(
                            f"Duplicate of annotation {dup_id} (IoU={max_iou:.3f})"
                        )
                        continue

                # Extract object with this mask
                try:
                    cropped, new_bbox = self.crop_with_mask(
                        image, instance.mask, instance.bbox, padding
                    )

                    if cropped is None:
                        results["failed"] += 1
                        results["errors"].append(f"Failed to crop annotation {ann_id}")
                        continue

                    # Encode to base64
                    success, encoded = cv2.imencode('.png', cropped)
                    if not success:
                        results["failed"] += 1
                        results["errors"].append(f"Failed to encode annotation {ann_id}")
                        continue

                    # Save PNG
                    png_path = output_path / cat_name / f"{ann_id}.png"
                    with open(png_path, 'wb') as f:
                        f.write(encoded.tobytes())

                    # Save JSON if requested
                    json_path = None
                    if save_individual_coco:
                        json_path = output_path / cat_name / f"{ann_id}.json"
                        cat = categories[cat_id]
                        coco_json = self.create_individual_coco(
                            annotation=ann,
                            category=cat,
                            image_shape=cropped.shape[:2],
                            original_image_filename=f"image_{img_id}"
                        )
                        with open(json_path, 'w') as f:
                            json.dump(coco_json, f, indent=2)

                    # Register extraction
                    if registry:
                        registry.register_extraction(
                            mask=instance.mask,
                            bbox=instance.bbox,
                            annotation_id=ann_id,
                            category_id=cat_id,
                            category_name=cat_name,
                            method=ExtractionMethod.SAM3_TEXT_PROMPT.value
                        )

                    # Update results
                    results["extracted"] += 1
                    results["by_category"][cat_name] = results["by_category"].get(cat_name, 0) + 1
                    results["by_method"][ExtractionMethod.SAM3_TEXT_PROMPT.value] = \
                        results["by_method"].get(ExtractionMethod.SAM3_TEXT_PROMPT.value, 0) + 1

                    results["extracted_files"].append({
                        "annotation_id": ann_id,
                        "category_name": cat_name,
                        "image_path": str(png_path),
                        "json_path": str(json_path) if json_path else None,
                        "method": ExtractionMethod.SAM3_TEXT_PROMPT.value,
                        "original_bbox": instance.bbox,
                        "extracted_size": [cropped.shape[1], cropped.shape[0]]
                    })

                    # Progress callback
                    if progress_callback:
                        progress_callback({
                            "extracted": results["extracted"],
                            "failed": results["failed"],
                            "current_category": cat_name,
                            "by_category": results["by_category"]
                        })

                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append(f"Annotation {ann_id}: {str(e)}")
                    logger.error(f"Error extracting annotation {ann_id}: {e}")

    def _process_bbox_mask_mode(
        self,
        image: np.ndarray,
        img_id: int,
        img_info: Dict,
        annotations_grouped: Dict,
        registry: Optional['ExtractionRegistry'],
        results: Dict,
        categories: Dict,
        output_path: 'Path',
        use_sam3_for_bbox: bool,
        force_bbox_only: bool,
        force_sam3_resegmentation: bool,
        padding: int,
        save_individual_coco: bool,
        progress_callback: Callable
    ) -> None:
        """
        Process annotations in bbox/mask mode with optional deduplication.

        Algorithm:
        1. For each annotation:
           - Call _extract_single_object_sync (existing logic)
           - If dedup enabled: check IoU with registry
           - Save if unique, register

        Args:
            image: Loaded image
            img_info: Image metadata from COCO JSON
            img_id: Image ID
            annotations_grouped: Dict with img_id keys
            registry: ExtractionRegistry or None
            results: Results dict to update
            categories: Category lookup dict
            output_path: Output directory Path
            use_sam3_for_bbox: Use SAM3 for bbox-only
            force_bbox_only: Force bbox-only mode
            force_sam3_resegmentation: Force SAM3 reseg
            padding: Padding pixels
            save_individual_coco: Whether to save individual JSON
            progress_callback: Progress callback function
        """
        # Get annotations for this image
        annotations = annotations_grouped.get(img_id, [])

        for ann in annotations:
            cat_id = ann["category_id"]
            cat_name = categories[cat_id]["name"]
            ann_id = ann["id"]

            try:
                # Extract using existing logic
                extraction_result = self._extract_single_object_sync(
                    image=image,
                    annotation=ann,
                    category_name=cat_name,
                    use_sam3=use_sam3_for_bbox,
                    padding=padding,
                    force_bbox_only=force_bbox_only,
                    force_sam3_resegmentation=force_sam3_resegmentation,
                    force_sam3_text_prompt=False
                )

                if not extraction_result["success"]:
                    results["failed"] += 1
                    results["errors"].append(extraction_result.get("error", "Unknown error"))
                    continue

                # Deduplication check
                if registry and extraction_result.get("mask_base64"):
                    # Decode mask from base64
                    mask_data = base64.b64decode(extraction_result["mask_base64"])
                    mask_img = cv2.imdecode(np.frombuffer(mask_data, np.uint8), cv2.IMREAD_GRAYSCALE)

                    bbox = extraction_result["original_bbox"]

                    is_dup, dup_id, max_iou = registry.is_duplicate(
                        mask=mask_img,
                        bbox=bbox,
                        category_id=cat_id,
                        category_name=cat_name
                    )

                    if is_dup:
                        logger.info(
                            f"Skipping duplicate: ann_id={ann_id} (IoU={max_iou:.3f} "
                            f"with ann_id={dup_id}, class='{cat_name}')"
                        )
                        results["failed"] += 1
                        results["errors"].append(
                            f"Duplicate of annotation {dup_id} (IoU={max_iou:.3f})"
                        )
                        continue

                # Save PNG
                png_path = output_path / cat_name / f"{ann_id}.png"
                img_data = base64.b64decode(extraction_result["cropped_image_base64"])
                with open(png_path, 'wb') as f:
                    f.write(img_data)

                # Save JSON
                json_path = None
                if save_individual_coco:
                    json_path = output_path / cat_name / f"{ann_id}.json"
                    cat = categories[cat_id]
                    coco_json = self.create_individual_coco(
                        annotation=ann,
                        category=cat,
                        image_shape=extraction_result["extracted_size"][::-1],
                        original_image_filename=f"image_{img_id}"
                    )
                    with open(json_path, 'w') as f:
                        json.dump(coco_json, f, indent=2)

                # Register
                if registry and extraction_result.get("mask_base64"):
                    registry.register_extraction(
                        mask=mask_img,
                        bbox=bbox,
                        annotation_id=ann_id,
                        category_id=cat_id,
                        category_name=cat_name,
                        method=extraction_result["method_used"]
                    )

                # Update results
                results["extracted"] += 1
                results["by_category"][cat_name] = results["by_category"].get(cat_name, 0) + 1
                method = extraction_result["method_used"]
                results["by_method"][method] = results["by_method"].get(method, 0) + 1

                results["extracted_files"].append({
                    "annotation_id": ann_id,
                    "category_name": cat_name,
                    "image_path": str(png_path),
                    "json_path": str(json_path) if json_path else None,
                    "method": method,
                    "original_bbox": extraction_result["original_bbox"],
                    "extracted_size": extraction_result["extracted_size"]
                })

                # Progress
                if progress_callback:
                    progress_callback({
                        "extracted": results["extracted"],
                        "failed": results["failed"],
                        "current_category": cat_name,
                        "by_category": results["by_category"]
                    })

            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Annotation {ann_id}: {str(e)}")
                logger.error(f"Error extracting annotation {ann_id}: {e}")

    async def extract_from_dataset(
        self,
        coco_data: Dict[str, Any],
        images_dir: str,
        output_dir: str,
        categories_to_extract: List[str] = None,
        use_sam3_for_bbox: bool = True,
        force_bbox_only: bool = False,
        force_sam3_resegmentation: bool = False,
        force_sam3_text_prompt: bool = False,
        padding: int = 5,
        min_object_area: int = 100,
        save_individual_coco: bool = True,
        progress_callback: Callable[[Dict[str, Any]], None] = None,
        deduplication_config: Optional['DeduplicationConfig'] = None
    ) -> Dict[str, Any]:
        """
        Extract all objects from a COCO dataset with deduplication support.
        Runs in thread pool to avoid blocking the event loop.

        Args:
            coco_data: COCO format dataset
            images_dir: Directory containing source images
            output_dir: Output directory for extracted objects
            categories_to_extract: List of category names to extract (None = all)
            use_sam3_for_bbox: Use SAM3 for bbox-only annotations
            force_bbox_only: Force extraction using only bbox, ignore existing masks
            force_sam3_resegmentation: Force SAM3 to regenerate masks even if polygon/RLE exist
            force_sam3_text_prompt: Use only class label with SAM3, ignore bbox and masks
            padding: Padding around objects
            min_object_area: Minimum area to extract
            save_individual_coco: Save JSON for each object
            progress_callback: Callback function for progress updates
            deduplication_config: Configuration for deduplication (None = default enabled)

        Returns:
            Extraction results dictionary with deduplication stats
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(
                self._extract_from_dataset_sync,
                coco_data=coco_data,
                images_dir=images_dir,
                output_dir=output_dir,
                categories_to_extract=categories_to_extract,
                use_sam3_for_bbox=use_sam3_for_bbox,
                force_bbox_only=force_bbox_only,
                force_sam3_resegmentation=force_sam3_resegmentation,
                force_sam3_text_prompt=force_sam3_text_prompt,
                padding=padding,
                min_object_area=min_object_area,
                save_individual_coco=save_individual_coco,
                progress_callback=progress_callback,
                deduplication_config=deduplication_config
            )
        )

    # =========================================================================
    # IMAGENET-STYLE EXTRACTION
    # =========================================================================

    def _extract_from_imagenet_structure_sync(
        self,
        root_dir: str,
        output_dir: str,
        padding: int = 5,
        min_object_area: int = 100,
        max_objects_per_class: Optional[int] = None,
        progress_callback: Callable[[Dict[str, Any]], None] = None
    ) -> Dict[str, Any]:
        """
        Extract objects from ImageNet-style directory structure (synchronous).

        Expected structure:
            root_dir/
            ├── class1/
            │   ├── img001.jpg
            │   ├── img002.jpg
            │   └── ...
            ├── class2/
            │   └── ...

        For each image:
        1. Use SAM3 with class name as text prompt
        2. Extract segmented object
        3. Save to output_dir/class_name/

        Args:
            root_dir: Root directory with class subdirectories
            output_dir: Output directory for extracted objects
            padding: Padding around extracted objects
            min_object_area: Minimum area filter
            max_objects_per_class: Limit objects per class (None=all)
            progress_callback: Callback for progress updates

        Returns:
            Extraction summary with stats per class
        """
        if not self.sam3_available:
            logger.error("SAM3 is required for ImageNet structure extraction but not available")
            return {
                "success": False,
                "error": "SAM3 not available",
                "total_classes": 0,
                "classes": {},
                "total_extracted": 0,
                "total_failed": 0
            }

        logger.info(f"Starting ImageNet extraction from {root_dir}")

        # Discover classes (subdirectories)
        try:
            classes = [d for d in os.listdir(root_dir)
                      if os.path.isdir(os.path.join(root_dir, d))]
        except Exception as e:
            logger.error(f"Failed to list root directory: {e}")
            return {
                "success": False,
                "error": f"Failed to access root directory: {e}",
                "total_classes": 0,
                "classes": {},
                "total_extracted": 0,
                "total_failed": 0
            }

        results = {
            "success": True,
            "total_classes": len(classes),
            "classes": {},
            "total_extracted": 0,
            "total_failed": 0,
            "errors": []
        }

        for class_idx, class_name in enumerate(classes):
            class_dir = os.path.join(root_dir, class_name)
            output_class_dir = os.path.join(output_dir, class_name)
            os.makedirs(output_class_dir, exist_ok=True)

            # List images in class directory
            try:
                image_files = [f for f in os.listdir(class_dir)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            except Exception as e:
                logger.error(f"Failed to list class directory {class_name}: {e}")
                results["errors"].append(f"Class '{class_name}': {str(e)}")
                continue

            if max_objects_per_class:
                image_files = image_files[:max_objects_per_class]

            class_stats = {
                "total_images": len(image_files),
                "extracted": 0,
                "failed": 0
            }

            logger.info(f"Processing class '{class_name}': {len(image_files)} images")

            for img_idx, img_file in enumerate(image_files):
                try:
                    img_path = os.path.join(class_dir, img_file)
                    img = cv2.imread(img_path)

                    if img is None:
                        logger.warning(f"Failed to read image: {img_path}")
                        class_stats["failed"] += 1
                        continue

                    # Segment with SAM3 text prompt
                    mask = self._segment_with_sam3_text_prompt_sync(
                        image=img,
                        class_name=class_name,
                        min_area=min_object_area
                    )

                    if mask is None:
                        class_stats["failed"] += 1
                        continue

                    # Calculate bbox from mask
                    y_indices, x_indices = np.where(mask > 0)
                    if len(x_indices) == 0 or len(y_indices) == 0:
                        class_stats["failed"] += 1
                        continue

                    x = int(x_indices.min())
                    y = int(y_indices.min())
                    w = int(x_indices.max() - x + 1)
                    h = int(y_indices.max() - y + 1)
                    bbox = [x, y, w, h]

                    # Extract object
                    cropped, new_bbox = self.crop_with_mask(img, mask, bbox, padding)

                    if cropped is None:
                        class_stats["failed"] += 1
                        continue

                    # Save extracted object
                    output_filename = f"{Path(img_file).stem}.png"
                    output_path = os.path.join(output_class_dir, output_filename)
                    cv2.imwrite(output_path, cropped)

                    class_stats["extracted"] += 1
                    results["total_extracted"] += 1

                    # Progress callback
                    if progress_callback and (img_idx + 1) % 10 == 0:
                        progress_callback({
                            "current_class": class_name,
                            "class_progress": f"{img_idx + 1}/{len(image_files)}",
                            "total_progress": f"{class_idx + 1}/{len(classes)} classes",
                            "extracted": results["total_extracted"],
                            "failed": results["total_failed"]
                        })

                except Exception as e:
                    logger.error(f"Failed to process {img_file}: {e}")
                    class_stats["failed"] += 1
                    results["total_failed"] += 1
                    results["errors"].append(f"{class_name}/{img_file}: {str(e)}")

            results["classes"][class_name] = class_stats
            results["total_failed"] += class_stats["failed"]

            logger.info(
                f"Class '{class_name}': {class_stats['extracted']}/{class_stats['total_images']} extracted"
            )

        logger.info(
            f"ImageNet extraction complete: {results['total_extracted']} extracted, "
            f"{results['total_failed']} failed from {results['total_classes']} classes"
        )

        return results

    async def extract_from_imagenet_structure(
        self,
        root_dir: str,
        output_dir: str,
        padding: int = 5,
        min_object_area: int = 100,
        max_objects_per_class: Optional[int] = None,
        progress_callback: Callable[[Dict[str, Any]], None] = None
    ) -> Dict[str, Any]:
        """
        Extract objects from ImageNet-style directory structure (async).

        Args:
            root_dir: Root directory with class subdirectories
            output_dir: Output directory for extracted objects
            padding: Padding around extracted objects
            min_object_area: Minimum object area filter
            max_objects_per_class: Limit objects per class (None=all)
            progress_callback: Callback for progress updates

        Returns:
            Dictionary with extraction results and statistics
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(
                self._extract_from_imagenet_structure_sync,
                root_dir=root_dir,
                output_dir=output_dir,
                padding=padding,
                min_object_area=min_object_area,
                max_objects_per_class=max_objects_per_class,
                progress_callback=progress_callback
            )
        )

    # =========================================================================
    # DATASET CONVERSION (BBOX -> SEGMENTATION)
    # =========================================================================

    def _convert_bbox_to_segmentation_sync(
        self,
        coco_data: Dict[str, Any],
        images_dir: str,
        output_path: str,
        categories_to_convert: List[str] = None,
        overwrite_existing: bool = False,
        simplify_polygons: bool = True,
        simplify_tolerance: float = 2.0,
        progress_callback: Callable[[Dict[str, Any]], None] = None,
    ) -> Dict[str, Any]:
        """
        Convert bbox-only annotations to segmentations using SAM3.

        Args:
            coco_data: Input COCO dataset
            images_dir: Directory containing images
            output_path: Path for output COCO JSON
            categories_to_convert: Categories to convert (None = all)
            overwrite_existing: Overwrite existing segmentations
            simplify_polygons: Simplify generated polygons
            simplify_tolerance: Simplification tolerance
            progress_callback: Progress callback function

        Returns:
            Conversion results
        """
        if not self.sam3_available:
            return {
                "success": False,
                "error": "SAM3 not available for conversion"
            }

        start_time = datetime.now()

        # Deep copy to avoid modifying original
        import copy
        output_data = copy.deepcopy(coco_data)

        # Build lookups
        categories = {c["id"]: c for c in output_data.get("categories", [])}
        images = {i["id"]: i for i in output_data.get("images", [])}

        # Filter categories
        if categories_to_convert:
            category_ids = {
                cid for cid, cat in categories.items()
                if cat["name"] in categories_to_convert
            }
        else:
            category_ids = set(categories.keys())

        results = {
            "converted": 0,
            "skipped": 0,
            "failed": 0,
            "errors": [],
            "by_category": {}
        }

        total = len(output_data.get("annotations", []))
        processed = 0

        # Group by image
        annotations_by_image = {}
        for idx, ann in enumerate(output_data["annotations"]):
            img_id = ann.get("image_id")
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append((idx, ann))

        for img_id, indexed_annotations in annotations_by_image.items():
            if img_id not in images:
                continue

            img_info = images[img_id]
            img_path = self.resolve_image_path(images_dir, img_info)

            if not img_path.exists():
                error_msg = f"Image not found: {img_path}"
                logger.error(error_msg)
                for idx, ann in indexed_annotations:
                    results["failed"] += 1
                    results["errors"].append(error_msg)
                continue

            image = cv2.imread(str(img_path))
            if image is None or image.size == 0:
                error_msg = f"cv2.imread failed for: {img_path}"
                logger.error(error_msg)
                for idx, ann in indexed_annotations:
                    results["failed"] += 1
                    results["errors"].append(error_msg)
                continue

            h, w = image.shape[:2]

            for idx, ann in indexed_annotations:
                processed += 1

                cat_id = ann.get("category_id")
                if cat_id not in category_ids:
                    results["skipped"] += 1
                    continue

                cat_name = categories[cat_id]["name"]
                ann_type = self.detect_annotation_type(ann)

                # Skip if already has segmentation and not overwriting
                if ann_type != AnnotationType.BBOX_ONLY and not overwrite_existing:
                    results["skipped"] += 1
                    continue

                bbox = ann.get("bbox")
                if not bbox or len(bbox) < 4:
                    results["skipped"] += 1
                    continue

                # Validate bbox dimensions
                bx, by, bw_val, bh_val = bbox[:4]
                if bw_val <= 0 or bh_val <= 0:
                    results["skipped"] += 1
                    results["errors"].append(f"Invalid bbox dimensions for annotation {ann.get('id')}: {bbox}")
                    continue

                # Check if bbox has overlap with image (intersection-based)
                x1_valid = max(0, int(bx))
                y1_valid = max(0, int(by))
                x2_valid = min(w, int(bx + bw_val))
                y2_valid = min(h, int(by + bh_val))

                if x1_valid >= x2_valid or y1_valid >= y2_valid:
                    results["skipped"] += 1
                    results["errors"].append(f"Annotation {ann.get('id')}: bbox outside image bounds - bbox=({bx},{by},{bw_val},{bh_val}) image=({w},{h})")
                    continue

                try:
                    # Segment with SAM3 (uses clamped bbox internally)
                    # Call sync version directly since we're in a thread
                    mask = self._segment_with_sam3_sync(image, bbox)

                    if mask is None:
                        results["failed"] += 1
                        results["errors"].append(f"SAM3 failed for annotation {ann.get('id')}")
                        continue

                    # Convert mask to polygon
                    polygons = self.mask_to_polygon(mask, simplify_polygons, simplify_tolerance)

                    if not polygons:
                        results["failed"] += 1
                        results["errors"].append(f"No valid polygon for annotation {ann.get('id')}")
                        continue

                    # Update annotation
                    output_data["annotations"][idx]["segmentation"] = polygons
                    output_data["annotations"][idx]["area"] = int(np.sum(mask > 0))

                    results["converted"] += 1
                    results["by_category"][cat_name] = results["by_category"].get(cat_name, 0) + 1

                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append(f"Error on annotation {ann.get('id')}: {str(e)}")

                if progress_callback:
                    progress_callback({
                        "converted": results["converted"],
                        "skipped": results["skipped"],
                        "failed": results["failed"],
                        "total": total,
                        "current_image": img_info.get("file_name", "")
                    })

        # Save output
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        end_time = datetime.now()

        results["success"] = True
        results["output_path"] = str(output_file)
        results["processing_time_seconds"] = (end_time - start_time).total_seconds()

        return results

    async def convert_bbox_to_segmentation(
        self,
        coco_data: Dict[str, Any],
        images_dir: str,
        output_path: str,
        categories_to_convert: List[str] = None,
        overwrite_existing: bool = False,
        simplify_polygons: bool = True,
        simplify_tolerance: float = 2.0,
        progress_callback: Callable[[Dict[str, Any]], None] = None,
    ) -> Dict[str, Any]:
        """
        Convert bbox-only annotations to segmentations using SAM3.
        Runs in thread pool to avoid blocking the event loop.

        Args:
            coco_data: Input COCO dataset
            images_dir: Directory containing images
            output_path: Path for output COCO JSON
            categories_to_convert: Categories to convert (None = all)
            overwrite_existing: Overwrite existing segmentations
            simplify_polygons: Simplify generated polygons
            simplify_tolerance: Simplification tolerance
            progress_callback: Progress callback function

        Returns:
            Conversion results
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(
                self._convert_bbox_to_segmentation_sync,
                coco_data=coco_data,
                images_dir=images_dir,
                output_path=output_path,
                categories_to_convert=categories_to_convert,
                overwrite_existing=overwrite_existing,
                simplify_polygons=simplify_polygons,
                simplify_tolerance=simplify_tolerance,
                progress_callback=progress_callback,
            )
        )

    # =========================================================================
    # CUSTOM OBJECT EXTRACTION (TEXT PROMPT MODE)
    # =========================================================================

    def _extract_custom_objects_sync(
        self,
        images_dir: str,
        output_dir: str,
        object_names: List[str],
        padding: int = 5,
        min_object_area: int = 100,
        save_individual_coco: bool = True,
        deduplication_config: Optional[Dict] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Extract custom objects using text prompts only (no COCO JSON).
        Synchronous implementation for thread pool execution.

        Args:
            images_dir: Directory containing images
            output_dir: Output directory for extracted objects
            object_names: List of object names to segment
            padding: Pixels of padding around objects
            min_object_area: Minimum object area in pixels
            save_individual_coco: Save individual COCO JSON per object
            deduplication_config: Deduplication configuration dict
            progress_callback: Progress callback function

        Returns:
            Extraction results dictionary
        """
        from app.utils import ExtractionRegistry, SAM3Instance
        from app.models.extraction_schemas import DeduplicationConfig

        start_time = datetime.now()

        # Parse deduplication config
        if deduplication_config:
            dedup_cfg = DeduplicationConfig(**deduplication_config)
        else:
            dedup_cfg = DeduplicationConfig(enabled=True, iou_threshold=0.7)

        # Initialize results
        results = {
            "success": False,
            "total_images": 0,
            "total_objects_extracted": 0,
            "failed_extractions": 0,
            "duplicates_prevented": 0,
            "by_category": {},
            "by_method": {},
            "errors": [],
            "output_dir": output_dir
        }

        # Scan images directory
        images_dir_path = Path(images_dir)
        if not images_dir_path.exists():
            results["errors"].append(f"Images directory not found: {images_dir}")
            return results

        # Find all images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_files = [
            f for f in images_dir_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        if not image_files:
            results["errors"].append(f"No images found in {images_dir}")
            return results

        results["total_images"] = len(image_files)
        logger.info(f"Found {len(image_files)} images in {images_dir}")

        # Create output directories for each object type
        output_dir_path = Path(output_dir)
        category_dirs = {}
        for obj_name in object_names:
            # Sanitize object name for directory
            safe_name = obj_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
            cat_dir = output_dir_path / safe_name
            cat_dir.mkdir(parents=True, exist_ok=True)
            category_dirs[obj_name] = cat_dir
            results["by_category"][obj_name] = 0

        # Counter for synthetic annotation IDs
        annotation_id_counter = 0

        # Process each image
        for img_idx, img_path in enumerate(image_files):
            logger.info(f"Processing image {img_idx + 1}/{len(image_files)}: {img_path.name}")

            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                error_msg = f"Failed to load image: {img_path}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
                continue

            img_h, img_w = image.shape[:2]

            # Initialize extraction registry for this image (deduplication per-image)
            registry = ExtractionRegistry(
                iou_threshold=dedup_cfg.iou_threshold,
                cross_category_dedup=dedup_cfg.cross_category_dedup,
                matching_strategy=dedup_cfg.matching_strategy
            )

            # Process each object name
            for obj_name in object_names:
                logger.debug(f"  Segmenting '{obj_name}' in {img_path.name}")

                # Run SAM3 text prompt segmentation
                instances = self._segment_with_sam3_text_prompt_sync(
                    image=image,
                    class_name=obj_name,
                    min_area=min_object_area,
                    return_all_instances=True
                )

                if instances is None or len(instances) == 0:
                    logger.debug(f"    No instances of '{obj_name}' found")
                    continue

                logger.info(f"    Found {len(instances)} instance(s) of '{obj_name}'")

                # Extract each instance
                for inst_idx, instance in enumerate(instances):
                    annotation_id_counter += 1

                    # Check for duplicates
                    if dedup_cfg.enabled:
                        is_dup, dup_id, iou = registry.is_duplicate(
                            mask=instance.mask,
                            bbox=instance.bbox,
                            category_id=0,  # synthetic
                            category_name=obj_name
                        )

                        if is_dup:
                            logger.debug(
                                f"      Instance {inst_idx} is duplicate (IoU={iou:.3f} with previous), skipping"
                            )
                            results["duplicates_prevented"] += 1
                            continue

                    # Extract object with mask
                    try:
                        extracted_img, mask_img, bbox_extracted = self._extract_object_with_mask(
                            image=image,
                            mask=instance.mask,
                            padding=padding
                        )

                        if extracted_img is None:
                            logger.warning(f"      Failed to extract instance {inst_idx}")
                            results["failed_extractions"] += 1
                            continue

                        # Save extracted image
                        safe_obj_name = obj_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
                        output_filename = f"{img_path.stem}_{safe_obj_name}_instance{inst_idx:03d}.png"
                        output_path = category_dirs[obj_name] / output_filename

                        cv2.imwrite(str(output_path), extracted_img)

                        # Save individual COCO JSON if requested
                        if save_individual_coco:
                            json_filename = output_path.stem + ".json"
                            json_path = category_dirs[obj_name] / json_filename

                            # Get polygon from mask
                            polygons = self.mask_to_polygon(instance.mask, simplify=True, tolerance=2.0)

                            coco_json = {
                                "info": {
                                    "description": f"Extracted {obj_name} from {img_path.name}",
                                    "date_created": datetime.now().isoformat()
                                },
                                "images": [{
                                    "id": 1,
                                    "file_name": output_filename,
                                    "width": extracted_img.shape[1],
                                    "height": extracted_img.shape[0]
                                }],
                                "annotations": [{
                                    "id": annotation_id_counter,
                                    "image_id": 1,
                                    "category_id": 1,
                                    "bbox": bbox_extracted,
                                    "area": int(np.sum(instance.mask > 0)),
                                    "segmentation": polygons if polygons else [],
                                    "iscrowd": 0
                                }],
                                "categories": [{
                                    "id": 1,
                                    "name": obj_name,
                                    "supercategory": "object"
                                }]
                            }

                            with open(json_path, 'w') as f:
                                json.dump(coco_json, f, indent=2)

                        # Register extraction
                        registry.register_extraction(
                            mask=instance.mask,
                            bbox=instance.bbox,
                            annotation_id=annotation_id_counter,
                            category_id=0,
                            category_name=obj_name,
                            method="sam3_text_prompt"
                        )

                        # Update stats
                        results["total_objects_extracted"] += 1
                        results["by_category"][obj_name] += 1
                        results["by_method"]["sam3_text_prompt"] = results["by_method"].get("sam3_text_prompt", 0) + 1

                        logger.debug(f"      Extracted instance {inst_idx} → {output_path.name}")

                    except Exception as e:
                        error_msg = f"Error extracting instance {inst_idx} of '{obj_name}' from {img_path.name}: {str(e)}"
                        logger.error(error_msg)
                        results["errors"].append(error_msg)
                        results["failed_extractions"] += 1

            # Progress callback
            if progress_callback:
                progress_callback({
                    "extracted": results["total_objects_extracted"],
                    "failed": results["failed_extractions"],
                    "duplicates_prevented": results["duplicates_prevented"],
                    "total_images": results["total_images"],
                    "current_image": img_path.name,
                    "image_progress": f"{img_idx + 1}/{len(image_files)}"
                })

        # Save extraction summary
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        summary = {
            "extraction_date": start_time.isoformat(),
            "images_dir": images_dir,
            "output_dir": output_dir,
            "object_names": object_names,
            "total_images_processed": results["total_images"],
            "total_objects_extracted": results["total_objects_extracted"],
            "failed_extractions": results["failed_extractions"],
            "duplicates_prevented": results["duplicates_prevented"],
            "by_category": results["by_category"],
            "by_method": results["by_method"],
            "errors": results["errors"],
            "processing_time_seconds": processing_time,
            "settings": {
                "padding": padding,
                "min_object_area": min_object_area,
                "deduplication_enabled": dedup_cfg.enabled,
                "deduplication_iou_threshold": dedup_cfg.iou_threshold
            }
        }

        summary_path = output_dir_path / "extraction_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Extraction complete: {results['total_objects_extracted']} objects extracted")
        logger.info(f"Summary saved to: {summary_path}")

        results["success"] = True
        results["processing_time_seconds"] = processing_time
        results["summary_path"] = str(summary_path)

        return results

    async def extract_custom_objects(
        self,
        images_dir: str,
        output_dir: str,
        object_names: List[str],
        padding: int = 5,
        min_object_area: int = 100,
        save_individual_coco: bool = True,
        deduplication_config: Optional[Dict] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Extract custom objects using text prompts only (no COCO JSON).
        Async wrapper that runs in thread pool.

        Args:
            images_dir: Directory containing images
            output_dir: Output directory for extracted objects
            object_names: List of object names to segment
            padding: Pixels of padding around objects
            min_object_area: Minimum object area in pixels
            save_individual_coco: Save individual COCO JSON per object
            deduplication_config: Deduplication configuration dict
            progress_callback: Progress callback function

        Returns:
            Extraction results dictionary
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(
                self._extract_custom_objects_sync,
                images_dir=images_dir,
                output_dir=output_dir,
                object_names=object_names,
                padding=padding,
                min_object_area=min_object_area,
                save_individual_coco=save_individual_coco,
                deduplication_config=deduplication_config,
                progress_callback=progress_callback
            )
        )
