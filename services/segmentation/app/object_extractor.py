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
        padding: int = 5
    ) -> Dict[str, Any]:
        """
        Synchronous version of extract_single_object (runs in thread pool).

        Args:
            image: Source image as numpy array
            annotation: COCO annotation for the object
            category_name: Name of the category
            use_sam3: Whether to use SAM3 for bbox-only annotations
            padding: Padding around extracted object

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

        # Validate bbox
        if not bbox or len(bbox) < 4:
            return {"success": False, "error": f"Invalid bbox: {bbox}"}

        # Ensure bbox values are valid
        bx, by, bw, bh = bbox[:4]
        if bw <= 0 or bh <= 0:
            return {"success": False, "error": f"Invalid bbox dimensions: {bbox}"}

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
        padding: int = 5
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
                padding=padding
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
        padding: int = 5,
        min_object_area: int = 100,
        save_individual_coco: bool = True,
        progress_callback: Callable[[Dict[str, Any]], None] = None
    ) -> Dict[str, Any]:
        """
        Extract all objects from a COCO dataset.

        Args:
            coco_data: COCO format dataset
            images_dir: Directory containing source images
            output_dir: Output directory for extracted objects
            categories_to_extract: List of category names to extract (None = all)
            use_sam3_for_bbox: Use SAM3 for bbox-only annotations
            padding: Padding around objects
            min_object_area: Minimum area to extract
            save_individual_coco: Save JSON for each object
            progress_callback: Callback function for progress updates

        Returns:
            Extraction results dictionary
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
            "extracted_files": []
        }

        # Group annotations by image for efficiency
        annotations_by_image = {}
        for ann in annotations:
            if ann.get("category_id") not in category_ids:
                continue

            # Check minimum area
            bbox = ann.get("bbox", [0, 0, 0, 0])
            if len(bbox) >= 4 and bbox[2] * bbox[3] < min_object_area:
                continue

            img_id = ann.get("image_id")
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)

        total_objects = sum(len(anns) for anns in annotations_by_image.values())
        processed = 0

        # Cache for loaded images
        current_image = None
        current_image_id = None

        for img_id, img_annotations in annotations_by_image.items():
            if img_id not in images:
                for ann in img_annotations:
                    results["failed"] += 1
                    results["errors"].append(f"Image ID {img_id} not found")
                continue

            img_info = images[img_id]
            img_path = self.resolve_image_path(images_dir, img_info)

            # Load image
            if not img_path.exists():
                error_msg = f"Image not found: {img_path} (file_name in JSON: {img_info.get('file_name', '?')}, images_dir: {images_dir})"
                logger.error(error_msg)
                for ann in img_annotations:
                    results["failed"] += 1
                    results["errors"].append(error_msg)
                continue

            try:
                image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            except Exception as e:
                error_msg = f"Exception loading image {img_path}: {e}"
                logger.error(error_msg)
                for ann in img_annotations:
                    results["failed"] += 1
                    results["errors"].append(error_msg)
                continue

            if image is None:
                error_msg = f"cv2.imread returned None for: {img_path} (file exists: {img_path.exists()}, size: {img_path.stat().st_size if img_path.exists() else 0} bytes)"
                logger.error(error_msg)
                for ann in img_annotations:
                    results["failed"] += 1
                    results["errors"].append(error_msg)
                continue

            # Log image info for debugging
            logger.debug(f"Loaded image {img_path}: shape={image.shape}, dtype={image.dtype}")

            # Process each annotation
            for ann in img_annotations:
                try:
                    cat = categories.get(ann["category_id"])
                    if not cat:
                        continue

                    cat_name = cat["name"]
                    ann_id = ann["id"]

                    # Extract object (call sync version directly since we're in a thread)
                    result = self._extract_single_object_sync(
                        image=image,
                        annotation=ann,
                        category_name=cat_name,
                        use_sam3=use_sam3_for_bbox,
                        padding=padding
                    )

                    if not result["success"]:
                        results["failed"] += 1
                        results["errors"].append(f"Annotation {ann_id}: {result.get('error', 'Unknown error')}")
                        continue

                    # Save PNG
                    png_path = output_path / cat_name / f"{ann_id}.png"
                    img_data = base64.b64decode(result["cropped_image_base64"])
                    with open(png_path, 'wb') as f:
                        f.write(img_data)

                    # Save individual COCO JSON
                    json_path = None
                    if save_individual_coco:
                        json_path = output_path / cat_name / f"{ann_id}.json"
                        coco_json = self.create_individual_coco(
                            annotation=ann,
                            category=cat,
                            image_shape=result["extracted_size"][::-1],  # [w,h] -> [h,w]
                            original_image_filename=img_info.get("file_name", "")
                        )
                        with open(json_path, 'w') as f:
                            json.dump(coco_json, f, indent=2)

                    # Update results
                    results["extracted"] += 1
                    results["by_category"][cat_name] += 1

                    method = result["method_used"]
                    results["by_method"][method] = results["by_method"].get(method, 0) + 1

                    results["extracted_files"].append({
                        "annotation_id": ann_id,
                        "category_name": cat_name,
                        "image_path": str(png_path),
                        "json_path": str(json_path) if json_path else None,
                        "method": method,
                        "original_bbox": result["original_bbox"],
                        "extracted_size": result["extracted_size"]
                    })

                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append(f"Annotation {ann.get('id', '?')}: {str(e)}")
                    logger.error(f"Error extracting annotation {ann.get('id')}: {e}")

                processed += 1

                # Progress callback
                if progress_callback:
                    progress_callback({
                        "extracted": results["extracted"],
                        "failed": results["failed"],
                        "total": total_objects,
                        "current_category": cat_name if 'cat_name' in dir() else "",
                        "by_category": results["by_category"]
                    })

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

        return results

    async def extract_from_dataset(
        self,
        coco_data: Dict[str, Any],
        images_dir: str,
        output_dir: str,
        categories_to_extract: List[str] = None,
        use_sam3_for_bbox: bool = True,
        padding: int = 5,
        min_object_area: int = 100,
        save_individual_coco: bool = True,
        progress_callback: Callable[[Dict[str, Any]], None] = None
    ) -> Dict[str, Any]:
        """
        Extract all objects from a COCO dataset.
        Runs in thread pool to avoid blocking the event loop.

        Args:
            coco_data: COCO format dataset
            images_dir: Directory containing source images
            output_dir: Output directory for extracted objects
            categories_to_extract: List of category names to extract (None = all)
            use_sam3_for_bbox: Use SAM3 for bbox-only annotations
            padding: Padding around objects
            min_object_area: Minimum area to extract
            save_individual_coco: Save JSON for each object
            progress_callback: Callback function for progress updates

        Returns:
            Extraction results dictionary
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
                padding=padding,
                min_object_area=min_object_area,
                save_individual_coco=save_individual_coco,
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
        progress_callback: Callable[[Dict[str, Any]], None] = None
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
        progress_callback: Callable[[Dict[str, Any]], None] = None
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
                progress_callback=progress_callback
            )
        )
