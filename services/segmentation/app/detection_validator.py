"""
Detection Validator for SAM3 Auto-Labeling

This module provides post-detection validation to filter out false positives
and improve the quality of auto-generated annotations.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


class DetectionValidator:
    """
    Validates SAM3 detections to reduce false positives.

    Applies multiple validation filters:
    - Aspect ratio validation per class
    - Relative area validation (not too small or too large)
    - Mask solidity check (avoid fragmented detections)
    - Edge detection (avoid detections that are mostly at image edges)
    """

    # Aspect ratio constraints per class (min, max) where ratio = width/height
    ASPECT_RATIO_CONSTRAINTS: Dict[str, Tuple[float, float]] = {
        # Marine life
        "fish": (0.3, 5.0),
        "coral": (0.2, 5.0),
        "shark": (0.5, 6.0),
        "turtle": (0.5, 2.5),
        "jellyfish": (0.3, 3.0),
        "octopus": (0.3, 3.0),
        "starfish": (0.5, 2.0),
        "crab": (0.5, 2.5),
        "dolphin": (0.5, 4.0),
        "whale": (0.3, 5.0),

        # Vehicles
        "car": (0.8, 3.5),
        "truck": (0.5, 5.0),
        "bus": (0.5, 4.0),
        "motorcycle": (0.5, 2.5),
        "bicycle": (0.6, 2.5),
        "boat": (0.3, 5.0),
        "airplane": (0.3, 4.0),
        "train": (0.2, 8.0),

        # People
        "person": (0.15, 1.5),
        "face": (0.5, 2.0),
        "hand": (0.3, 3.0),
        "head": (0.5, 1.8),

        # Default for unknown classes
        "_default": (0.1, 10.0),
    }

    # Minimum relative area (fraction of image area)
    MIN_RELATIVE_AREA: Dict[str, float] = {
        "fish": 0.0005,
        "person": 0.002,
        "car": 0.003,
        "face": 0.001,
        "_default": 0.001,
    }

    # Maximum relative area (fraction of image area)
    MAX_RELATIVE_AREA: Dict[str, float] = {
        "fish": 0.6,
        "person": 0.8,
        "car": 0.7,
        "background": 0.95,
        "_default": 0.8,
    }

    # Minimum mask solidity (area / convex hull area)
    MIN_SOLIDITY: Dict[str, float] = {
        "fish": 0.4,
        "person": 0.3,
        "car": 0.5,
        "coral": 0.2,  # Coral can be irregular
        "_default": 0.3,
    }

    def __init__(
        self,
        enable_aspect_ratio: bool = True,
        enable_area_check: bool = True,
        enable_solidity_check: bool = True,
        enable_edge_check: bool = True,
        edge_margin_ratio: float = 0.02,
    ):
        """
        Initialize the DetectionValidator.

        Args:
            enable_aspect_ratio: Enable aspect ratio validation
            enable_area_check: Enable relative area validation
            enable_solidity_check: Enable mask solidity validation
            enable_edge_check: Enable edge boundary check
            edge_margin_ratio: Margin from image edge as ratio of dimension
        """
        self.enable_aspect_ratio = enable_aspect_ratio
        self.enable_area_check = enable_area_check
        self.enable_solidity_check = enable_solidity_check
        self.enable_edge_check = enable_edge_check
        self.edge_margin_ratio = edge_margin_ratio

    def validate_detection(
        self,
        mask: np.ndarray,
        bbox: List[float],
        class_name: str,
        image_size: Tuple[int, int],
        score: float = 1.0,
    ) -> Tuple[bool, str, float]:
        """
        Validate a single detection.

        Args:
            mask: Binary mask array (H, W)
            bbox: Bounding box [x, y, w, h]
            class_name: Detected class name
            image_size: Image dimensions (width, height)
            score: Detection confidence score

        Returns:
            Tuple of (is_valid, rejection_reason, adjusted_score)
            - is_valid: Whether the detection passed validation
            - rejection_reason: Why it was rejected (empty if valid)
            - adjusted_score: Potentially adjusted confidence score
        """
        normalized_class = class_name.lower().strip().replace("_", " ")
        img_w, img_h = image_size
        img_area = img_w * img_h
        x, y, w, h = bbox

        # Check aspect ratio
        if self.enable_aspect_ratio:
            if h <= 0:
                return False, "invalid_height", 0.0

            aspect_ratio = w / h
            min_ratio, max_ratio = self.ASPECT_RATIO_CONSTRAINTS.get(
                normalized_class,
                self.ASPECT_RATIO_CONSTRAINTS["_default"]
            )

            if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
                return False, f"aspect_ratio_out_of_range_{aspect_ratio:.2f}", 0.0

        # Check relative area
        if self.enable_area_check:
            bbox_area = w * h
            rel_area = bbox_area / img_area

            min_area = self.MIN_RELATIVE_AREA.get(
                normalized_class,
                self.MIN_RELATIVE_AREA["_default"]
            )
            max_area = self.MAX_RELATIVE_AREA.get(
                normalized_class,
                self.MAX_RELATIVE_AREA["_default"]
            )

            if rel_area < min_area:
                return False, f"too_small_{rel_area:.4f}", 0.0

            if rel_area > max_area:
                return False, f"too_large_{rel_area:.2f}", 0.0

        # Check mask solidity
        if self.enable_solidity_check and mask is not None:
            solidity = self._calculate_solidity(mask)
            min_solidity = self.MIN_SOLIDITY.get(
                normalized_class,
                self.MIN_SOLIDITY["_default"]
            )

            if solidity < min_solidity:
                return False, f"low_solidity_{solidity:.2f}", 0.0

        # Check if detection is mostly at image edge (likely partial/cut-off)
        if self.enable_edge_check:
            margin_x = img_w * self.edge_margin_ratio
            margin_y = img_h * self.edge_margin_ratio

            # Check if bbox touches multiple edges (likely background or partial)
            touches_left = x <= margin_x
            touches_right = (x + w) >= (img_w - margin_x)
            touches_top = y <= margin_y
            touches_bottom = (y + h) >= (img_h - margin_y)

            edge_count = sum([touches_left, touches_right, touches_top, touches_bottom])

            # If touches 3+ edges, likely a false positive (background, etc.)
            if edge_count >= 3:
                return False, "touches_too_many_edges", 0.0

            # Reduce score slightly if detection touches edges
            if edge_count >= 2:
                score *= 0.9

        return True, "", score

    def _calculate_solidity(self, mask: np.ndarray) -> float:
        """
        Calculate mask solidity (area / convex hull area).

        High solidity means compact shape, low means fragmented.

        Args:
            mask: Binary mask array

        Returns:
            Solidity value between 0 and 1
        """
        if mask is None or mask.size == 0:
            return 0.0

        # Ensure mask is binary uint8
        mask_binary = (mask > 0).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(
            mask_binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return 0.0

        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)

        if contour_area <= 0:
            return 0.0

        # Calculate convex hull area
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)

        if hull_area <= 0:
            return 0.0

        return contour_area / hull_area

    def validate_batch(
        self,
        detections: List[Dict],
        class_name: str,
        image_size: Tuple[int, int],
    ) -> List[Dict]:
        """
        Validate a batch of detections for a single class.

        Args:
            detections: List of detection dicts with 'mask', 'bbox', 'score'
            class_name: The detected class
            image_size: Image dimensions (width, height)

        Returns:
            Filtered list of valid detections with adjusted scores
        """
        valid_detections = []

        for det in detections:
            is_valid, reason, adjusted_score = self.validate_detection(
                mask=det.get("mask"),
                bbox=det["bbox"],
                class_name=class_name,
                image_size=image_size,
                score=det.get("score", 1.0),
            )

            if is_valid:
                det["score"] = adjusted_score
                valid_detections.append(det)
            else:
                logger.debug(
                    f"Rejected detection for '{class_name}': {reason}"
                )

        return valid_detections


def deduplicate_annotations(
    annotations: List[Dict],
    iou_threshold: float = 0.7,
) -> List[Dict]:
    """
    Remove duplicate annotations based on bbox IoU.

    Args:
        annotations: List of COCO-format annotations
        iou_threshold: IoU threshold for considering duplicates

    Returns:
        Deduplicated list of annotations
    """
    if len(annotations) <= 1:
        return annotations

    # Sort by area (larger first) to keep more complete detections
    sorted_anns = sorted(
        annotations,
        key=lambda a: a.get("area", 0),
        reverse=True
    )

    keep = []
    for ann in sorted_anns:
        is_duplicate = False
        for kept in keep:
            iou = _calculate_bbox_iou(ann["bbox"], kept["bbox"])
            if iou > iou_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            keep.append(ann)

    removed_count = len(annotations) - len(keep)
    if removed_count > 0:
        logger.debug(f"Deduplicated {removed_count} annotations (IoU > {iou_threshold})")

    return keep


def _calculate_bbox_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate IoU between two bounding boxes in COCO format [x, y, w, h].

    Args:
        bbox1, bbox2: Bounding boxes in [x, y, w, h] format

    Returns:
        IoU value between 0 and 1
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Convert to corners
    box1_x2, box1_y2 = x1 + w1, y1 + h1
    box2_x2, box2_y2 = x2 + w2, y2 + h2

    # Intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)

    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    intersection = (xi2 - xi1) * (yi2 - yi1)

    # Union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# Global instance
_detection_validator: Optional[DetectionValidator] = None


def get_detection_validator() -> DetectionValidator:
    """Get or create the global DetectionValidator instance."""
    global _detection_validator
    if _detection_validator is None:
        _detection_validator = DetectionValidator()
    return _detection_validator
