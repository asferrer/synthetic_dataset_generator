"""
Extraction Registry
===================
Per-image registry for duplicate detection and SAM3 result caching.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

from .mask_utils import (
    ExtractedMask,
    SAM3Instance,
    MatchingStrategy,
    calculate_bbox_iou,
    calculate_mask_iou
)

logger = logging.getLogger(__name__)


class ExtractionRegistry:
    """
    Per-image registry for duplicate detection and SAM3 caching.

    Purpose:
    - Track extracted masks to prevent duplicates
    - Cache SAM3 results per category (avoid re-running SAM3)
    - Provide statistics on duplicates prevented

    Lifecycle:
    - Created once per image during batch extraction
    - Cleared when moving to next image (no cross-image state)

    Example:
        >>> registry = ExtractionRegistry(iou_threshold=0.7)
        >>>
        >>> # Extract first object
        >>> registry.register_extraction(
        ...     mask=mask1,
        ...     bbox=[10, 10, 50, 50],
        ...     annotation_id=1,
        ...     category_id=5,
        ...     category_name="fish",
        ...     method="sam3"
        ... )
        >>>
        >>> # Check if second object is duplicate
        >>> is_dup, dup_id, iou = registry.is_duplicate(
        ...     mask=mask2,
        ...     bbox=[12, 12, 48, 48],
        ...     category_id=5,
        ...     category_name="fish"
        ... )
        >>> if is_dup:
        ...     print(f"Duplicate of annotation {dup_id} (IoU={iou:.3f})")
    """

    def __init__(
        self,
        iou_threshold: float = 0.7,
        cross_category_dedup: bool = False,
        matching_strategy: MatchingStrategy = MatchingStrategy.BBOX_IOU
    ):
        """
        Initialize extraction registry.

        Args:
            iou_threshold: IoU threshold for duplicate detection (0.0-1.0)
                          Higher = stricter (only marks obvious duplicates)
                          Default: 0.7 (70% overlap)
            cross_category_dedup: Check for duplicates across different categories
                                 Default: False (only within same category)
            matching_strategy: Strategy for matching SAM3 instances to annotations
                              Default: BBOX_IOU (fast, accurate enough)
        """
        self.iou_threshold = iou_threshold
        self.cross_category_dedup = cross_category_dedup
        self.matching_strategy = matching_strategy

        # List of extracted masks
        self.extracted_masks: List[ExtractedMask] = []

        # Cache: category_name -> List[SAM3Instance]
        # Avoids re-running SAM3 for same category in same image
        self.sam3_cache: Dict[str, List[SAM3Instance]] = {}

        # Statistics
        self.stats = {
            "total_checked": 0,
            "duplicates_prevented": 0,
            "unique_extracted": 0,
        }

        logger.debug(
            f"Created ExtractionRegistry (iou_threshold={iou_threshold}, "
            f"cross_category={cross_category_dedup}, strategy={matching_strategy.value})"
        )

    def is_duplicate(
        self,
        mask: np.ndarray,
        bbox: List[float],
        category_id: int,
        category_name: str
    ) -> Tuple[bool, Optional[int], float]:
        """
        Check if mask duplicates any previously extracted mask.

        Algorithm:
        1. Filter candidates by category (unless cross_category_dedup=True)
        2. Fast bbox IoU pre-filter (skip if bbox_iou < iou_threshold - 0.1)
        3. Calculate full mask IoU for remaining candidates
        4. If any mask_iou >= iou_threshold: return (True, dup_ann_id, max_iou)
        5. Else: return (False, None, max_iou)

        Args:
            mask: Binary mask (H, W) with values 0 or 255
            bbox: Bounding box [x, y, width, height]
            category_id: Category ID
            category_name: Category name

        Returns:
            Tuple of (is_duplicate, duplicate_annotation_id, max_iou)
            - is_duplicate: True if mask duplicates existing extraction
            - duplicate_annotation_id: ID of annotation it duplicates (or None)
            - max_iou: Highest IoU found with existing masks

        Example:
            >>> is_dup, dup_id, iou = registry.is_duplicate(
            ...     mask=new_mask,
            ...     bbox=[10, 10, 50, 50],
            ...     category_id=5,
            ...     category_name="fish"
            ... )
            >>> if is_dup:
            ...     print(f"Skip: duplicate of annotation {dup_id} (IoU={iou:.3f})")
        """
        self.stats["total_checked"] += 1

        # Filter candidates
        candidates = self.extracted_masks

        if not self.cross_category_dedup:
            # Only check within same category
            candidates = [
                em for em in self.extracted_masks
                if em.category_name == category_name
            ]

        if not candidates:
            return (False, None, 0.0)

        max_iou = 0.0
        duplicate_of = None

        # Check each candidate
        for existing in candidates:
            # Fast bbox IoU pre-filter (cheap operation)
            bbox_iou = calculate_bbox_iou(bbox, existing.bbox)

            # Skip if bboxes don't overlap enough
            # Use threshold - 0.1 as pre-filter to avoid false negatives
            if bbox_iou < (self.iou_threshold - 0.1):
                continue

            # Calculate full mask IoU (expensive operation)
            mask_iou = calculate_mask_iou(mask, existing.mask)

            # Track max IoU
            if mask_iou > max_iou:
                max_iou = mask_iou

            # Check if duplicate
            if mask_iou >= self.iou_threshold:
                duplicate_of = existing.annotation_id
                logger.debug(
                    f"Duplicate detected: new mask vs annotation {existing.annotation_id} "
                    f"(mask_iou={mask_iou:.3f}, bbox_iou={bbox_iou:.3f}, "
                    f"category='{category_name}', threshold={self.iou_threshold})"
                )
                break

        is_duplicate = (duplicate_of is not None)

        if is_duplicate:
            self.stats["duplicates_prevented"] += 1

        return (is_duplicate, duplicate_of, max_iou)

    def register_extraction(
        self,
        mask: np.ndarray,
        bbox: List[float],
        annotation_id: int,
        category_id: int,
        category_name: str,
        method: str
    ) -> None:
        """
        Register successfully extracted mask to prevent future duplicates.

        Args:
            mask: Binary mask (H, W)
            bbox: Bounding box [x, y, width, height]
            annotation_id: Annotation ID
            category_id: Category ID
            category_name: Category name
            method: Extraction method (e.g., "sam3_from_bbox", "polygon_mask")

        Example:
            >>> registry.register_extraction(
            ...     mask=mask,
            ...     bbox=[10, 10, 50, 50],
            ...     annotation_id=123,
            ...     category_id=5,
            ...     category_name="fish",
            ...     method="sam3_text_prompt"
            ... )
        """
        extracted_mask = ExtractedMask(
            mask=mask,
            bbox=bbox,
            annotation_id=annotation_id,
            category_id=category_id,
            category_name=category_name,
            method=method
        )

        self.extracted_masks.append(extracted_mask)
        self.stats["unique_extracted"] += 1

        logger.debug(
            f"Registered extraction: ann_id={annotation_id}, category='{category_name}', "
            f"method='{method}', bbox={bbox}"
        )

    def cache_sam3_results(
        self,
        category_name: str,
        instances: List[SAM3Instance]
    ) -> None:
        """
        Cache SAM3 instances for category to avoid re-running SAM3.

        When using text prompt mode, SAM3 is run once per (image, category)
        and results are cached. Subsequent annotations of the same category
        in the same image reuse these instances.

        Args:
            category_name: Category name (e.g., "fish", "coral")
            instances: List of SAM3Instance objects from segmentation

        Example:
            >>> instances = segment_with_sam3_text_prompt(
            ...     image=img,
            ...     class_name="fish",
            ...     return_all_instances=True
            ... )
            >>> registry.cache_sam3_results("fish", instances)
        """
        self.sam3_cache[category_name] = instances

        logger.debug(
            f"Cached SAM3 results: category='{category_name}', "
            f"instances={len(instances)}"
        )

    def get_sam3_instances(
        self,
        category_name: str
    ) -> Optional[List[SAM3Instance]]:
        """
        Retrieve cached SAM3 instances for category.

        Args:
            category_name: Category name

        Returns:
            List of SAM3Instance objects if cached, None otherwise

        Example:
            >>> instances = registry.get_sam3_instances("fish")
            >>> if instances is not None:
            ...     print(f"Using {len(instances)} cached instances")
            ... else:
            ...     # Need to run SAM3
            ...     instances = segment_with_sam3_text_prompt(...)
        """
        return self.sam3_cache.get(category_name)

    def get_stats(self) -> Dict[str, int]:
        """
        Get deduplication statistics.

        Returns:
            Dictionary with:
            - total_checked: Number of masks checked for duplicates
            - duplicates_prevented: Number of duplicates prevented
            - unique_extracted: Number of unique extractions registered

        Example:
            >>> stats = registry.get_stats()
            >>> print(f"Prevented {stats['duplicates_prevented']} duplicates")
        """
        return self.stats.copy()

    def clear(self) -> None:
        """
        Clear registry for next image.

        Call this when moving to a new image to reset state.
        """
        self.extracted_masks.clear()
        self.sam3_cache.clear()
        self.stats = {
            "total_checked": 0,
            "duplicates_prevented": 0,
            "unique_extracted": 0,
        }

        logger.debug("Cleared ExtractionRegistry")

    def __repr__(self) -> str:
        return (
            f"ExtractionRegistry("
            f"iou_threshold={self.iou_threshold}, "
            f"extractions={len(self.extracted_masks)}, "
            f"cached_categories={len(self.sam3_cache)}, "
            f"duplicates_prevented={self.stats['duplicates_prevented']}"
            f")"
        )
