"""
Utilities Module
================
Utility functions for object extraction and deduplication.
"""

from .mask_utils import (
    MatchingStrategy,
    SAM3Instance,
    ExtractedMask,
    calculate_bbox_iou,
    calculate_mask_iou,
    calculate_bbox_center_distance,
    match_instances_to_annotations,
    get_bbox_from_mask
)

from .extraction_registry import ExtractionRegistry

__all__ = [
    "MatchingStrategy",
    "SAM3Instance",
    "ExtractedMask",
    "calculate_bbox_iou",
    "calculate_mask_iou",
    "calculate_bbox_center_distance",
    "match_instances_to_annotations",
    "get_bbox_from_mask",
    "ExtractionRegistry"
]
