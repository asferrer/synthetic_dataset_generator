"""
Base exporter class and common dataclasses for annotation export.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for dataset export."""
    output_dir: str
    include_images: bool = True
    copy_images: bool = False  # True=copy, False=symlink or reference only
    normalize_coords: bool = False
    image_format: str = 'jpg'  # jpg, png
    create_subdirs: bool = True  # Create images/ and labels/ subdirectories


@dataclass
class ExportResult:
    """Result of an export operation."""
    success: bool
    output_path: str
    num_images: int
    num_annotations: int
    format_name: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return (f"ExportResult({status}): {self.format_name} -> {self.output_path}\n"
                f"  Images: {self.num_images}, Annotations: {self.num_annotations}\n"
                f"  Errors: {len(self.errors)}, Warnings: {len(self.warnings)}")


class BaseExporter(ABC):
    """Abstract base class for all annotation format exporters."""

    def __init__(self, config: ExportConfig):
        """
        Initialize the exporter.

        Args:
            config: Export configuration
        """
        self.config = config
        self._ensure_output_dir()

    def _ensure_output_dir(self) -> None:
        """Create output directory if it doesn't exist."""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def export(self,
               coco_data: Dict[str, Any],
               images_dir: str,
               output_name: str = "dataset") -> ExportResult:
        """
        Export the dataset to the specific format.

        Args:
            coco_data: Dataset in COCO format (dict with images, annotations, categories)
            images_dir: Directory containing the source images
            output_name: Base name for output files/directories

        Returns:
            ExportResult with details about the export operation
        """
        pass

    @abstractmethod
    def get_format_name(self) -> str:
        """
        Return the name of the export format.

        Returns:
            Format name string (e.g., 'COCO', 'YOLO', 'COCO_Segmentation')
        """
        pass

    def validate_input(self, coco_data: Dict[str, Any]) -> bool:
        """
        Validate that input has valid COCO structure.

        Args:
            coco_data: Dataset to validate

        Returns:
            True if valid, False otherwise
        """
        required_keys = ['images', 'annotations', 'categories']

        for key in required_keys:
            if key not in coco_data:
                logger.error(f"Missing required key in COCO data: {key}")
                return False

        if not isinstance(coco_data['images'], list):
            logger.error("'images' must be a list")
            return False

        if not isinstance(coco_data['annotations'], list):
            logger.error("'annotations' must be a list")
            return False

        if not isinstance(coco_data['categories'], list):
            logger.error("'categories' must be a list")
            return False

        return True

    def _build_image_lookup(self, images: List[Dict]) -> Dict[int, Dict]:
        """
        Build a lookup dictionary for images by ID.

        Args:
            images: List of image dictionaries

        Returns:
            Dictionary mapping image_id to image info
        """
        return {img['id']: img for img in images}

    def _build_category_lookup(self, categories: List[Dict]) -> Dict[int, Dict]:
        """
        Build a lookup dictionary for categories by ID.

        Args:
            categories: List of category dictionaries

        Returns:
            Dictionary mapping category_id to category info
        """
        return {cat['id']: cat for cat in categories}

    def _group_annotations_by_image(self, annotations: List[Dict]) -> Dict[int, List[Dict]]:
        """
        Group annotations by image_id.

        Args:
            annotations: List of annotation dictionaries

        Returns:
            Dictionary mapping image_id to list of annotations
        """
        grouped = {}
        for ann in annotations:
            img_id = ann['image_id']
            if img_id not in grouped:
                grouped[img_id] = []
            grouped[img_id].append(ann)
        return grouped
