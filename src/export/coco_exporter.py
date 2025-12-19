"""
Enhanced COCO format exporter with complete metadata support.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import shutil

from .base_exporter import BaseExporter, ExportConfig, ExportResult

logger = logging.getLogger(__name__)


class COCOExporter(BaseExporter):
    """Exporter for enhanced COCO JSON format with complete metadata."""

    def __init__(self, config: ExportConfig):
        """
        Initialize COCO exporter.

        Args:
            config: Export configuration
        """
        super().__init__(config)

    def get_format_name(self) -> str:
        """Return format name."""
        return "COCO"

    def export(self,
               coco_data: Dict[str, Any],
               images_dir: str,
               output_name: str = "dataset") -> ExportResult:
        """
        Export to enhanced COCO format.

        Enhancements over basic COCO:
        - Complete metadata (info, licenses)
        - Automatic area calculation if missing
        - Integrity validation

        Args:
            coco_data: Dataset in COCO format
            images_dir: Directory containing source images
            output_name: Base name for output

        Returns:
            ExportResult with export details
        """
        errors = []
        warnings = []

        # Validate input
        if not self.validate_input(coco_data):
            return ExportResult(
                success=False,
                output_path="",
                num_images=0,
                num_annotations=0,
                format_name=self.get_format_name(),
                errors=["Invalid COCO data structure"]
            )

        try:
            # Create enhanced copy
            enhanced_data = self._enhance_coco_data(coco_data, warnings)

            # Setup output paths
            output_base = Path(self.config.output_dir) / output_name
            output_base.mkdir(parents=True, exist_ok=True)

            # Write JSON
            json_path = output_base / f"{output_name}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_data, f, indent=2, ensure_ascii=False)

            logger.info(f"COCO JSON saved to {json_path}")

            # Copy images if configured
            if self.config.include_images and self.config.copy_images:
                images_out = output_base / "images"
                images_out.mkdir(exist_ok=True)
                self._copy_images(coco_data['images'], images_dir, images_out, errors)

            return ExportResult(
                success=len(errors) == 0,
                output_path=str(output_base),
                num_images=len(enhanced_data['images']),
                num_annotations=len(enhanced_data['annotations']),
                format_name=self.get_format_name(),
                errors=errors,
                warnings=warnings
            )

        except Exception as e:
            logger.error(f"COCO export failed: {e}")
            return ExportResult(
                success=False,
                output_path="",
                num_images=0,
                num_annotations=0,
                format_name=self.get_format_name(),
                errors=[str(e)]
            )

    def _enhance_coco_data(self, data: Dict[str, Any], warnings: List[str]) -> Dict[str, Any]:
        """
        Enhance COCO data with metadata and calculated fields.

        Args:
            data: Original COCO data
            warnings: List to append warnings to

        Returns:
            Enhanced COCO data
        """
        enhanced = {
            'info': self._get_info(data),
            'licenses': data.get('licenses', []),
            'categories': data['categories'],
            'images': data['images'],
            'annotations': self._ensure_annotations_complete(data['annotations'], warnings)
        }
        return enhanced

    def _get_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get or create info section."""
        if 'info' in data and data['info']:
            return data['info']

        return {
            'description': 'Synthetic Dataset',
            'url': '',
            'version': '1.0',
            'year': datetime.now().year,
            'contributor': 'Synthetic Dataset Generator',
            'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def _ensure_annotations_complete(self, annotations: List[Dict], warnings: List[str]) -> List[Dict]:
        """
        Ensure all annotations have required fields.

        Args:
            annotations: List of annotations
            warnings: List to append warnings

        Returns:
            List of complete annotations
        """
        complete_annotations = []

        for ann in annotations:
            ann_copy = ann.copy()

            # Ensure area is calculated
            if 'area' not in ann_copy or ann_copy['area'] == 0:
                if 'bbox' in ann_copy and len(ann_copy['bbox']) == 4:
                    _, _, w, h = ann_copy['bbox']
                    ann_copy['area'] = w * h
                    warnings.append(f"Calculated area for annotation {ann_copy.get('id', 'unknown')}")

            # Ensure iscrowd field exists
            if 'iscrowd' not in ann_copy:
                ann_copy['iscrowd'] = 0

            # Ensure segmentation field exists (empty if not present)
            if 'segmentation' not in ann_copy:
                ann_copy['segmentation'] = []

            complete_annotations.append(ann_copy)

        return complete_annotations

    def _copy_images(self, images: List[Dict], src_dir: str, dst_dir: Path, errors: List[str]) -> None:
        """
        Copy images to output directory.

        Args:
            images: List of image info dicts
            src_dir: Source directory
            dst_dir: Destination directory
            errors: List to append errors
        """
        for img in images:
            try:
                src_path = Path(src_dir) / img['file_name']
                dst_path = dst_dir / img['file_name']

                if src_path.exists():
                    shutil.copy2(src_path, dst_path)
                else:
                    errors.append(f"Source image not found: {src_path}")
            except Exception as e:
                errors.append(f"Error copying {img['file_name']}: {e}")
