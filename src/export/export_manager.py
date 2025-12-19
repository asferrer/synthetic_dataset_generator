"""
Export Manager - Orchestrator for multi-format dataset exports.
"""

import logging
from typing import Dict, List, Optional, Any, Type
from pathlib import Path

from .base_exporter import BaseExporter, ExportConfig, ExportResult
from .coco_exporter import COCOExporter
from .yolo_exporter import YOLOExporter
from .coco_segmentation import COCOSegmentationExporter
from .pascal_voc_exporter import PascalVOCExporter

logger = logging.getLogger(__name__)


class ExportManager:
    """
    Orchestrator for exporting datasets to multiple formats.

    Usage:
        manager = ExportManager("/output/path")
        manager.configure_format('yolo', ExportConfig(...))
        results = manager.export_all(coco_data, images_dir, ['coco', 'yolo'])
    """

    AVAILABLE_FORMATS: Dict[str, Type[BaseExporter]] = {
        'coco': COCOExporter,
        'yolo': YOLOExporter,
        'coco_segmentation': COCOSegmentationExporter,
        'pascal_voc': PascalVOCExporter
    }

    def __init__(self, output_dir: str):
        """
        Initialize export manager.

        Args:
            output_dir: Base directory for exports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.exporters: Dict[str, BaseExporter] = {}
        self._configs: Dict[str, ExportConfig] = {}

    def configure_format(self, format_name: str,
                         config: Optional[ExportConfig] = None,
                         **kwargs) -> None:
        """
        Configure an export format.

        Args:
            format_name: Name of format ('coco', 'yolo', 'coco_segmentation')
            config: Export configuration (creates default if None)
            **kwargs: Additional kwargs passed to exporter constructor
        """
        if format_name not in self.AVAILABLE_FORMATS:
            raise ValueError(f"Unknown format: {format_name}. Available: {list(self.AVAILABLE_FORMATS.keys())}")

        if config is None:
            config = ExportConfig(output_dir=str(self.output_dir / format_name))
        else:
            # Update output_dir to be format-specific
            config.output_dir = str(self.output_dir / format_name)

        self._configs[format_name] = config

        exporter_cls = self.AVAILABLE_FORMATS[format_name]
        self.exporters[format_name] = exporter_cls(config, **kwargs)

        logger.info(f"Configured exporter for format: {format_name}")

    def export_format(self, format_name: str,
                      coco_data: Dict[str, Any],
                      images_dir: str,
                      output_name: str = "dataset",
                      **kwargs) -> ExportResult:
        """
        Export to a single format.

        Args:
            format_name: Format to export to
            coco_data: Dataset in COCO format
            images_dir: Directory containing images
            output_name: Base name for output
            **kwargs: Additional kwargs for specific exporters

        Returns:
            ExportResult
        """
        if format_name not in self.exporters:
            self.configure_format(format_name)

        exporter = self.exporters[format_name]

        logger.info(f"Exporting to {format_name} format...")
        result = exporter.export(coco_data, images_dir, output_name, **kwargs)

        if result.success:
            logger.info(f"Successfully exported to {format_name}: {result.output_path}")
        else:
            logger.error(f"Failed to export to {format_name}: {result.errors}")

        return result

    def export_all(self, coco_data: Dict[str, Any],
                   images_dir: str,
                   formats: Optional[List[str]] = None,
                   output_name: str = "dataset") -> Dict[str, ExportResult]:
        """
        Export to multiple formats.

        Args:
            coco_data: Dataset in COCO format
            images_dir: Directory containing images
            formats: List of formats to export (None = all configured)
            output_name: Base name for output

        Returns:
            Dictionary mapping format name to ExportResult
        """
        if formats is None:
            formats = list(self.exporters.keys()) if self.exporters else list(self.AVAILABLE_FORMATS.keys())

        results = {}
        for fmt in formats:
            results[fmt] = self.export_format(fmt, coco_data, images_dir, output_name)

        # Summary logging
        successful = sum(1 for r in results.values() if r.success)
        logger.info(f"Export complete: {successful}/{len(results)} formats successful")

        return results

    def export_splits(self, splits: Dict[str, Dict[str, Any]],
                      images_base_dir: str,
                      formats: Optional[List[str]] = None) -> Dict[str, Dict[str, ExportResult]]:
        """
        Export multiple splits (train/val/test) to multiple formats.

        Args:
            splits: Dictionary with split names as keys and COCO data as values
                   e.g., {'train': {...}, 'val': {...}, 'test': {...}}
            images_base_dir: Base directory for images (split subdirs expected)
            formats: List of formats to export

        Returns:
            Nested dictionary: {split_name: {format_name: ExportResult}}
        """
        if formats is None:
            formats = list(self.AVAILABLE_FORMATS.keys())

        all_results = {}

        for split_name, split_data in splits.items():
            logger.info(f"Exporting split: {split_name}")

            # Determine images directory for this split
            split_images_dir = Path(images_base_dir) / split_name / "images"
            if not split_images_dir.exists():
                split_images_dir = Path(images_base_dir)  # Fall back to base dir

            split_results = {}
            for fmt in formats:
                # Configure with split-specific output
                split_output_dir = self.output_dir / split_name / fmt
                config = ExportConfig(output_dir=str(split_output_dir))

                exporter_cls = self.AVAILABLE_FORMATS[fmt]
                exporter = exporter_cls(config)

                result = exporter.export(split_data, str(split_images_dir), split_name)
                split_results[fmt] = result

            all_results[split_name] = split_results

        return all_results

    @classmethod
    def get_available_formats(cls) -> List[str]:
        """
        Get list of available export formats.

        Returns:
            List of format names
        """
        return list(cls.AVAILABLE_FORMATS.keys())

    @classmethod
    def get_format_description(cls, format_name: str) -> str:
        """
        Get description of a format.

        Args:
            format_name: Name of format

        Returns:
            Description string
        """
        descriptions = {
            'coco': 'COCO JSON format with enhanced metadata',
            'yolo': 'YOLO format (.txt per image, normalized coordinates)',
            'coco_segmentation': 'COCO format with polygon segmentations',
            'pascal_voc': 'Pascal VOC XML format (per image annotations)'
        }
        return descriptions.get(format_name, f"Unknown format: {format_name}")

    def get_export_summary(self, results: Dict[str, ExportResult]) -> str:
        """
        Generate summary of export results.

        Args:
            results: Export results dictionary

        Returns:
            Formatted summary string
        """
        lines = ["Export Summary", "=" * 40]

        for fmt, result in results.items():
            status = "OK" if result.success else "FAILED"
            lines.append(f"\n{fmt.upper()} [{status}]")
            lines.append(f"  Output: {result.output_path}")
            lines.append(f"  Images: {result.num_images}")
            lines.append(f"  Annotations: {result.num_annotations}")

            if result.errors:
                lines.append(f"  Errors: {len(result.errors)}")
                for err in result.errors[:3]:  # Show first 3 errors
                    lines.append(f"    - {err}")
                if len(result.errors) > 3:
                    lines.append(f"    ... and {len(result.errors) - 3} more")

            if result.warnings:
                lines.append(f"  Warnings: {len(result.warnings)}")

        return "\n".join(lines)
