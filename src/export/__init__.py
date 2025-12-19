"""
Export module for converting datasets to multiple annotation formats.

Supported formats:
- COCO: Standard COCO JSON format with enhanced metadata
- YOLO: Per-image .txt files with normalized coordinates
- COCO Segmentation: COCO format with polygon segmentations
- Pascal VOC: XML format per image
"""

from .base_exporter import BaseExporter, ExportConfig, ExportResult
from .coco_exporter import COCOExporter
from .yolo_exporter import YOLOExporter
from .coco_segmentation import COCOSegmentationExporter
from .pascal_voc_exporter import PascalVOCExporter
from .export_manager import ExportManager

__all__ = [
    'BaseExporter',
    'ExportConfig',
    'ExportResult',
    'COCOExporter',
    'YOLOExporter',
    'COCOSegmentationExporter',
    'PascalVOCExporter',
    'ExportManager'
]
