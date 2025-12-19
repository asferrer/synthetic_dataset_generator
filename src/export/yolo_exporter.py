"""
YOLO format exporter - creates per-image .txt label files.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List
import shutil

from .base_exporter import BaseExporter, ExportConfig, ExportResult

logger = logging.getLogger(__name__)


class YOLOExporter(BaseExporter):
    """
    Exporter for YOLO format (per-image .txt label files).

    Output structure:
        dataset_yolo/
        ├── images/
        │   ├── img001.jpg
        │   └── ...
        ├── labels/
        │   ├── img001.txt
        │   └── ...
        ├── classes.txt
        └── data.yaml (optional, for YOLOv5+)
    """

    def __init__(self, config: ExportConfig,
                 create_classes_file: bool = True,
                 create_yaml: bool = True):
        """
        Initialize YOLO exporter.

        Args:
            config: Export configuration
            create_classes_file: Whether to create classes.txt
            create_yaml: Whether to create data.yaml for YOLOv5+
        """
        super().__init__(config)
        self.create_classes_file = create_classes_file
        self.create_yaml = create_yaml

    def get_format_name(self) -> str:
        """Return format name."""
        return "YOLO"

    def export(self,
               coco_data: Dict[str, Any],
               images_dir: str,
               output_name: str = "dataset") -> ExportResult:
        """
        Export to YOLO format.

        YOLO format per line: class_id x_center y_center width height
        All coordinates are normalized to [0, 1].

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
            # Setup output directories
            output_base = Path(self.config.output_dir) / output_name
            images_out = output_base / "images"
            labels_out = output_base / "labels"

            images_out.mkdir(parents=True, exist_ok=True)
            labels_out.mkdir(parents=True, exist_ok=True)

            # Create category mapping (COCO ID -> YOLO 0-based index)
            sorted_categories = sorted(coco_data['categories'], key=lambda x: x['id'])
            cat_id_to_yolo_idx = {cat['id']: idx for idx, cat in enumerate(sorted_categories)}

            # Build lookups
            images_dict = self._build_image_lookup(coco_data['images'])
            anns_by_image = self._group_annotations_by_image(coco_data['annotations'])

            # Process each image
            processed_images = 0
            total_annotations = 0

            for img_id, img_info in images_dict.items():
                try:
                    # Copy/link image if configured
                    if self.config.include_images:
                        src_img = Path(images_dir) / img_info['file_name']
                        dst_img = images_out / img_info['file_name']

                        if src_img.exists():
                            if self.config.copy_images:
                                shutil.copy2(src_img, dst_img)
                        else:
                            warnings.append(f"Source image not found: {src_img}")

                    # Create label file
                    label_name = Path(img_info['file_name']).stem + ".txt"
                    label_path = labels_out / label_name

                    img_w = img_info['width']
                    img_h = img_info['height']

                    with open(label_path, 'w') as f:
                        annotations = anns_by_image.get(img_id, [])
                        for ann in annotations:
                            if ann['category_id'] not in cat_id_to_yolo_idx:
                                warnings.append(f"Unknown category_id {ann['category_id']} in annotation {ann['id']}")
                                continue

                            yolo_line = self._coco_to_yolo(ann, cat_id_to_yolo_idx, img_w, img_h)
                            if yolo_line:
                                f.write(yolo_line + "\n")
                                total_annotations += 1

                    processed_images += 1

                except Exception as e:
                    errors.append(f"Error processing {img_info['file_name']}: {e}")

            # Create classes.txt
            if self.create_classes_file:
                self._write_classes_file(sorted_categories, output_base)

            # Create data.yaml for YOLOv5+
            if self.create_yaml:
                self._write_yaml_file(sorted_categories, output_base, output_name)

            logger.info(f"YOLO export complete: {processed_images} images, {total_annotations} annotations")

            return ExportResult(
                success=len(errors) == 0,
                output_path=str(output_base),
                num_images=processed_images,
                num_annotations=total_annotations,
                format_name=self.get_format_name(),
                errors=errors,
                warnings=warnings
            )

        except Exception as e:
            logger.error(f"YOLO export failed: {e}")
            return ExportResult(
                success=False,
                output_path="",
                num_images=0,
                num_annotations=0,
                format_name=self.get_format_name(),
                errors=[str(e)]
            )

    def _coco_to_yolo(self, ann: Dict, cat_map: Dict[int, int],
                      img_w: int, img_h: int) -> str:
        """
        Convert COCO annotation to YOLO format line.

        Args:
            ann: COCO annotation dict
            cat_map: Mapping from COCO category_id to YOLO class index
            img_w: Image width
            img_h: Image height

        Returns:
            YOLO format string: "class_id x_center y_center width height"
        """
        if 'bbox' not in ann or len(ann['bbox']) != 4:
            return ""

        class_id = cat_map[ann['category_id']]
        x, y, w, h = ann['bbox']

        # Convert to center coordinates and normalize
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        # Clamp values to [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        w_norm = max(0, min(1, w_norm))
        h_norm = max(0, min(1, h_norm))

        return f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"

    def _write_classes_file(self, categories: List[Dict], output_dir: Path) -> None:
        """
        Write classes.txt file with category names.

        Args:
            categories: Sorted list of categories
            output_dir: Output directory
        """
        classes_path = output_dir / "classes.txt"
        with open(classes_path, 'w') as f:
            for cat in categories:
                f.write(cat['name'] + "\n")
        logger.info(f"Classes file saved to {classes_path}")

    def _write_yaml_file(self, categories: List[Dict], output_dir: Path, name: str) -> None:
        """
        Write data.yaml file for YOLOv5/v8 training.

        Args:
            categories: Sorted list of categories
            output_dir: Output directory
            name: Dataset name
        """
        yaml_path = output_dir / "data.yaml"

        # Build class names list
        names = [cat['name'] for cat in categories]

        yaml_content = f"""# Dataset configuration for YOLO training
# Generated by Synthetic Dataset Generator

path: {output_dir.absolute()}
train: images
val: images
test: images

# Number of classes
nc: {len(names)}

# Class names
names: {names}
"""

        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        logger.info(f"YAML config saved to {yaml_path}")
