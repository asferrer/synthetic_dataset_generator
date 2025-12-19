"""
Pascal VOC format exporter.

Exports annotations in Pascal VOC XML format.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from xml.etree import ElementTree as ET
from xml.dom import minidom

from .base_exporter import BaseExporter, ExportConfig, ExportResult

logger = logging.getLogger(__name__)


class PascalVOCExporter(BaseExporter):
    """
    Exports annotations in Pascal VOC XML format.

    Output structure:
    output_dir/
    ├── Annotations/
    │   ├── image_001.xml
    │   └── ...
    ├── JPEGImages/
    │   ├── image_001.jpg
    │   └── ...
    └── ImageSets/
        └── Main/
            └── trainval.txt
    """

    def __init__(self, config: ExportConfig):
        """
        Initialize Pascal VOC exporter.

        Args:
            config: Export configuration
        """
        super().__init__(config)
        self.include_difficult = config.extra_params.get('include_difficult', False)
        self.include_truncated = config.extra_params.get('include_truncated', False)
        self.include_segmented = config.extra_params.get('include_segmented', False)

    def get_format_name(self) -> str:
        return "pascal_voc"

    def export(self, coco_data: Dict[str, Any],
               images_dir: str,
               output_name: str = "dataset") -> ExportResult:
        """
        Export COCO data to Pascal VOC format.

        Args:
            coco_data: COCO format dataset
            images_dir: Directory containing source images
            output_name: Base name for output

        Returns:
            ExportResult with status and details
        """
        try:
            # Create output directories
            annotations_dir = self.output_dir / "Annotations"
            images_out_dir = self.output_dir / "JPEGImages"
            imagesets_dir = self.output_dir / "ImageSets" / "Main"

            annotations_dir.mkdir(parents=True, exist_ok=True)
            images_out_dir.mkdir(parents=True, exist_ok=True)
            imagesets_dir.mkdir(parents=True, exist_ok=True)

            # Build category mapping
            cat_id_to_name = {
                cat['id']: cat['name']
                for cat in coco_data['categories']
            }

            # Build image to annotations mapping
            img_to_anns = {}
            for ann in coco_data['annotations']:
                img_id = ann['image_id']
                if img_id not in img_to_anns:
                    img_to_anns[img_id] = []
                img_to_anns[img_id].append(ann)

            # Process each image
            exported_images = []
            errors = []

            for img in coco_data['images']:
                img_id = img['id']
                file_name = img['file_name']

                # Get base name without extension
                base_name = Path(file_name).stem

                # Create XML annotation
                xml_content = self._create_xml_annotation(
                    img,
                    img_to_anns.get(img_id, []),
                    cat_id_to_name
                )

                # Save XML
                xml_path = annotations_dir / f"{base_name}.xml"
                with open(xml_path, 'w', encoding='utf-8') as f:
                    f.write(xml_content)

                # Copy image if requested
                if self.config.copy_images:
                    src_path = Path(images_dir) / file_name
                    dst_path = images_out_dir / file_name

                    if src_path.exists():
                        shutil.copy2(src_path, dst_path)
                    else:
                        errors.append(f"Image not found: {src_path}")

                exported_images.append(base_name)

            # Create ImageSets file
            imageset_path = imagesets_dir / "trainval.txt"
            with open(imageset_path, 'w') as f:
                f.write('\n'.join(exported_images))

            # Also create individual train/val/test files if split info exists
            splits = self._identify_splits(coco_data['images'])
            for split_name, split_images in splits.items():
                split_path = imagesets_dir / f"{split_name}.txt"
                with open(split_path, 'w') as f:
                    f.write('\n'.join(split_images))

            logger.info(f"Exported {len(exported_images)} images to Pascal VOC format")

            return ExportResult(
                success=True,
                format_name=self.get_format_name(),
                output_path=str(self.output_dir),
                files_created=[
                    str(annotations_dir),
                    str(images_out_dir),
                    str(imageset_path)
                ],
                stats={
                    'images_exported': len(exported_images),
                    'annotations_exported': len(coco_data['annotations']),
                    'categories': len(coco_data['categories'])
                },
                errors=errors if errors else None
            )

        except Exception as e:
            logger.error(f"Pascal VOC export failed: {e}")
            return ExportResult(
                success=False,
                format_name=self.get_format_name(),
                output_path=str(self.output_dir),
                errors=[str(e)]
            )

    def _create_xml_annotation(self,
                                img: Dict[str, Any],
                                annotations: List[Dict[str, Any]],
                                cat_id_to_name: Dict[int, str]) -> str:
        """
        Create Pascal VOC XML annotation for an image.

        Args:
            img: Image info dict
            annotations: List of annotations for this image
            cat_id_to_name: Category ID to name mapping

        Returns:
            XML string
        """
        # Create root element
        annotation = ET.Element('annotation')

        # Add folder
        folder = ET.SubElement(annotation, 'folder')
        folder.text = 'JPEGImages'

        # Add filename
        filename = ET.SubElement(annotation, 'filename')
        filename.text = img['file_name']

        # Add path
        path = ET.SubElement(annotation, 'path')
        path.text = f"JPEGImages/{img['file_name']}"

        # Add source
        source = ET.SubElement(annotation, 'source')
        database = ET.SubElement(source, 'database')
        database.text = 'Synthetic Dataset Generator'

        # Add size
        size = ET.SubElement(annotation, 'size')
        width_el = ET.SubElement(size, 'width')
        width_el.text = str(img.get('width', 0))
        height_el = ET.SubElement(size, 'height')
        height_el.text = str(img.get('height', 0))
        depth = ET.SubElement(size, 'depth')
        depth.text = '3'

        # Add segmented flag
        segmented = ET.SubElement(annotation, 'segmented')
        segmented.text = '1' if self.include_segmented else '0'

        # Add objects
        for ann in annotations:
            self._add_object_element(annotation, ann, cat_id_to_name)

        # Pretty print
        xml_str = ET.tostring(annotation, encoding='unicode')
        dom = minidom.parseString(xml_str)
        return dom.toprettyxml(indent='    ')

    def _add_object_element(self,
                            parent: ET.Element,
                            ann: Dict[str, Any],
                            cat_id_to_name: Dict[int, str]) -> None:
        """
        Add object element to XML annotation.

        Args:
            parent: Parent XML element
            ann: Annotation dict
            cat_id_to_name: Category ID to name mapping
        """
        obj = ET.SubElement(parent, 'object')

        # Name (class)
        name = ET.SubElement(obj, 'name')
        name.text = cat_id_to_name.get(ann['category_id'], 'unknown')

        # Pose
        pose = ET.SubElement(obj, 'pose')
        pose.text = 'Unspecified'

        # Truncated
        truncated = ET.SubElement(obj, 'truncated')
        truncated.text = '1' if ann.get('iscrowd', 0) or self.include_truncated else '0'

        # Difficult
        difficult = ET.SubElement(obj, 'difficult')
        difficult.text = '1' if self.include_difficult else '0'

        # Bounding box
        bbox = ann.get('bbox', [0, 0, 0, 0])
        x, y, w, h = bbox

        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(int(x))
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(int(y))
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(int(x + w))
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(int(y + h))

    def _identify_splits(self, images: List[Dict]) -> Dict[str, List[str]]:
        """
        Identify splits from image metadata if available.

        Args:
            images: List of image dicts

        Returns:
            Dict of split_name -> list of image base names
        """
        splits = {}

        for img in images:
            split = img.get('split', img.get('source_dataset'))
            if split:
                if split not in splits:
                    splits[split] = []
                splits[split].append(Path(img['file_name']).stem)

        return splits
