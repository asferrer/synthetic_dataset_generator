"""
Dataset Exporters - Export to multiple annotation formats.

Supported formats:
- COCO JSON
- YOLO (txt per image)
- Pascal VOC (XML per image)
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from xml.etree import ElementTree as ET
from xml.dom import minidom
from copy import deepcopy

logger = logging.getLogger(__name__)


def export_to_coco(coco_data: Dict[str, Any],
                   output_dir: str,
                   output_name: str = "dataset",
                   copy_images: bool = False,
                   images_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Export dataset in COCO JSON format.

    Args:
        coco_data: COCO format dataset
        output_dir: Output directory
        output_name: Base name for output files
        copy_images: Whether to copy images
        images_dir: Source images directory (required if copy_images=True)

    Returns:
        Export result with status and paths
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = output_path / f"{output_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)

        # Copy images if requested
        if copy_images and images_dir:
            images_out = output_path / "images"
            images_out.mkdir(exist_ok=True)

            copied = 0
            for img in coco_data.get('images', []):
                src = Path(images_dir) / img['file_name']
                if src.exists():
                    shutil.copy2(src, images_out / img['file_name'])
                    copied += 1

            logger.info(f"Copied {copied} images")

        return {
            'success': True,
            'format': 'coco',
            'output_path': str(json_path),
            'num_images': len(coco_data.get('images', [])),
            'num_annotations': len(coco_data.get('annotations', [])),
            'num_categories': len(coco_data.get('categories', []))
        }

    except Exception as e:
        logger.error(f"COCO export failed: {e}")
        return {'success': False, 'error': str(e)}


def export_to_yolo(coco_data: Dict[str, Any],
                   output_dir: str,
                   copy_images: bool = False,
                   images_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Export dataset in YOLO format.

    Output structure:
    output_dir/
    ├── images/
    │   └── *.jpg
    ├── labels/
    │   └── *.txt
    ├── classes.txt
    └── data.yaml

    Args:
        coco_data: COCO format dataset
        output_dir: Output directory
        copy_images: Whether to copy images
        images_dir: Source images directory

    Returns:
        Export result with status and paths
    """
    try:
        output_path = Path(output_dir)
        images_out = output_path / "images"
        labels_out = output_path / "labels"

        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        # Build category mapping (YOLO uses 0-indexed class IDs)
        categories = sorted(coco_data['categories'], key=lambda x: x['id'])
        cat_id_to_yolo_id = {cat['id']: i for i, cat in enumerate(categories)}
        cat_names = [cat['name'] for cat in categories]

        # Build image info mapping
        img_info = {img['id']: img for img in coco_data['images']}

        # Group annotations by image
        img_to_anns = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)

        # Process each image
        exported_count = 0
        for img in coco_data['images']:
            img_id = img['id']
            file_name = img['file_name']
            base_name = Path(file_name).stem

            img_w = img.get('width', 1)
            img_h = img.get('height', 1)

            # Create label file
            label_lines = []
            for ann in img_to_anns.get(img_id, []):
                bbox = ann.get('bbox', [])
                if len(bbox) != 4:
                    continue

                x, y, w, h = bbox
                cat_id = ann['category_id']
                yolo_id = cat_id_to_yolo_id.get(cat_id)

                if yolo_id is None:
                    continue

                # Convert to YOLO format (normalized x_center, y_center, width, height)
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h

                label_lines.append(f"{yolo_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

            # Write label file
            label_path = labels_out / f"{base_name}.txt"
            with open(label_path, 'w') as f:
                f.write('\n'.join(label_lines))

            # Copy image if requested
            if copy_images and images_dir:
                src = Path(images_dir) / file_name
                if src.exists():
                    shutil.copy2(src, images_out / file_name)

            exported_count += 1

        # Write classes.txt
        classes_path = output_path / "classes.txt"
        with open(classes_path, 'w') as f:
            f.write('\n'.join(cat_names))

        # Write data.yaml for YOLO training
        yaml_path = output_path / "data.yaml"
        yaml_content = f"""# YOLO Dataset Configuration
path: {output_path.absolute()}
train: images
val: images

nc: {len(cat_names)}
names: {cat_names}
"""
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

        return {
            'success': True,
            'format': 'yolo',
            'output_path': str(output_path),
            'num_images': exported_count,
            'num_annotations': len(coco_data.get('annotations', [])),
            'files': {
                'images': str(images_out),
                'labels': str(labels_out),
                'classes': str(classes_path),
                'yaml': str(yaml_path)
            }
        }

    except Exception as e:
        logger.error(f"YOLO export failed: {e}")
        return {'success': False, 'error': str(e)}


def export_to_pascal_voc(coco_data: Dict[str, Any],
                         output_dir: str,
                         copy_images: bool = False,
                         images_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Export dataset in Pascal VOC XML format.

    Output structure:
    output_dir/
    ├── Annotations/
    │   └── *.xml
    ├── JPEGImages/
    │   └── *.jpg
    └── ImageSets/
        └── Main/
            └── trainval.txt

    Args:
        coco_data: COCO format dataset
        output_dir: Output directory
        copy_images: Whether to copy images
        images_dir: Source images directory

    Returns:
        Export result with status and paths
    """
    try:
        output_path = Path(output_dir)
        annotations_dir = output_path / "Annotations"
        images_out_dir = output_path / "JPEGImages"
        imagesets_dir = output_path / "ImageSets" / "Main"

        annotations_dir.mkdir(parents=True, exist_ok=True)
        images_out_dir.mkdir(parents=True, exist_ok=True)
        imagesets_dir.mkdir(parents=True, exist_ok=True)

        # Build category mapping
        cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

        # Group annotations by image
        img_to_anns = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)

        # Process each image
        exported_images = []
        for img in coco_data['images']:
            img_id = img['id']
            file_name = img['file_name']
            base_name = Path(file_name).stem

            # Create XML annotation
            annotation = ET.Element('annotation')

            # Folder
            folder = ET.SubElement(annotation, 'folder')
            folder.text = 'JPEGImages'

            # Filename
            filename = ET.SubElement(annotation, 'filename')
            filename.text = file_name

            # Path
            path_el = ET.SubElement(annotation, 'path')
            path_el.text = f"JPEGImages/{file_name}"

            # Source
            source = ET.SubElement(annotation, 'source')
            database = ET.SubElement(source, 'database')
            database.text = 'Synthetic Dataset Generator'

            # Size
            size = ET.SubElement(annotation, 'size')
            width_el = ET.SubElement(size, 'width')
            width_el.text = str(img.get('width', 0))
            height_el = ET.SubElement(size, 'height')
            height_el.text = str(img.get('height', 0))
            depth = ET.SubElement(size, 'depth')
            depth.text = '3'

            # Segmented
            segmented = ET.SubElement(annotation, 'segmented')
            segmented.text = '0'

            # Objects
            for ann in img_to_anns.get(img_id, []):
                obj = ET.SubElement(annotation, 'object')

                name = ET.SubElement(obj, 'name')
                name.text = cat_id_to_name.get(ann['category_id'], 'unknown')

                pose = ET.SubElement(obj, 'pose')
                pose.text = 'Unspecified'

                truncated = ET.SubElement(obj, 'truncated')
                truncated.text = '0'

                difficult = ET.SubElement(obj, 'difficult')
                difficult.text = '0'

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

            # Pretty print XML
            xml_str = ET.tostring(annotation, encoding='unicode')
            dom = minidom.parseString(xml_str)
            pretty_xml = dom.toprettyxml(indent='    ')

            # Save XML
            xml_path = annotations_dir / f"{base_name}.xml"
            with open(xml_path, 'w', encoding='utf-8') as f:
                f.write(pretty_xml)

            # Copy image if requested
            if copy_images and images_dir:
                src = Path(images_dir) / file_name
                if src.exists():
                    shutil.copy2(src, images_out_dir / file_name)

            exported_images.append(base_name)

        # Create ImageSets file
        imageset_path = imagesets_dir / "trainval.txt"
        with open(imageset_path, 'w') as f:
            f.write('\n'.join(exported_images))

        return {
            'success': True,
            'format': 'pascal_voc',
            'output_path': str(output_path),
            'num_images': len(exported_images),
            'num_annotations': len(coco_data.get('annotations', [])),
            'files': {
                'annotations': str(annotations_dir),
                'images': str(images_out_dir),
                'imageset': str(imageset_path)
            }
        }

    except Exception as e:
        logger.error(f"Pascal VOC export failed: {e}")
        return {'success': False, 'error': str(e)}


class ExportManager:
    """Manager for multi-format exports."""

    FORMATS = {
        'coco': export_to_coco,
        'yolo': export_to_yolo,
        'pascal_voc': export_to_pascal_voc
    }

    @classmethod
    def export(cls,
               coco_data: Dict[str, Any],
               output_dir: str,
               formats: List[str],
               copy_images: bool = False,
               images_dir: Optional[str] = None,
               output_name: str = "dataset") -> Dict[str, Any]:
        """
        Export to multiple formats.

        Args:
            coco_data: COCO format dataset
            output_dir: Base output directory
            formats: List of formats to export
            copy_images: Whether to copy images
            images_dir: Source images directory
            output_name: Base name for output

        Returns:
            Dictionary with results per format
        """
        results = {}

        for fmt in formats:
            if fmt not in cls.FORMATS:
                results[fmt] = {'success': False, 'error': f'Unknown format: {fmt}'}
                continue

            fmt_output = str(Path(output_dir) / fmt)

            if fmt == 'coco':
                results[fmt] = export_to_coco(
                    coco_data, fmt_output, output_name,
                    copy_images, images_dir
                )
            elif fmt == 'yolo':
                results[fmt] = export_to_yolo(
                    coco_data, fmt_output, copy_images, images_dir
                )
            elif fmt == 'pascal_voc':
                results[fmt] = export_to_pascal_voc(
                    coco_data, fmt_output, copy_images, images_dir
                )

        return results

    @classmethod
    def get_available_formats(cls) -> List[str]:
        """Get list of available export formats."""
        return list(cls.FORMATS.keys())
