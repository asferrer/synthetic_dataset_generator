"""
Label Manager - Tools for modifying dataset labels.

Provides functionality for:
- Renaming labels
- Deleting labels and their annotations
- Adding new empty classes
- Merging labels
- Converting segmentations to bounding boxes
"""

import logging
from typing import Dict, List, Any, Optional
from copy import deepcopy

logger = logging.getLogger(__name__)


class LabelManager:
    """
    Manager for dataset label operations.

    All operations return a new dataset without modifying the original.
    """

    @staticmethod
    def rename_label(coco_data: Dict[str, Any],
                     old_name: str,
                     new_name: str) -> Dict[str, Any]:
        """
        Rename a label in the dataset.

        Args:
            coco_data: COCO format dataset
            old_name: Current label name
            new_name: New label name

        Returns:
            Modified COCO dataset
        """
        result = deepcopy(coco_data)

        renamed = False
        for cat in result['categories']:
            if cat['name'] == old_name:
                cat['name'] = new_name
                renamed = True
                logger.info(f"Renamed label '{old_name}' to '{new_name}'")
                break

        if not renamed:
            logger.warning(f"Label '{old_name}' not found in dataset")

        return result

    @staticmethod
    def rename_labels_batch(coco_data: Dict[str, Any],
                            name_mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        Rename multiple labels in the dataset.

        Args:
            coco_data: COCO format dataset
            name_mapping: Dict of old_name -> new_name

        Returns:
            Modified COCO dataset
        """
        result = deepcopy(coco_data)

        for cat in result['categories']:
            if cat['name'] in name_mapping:
                old_name = cat['name']
                cat['name'] = name_mapping[old_name]
                logger.info(f"Renamed label '{old_name}' to '{cat['name']}'")

        return result

    @staticmethod
    def delete_label(coco_data: Dict[str, Any],
                     label_name: str,
                     delete_annotations: bool = True) -> Dict[str, Any]:
        """
        Delete a label from the dataset.

        Args:
            coco_data: COCO format dataset
            label_name: Name of label to delete
            delete_annotations: If True, also delete all annotations with this label

        Returns:
            Modified COCO dataset
        """
        result = deepcopy(coco_data)

        cat_id_to_delete = None
        for cat in result['categories']:
            if cat['name'] == label_name:
                cat_id_to_delete = cat['id']
                break

        if cat_id_to_delete is None:
            logger.warning(f"Label '{label_name}' not found in dataset")
            return result

        result['categories'] = [
            cat for cat in result['categories']
            if cat['name'] != label_name
        ]

        if delete_annotations:
            original_count = len(result['annotations'])
            result['annotations'] = [
                ann for ann in result['annotations']
                if ann['category_id'] != cat_id_to_delete
            ]
            removed_count = original_count - len(result['annotations'])
            logger.info(f"Deleted label '{label_name}' and {removed_count} associated annotations")
        else:
            logger.info(f"Deleted label '{label_name}' (annotations kept but orphaned)")

        return result

    @staticmethod
    def delete_labels_batch(coco_data: Dict[str, Any],
                            label_names: List[str],
                            delete_annotations: bool = True) -> Dict[str, Any]:
        """
        Delete multiple labels from the dataset.

        Args:
            coco_data: COCO format dataset
            label_names: List of label names to delete
            delete_annotations: If True, also delete all annotations

        Returns:
            Modified COCO dataset
        """
        result = deepcopy(coco_data)

        cat_ids_to_delete = set()
        for cat in result['categories']:
            if cat['name'] in label_names:
                cat_ids_to_delete.add(cat['id'])

        result['categories'] = [
            cat for cat in result['categories']
            if cat['name'] not in label_names
        ]

        if delete_annotations and cat_ids_to_delete:
            original_count = len(result['annotations'])
            result['annotations'] = [
                ann for ann in result['annotations']
                if ann['category_id'] not in cat_ids_to_delete
            ]
            removed_count = original_count - len(result['annotations'])
            logger.info(f"Deleted {len(label_names)} labels and {removed_count} annotations")

        return result

    @staticmethod
    def add_label(coco_data: Dict[str, Any],
                  label_name: str,
                  supercategory: Optional[str] = None) -> Dict[str, Any]:
        """
        Add a new empty label to the dataset.

        Args:
            coco_data: COCO format dataset
            label_name: Name of new label
            supercategory: Optional supercategory for the label

        Returns:
            Modified COCO dataset
        """
        result = deepcopy(coco_data)

        existing_names = {cat['name'] for cat in result['categories']}
        if label_name in existing_names:
            logger.warning(f"Label '{label_name}' already exists in dataset")
            return result

        max_id = max((cat['id'] for cat in result['categories']), default=0)

        new_category = {
            'id': max_id + 1,
            'name': label_name,
            'supercategory': supercategory or label_name
        }
        result['categories'].append(new_category)

        logger.info(f"Added new label '{label_name}' with ID {max_id + 1}")

        return result

    @staticmethod
    def add_labels_batch(coco_data: Dict[str, Any],
                         label_names: List[str],
                         supercategory: Optional[str] = None) -> Dict[str, Any]:
        """
        Add multiple new empty labels to the dataset.

        Args:
            coco_data: COCO format dataset
            label_names: List of new label names
            supercategory: Optional supercategory for all labels

        Returns:
            Modified COCO dataset
        """
        result = deepcopy(coco_data)

        existing_names = {cat['name'] for cat in result['categories']}
        max_id = max((cat['id'] for cat in result['categories']), default=0)

        added_count = 0
        for label_name in label_names:
            if label_name in existing_names:
                logger.warning(f"Label '{label_name}' already exists, skipping")
                continue

            max_id += 1
            new_category = {
                'id': max_id,
                'name': label_name,
                'supercategory': supercategory or label_name
            }
            result['categories'].append(new_category)
            existing_names.add(label_name)
            added_count += 1

        logger.info(f"Added {added_count} new labels")

        return result

    @staticmethod
    def merge_labels(coco_data: Dict[str, Any],
                     source_labels: List[str],
                     target_label: str) -> Dict[str, Any]:
        """
        Merge multiple labels into a single label.

        Args:
            coco_data: COCO format dataset
            source_labels: Labels to merge (will be deleted)
            target_label: Label to merge into (will be created if doesn't exist)

        Returns:
            Modified COCO dataset
        """
        result = deepcopy(coco_data)

        target_cat_id = None
        for cat in result['categories']:
            if cat['name'] == target_label:
                target_cat_id = cat['id']
                break

        if target_cat_id is None:
            max_id = max((cat['id'] for cat in result['categories']), default=0)
            target_cat_id = max_id + 1
            result['categories'].append({
                'id': target_cat_id,
                'name': target_label,
                'supercategory': target_label
            })

        source_cat_ids = set()
        for cat in result['categories']:
            if cat['name'] in source_labels:
                source_cat_ids.add(cat['id'])

        updated_count = 0
        for ann in result['annotations']:
            if ann['category_id'] in source_cat_ids:
                ann['category_id'] = target_cat_id
                updated_count += 1

        result['categories'] = [
            cat for cat in result['categories']
            if cat['name'] not in source_labels
        ]

        logger.info(f"Merged {len(source_labels)} labels into '{target_label}', updated {updated_count} annotations")

        return result

    @staticmethod
    def get_label_statistics(coco_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for each label in the dataset.

        Args:
            coco_data: COCO format dataset

        Returns:
            Dict mapping label name to statistics
        """
        cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

        annotation_counts = {}
        image_counts = {}

        for ann in coco_data['annotations']:
            cat_id = ann['category_id']
            cat_name = cat_id_to_name.get(cat_id)
            if cat_name:
                annotation_counts[cat_name] = annotation_counts.get(cat_name, 0) + 1

                img_id = ann['image_id']
                if cat_name not in image_counts:
                    image_counts[cat_name] = set()
                image_counts[cat_name].add(img_id)

        stats = {}
        total_anns = len(coco_data['annotations'])
        for cat in coco_data['categories']:
            name = cat['name']
            stats[name] = {
                'id': cat['id'],
                'supercategory': cat.get('supercategory', ''),
                'annotation_count': annotation_counts.get(name, 0),
                'image_count': len(image_counts.get(name, set())),
                'percentage': annotation_counts.get(name, 0) / total_anns * 100
                if total_anns else 0
            }

        return stats

    @staticmethod
    def segmentation_to_bbox(coco_data: Dict[str, Any],
                             overwrite_existing: bool = False) -> Dict[str, Any]:
        """
        Convert segmentation masks to bounding boxes.

        Args:
            coco_data: COCO format dataset
            overwrite_existing: If True, recalculate bbox even if one exists

        Returns:
            Modified COCO dataset
        """
        result = deepcopy(coco_data)

        converted_count = 0
        for ann in result['annotations']:
            segmentation = ann.get('segmentation')
            if not segmentation:
                continue

            has_bbox = 'bbox' in ann and ann['bbox'] and len(ann['bbox']) == 4
            if has_bbox and not overwrite_existing:
                continue

            try:
                if isinstance(segmentation, list) and len(segmentation) > 0:
                    all_x = []
                    all_y = []

                    for poly in segmentation:
                        if isinstance(poly, list) and len(poly) >= 6:
                            all_x.extend(poly[0::2])
                            all_y.extend(poly[1::2])

                    if all_x and all_y:
                        x_min = min(all_x)
                        y_min = min(all_y)
                        x_max = max(all_x)
                        y_max = max(all_y)

                        ann['bbox'] = [x_min, y_min, x_max - x_min, y_max - y_min]
                        ann['area'] = (x_max - x_min) * (y_max - y_min)
                        converted_count += 1

                elif isinstance(segmentation, dict):
                    logger.warning("RLE segmentation format not supported for bbox conversion")

            except Exception as e:
                logger.warning(f"Failed to convert segmentation for annotation {ann.get('id')}: {e}")

        logger.info(f"Converted {converted_count} segmentations to bounding boxes")

        return result

    @staticmethod
    def filter_by_labels(coco_data: Dict[str, Any],
                         keep_labels: List[str]) -> Dict[str, Any]:
        """
        Filter dataset to keep only specified labels.

        Args:
            coco_data: COCO format dataset
            keep_labels: List of label names to keep

        Returns:
            Filtered COCO dataset
        """
        result = deepcopy(coco_data)

        keep_set = set(keep_labels)
        keep_cat_ids = {
            cat['id'] for cat in result['categories']
            if cat['name'] in keep_set
        }

        result['categories'] = [
            cat for cat in result['categories']
            if cat['name'] in keep_set
        ]

        original_count = len(result['annotations'])
        result['annotations'] = [
            ann for ann in result['annotations']
            if ann['category_id'] in keep_cat_ids
        ]

        remaining_img_ids = {ann['image_id'] for ann in result['annotations']}
        result['images'] = [
            img for img in result['images']
            if img['id'] in remaining_img_ids
        ]

        logger.info(
            f"Filtered to {len(keep_labels)} labels: "
            f"{len(result['images'])} images, "
            f"{len(result['annotations'])} annotations "
            f"(removed {original_count - len(result['annotations'])})"
        )

        return result
