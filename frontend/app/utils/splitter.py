"""
Dataset Splitter - Train/Val/Test splits with stratification support.
"""

import random
import logging
from typing import Dict, List, Any, Optional, Tuple
from copy import deepcopy
from collections import Counter

logger = logging.getLogger(__name__)

# Try to import sklearn for stratified splits
try:
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class DatasetSplitter:
    """
    Splitter for creating train/val/test splits from COCO datasets.

    Supports:
    - Random splits
    - Stratified splits (maintains class distribution)
    """

    @staticmethod
    def split_dataset(coco_data: Dict[str, Any],
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.2,
                      test_ratio: float = 0.1,
                      strategy: str = 'stratified',
                      random_seed: int = 42) -> Dict[str, Dict[str, Any]]:
        """
        Split dataset into train/val/test sets.

        Args:
            coco_data: COCO format dataset
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for test
            strategy: 'random' or 'stratified'
            random_seed: Random seed for reproducibility

        Returns:
            Dictionary with 'train', 'val', 'test' COCO datasets
        """
        random.seed(random_seed)

        images = coco_data['images']
        annotations = coco_data['annotations']
        categories = coco_data['categories']

        n_images = len(images)

        if n_images == 0:
            empty_split = {
                'info': coco_data.get('info', {}),
                'licenses': coco_data.get('licenses', []),
                'categories': categories,
                'images': [],
                'annotations': []
            }
            return {'train': empty_split, 'val': deepcopy(empty_split), 'test': deepcopy(empty_split)}

        # Build image ID to annotations mapping
        img_to_anns = {}
        for ann in annotations:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)

        # Get image indices
        indices = list(range(n_images))

        if strategy == 'stratified' and SKLEARN_AVAILABLE:
            # Get predominant class for each image for stratification
            labels = DatasetSplitter._get_image_labels(images, annotations)

            try:
                # First split: train vs (val+test)
                train_idx, temp_idx = train_test_split(
                    indices,
                    test_size=(1 - train_ratio),
                    stratify=labels,
                    random_state=random_seed
                )

                # Get labels for temp set
                temp_labels = [labels[i] for i in temp_idx]

                # Second split: val vs test
                val_size = val_ratio / (val_ratio + test_ratio)
                val_idx, test_idx = train_test_split(
                    temp_idx,
                    test_size=(1 - val_size),
                    stratify=temp_labels,
                    random_state=random_seed
                )

            except ValueError as e:
                logger.warning(f"Stratified split failed ({e}), falling back to random")
                train_idx, val_idx, test_idx = DatasetSplitter._random_split(
                    indices, train_ratio, val_ratio, random_seed
                )
        else:
            if strategy == 'stratified' and not SKLEARN_AVAILABLE:
                logger.warning("scikit-learn not available, using random split")

            train_idx, val_idx, test_idx = DatasetSplitter._random_split(
                indices, train_ratio, val_ratio, random_seed
            )

        # Build split datasets
        splits = {}
        for split_name, split_indices in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
            split_images = [images[i] for i in split_indices]
            split_img_ids = {img['id'] for img in split_images}

            split_annotations = [
                ann for ann in annotations
                if ann['image_id'] in split_img_ids
            ]

            splits[split_name] = {
                'info': coco_data.get('info', {}),
                'licenses': coco_data.get('licenses', []),
                'categories': categories,
                'images': split_images,
                'annotations': split_annotations
            }

        logger.info(
            f"Split complete: train={len(splits['train']['images'])}, "
            f"val={len(splits['val']['images'])}, test={len(splits['test']['images'])}"
        )

        return splits

    @staticmethod
    def _random_split(indices: List[int],
                      train_ratio: float,
                      val_ratio: float,
                      random_seed: int) -> Tuple[List[int], List[int], List[int]]:
        """Perform random split."""
        random.seed(random_seed)
        shuffled = indices.copy()
        random.shuffle(shuffled)

        n = len(indices)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]

    @staticmethod
    def _get_image_labels(images: List[Dict], annotations: List[Dict]) -> List[int]:
        """
        Get predominant class label for each image for stratification.
        """
        img_id_to_idx = {img['id']: i for i, img in enumerate(images)}

        # Count classes per image
        class_counts = {i: Counter() for i in range(len(images))}

        for ann in annotations:
            idx = img_id_to_idx.get(ann['image_id'])
            if idx is not None:
                class_counts[idx][ann['category_id']] += 1

        # Get predominant class for each image
        labels = []
        for i in range(len(images)):
            if class_counts[i]:
                labels.append(class_counts[i].most_common(1)[0][0])
            else:
                labels.append(0)  # Default for images without annotations

        return labels

    @staticmethod
    def get_split_statistics(splits: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics for split datasets.

        Args:
            splits: Dictionary of split datasets

        Returns:
            Statistics dictionary
        """
        stats = {}

        for split_name, split_data in splits.items():
            # Count annotations per class
            cat_id_to_name = {
                cat['id']: cat['name']
                for cat in split_data['categories']
            }

            class_counts = Counter()
            for ann in split_data['annotations']:
                cat_name = cat_id_to_name.get(ann['category_id'], 'unknown')
                class_counts[cat_name] += 1

            stats[split_name] = {
                'num_images': len(split_data['images']),
                'num_annotations': len(split_data['annotations']),
                'class_distribution': dict(class_counts)
            }

        return stats
