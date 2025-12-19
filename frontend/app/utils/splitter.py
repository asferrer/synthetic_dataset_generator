"""
Dataset Splitter - Train/Val/Test splits and K-Fold cross-validation with stratification support.
"""

import random
import logging
from typing import Dict, List, Any, Optional, Tuple, Generator
from copy import deepcopy
from collections import Counter
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Try to import sklearn for stratified splits
try:
    from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
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


# =============================================================================
# K-Fold Cross-Validation
# =============================================================================

@dataclass
class KFoldConfig:
    """Configuration for K-Fold cross-validation."""
    n_folds: int = 5
    stratified: bool = True
    random_seed: int = 42
    shuffle: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {self.n_folds}")


@dataclass
class FoldResult:
    """Result for a single fold."""
    fold_number: int
    train_indices: List[int]
    val_indices: List[int]
    train_images: List[Dict]
    val_images: List[Dict]
    n_train: int = 0
    n_val: int = 0

    def __post_init__(self):
        """Set counts."""
        self.n_train = len(self.train_indices)
        self.n_val = len(self.val_indices)

    def get_summary(self) -> str:
        """Get human-readable summary."""
        return f"Fold {self.fold_number}: train={self.n_train}, val={self.n_val}"


class KFoldGenerator:
    """
    Generator for K-Fold cross-validation splits.

    Supports:
    - Standard K-Fold splits
    - Stratified K-Fold (maintains class distribution in each fold)
    """

    def __init__(self, config: Optional[KFoldConfig] = None, n_folds: int = 5):
        """
        Initialize K-Fold generator.

        Args:
            config: K-Fold configuration (optional)
            n_folds: Number of folds (used if config not provided)
        """
        if config is not None:
            self.config = config
        else:
            self.config = KFoldConfig(n_folds=n_folds)

    def generate_folds(self, images: List[Dict],
                       annotations: List[Dict]) -> Generator[FoldResult, None, None]:
        """
        Generate K folds for cross-validation.

        Args:
            images: List of image dictionaries
            annotations: List of annotation dictionaries

        Yields:
            FoldResult for each fold
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for K-Fold cross-validation. "
                "Install it with: pip install scikit-learn"
            )

        n_samples = len(images)

        if n_samples < self.config.n_folds:
            raise ValueError(f"Cannot create {self.config.n_folds} folds with only {n_samples} samples")

        indices = np.arange(n_samples)

        if self.config.stratified:
            labels = self._get_image_labels(images, annotations)

            # Check if stratification is possible
            unique_labels, counts = np.unique(labels, return_counts=True)
            if counts.min() < self.config.n_folds:
                logger.warning(
                    f"Some classes have fewer than {self.config.n_folds} samples, "
                    "falling back to non-stratified K-Fold"
                )
                kfold = KFold(
                    n_splits=self.config.n_folds,
                    shuffle=self.config.shuffle,
                    random_state=self.config.random_seed
                )
                splits = kfold.split(indices)
            else:
                kfold = StratifiedKFold(
                    n_splits=self.config.n_folds,
                    shuffle=self.config.shuffle,
                    random_state=self.config.random_seed
                )
                splits = kfold.split(indices, labels)
        else:
            kfold = KFold(
                n_splits=self.config.n_folds,
                shuffle=self.config.shuffle,
                random_state=self.config.random_seed
            )
            splits = kfold.split(indices)

        for fold_num, (train_idx, val_idx) in enumerate(splits, 1):
            logger.info(f"Generated fold {fold_num}/{self.config.n_folds}: "
                       f"train={len(train_idx)}, val={len(val_idx)}")

            yield FoldResult(
                fold_number=fold_num,
                train_indices=train_idx.tolist(),
                val_indices=val_idx.tolist(),
                train_images=[images[i] for i in train_idx],
                val_images=[images[i] for i in val_idx]
            )

    def _get_image_labels(self, images: List[Dict], annotations: List[Dict]) -> np.ndarray:
        """
        Get class label for each image for stratification.

        Args:
            images: List of images
            annotations: List of annotations

        Returns:
            Array of class labels
        """
        img_id_to_idx = {img['id']: i for i, img in enumerate(images)}
        labels = np.zeros(len(images), dtype=int)

        # Count classes per image
        class_counts = {i: {} for i in range(len(images))}

        for ann in annotations:
            idx = img_id_to_idx.get(ann['image_id'])
            if idx is not None:
                cat_id = ann['category_id']
                class_counts[idx][cat_id] = class_counts[idx].get(cat_id, 0) + 1

        # Assign predominant class
        for i in range(len(images)):
            if class_counts[i]:
                labels[i] = max(class_counts[i].keys(), key=lambda k: class_counts[i][k])

        return labels

    def export_fold(self, fold: FoldResult,
                    coco_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Export a fold as two separate COCO datasets.

        Args:
            fold: FoldResult to export
            coco_data: Original COCO data

        Returns:
            Tuple of (train_coco_data, val_coco_data)
        """
        # Get image IDs for each split
        train_img_ids = {img['id'] for img in fold.train_images}
        val_img_ids = {img['id'] for img in fold.val_images}

        # Build train dataset
        train_data = {
            'info': coco_data.get('info', {}),
            'licenses': coco_data.get('licenses', []),
            'categories': coco_data['categories'],
            'images': fold.train_images,
            'annotations': [
                ann for ann in coco_data['annotations']
                if ann['image_id'] in train_img_ids
            ]
        }

        # Build val dataset
        val_data = {
            'info': coco_data.get('info', {}),
            'licenses': coco_data.get('licenses', []),
            'categories': coco_data['categories'],
            'images': fold.val_images,
            'annotations': [
                ann for ann in coco_data['annotations']
                if ann['image_id'] in val_img_ids
            ]
        }

        return train_data, val_data

    def get_all_folds(self, images: List[Dict],
                      annotations: List[Dict]) -> List[FoldResult]:
        """
        Get all folds as a list (non-generator version).

        Args:
            images: List of images
            annotations: List of annotations

        Returns:
            List of FoldResults
        """
        return list(self.generate_folds(images, annotations))

    def get_fold_statistics(self, folds: List[FoldResult],
                            annotations: List[Dict],
                            categories: List[Dict]) -> Dict:
        """
        Compute statistics across all folds.

        Args:
            folds: List of FoldResults
            annotations: All annotations
            categories: All categories

        Returns:
            Statistics dictionary
        """
        cat_id_to_name = {cat['id']: cat['name'] for cat in categories}

        stats = {
            'n_folds': len(folds),
            'folds': [],
            'avg_train_size': 0,
            'avg_val_size': 0
        }

        total_train = 0
        total_val = 0

        for fold in folds:
            fold_stats = {
                'fold': fold.fold_number,
                'train_size': fold.n_train,
                'val_size': fold.n_val,
                'train_class_counts': {},
                'val_class_counts': {}
            }

            # Count classes in train
            train_img_ids = {img['id'] for img in fold.train_images}
            for ann in annotations:
                if ann['image_id'] in train_img_ids:
                    cat_name = cat_id_to_name.get(ann['category_id'], f"class_{ann['category_id']}")
                    fold_stats['train_class_counts'][cat_name] = \
                        fold_stats['train_class_counts'].get(cat_name, 0) + 1

            # Count classes in val
            val_img_ids = {img['id'] for img in fold.val_images}
            for ann in annotations:
                if ann['image_id'] in val_img_ids:
                    cat_name = cat_id_to_name.get(ann['category_id'], f"class_{ann['category_id']}")
                    fold_stats['val_class_counts'][cat_name] = \
                        fold_stats['val_class_counts'].get(cat_name, 0) + 1

            stats['folds'].append(fold_stats)
            total_train += fold.n_train
            total_val += fold.n_val

        stats['avg_train_size'] = total_train / len(folds) if folds else 0
        stats['avg_val_size'] = total_val / len(folds) if folds else 0

        return stats
