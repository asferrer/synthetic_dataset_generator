"""
K-Fold cross-validation generator for datasets.
"""

from typing import List, Dict, Generator, Tuple, Any
from dataclasses import dataclass
import numpy as np
import logging

# Lazy import of sklearn to avoid import errors when not installed
_sklearn_available = False
StratifiedKFold = None
KFold = None

try:
    from sklearn.model_selection import StratifiedKFold, KFold
    _sklearn_available = True
except ImportError:
    pass


def _check_sklearn():
    """Check if sklearn is available and raise helpful error if not."""
    if not _sklearn_available:
        raise ImportError(
            "scikit-learn is required for K-Fold cross-validation. "
            "Install it with: pip install scikit-learn"
        )


logger = logging.getLogger(__name__)


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
    """Generator for K-Fold cross-validation splits."""

    def __init__(self, config: KFoldConfig):
        """
        Initialize K-Fold generator.

        Args:
            config: K-Fold configuration
        """
        self.config = config

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
        # Check sklearn availability
        _check_sklearn()

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
                            images: List[Dict]) -> Dict:
        """
        Compute statistics across all folds.

        Args:
            folds: List of FoldResults
            annotations: All annotations
            images: All images

        Returns:
            Statistics dictionary
        """
        img_id_to_idx = {img['id']: i for i, img in enumerate(images)}

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
                    cat = ann['category_id']
                    fold_stats['train_class_counts'][cat] = \
                        fold_stats['train_class_counts'].get(cat, 0) + 1

            # Count classes in val
            val_img_ids = {img['id'] for img in fold.val_images}
            for ann in annotations:
                if ann['image_id'] in val_img_ids:
                    cat = ann['category_id']
                    fold_stats['val_class_counts'][cat] = \
                        fold_stats['val_class_counts'].get(cat, 0) + 1

            stats['folds'].append(fold_stats)
            total_train += fold.n_train
            total_val += fold.n_val

        stats['avg_train_size'] = total_train / len(folds) if folds else 0
        stats['avg_val_size'] = total_val / len(folds) if folds else 0

        return stats
