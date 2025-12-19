"""
Split strategies for train/val/test dataset splitting.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np
import logging

# Lazy import of sklearn to avoid import errors when not installed
_sklearn_available = False
train_test_split = None

try:
    from sklearn.model_selection import train_test_split
    _sklearn_available = True
except ImportError:
    pass


def _check_sklearn():
    """Check if sklearn is available and raise helpful error if not."""
    if not _sklearn_available:
        raise ImportError(
            "scikit-learn is required for stratified splits. "
            "Install it with: pip install scikit-learn"
        )

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """Configuration for dataset splits."""
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    random_seed: int = 42

    def __post_init__(self):
        """Validate ratios sum to 1.0."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.3f}")

        if any(r < 0 for r in [self.train_ratio, self.val_ratio, self.test_ratio]):
            raise ValueError("Split ratios must be non-negative")


@dataclass
class SplitResult:
    """Result of a dataset split operation."""
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]
    train_images: List[Dict]
    val_images: List[Dict]
    test_images: List[Dict]
    statistics: Dict = field(default_factory=dict)
    strategy_name: str = ""

    @property
    def total_count(self) -> int:
        """Total number of images."""
        return len(self.train_indices) + len(self.val_indices) + len(self.test_indices)

    def get_summary(self) -> str:
        """Get human-readable summary."""
        return (
            f"Split Summary ({self.strategy_name}):\n"
            f"  Train: {len(self.train_indices)} ({len(self.train_indices)/self.total_count*100:.1f}%)\n"
            f"  Val:   {len(self.val_indices)} ({len(self.val_indices)/self.total_count*100:.1f}%)\n"
            f"  Test:  {len(self.test_indices)} ({len(self.test_indices)/self.total_count*100:.1f}%)"
        )


class BaseSplitStrategy(ABC):
    """Abstract base class for split strategies."""

    def __init__(self, config: SplitConfig):
        """
        Initialize split strategy.

        Args:
            config: Split configuration
        """
        self.config = config
        np.random.seed(config.random_seed)

    @abstractmethod
    def split(self, images: List[Dict], annotations: List[Dict]) -> SplitResult:
        """
        Perform the dataset split.

        Args:
            images: List of image info dictionaries
            annotations: List of annotation dictionaries

        Returns:
            SplitResult with indices and images for each split
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return name of this strategy."""
        pass

    def _compute_basic_stats(self, images: List[Dict],
                             train_idx: List[int],
                             val_idx: List[int],
                             test_idx: List[int]) -> Dict:
        """
        Compute basic split statistics.

        Args:
            images: All images
            train_idx: Training indices
            val_idx: Validation indices
            test_idx: Test indices

        Returns:
            Statistics dictionary
        """
        n = len(images)
        return {
            'total': n,
            'train_count': len(train_idx),
            'val_count': len(val_idx),
            'test_count': len(test_idx),
            'train_ratio_actual': len(train_idx) / n if n > 0 else 0,
            'val_ratio_actual': len(val_idx) / n if n > 0 else 0,
            'test_ratio_actual': len(test_idx) / n if n > 0 else 0
        }


class RandomSplitStrategy(BaseSplitStrategy):
    """Random split without considering class distribution."""

    def get_strategy_name(self) -> str:
        """Return strategy name."""
        return "random"

    def split(self, images: List[Dict], annotations: List[Dict]) -> SplitResult:
        """
        Perform random split.

        Args:
            images: List of image dictionaries
            annotations: List of annotation dictionaries (not used for random split)

        Returns:
            SplitResult
        """
        n = len(images)
        indices = np.arange(n)
        np.random.shuffle(indices)

        train_end = int(n * self.config.train_ratio)
        val_end = train_end + int(n * self.config.val_ratio)

        train_idx = indices[:train_end].tolist()
        val_idx = indices[train_end:val_end].tolist()
        test_idx = indices[val_end:].tolist()

        logger.info(f"Random split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        return SplitResult(
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
            train_images=[images[i] for i in train_idx],
            val_images=[images[i] for i in val_idx],
            test_images=[images[i] for i in test_idx],
            statistics=self._compute_basic_stats(images, train_idx, val_idx, test_idx),
            strategy_name=self.get_strategy_name()
        )


class StratifiedSplitStrategy(BaseSplitStrategy):
    """Stratified split maintaining class distribution across splits."""

    def get_strategy_name(self) -> str:
        """Return strategy name."""
        return "stratified"

    def split(self, images: List[Dict], annotations: List[Dict]) -> SplitResult:
        """
        Perform stratified split.

        Uses the predominant class in each image for stratification.

        Args:
            images: List of image dictionaries
            annotations: List of annotation dictionaries

        Returns:
            SplitResult
        """
        # Check sklearn availability
        _check_sklearn()

        n = len(images)

        if n == 0:
            return SplitResult(
                train_indices=[], val_indices=[], test_indices=[],
                train_images=[], val_images=[], test_images=[],
                statistics={}, strategy_name=self.get_strategy_name()
            )

        # Get class label for each image (predominant class)
        image_labels = self._get_image_labels(images, annotations)

        indices = np.arange(n)

        # Check if we have enough samples per class
        unique_labels, counts = np.unique(image_labels, return_counts=True)
        min_count = counts.min()

        # If any class has too few samples, fall back to random
        if min_count < 2:
            logger.warning("Some classes have <2 samples, falling back to random split")
            random_strategy = RandomSplitStrategy(self.config)
            return random_strategy.split(images, annotations)

        try:
            # First split: train vs (val+test)
            train_idx, temp_idx, train_labels, temp_labels = train_test_split(
                indices,
                image_labels,
                test_size=(1 - self.config.train_ratio),
                stratify=image_labels,
                random_state=self.config.random_seed
            )

            # Second split: val vs test
            val_ratio_adjusted = self.config.val_ratio / (self.config.val_ratio + self.config.test_ratio)

            # Check if temp set can be split
            unique_temp, counts_temp = np.unique(temp_labels, return_counts=True)
            if counts_temp.min() < 2:
                # Cannot stratify second split, do random
                n_val = int(len(temp_idx) * val_ratio_adjusted)
                np.random.shuffle(temp_idx)
                val_idx = temp_idx[:n_val].tolist()
                test_idx = temp_idx[n_val:].tolist()
            else:
                val_idx, test_idx = train_test_split(
                    temp_idx,
                    test_size=(1 - val_ratio_adjusted),
                    stratify=temp_labels,
                    random_state=self.config.random_seed
                )

            train_idx = train_idx.tolist() if hasattr(train_idx, 'tolist') else list(train_idx)
            val_idx = val_idx.tolist() if hasattr(val_idx, 'tolist') else list(val_idx)
            test_idx = test_idx.tolist() if hasattr(test_idx, 'tolist') else list(test_idx)

        except ValueError as e:
            logger.warning(f"Stratified split failed ({e}), falling back to random")
            random_strategy = RandomSplitStrategy(self.config)
            return random_strategy.split(images, annotations)

        logger.info(f"Stratified split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        return SplitResult(
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
            train_images=[images[i] for i in train_idx],
            val_images=[images[i] for i in val_idx],
            test_images=[images[i] for i in test_idx],
            statistics=self._compute_stratified_stats(
                images, annotations, train_idx, val_idx, test_idx
            ),
            strategy_name=self.get_strategy_name()
        )

    def _get_image_labels(self, images: List[Dict], annotations: List[Dict]) -> np.ndarray:
        """
        Get predominant class label for each image.

        Args:
            images: List of images
            annotations: List of annotations

        Returns:
            Array of class labels (one per image)
        """
        img_id_to_idx = {img['id']: i for i, img in enumerate(images)}

        # Count classes per image
        class_counts = {i: {} for i in range(len(images))}

        for ann in annotations:
            img_idx = img_id_to_idx.get(ann['image_id'])
            if img_idx is not None:
                cat_id = ann['category_id']
                class_counts[img_idx][cat_id] = class_counts[img_idx].get(cat_id, 0) + 1

        # Select predominant class
        labels = np.zeros(len(images), dtype=int)
        for i in range(len(images)):
            if class_counts[i]:
                # Get class with most annotations
                labels[i] = max(class_counts[i].keys(), key=lambda k: class_counts[i][k])
            else:
                labels[i] = 0  # Default for images without annotations

        return labels

    def _compute_stratified_stats(self, images: List[Dict], annotations: List[Dict],
                                   train_idx: List[int], val_idx: List[int],
                                   test_idx: List[int]) -> Dict:
        """
        Compute statistics including class distribution per split.

        Args:
            images: All images
            annotations: All annotations
            train_idx: Training indices
            val_idx: Validation indices
            test_idx: Test indices

        Returns:
            Statistics dictionary with class distributions
        """
        stats = self._compute_basic_stats(images, train_idx, val_idx, test_idx)

        # Build image ID sets for each split
        train_img_ids = {images[i]['id'] for i in train_idx}
        val_img_ids = {images[i]['id'] for i in val_idx}
        test_img_ids = {images[i]['id'] for i in test_idx}

        # Count classes per split
        class_distribution = {'train': {}, 'val': {}, 'test': {}}

        for ann in annotations:
            img_id = ann['image_id']
            cat_id = ann['category_id']

            if img_id in train_img_ids:
                class_distribution['train'][cat_id] = class_distribution['train'].get(cat_id, 0) + 1
            elif img_id in val_img_ids:
                class_distribution['val'][cat_id] = class_distribution['val'].get(cat_id, 0) + 1
            elif img_id in test_img_ids:
                class_distribution['test'][cat_id] = class_distribution['test'].get(cat_id, 0) + 1

        stats['class_distribution'] = class_distribution

        return stats
