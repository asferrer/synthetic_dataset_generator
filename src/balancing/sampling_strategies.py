"""
Sampling strategies for class balancing.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import random
import logging

logger = logging.getLogger(__name__)


@dataclass
class BalancingConfig:
    """Configuration for class balancing."""
    target_count: Optional[int] = None  # None = use max/min/median depending on strategy
    strategy: str = 'oversample'  # 'oversample', 'undersample', 'hybrid'
    random_seed: int = 42
    max_oversample_ratio: float = 5.0  # Maximum times a sample can be duplicated

    def __post_init__(self):
        """Validate configuration."""
        if self.max_oversample_ratio < 1.0:
            raise ValueError("max_oversample_ratio must be >= 1.0")


@dataclass
class BalancingResult:
    """Result of a balancing operation."""
    original_counts: Dict[int, int]  # category_id -> count
    balanced_counts: Dict[int, int]  # category_id -> count after balancing
    oversampled_indices: Dict[int, int]  # annotation_index -> n_copies
    undersampled_indices: List[int]  # annotation indices to remove
    class_weights: Dict[int, float]  # category_id -> weight
    strategy_name: str = ""

    @property
    def total_original(self) -> int:
        """Total original annotations."""
        return sum(self.original_counts.values())

    @property
    def total_balanced(self) -> int:
        """Total balanced annotations."""
        return sum(self.balanced_counts.values())

    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Balancing Summary ({self.strategy_name})",
            f"Original: {self.total_original} annotations",
            f"Balanced: {self.total_balanced} annotations",
            f"Oversampled: {sum(self.oversampled_indices.values())} copies",
            f"Undersampled: {len(self.undersampled_indices)} removed"
        ]
        return "\n".join(lines)


class BaseSamplingStrategy(ABC):
    """Abstract base class for sampling strategies."""

    def __init__(self, config: BalancingConfig):
        """
        Initialize sampling strategy.

        Args:
            config: Balancing configuration
        """
        self.config = config
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)

    @abstractmethod
    def balance(self, annotations: List[Dict],
                categories: List[Dict]) -> BalancingResult:
        """
        Perform class balancing.

        Args:
            annotations: List of annotation dictionaries
            categories: List of category dictionaries

        Returns:
            BalancingResult with sampling decisions
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return name of this strategy."""
        pass

    def _count_by_class(self, annotations: List[Dict]) -> Dict[int, int]:
        """
        Count annotations per class.

        Args:
            annotations: List of annotations

        Returns:
            Dictionary of category_id -> count
        """
        counts = defaultdict(int)
        for ann in annotations:
            counts[ann['category_id']] += 1
        return dict(counts)

    def _get_indices_by_class(self, annotations: List[Dict]) -> Dict[int, List[int]]:
        """
        Get annotation indices grouped by class.

        Args:
            annotations: List of annotations

        Returns:
            Dictionary of category_id -> list of indices
        """
        indices = defaultdict(list)
        for i, ann in enumerate(annotations):
            indices[ann['category_id']].append(i)
        return dict(indices)

    def _calculate_weights(self, counts: Dict[int, int]) -> Dict[int, float]:
        """
        Calculate inverse frequency weights.

        Args:
            counts: Class counts

        Returns:
            Dictionary of category_id -> weight
        """
        total = sum(counts.values())
        n_classes = len(counts)

        if total == 0 or n_classes == 0:
            return {}

        weights = {}
        for cat_id, count in counts.items():
            if count > 0:
                weights[cat_id] = total / (n_classes * count)
            else:
                weights[cat_id] = 1.0

        return weights


class OversamplingStrategy(BaseSamplingStrategy):
    """Oversample minority classes to match majority."""

    def get_strategy_name(self) -> str:
        """Return strategy name."""
        return "oversample"

    def balance(self, annotations: List[Dict],
                categories: List[Dict]) -> BalancingResult:
        """
        Oversample minority classes.

        Duplicates annotations from minority classes until they reach
        the target count (default: maximum class count).

        Args:
            annotations: List of annotations
            categories: List of categories

        Returns:
            BalancingResult
        """
        class_counts = self._count_by_class(annotations)
        indices_by_class = self._get_indices_by_class(annotations)

        # Determine target (max count or user-specified)
        target = self.config.target_count
        if target is None:
            target = max(class_counts.values()) if class_counts else 0

        oversampled = {}
        balanced_counts = {}

        for cat_id, count in class_counts.items():
            balanced_counts[cat_id] = count

            if count < target and count > 0:
                needed = target - count

                # Limit by max oversample ratio
                max_copies = int(count * self.config.max_oversample_ratio) - count
                actual_needed = min(needed, max_copies)

                if actual_needed > 0:
                    # Select indices to duplicate
                    cat_indices = indices_by_class.get(cat_id, [])

                    for _ in range(actual_needed):
                        idx = random.choice(cat_indices)
                        oversampled[idx] = oversampled.get(idx, 0) + 1

                    balanced_counts[cat_id] = count + actual_needed

        logger.info(f"Oversampling: target={target}, added {sum(oversampled.values())} copies")

        return BalancingResult(
            original_counts=class_counts,
            balanced_counts=balanced_counts,
            oversampled_indices=oversampled,
            undersampled_indices=[],
            class_weights=self._calculate_weights(class_counts),
            strategy_name=self.get_strategy_name()
        )


class UndersamplingStrategy(BaseSamplingStrategy):
    """Undersample majority classes to match minority."""

    def get_strategy_name(self) -> str:
        """Return strategy name."""
        return "undersample"

    def balance(self, annotations: List[Dict],
                categories: List[Dict]) -> BalancingResult:
        """
        Undersample majority classes.

        Removes annotations from majority classes until they reach
        the target count (default: minimum class count).

        Args:
            annotations: List of annotations
            categories: List of categories

        Returns:
            BalancingResult
        """
        class_counts = self._count_by_class(annotations)
        indices_by_class = self._get_indices_by_class(annotations)

        # Determine target (min count or user-specified)
        target = self.config.target_count
        if target is None:
            target = min(class_counts.values()) if class_counts else 0

        undersampled = []
        balanced_counts = {}

        for cat_id, count in class_counts.items():
            balanced_counts[cat_id] = min(count, target)

            if count > target:
                # Select indices to remove
                cat_indices = indices_by_class.get(cat_id, [])
                n_remove = count - target

                to_remove = random.sample(cat_indices, n_remove)
                undersampled.extend(to_remove)

        logger.info(f"Undersampling: target={target}, removed {len(undersampled)} annotations")

        return BalancingResult(
            original_counts=class_counts,
            balanced_counts=balanced_counts,
            oversampled_indices={},
            undersampled_indices=undersampled,
            class_weights=self._calculate_weights(class_counts),
            strategy_name=self.get_strategy_name()
        )


class HybridSamplingStrategy(BaseSamplingStrategy):
    """
    Hybrid strategy: oversample minority and undersample majority.

    Target is typically the median class count.
    """

    def __init__(self, config: BalancingConfig, use_median: bool = True):
        """
        Initialize hybrid strategy.

        Args:
            config: Balancing configuration
            use_median: If True, target is median; if False, uses mean
        """
        super().__init__(config)
        self.use_median = use_median

    def get_strategy_name(self) -> str:
        """Return strategy name."""
        return "hybrid"

    def balance(self, annotations: List[Dict],
                categories: List[Dict]) -> BalancingResult:
        """
        Apply hybrid balancing.

        Classes below target are oversampled, classes above are undersampled.

        Args:
            annotations: List of annotations
            categories: List of categories

        Returns:
            BalancingResult
        """
        class_counts = self._count_by_class(annotations)
        indices_by_class = self._get_indices_by_class(annotations)

        # Determine target
        if self.config.target_count is not None:
            target = self.config.target_count
        else:
            counts = list(class_counts.values())
            if self.use_median:
                target = int(np.median(counts)) if counts else 0
            else:
                target = int(np.mean(counts)) if counts else 0

        oversampled = {}
        undersampled = []
        balanced_counts = {}

        for cat_id, count in class_counts.items():
            cat_indices = indices_by_class.get(cat_id, [])

            if count < target and count > 0:
                # Oversample
                needed = target - count
                max_copies = int(count * self.config.max_oversample_ratio) - count
                actual_needed = min(needed, max_copies)

                for _ in range(actual_needed):
                    idx = random.choice(cat_indices)
                    oversampled[idx] = oversampled.get(idx, 0) + 1

                balanced_counts[cat_id] = count + actual_needed

            elif count > target:
                # Undersample
                n_remove = count - target
                to_remove = random.sample(cat_indices, n_remove)
                undersampled.extend(to_remove)
                balanced_counts[cat_id] = target

            else:
                balanced_counts[cat_id] = count

        logger.info(f"Hybrid balancing: target={target}, "
                   f"added {sum(oversampled.values())} copies, "
                   f"removed {len(undersampled)} annotations")

        return BalancingResult(
            original_counts=class_counts,
            balanced_counts=balanced_counts,
            oversampled_indices=oversampled,
            undersampled_indices=undersampled,
            class_weights=self._calculate_weights(class_counts),
            strategy_name=self.get_strategy_name()
        )
