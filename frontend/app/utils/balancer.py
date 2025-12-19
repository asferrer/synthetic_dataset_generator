"""
Class Balancer - Strategies for handling imbalanced datasets.

Provides:
- Oversampling minority classes
- Undersampling majority classes
- Hybrid balancing
- Class weights calculation for training
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from copy import deepcopy
import random
import logging

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration and Result Classes
# =============================================================================

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


# =============================================================================
# Sampling Strategies
# =============================================================================

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
        """Count annotations per class."""
        counts = defaultdict(int)
        for ann in annotations:
            counts[ann['category_id']] += 1
        return dict(counts)

    def _get_indices_by_class(self, annotations: List[Dict]) -> Dict[int, List[int]]:
        """Get annotation indices grouped by class."""
        indices = defaultdict(list)
        for i, ann in enumerate(annotations):
            indices[ann['category_id']].append(i)
        return dict(indices)

    def _calculate_weights(self, counts: Dict[int, int]) -> Dict[int, float]:
        """Calculate inverse frequency weights."""
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
        return "oversample"

    def balance(self, annotations: List[Dict],
                categories: List[Dict]) -> BalancingResult:
        """
        Oversample minority classes.

        Duplicates annotations from minority classes until they reach
        the target count (default: maximum class count).
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
        return "undersample"

    def balance(self, annotations: List[Dict],
                categories: List[Dict]) -> BalancingResult:
        """
        Undersample majority classes.

        Removes annotations from majority classes until they reach
        the target count (default: minimum class count).
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
    """Hybrid strategy: oversample minority and undersample majority."""

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
        return "hybrid"

    def balance(self, annotations: List[Dict],
                categories: List[Dict]) -> BalancingResult:
        """
        Apply hybrid balancing.

        Classes below target are oversampled, classes above are undersampled.
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


# =============================================================================
# Class Balancer (Main Interface)
# =============================================================================

class ClassBalancer:
    """
    Main class for balancing datasets by class.

    Supports:
    - Oversampling minority classes
    - Undersampling majority classes
    - Hybrid balancing
    """

    STRATEGIES = {
        'oversample': OversamplingStrategy,
        'undersample': UndersamplingStrategy,
        'hybrid': HybridSamplingStrategy
    }

    def __init__(self, config: Optional[BalancingConfig] = None):
        """
        Initialize class balancer.

        Args:
            config: Balancing configuration (optional)
        """
        self.config = config or BalancingConfig()

    def balance(self, coco_data: Dict[str, Any],
                strategy: str = 'oversample',
                target_count: Optional[int] = None) -> Tuple[Dict[str, Any], BalancingResult]:
        """
        Balance a COCO dataset.

        Args:
            coco_data: COCO format dataset
            strategy: 'oversample', 'undersample', or 'hybrid'
            target_count: Target count per class (None = auto)

        Returns:
            Tuple of (balanced_coco_data, BalancingResult)
        """
        config = BalancingConfig(
            strategy=strategy,
            target_count=target_count,
            random_seed=self.config.random_seed,
            max_oversample_ratio=self.config.max_oversample_ratio
        )

        strategy_cls = self.STRATEGIES.get(strategy)
        if not strategy_cls:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(self.STRATEGIES.keys())}")

        # Perform balancing
        balancer = strategy_cls(config)
        result = balancer.balance(coco_data['annotations'], coco_data['categories'])

        logger.info(result.get_summary())

        # Apply balancing to create new dataset
        balanced_data = self._apply_balancing(coco_data, result)

        return balanced_data, result

    def _apply_balancing(self, coco_data: Dict[str, Any],
                         result: BalancingResult) -> Dict[str, Any]:
        """Apply balancing result to create a new balanced dataset."""
        balanced = deepcopy(coco_data)

        # Remove undersampled annotations
        if result.undersampled_indices:
            remove_set = set(result.undersampled_indices)
            balanced['annotations'] = [
                ann for i, ann in enumerate(balanced['annotations'])
                if i not in remove_set
            ]

        # Add oversampled annotations (duplicates)
        if result.oversampled_indices:
            max_ann_id = max(ann['id'] for ann in balanced['annotations'])
            for idx, n_copies in result.oversampled_indices.items():
                original_ann = coco_data['annotations'][idx]
                for _ in range(n_copies):
                    max_ann_id += 1
                    new_ann = deepcopy(original_ann)
                    new_ann['id'] = max_ann_id
                    balanced['annotations'].append(new_ann)

        # Remove orphaned images (images without annotations)
        annotated_img_ids = {ann['image_id'] for ann in balanced['annotations']}
        balanced['images'] = [
            img for img in balanced['images']
            if img['id'] in annotated_img_ids
        ]

        return balanced


# =============================================================================
# Class Weights Calculator
# =============================================================================

class ClassWeightsCalculator:
    """
    Calculator for class weights to use during model training.

    Supports multiple weighting schemes:
    - Inverse frequency
    - Effective number of samples
    - Focal loss weights
    - Square root inverse
    """

    @staticmethod
    def compute_inverse_frequency(class_counts: Dict[int, int],
                                   normalize: bool = True) -> Dict[int, float]:
        """
        Compute weights inversely proportional to class frequency.

        Formula: weight = total / (n_classes * count)

        Use for: Standard weighted cross-entropy loss

        Args:
            class_counts: Dictionary of category_id -> count
            normalize: If True, normalize weights to sum to n_classes

        Returns:
            Dictionary of category_id -> weight
        """
        total = sum(class_counts.values())
        n_classes = len(class_counts)

        if total == 0 or n_classes == 0:
            return {}

        weights = {}
        for cat_id, count in class_counts.items():
            if count > 0:
                weights[cat_id] = total / (n_classes * count)
            else:
                weights[cat_id] = 1.0

        if normalize:
            weight_sum = sum(weights.values())
            if weight_sum > 0:
                weights = {k: v * n_classes / weight_sum for k, v in weights.items()}

        logger.info(f"Computed inverse frequency weights for {n_classes} classes")
        return weights

    @staticmethod
    def compute_effective_samples(class_counts: Dict[int, int],
                                   beta: float = 0.999) -> Dict[int, float]:
        """
        Compute weights based on effective number of samples.

        From paper: "Class-Balanced Loss Based on Effective Number of Samples"
        (Cui et al., CVPR 2019)

        Formula: effective_num = (1 - beta^n) / (1 - beta)
                weight = 1 / effective_num

        Args:
            class_counts: Dictionary of category_id -> count
            beta: Decay factor (typically 0.9, 0.99, or 0.999)

        Returns:
            Dictionary of category_id -> weight
        """
        if not 0 < beta < 1:
            raise ValueError(f"beta must be in (0, 1), got {beta}")

        weights = {}
        for cat_id, count in class_counts.items():
            if count > 0:
                effective_num = (1 - beta ** count) / (1 - beta)
                weights[cat_id] = 1.0 / effective_num
            else:
                weights[cat_id] = 1.0

        # Normalize weights
        total = sum(weights.values())
        n_classes = len(weights)
        if total > 0:
            weights = {k: v * n_classes / total for k, v in weights.items()}

        logger.info(f"Computed effective samples weights (beta={beta}) for {n_classes} classes")
        return weights

    @staticmethod
    def compute_focal_weights(class_counts: Dict[int, int],
                               gamma: float = 2.0) -> Dict[int, float]:
        """
        Compute weights for focal loss style weighting.

        Formula: weight = (1 - frequency)^gamma

        Rare classes get higher weights following a power law.

        Args:
            class_counts: Dictionary of category_id -> count
            gamma: Focusing parameter (higher = more focus on rare classes)

        Returns:
            Dictionary of category_id -> weight
        """
        total = sum(class_counts.values())

        if total == 0:
            return {}

        weights = {}
        for cat_id, count in class_counts.items():
            if count > 0:
                freq = count / total
                weights[cat_id] = (1 - freq) ** gamma
            else:
                weights[cat_id] = 1.0

        # Normalize
        n_classes = len(weights)
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            weights = {k: v * n_classes / weight_sum for k, v in weights.items()}

        logger.info(f"Computed focal weights (gamma={gamma}) for {n_classes} classes")
        return weights

    @staticmethod
    def compute_sqrt_inverse(class_counts: Dict[int, int]) -> Dict[int, float]:
        """
        Compute square root of inverse frequency weights.

        Less aggressive than pure inverse frequency.

        Formula: weight = sqrt(total / count)

        Args:
            class_counts: Dictionary of category_id -> count

        Returns:
            Dictionary of category_id -> weight
        """
        total = sum(class_counts.values())

        if total == 0:
            return {}

        weights = {}
        for cat_id, count in class_counts.items():
            if count > 0:
                weights[cat_id] = np.sqrt(total / count)
            else:
                weights[cat_id] = 1.0

        # Normalize
        n_classes = len(weights)
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            weights = {k: v * n_classes / weight_sum for k, v in weights.items()}

        logger.info(f"Computed sqrt inverse weights for {n_classes} classes")
        return weights

    @staticmethod
    def export_weights(weights: Dict[int, float],
                       categories: List[Dict],
                       format: str = 'dict') -> Any:
        """
        Export weights in format compatible with deep learning frameworks.

        Args:
            weights: Dictionary of category_id -> weight
            categories: List of category dictionaries
            format: 'dict', 'pytorch', 'tensorflow', or 'list'

        Returns:
            Weights in specified format
        """
        # Sort categories by ID for consistent ordering
        sorted_cats = sorted(categories, key=lambda x: x['id'])

        if format == 'dict':
            return {
                cat['name']: weights.get(cat['id'], 1.0)
                for cat in sorted_cats
            }

        elif format == 'pytorch':
            weights_list = [weights.get(cat['id'], 1.0) for cat in sorted_cats]
            return {
                'weights': weights_list,
                'class_to_idx': {cat['name']: i for i, cat in enumerate(sorted_cats)},
                'idx_to_class': {i: cat['name'] for i, cat in enumerate(sorted_cats)}
            }

        elif format == 'tensorflow':
            return {
                cat['id']: weights.get(cat['id'], 1.0)
                for cat in sorted_cats
            }

        elif format == 'list':
            return [weights.get(cat['id'], 1.0) for cat in sorted_cats]

        else:
            raise ValueError(f"Unknown format: {format}. Use 'dict', 'pytorch', 'tensorflow', or 'list'")

    @classmethod
    def from_coco_data(cls, coco_data: Dict[str, Any],
                       method: str = 'inverse_frequency',
                       **kwargs) -> Dict[int, float]:
        """
        Compute weights directly from COCO data.

        Args:
            coco_data: COCO format dataset
            method: 'inverse_frequency', 'effective_samples', 'focal', 'sqrt_inverse'
            **kwargs: Additional arguments for specific methods

        Returns:
            Dictionary of category_id -> weight
        """
        # Count annotations per class
        class_counts = {}
        for ann in coco_data['annotations']:
            cat_id = ann['category_id']
            class_counts[cat_id] = class_counts.get(cat_id, 0) + 1

        # Compute weights
        method_map = {
            'inverse_frequency': cls.compute_inverse_frequency,
            'effective_samples': cls.compute_effective_samples,
            'focal': cls.compute_focal_weights,
            'sqrt_inverse': cls.compute_sqrt_inverse
        }

        if method not in method_map:
            raise ValueError(f"Unknown method: {method}. Available: {list(method_map.keys())}")

        return method_map[method](class_counts, **kwargs)
