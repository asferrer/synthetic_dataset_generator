"""
Class weights calculator for training.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ClassWeightsCalculator:
    """
    Calculator for class weights to use during model training.

    Supports multiple weighting schemes:
    - Inverse frequency
    - Effective number of samples
    - Focal loss weights
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
            # Return dict with category names
            return {
                cat['name']: weights.get(cat['id'], 1.0)
                for cat in sorted_cats
            }

        elif format == 'pytorch':
            # Return tensor-ready format for PyTorch
            weights_list = [weights.get(cat['id'], 1.0) for cat in sorted_cats]
            return {
                'weights': weights_list,
                'class_to_idx': {cat['name']: i for i, cat in enumerate(sorted_cats)},
                'idx_to_class': {i: cat['name'] for i, cat in enumerate(sorted_cats)}
            }

        elif format == 'tensorflow':
            # Return dict with category IDs (TF typically uses integer keys)
            return {
                cat['id']: weights.get(cat['id'], 1.0)
                for cat in sorted_cats
            }

        elif format == 'list':
            # Simple ordered list
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
