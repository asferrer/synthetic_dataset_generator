"""
Balancing module for class balancing in datasets.

Supports:
- Oversampling (duplicate minority class samples)
- Undersampling (reduce majority class samples)
- Hybrid sampling (combination of both)
- Class weights calculation for training
"""

from .sampling_strategies import (
    BalancingConfig,
    BalancingResult,
    BaseSamplingStrategy,
    OversamplingStrategy,
    UndersamplingStrategy,
    HybridSamplingStrategy
)
from .weights_calculator import ClassWeightsCalculator
from .class_balancer import ClassBalancer

__all__ = [
    'BalancingConfig',
    'BalancingResult',
    'BaseSamplingStrategy',
    'OversamplingStrategy',
    'UndersamplingStrategy',
    'HybridSamplingStrategy',
    'ClassWeightsCalculator',
    'ClassBalancer'
]
