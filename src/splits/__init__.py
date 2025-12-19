"""
Splits module for train/val/test dataset splitting.

Supports:
- Random splits
- Stratified splits (maintains class distribution)
- Stratified K-Folds for cross-validation
"""

from .split_strategies import (
    SplitConfig,
    SplitResult,
    BaseSplitStrategy,
    RandomSplitStrategy,
    StratifiedSplitStrategy
)
from .kfold_generator import KFoldConfig, FoldResult, KFoldGenerator
from .dataset_splitter import DatasetSplitter

__all__ = [
    'SplitConfig',
    'SplitResult',
    'BaseSplitStrategy',
    'RandomSplitStrategy',
    'StratifiedSplitStrategy',
    'KFoldConfig',
    'FoldResult',
    'KFoldGenerator',
    'DatasetSplitter'
]
