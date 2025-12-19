"""
Main dataset splitter class for train/val/test splitting.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import shutil
import logging

from .split_strategies import (
    SplitConfig, SplitResult,
    RandomSplitStrategy, StratifiedSplitStrategy, BaseSplitStrategy
)
from .kfold_generator import KFoldGenerator, KFoldConfig

logger = logging.getLogger(__name__)


class DatasetSplitter:
    """
    Main class for splitting datasets into train/val/test sets.

    Supports:
    - Random splits
    - Stratified splits
    - K-Fold cross-validation
    - Combining original and synthetic datasets
    """

    STRATEGIES: Dict[str, type] = {
        'random': RandomSplitStrategy,
        'stratified': StratifiedSplitStrategy
    }

    def __init__(self, output_dir: str):
        """
        Initialize dataset splitter.

        Args:
            output_dir: Directory for split outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def split_dataset(self,
                      coco_data: Dict[str, Any],
                      images_dir: str,
                      strategy: str = 'stratified',
                      config: Optional[SplitConfig] = None,
                      combine_with_synthetic: bool = False,
                      synthetic_data: Optional[Dict[str, Any]] = None,
                      copy_images: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Split dataset into train/val/test sets.

        Args:
            coco_data: Original COCO dataset
            images_dir: Directory containing images
            strategy: 'random' or 'stratified'
            config: Split configuration (uses default if None)
            combine_with_synthetic: Whether to combine with synthetic data
            synthetic_data: Synthetic COCO data to combine
            copy_images: Whether to copy images to split directories

        Returns:
            Dictionary with 'train', 'val', 'test' COCO datasets
        """
        if config is None:
            config = SplitConfig()

        # Combine datasets if requested
        working_data = coco_data
        if combine_with_synthetic and synthetic_data:
            logger.info("Combining original and synthetic datasets...")
            working_data = self._merge_datasets(coco_data, synthetic_data)
            logger.info(f"Combined dataset: {len(working_data['images'])} images, "
                       f"{len(working_data['annotations'])} annotations")

        # Get strategy class
        strategy_cls = self.STRATEGIES.get(strategy)
        if not strategy_cls:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(self.STRATEGIES.keys())}")

        # Perform split
        splitter = strategy_cls(config)
        result = splitter.split(working_data['images'], working_data['annotations'])

        logger.info(result.get_summary())

        # Create COCO datasets for each split
        splits = self._create_split_datasets(working_data, result)

        # Save splits and optionally copy images
        self._save_splits(splits, images_dir, copy_images)

        return splits

    def split_synthetic_only(self,
                             synthetic_data: Dict[str, Any],
                             images_dir: str,
                             strategy: str = 'stratified',
                             config: Optional[SplitConfig] = None,
                             copy_images: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Split only synthetic dataset (convenience method).

        Args:
            synthetic_data: Synthetic COCO data
            images_dir: Directory containing synthetic images
            strategy: Split strategy
            config: Split configuration
            copy_images: Whether to copy images

        Returns:
            Dictionary with split datasets
        """
        return self.split_dataset(
            coco_data=synthetic_data,
            images_dir=images_dir,
            strategy=strategy,
            config=config,
            combine_with_synthetic=False,
            copy_images=copy_images
        )

    def create_kfolds(self,
                      coco_data: Dict[str, Any],
                      images_dir: str,
                      n_folds: int = 5,
                      stratified: bool = True,
                      copy_images: bool = False) -> List[Dict[str, Any]]:
        """
        Create K-Fold cross-validation splits.

        Args:
            coco_data: COCO dataset
            images_dir: Directory containing images
            n_folds: Number of folds
            stratified: Whether to use stratified K-Fold
            copy_images: Whether to copy images to fold directories

        Returns:
            List of fold dictionaries, each with 'train' and 'val' COCO data
        """
        config = KFoldConfig(
            n_folds=n_folds,
            stratified=stratified
        )

        generator = KFoldGenerator(config)
        folds = []

        for fold in generator.generate_folds(coco_data['images'], coco_data['annotations']):
            train_data, val_data = generator.export_fold(fold, coco_data)

            # Create fold directory
            fold_dir = self.output_dir / f"fold_{fold.fold_number}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            # Save JSON files
            train_json = fold_dir / "train.json"
            val_json = fold_dir / "val.json"

            with open(train_json, 'w') as f:
                json.dump(train_data, f, indent=2)

            with open(val_json, 'w') as f:
                json.dump(val_data, f, indent=2)

            # Copy images if requested
            if copy_images:
                self._copy_fold_images(fold, images_dir, fold_dir)

            folds.append({
                'fold': fold.fold_number,
                'train': train_data,
                'val': val_data,
                'train_path': str(train_json),
                'val_path': str(val_json),
                'output_dir': str(fold_dir)
            })

            logger.info(f"Saved fold {fold.fold_number} to {fold_dir}")

        return folds

    def _merge_datasets(self, original: Dict[str, Any],
                        synthetic: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge original and synthetic datasets.

        Handles ID conflicts by offsetting synthetic IDs.

        Args:
            original: Original COCO data
            synthetic: Synthetic COCO data

        Returns:
            Merged COCO data
        """
        # Find max IDs in original
        max_img_id = max((img['id'] for img in original['images']), default=0)
        max_ann_id = max((ann['id'] for ann in original['annotations']), default=0)

        merged = {
            'info': original.get('info', {}),
            'licenses': original.get('licenses', []),
            'categories': original['categories'],
            'images': list(original['images']),
            'annotations': list(original['annotations'])
        }

        # Add synthetic images with offset IDs
        img_id_map = {}  # Old ID -> New ID
        for img in synthetic['images']:
            new_id = img['id'] + max_img_id + 1
            img_id_map[img['id']] = new_id

            new_img = img.copy()
            new_img['id'] = new_id
            new_img['is_synthetic'] = True
            merged['images'].append(new_img)

        # Add synthetic annotations with offset IDs
        for ann in synthetic['annotations']:
            new_ann = ann.copy()
            new_ann['id'] = ann['id'] + max_ann_id + 1
            new_ann['image_id'] = img_id_map[ann['image_id']]
            new_ann['is_synthetic'] = True
            merged['annotations'].append(new_ann)

        return merged

    def _create_split_datasets(self, coco_data: Dict[str, Any],
                               result: SplitResult) -> Dict[str, Dict[str, Any]]:
        """
        Create separate COCO datasets for each split.

        Args:
            coco_data: Full COCO data
            result: SplitResult with indices

        Returns:
            Dictionary with 'train', 'val', 'test' datasets
        """
        splits = {}

        for split_name, indices, images in [
            ('train', result.train_indices, result.train_images),
            ('val', result.val_indices, result.val_images),
            ('test', result.test_indices, result.test_images)
        ]:
            # Get image IDs for this split
            img_ids = {coco_data['images'][i]['id'] for i in indices}

            splits[split_name] = {
                'info': coco_data.get('info', {}),
                'licenses': coco_data.get('licenses', []),
                'categories': coco_data['categories'],
                'images': images,
                'annotations': [
                    ann for ann in coco_data['annotations']
                    if ann['image_id'] in img_ids
                ]
            }

        return splits

    def _save_splits(self, splits: Dict[str, Dict[str, Any]],
                     images_dir: str, copy_images: bool) -> None:
        """
        Save split datasets to disk.

        Args:
            splits: Dictionary of split datasets
            images_dir: Source images directory
            copy_images: Whether to copy images
        """
        for split_name, data in splits.items():
            split_dir = self.output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            # Save JSON
            json_path = split_dir / "annotations.json"
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {split_name} annotations to {json_path}")

            # Copy images if requested
            if copy_images:
                images_out = split_dir / "images"
                images_out.mkdir(exist_ok=True)

                for img in data['images']:
                    src = Path(images_dir) / img['file_name']
                    dst = images_out / img['file_name']

                    if src.exists():
                        shutil.copy2(src, dst)
                    else:
                        logger.warning(f"Source image not found: {src}")

                logger.info(f"Copied {len(data['images'])} images to {images_out}")

    def _copy_fold_images(self, fold, images_dir: str, fold_dir: Path) -> None:
        """
        Copy images for a fold.

        Args:
            fold: FoldResult
            images_dir: Source directory
            fold_dir: Destination fold directory
        """
        for split_name, images in [('train', fold.train_images), ('val', fold.val_images)]:
            split_images_dir = fold_dir / split_name / "images"
            split_images_dir.mkdir(parents=True, exist_ok=True)

            for img in images:
                src = Path(images_dir) / img['file_name']
                dst = split_images_dir / img['file_name']

                if src.exists():
                    shutil.copy2(src, dst)

    def get_split_summary(self, splits: Dict[str, Dict[str, Any]]) -> str:
        """
        Get human-readable summary of splits.

        Args:
            splits: Split datasets dictionary

        Returns:
            Summary string
        """
        lines = ["Dataset Split Summary", "=" * 40]

        total_images = 0
        total_annotations = 0

        for split_name in ['train', 'val', 'test']:
            if split_name in splits:
                data = splits[split_name]
                n_images = len(data['images'])
                n_annotations = len(data['annotations'])

                total_images += n_images
                total_annotations += n_annotations

                lines.append(f"\n{split_name.upper()}:")
                lines.append(f"  Images: {n_images}")
                lines.append(f"  Annotations: {n_annotations}")

                # Class distribution
                class_counts = {}
                for ann in data['annotations']:
                    cat = ann['category_id']
                    class_counts[cat] = class_counts.get(cat, 0) + 1

                if class_counts:
                    lines.append("  Classes:")
                    for cat_id, count in sorted(class_counts.items()):
                        lines.append(f"    {cat_id}: {count}")

        lines.append(f"\nTOTAL: {total_images} images, {total_annotations} annotations")

        return "\n".join(lines)
