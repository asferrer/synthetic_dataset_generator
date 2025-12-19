"""
Unified post-processing pipeline for datasets.

Combines:
- Multi-format export (COCO, YOLO, COCO Segmentation)
- Train/Val/Test splits (Random, Stratified, K-Folds)
- Class balancing (Oversampling, Undersampling, Hybrid)
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field
import json
import logging

from ..export import ExportManager, ExportConfig, ExportResult
from ..splits import DatasetSplitter, SplitConfig, KFoldGenerator, KFoldConfig
from ..balancing import ClassBalancer, BalancingConfig, ClassWeightsCalculator

logger = logging.getLogger(__name__)


@dataclass
class PostProcessingConfig:
    """Configuration for post-processing pipeline."""
    output_dir: str

    # Export settings
    export_formats: List[str] = field(default_factory=lambda: ['coco', 'yolo'])
    copy_images: bool = True

    # Split settings
    enable_splits: bool = True
    split_strategy: str = 'stratified'
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1

    # K-Fold settings
    enable_kfolds: bool = False
    n_folds: int = 5
    stratified_folds: bool = True

    # Balancing settings
    enable_balancing: bool = False
    balance_strategy: str = 'oversample'
    balance_target: Optional[int] = None
    max_oversample_ratio: float = 5.0

    # Weights settings
    compute_weights: bool = True
    weights_method: str = 'inverse_frequency'

    # Dataset combination
    combine_synthetic: bool = False

    random_seed: int = 42


class PostProcessingPipeline:
    """
    Unified pipeline for post-processing generated datasets.

    Usage:
        config = PostProcessingConfig(
            output_dir="/output",
            export_formats=['coco', 'yolo'],
            enable_splits=True,
            enable_balancing=True
        )

        pipeline = PostProcessingPipeline(config)
        results = pipeline.run(
            coco_data=synthetic_data,
            images_dir="/path/to/images"
        )
    """

    def __init__(self, config: PostProcessingConfig):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.export_manager = ExportManager(str(self.output_dir / "exports"))
        self.splitter = DatasetSplitter(str(self.output_dir / "splits"))
        self.balancer = ClassBalancer(str(self.output_dir / "balanced"))
        self.weights_calculator = ClassWeightsCalculator()

    def run(self,
            coco_data: Dict[str, Any],
            images_dir: str,
            synthetic_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the complete post-processing pipeline.

        Args:
            coco_data: Original/main COCO dataset
            images_dir: Directory containing images
            synthetic_data: Optional synthetic data to combine

        Returns:
            Dictionary with all results from each stage
        """
        results = {
            'config': self._config_to_dict(),
            'stages': {}
        }

        working_data = coco_data

        # Stage 1: Combine datasets if requested
        if self.config.combine_synthetic and synthetic_data:
            logger.info("Stage 1: Combining datasets...")
            working_data = self._merge_datasets(coco_data, synthetic_data)
            results['stages']['combine'] = {
                'original_images': len(coco_data['images']),
                'synthetic_images': len(synthetic_data['images']),
                'combined_images': len(working_data['images']),
                'combined_annotations': len(working_data['annotations'])
            }
            logger.info(f"Combined: {len(working_data['images'])} images, "
                       f"{len(working_data['annotations'])} annotations")
        else:
            results['stages']['combine'] = {'skipped': True}

        # Stage 2: Balancing (before splits to ensure balanced distribution)
        if self.config.enable_balancing:
            logger.info("Stage 2: Balancing classes...")
            balance_config = BalancingConfig(
                target_count=self.config.balance_target,
                strategy=self.config.balance_strategy,
                random_seed=self.config.random_seed,
                max_oversample_ratio=self.config.max_oversample_ratio
            )
            working_data, balance_result = self.balancer.balance_dataset(
                working_data, images_dir,
                strategy=self.config.balance_strategy,
                config=balance_config,
                copy_images=False  # Don't copy yet, will copy during split
            )
            results['stages']['balancing'] = {
                'strategy': balance_result.strategy_name,
                'original_counts': balance_result.original_counts,
                'balanced_counts': balance_result.balanced_counts,
                'total_original': balance_result.total_original,
                'total_balanced': balance_result.total_balanced
            }
            logger.info(f"Balanced: {balance_result.total_balanced} annotations")
        else:
            results['stages']['balancing'] = {'skipped': True}

        # Stage 3: Compute class weights
        if self.config.compute_weights:
            logger.info("Stage 3: Computing class weights...")
            weights = self.balancer.compute_weights(
                working_data,
                method=self.config.weights_method,
                export_format='pytorch',
                save_to_file=True
            )
            results['stages']['weights'] = {
                'method': self.config.weights_method,
                'weights': weights
            }
        else:
            results['stages']['weights'] = {'skipped': True}

        # Stage 4: Split dataset
        if self.config.enable_splits:
            logger.info("Stage 4: Splitting dataset...")
            split_config = SplitConfig(
                train_ratio=self.config.train_ratio,
                val_ratio=self.config.val_ratio,
                test_ratio=self.config.test_ratio,
                random_seed=self.config.random_seed
            )
            splits = self.splitter.split_dataset(
                working_data, images_dir,
                strategy=self.config.split_strategy,
                config=split_config,
                copy_images=self.config.copy_images
            )
            results['stages']['splits'] = {
                'strategy': self.config.split_strategy,
                'train_images': len(splits['train']['images']),
                'val_images': len(splits['val']['images']),
                'test_images': len(splits['test']['images']),
                'train_annotations': len(splits['train']['annotations']),
                'val_annotations': len(splits['val']['annotations']),
                'test_annotations': len(splits['test']['annotations'])
            }
            logger.info(f"Split: train={len(splits['train']['images'])}, "
                       f"val={len(splits['val']['images'])}, "
                       f"test={len(splits['test']['images'])}")

            # Stage 5: Export each split
            if self.config.export_formats:
                logger.info("Stage 5: Exporting splits...")
                export_results = self._export_splits(splits, images_dir)
                results['stages']['exports'] = export_results
        else:
            results['stages']['splits'] = {'skipped': True}

            # Export without splits
            if self.config.export_formats:
                logger.info("Stage 4: Exporting dataset...")
                export_results = self.export_manager.export_all(
                    working_data, images_dir,
                    self.config.export_formats,
                    "dataset"
                )
                results['stages']['exports'] = {
                    fmt: {
                        'success': r.success,
                        'path': r.output_path,
                        'images': r.num_images,
                        'annotations': r.num_annotations
                    }
                    for fmt, r in export_results.items()
                }

        # Stage 6: K-Folds (optional)
        if self.config.enable_kfolds:
            logger.info("Stage 6: Generating K-Folds...")
            folds = self.splitter.create_kfolds(
                working_data, images_dir,
                n_folds=self.config.n_folds,
                stratified=self.config.stratified_folds,
                copy_images=False
            )

            # Export each fold
            folds_results = []
            for fold in folds:
                fold_exports = {}
                for split_name in ['train', 'val']:
                    fold_data = fold[split_name]
                    for fmt in self.config.export_formats:
                        fold_output = self.output_dir / "kfolds" / f"fold_{fold['fold']}" / split_name / fmt
                        fold_output.mkdir(parents=True, exist_ok=True)

                        export_config = ExportConfig(output_dir=str(fold_output))
                        exporter = self.export_manager.AVAILABLE_FORMATS[fmt](export_config)
                        result = exporter.export(fold_data, images_dir, split_name)

                        fold_exports[f"{split_name}_{fmt}"] = result.success

                folds_results.append({
                    'fold': fold['fold'],
                    'train_images': len(fold['train']['images']),
                    'val_images': len(fold['val']['images']),
                    'exports': fold_exports
                })

            results['stages']['kfolds'] = {
                'n_folds': self.config.n_folds,
                'stratified': self.config.stratified_folds,
                'folds': folds_results
            }
        else:
            results['stages']['kfolds'] = {'skipped': True}

        # Save results summary
        self._save_results(results)

        logger.info("Post-processing pipeline complete!")
        return results

    def _export_splits(self, splits: Dict[str, Dict], images_dir: str) -> Dict:
        """
        Export each split to all configured formats.

        Args:
            splits: Dictionary of split datasets
            images_dir: Images directory

        Returns:
            Export results dictionary
        """
        export_results = {}

        for split_name, split_data in splits.items():
            split_results = {}

            # Determine images directory for this split
            split_images_dir = self.output_dir / "splits" / split_name / "images"
            if not split_images_dir.exists():
                split_images_dir = Path(images_dir)

            for fmt in self.config.export_formats:
                # Configure export for this split/format
                export_dir = self.output_dir / "exports" / split_name / fmt
                export_dir.mkdir(parents=True, exist_ok=True)

                export_config = ExportConfig(
                    output_dir=str(export_dir),
                    copy_images=False  # Images already in split dir
                )

                exporter = self.export_manager.AVAILABLE_FORMATS[fmt](export_config)
                result = exporter.export(split_data, str(split_images_dir), split_name)

                split_results[fmt] = {
                    'success': result.success,
                    'path': result.output_path,
                    'images': result.num_images,
                    'annotations': result.num_annotations,
                    'errors': result.errors[:3] if result.errors else []
                }

            export_results[split_name] = split_results

        return export_results

    def _merge_datasets(self, original: Dict, synthetic: Dict) -> Dict:
        """
        Merge original and synthetic datasets.

        Args:
            original: Original COCO data
            synthetic: Synthetic COCO data

        Returns:
            Merged dataset
        """
        max_img_id = max((img['id'] for img in original['images']), default=0)
        max_ann_id = max((ann['id'] for ann in original['annotations']), default=0)

        merged = {
            'info': original.get('info', {}),
            'licenses': original.get('licenses', []),
            'categories': original['categories'],
            'images': list(original['images']),
            'annotations': list(original['annotations'])
        }

        # Add synthetic with offset IDs
        img_id_map = {}
        for img in synthetic['images']:
            new_id = img['id'] + max_img_id + 1
            img_id_map[img['id']] = new_id

            new_img = img.copy()
            new_img['id'] = new_id
            new_img['is_synthetic'] = True
            merged['images'].append(new_img)

        for ann in synthetic['annotations']:
            new_ann = ann.copy()
            new_ann['id'] = ann['id'] + max_ann_id + 1
            new_ann['image_id'] = img_id_map[ann['image_id']]
            merged['annotations'].append(new_ann)

        return merged

    def _config_to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'output_dir': str(self.config.output_dir),
            'export_formats': self.config.export_formats,
            'enable_splits': self.config.enable_splits,
            'split_strategy': self.config.split_strategy,
            'train_ratio': self.config.train_ratio,
            'val_ratio': self.config.val_ratio,
            'test_ratio': self.config.test_ratio,
            'enable_kfolds': self.config.enable_kfolds,
            'n_folds': self.config.n_folds,
            'enable_balancing': self.config.enable_balancing,
            'balance_strategy': self.config.balance_strategy,
            'compute_weights': self.config.compute_weights,
            'weights_method': self.config.weights_method
        }

    def _save_results(self, results: Dict) -> None:
        """Save results summary to file."""
        results_path = self.output_dir / "pipeline_results.json"

        # Convert any non-serializable objects
        serializable = self._make_serializable(results)

        with open(results_path, 'w') as f:
            json.dump(serializable, f, indent=2)

        logger.info(f"Saved pipeline results to {results_path}")

    def _make_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)

    @classmethod
    def from_config_file(cls, config_path: str) -> 'PostProcessingPipeline':
        """
        Create pipeline from YAML/JSON config file.

        Args:
            config_path: Path to config file

        Returns:
            PostProcessingPipeline instance
        """
        import yaml

        with open(config_path) as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)

        config = PostProcessingConfig(**config_dict)
        return cls(config)
