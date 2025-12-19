"""
Main class balancer for dataset balancing operations.
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import shutil
import logging

from .sampling_strategies import (
    BalancingConfig, BalancingResult,
    OversamplingStrategy, UndersamplingStrategy, HybridSamplingStrategy,
    BaseSamplingStrategy
)
from .weights_calculator import ClassWeightsCalculator

logger = logging.getLogger(__name__)


class ClassBalancer:
    """
    Main class for balancing datasets by class.

    Supports:
    - Oversampling minority classes
    - Undersampling majority classes
    - Hybrid balancing
    - Class weights calculation
    """

    STRATEGIES: Dict[str, type] = {
        'oversample': OversamplingStrategy,
        'undersample': UndersamplingStrategy,
        'hybrid': HybridSamplingStrategy
    }

    def __init__(self, output_dir: str):
        """
        Initialize class balancer.

        Args:
            output_dir: Directory for balanced dataset output
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.weights_calculator = ClassWeightsCalculator()

    def balance_dataset(self,
                        coco_data: Dict[str, Any],
                        images_dir: str,
                        strategy: str = 'oversample',
                        config: Optional[BalancingConfig] = None,
                        copy_images: bool = True) -> Tuple[Dict[str, Any], BalancingResult]:
        """
        Balance a COCO dataset.

        Args:
            coco_data: COCO format dataset
            images_dir: Directory containing images
            strategy: 'oversample', 'undersample', or 'hybrid'
            config: Balancing configuration
            copy_images: Whether to copy/duplicate image files

        Returns:
            Tuple of (balanced_coco_data, BalancingResult)
        """
        if config is None:
            config = BalancingConfig(strategy=strategy)

        strategy_cls = self.STRATEGIES.get(strategy)
        if not strategy_cls:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(self.STRATEGIES.keys())}")

        # Perform balancing
        balancer = strategy_cls(config)
        result = balancer.balance(coco_data['annotations'], coco_data['categories'])

        logger.info(result.get_summary())

        # Apply balancing to create new dataset
        balanced_data = self._apply_balancing(coco_data, result)

        # Save dataset
        self._save_balanced_dataset(balanced_data, images_dir, result, copy_images)

        return balanced_data, result

    def compute_weights(self,
                        coco_data: Dict[str, Any],
                        method: str = 'inverse_frequency',
                        export_format: str = 'dict',
                        save_to_file: bool = True,
                        **kwargs) -> Any:
        """
        Compute class weights without modifying the dataset.

        Args:
            coco_data: COCO format dataset
            method: 'inverse_frequency', 'effective_samples', 'focal', 'sqrt_inverse'
            export_format: 'dict', 'pytorch', 'tensorflow', 'list'
            save_to_file: Whether to save weights to JSON file
            **kwargs: Additional arguments for specific methods

        Returns:
            Weights in specified format
        """
        # Compute weights
        weights = self.weights_calculator.from_coco_data(coco_data, method, **kwargs)

        # Export to format
        exported = self.weights_calculator.export_weights(
            weights, coco_data['categories'], export_format
        )

        # Save to file
        if save_to_file:
            weights_path = self.output_dir / f"class_weights_{method}.json"

            # Convert to serializable format
            if export_format == 'pytorch':
                save_data = {
                    'weights': exported['weights'],
                    'class_to_idx': exported['class_to_idx'],
                    'idx_to_class': {str(k): v for k, v in exported['idx_to_class'].items()}
                }
            else:
                save_data = exported if isinstance(exported, dict) else {'weights': exported}

            with open(weights_path, 'w') as f:
                json.dump(save_data, f, indent=2)

            logger.info(f"Saved class weights to {weights_path}")

        return exported

    def _apply_balancing(self, coco_data: Dict[str, Any],
                         result: BalancingResult) -> Dict[str, Any]:
        """
        Apply balancing result to create new dataset.

        Args:
            coco_data: Original COCO data
            result: Balancing result

        Returns:
            Balanced COCO data
        """
        # Build mappings
        img_id_to_info = {img['id']: img for img in coco_data['images']}
        ann_to_img = {i: ann['image_id'] for i, ann in enumerate(coco_data['annotations'])}

        # Initialize balanced data
        balanced = {
            'info': coco_data.get('info', {}),
            'licenses': coco_data.get('licenses', []),
            'categories': coco_data['categories'],
            'images': [],
            'annotations': []
        }

        # Set of indices to keep (all except undersampled)
        undersampled_set = set(result.undersampled_indices)
        keep_indices = [i for i in range(len(coco_data['annotations']))
                        if i not in undersampled_set]

        # Track which images are needed
        needed_img_ids = set()
        for idx in keep_indices:
            needed_img_ids.add(ann_to_img[idx])

        # Find max IDs for new entries
        max_img_id = max((img['id'] for img in coco_data['images']), default=0)
        max_ann_id = max((ann['id'] for ann in coco_data['annotations']), default=0)

        new_img_id = max_img_id + 1
        new_ann_id = max_ann_id + 1

        # Add images for kept annotations
        added_img_ids = set()
        for img_id in needed_img_ids:
            if img_id not in added_img_ids:
                balanced['images'].append(img_id_to_info[img_id])
                added_img_ids.add(img_id)

        # Add kept annotations
        for idx in keep_indices:
            balanced['annotations'].append(coco_data['annotations'][idx])

        # Add oversampled copies
        for orig_idx, n_copies in result.oversampled_indices.items():
            orig_ann = coco_data['annotations'][orig_idx]
            orig_img_id = orig_ann['image_id']
            orig_img = img_id_to_info[orig_img_id]

            for copy_num in range(n_copies):
                # Create new image entry (copy)
                new_img = orig_img.copy()
                new_img['id'] = new_img_id
                new_img['file_name'] = f"bal_{new_img_id}_{orig_img['file_name']}"
                new_img['is_balanced_copy'] = True
                new_img['original_image_id'] = orig_img_id
                balanced['images'].append(new_img)

                # Create new annotation
                new_ann = orig_ann.copy()
                new_ann['id'] = new_ann_id
                new_ann['image_id'] = new_img_id
                new_ann['is_balanced_copy'] = True
                balanced['annotations'].append(new_ann)

                new_img_id += 1
                new_ann_id += 1

        return balanced

    def _save_balanced_dataset(self, data: Dict[str, Any],
                               images_dir: str,
                               result: BalancingResult,
                               copy_images: bool) -> None:
        """
        Save balanced dataset to disk.

        Args:
            data: Balanced COCO data
            images_dir: Source images directory
            result: Balancing result
            copy_images: Whether to copy image files
        """
        # Save JSON
        json_path = self.output_dir / "balanced_dataset.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved balanced dataset to {json_path}")

        # Save class weights
        weights_path = self.output_dir / "class_weights.json"
        with open(weights_path, 'w') as f:
            json.dump(result.class_weights, f, indent=2)

        logger.info(f"Saved class weights to {weights_path}")

        # Save balancing statistics
        stats_path = self.output_dir / "balancing_stats.json"
        stats = {
            'strategy': result.strategy_name,
            'original_counts': result.original_counts,
            'balanced_counts': result.balanced_counts,
            'total_original': result.total_original,
            'total_balanced': result.total_balanced,
            'oversampled_count': sum(result.oversampled_indices.values()),
            'undersampled_count': len(result.undersampled_indices)
        }
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        # Copy images if requested
        if copy_images:
            self._copy_images(data, images_dir)

    def _copy_images(self, data: Dict[str, Any], images_dir: str) -> None:
        """
        Copy images for balanced dataset.

        Args:
            data: Balanced COCO data
            images_dir: Source directory
        """
        images_out = self.output_dir / "images"
        images_out.mkdir(exist_ok=True)

        for img in data['images']:
            dst = images_out / img['file_name']

            if img.get('is_balanced_copy'):
                # This is a duplicate, copy from original
                original_name = img['file_name'].split('_', 2)[-1]  # Remove "bal_N_" prefix
                src = Path(images_dir) / original_name
            else:
                src = Path(images_dir) / img['file_name']

            if src.exists():
                shutil.copy2(src, dst)
            else:
                logger.warning(f"Source image not found: {src}")

        logger.info(f"Copied {len(data['images'])} images to {images_out}")

    def get_class_distribution(self, coco_data: Dict[str, Any]) -> Dict[str, int]:
        """
        Get class distribution of a dataset.

        Args:
            coco_data: COCO format dataset

        Returns:
            Dictionary of category_name -> count
        """
        cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

        distribution = {}
        for ann in coco_data['annotations']:
            cat_name = cat_id_to_name.get(ann['category_id'], f"unknown_{ann['category_id']}")
            distribution[cat_name] = distribution.get(cat_name, 0) + 1

        return distribution

    def get_imbalance_ratio(self, coco_data: Dict[str, Any]) -> float:
        """
        Calculate imbalance ratio (max_count / min_count).

        Args:
            coco_data: COCO format dataset

        Returns:
            Imbalance ratio (1.0 = perfectly balanced)
        """
        distribution = self.get_class_distribution(coco_data)
        counts = list(distribution.values())

        if not counts or min(counts) == 0:
            return float('inf')

        return max(counts) / min(counts)

    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Get list of available balancing strategies."""
        return list(cls.STRATEGIES.keys())
