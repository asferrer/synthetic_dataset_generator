"""
Unit tests for Dataset Splitter and K-Fold Generator.

These tests verify the splitting and cross-validation functionality.
"""
import pytest
import sys
from pathlib import Path

# Add frontend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "frontend"))

from app.utils.splitter import DatasetSplitter, KFoldGenerator, KFoldConfig, FoldResult


@pytest.mark.unit
class TestDatasetSplitter:
    """Tests for DatasetSplitter class."""

    def test_random_split_default_ratios(self, sample_coco_data):
        """Test random split with default ratios (70/20/10)."""
        splits = DatasetSplitter.split_dataset(
            sample_coco_data,
            strategy='random'
        )

        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits

        total_images = len(sample_coco_data['images'])
        assert len(splits['train']['images']) == int(total_images * 0.7)
        assert len(splits['val']['images']) == int(total_images * 0.2)
        assert len(splits['test']['images']) == total_images - int(total_images * 0.7) - int(total_images * 0.2)

    def test_random_split_custom_ratios(self, sample_coco_data):
        """Test random split with custom ratios."""
        splits = DatasetSplitter.split_dataset(
            sample_coco_data,
            train_ratio=0.6,
            val_ratio=0.3,
            test_ratio=0.1,
            strategy='random'
        )

        total_images = len(sample_coco_data['images'])
        assert len(splits['train']['images']) == int(total_images * 0.6)
        assert len(splits['val']['images']) == int(total_images * 0.3)

    def test_stratified_split(self, sample_coco_data):
        """Test stratified split maintains class distribution."""
        splits = DatasetSplitter.split_dataset(
            sample_coco_data,
            strategy='stratified'
        )

        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits

        # All images should be distributed
        total = (
            len(splits['train']['images']) +
            len(splits['val']['images']) +
            len(splits['test']['images'])
        )
        assert total == len(sample_coco_data['images'])

    def test_split_preserves_categories(self, sample_coco_data):
        """Test that split preserves category information."""
        splits = DatasetSplitter.split_dataset(sample_coco_data, strategy='random')

        for split_name in ['train', 'val', 'test']:
            assert splits[split_name]['categories'] == sample_coco_data['categories']

    def test_split_preserves_info(self, sample_coco_data):
        """Test that split preserves info metadata."""
        splits = DatasetSplitter.split_dataset(sample_coco_data, strategy='random')

        for split_name in ['train', 'val', 'test']:
            assert splits[split_name]['info'] == sample_coco_data.get('info', {})

    def test_annotations_match_images(self, sample_coco_data):
        """Test that annotations in each split match the images."""
        splits = DatasetSplitter.split_dataset(sample_coco_data, strategy='random')

        for split_name, split_data in splits.items():
            image_ids = {img['id'] for img in split_data['images']}
            for ann in split_data['annotations']:
                assert ann['image_id'] in image_ids, \
                    f"Annotation {ann['id']} references image not in {split_name} split"

    def test_empty_dataset_handling(self):
        """Test handling of empty dataset."""
        empty_data = {
            'images': [],
            'annotations': [],
            'categories': [{'id': 1, 'name': 'test'}]
        }

        splits = DatasetSplitter.split_dataset(empty_data, strategy='random')

        assert len(splits['train']['images']) == 0
        assert len(splits['val']['images']) == 0
        assert len(splits['test']['images']) == 0

    def test_reproducibility_with_seed(self, sample_coco_data):
        """Test that same seed produces same splits."""
        splits1 = DatasetSplitter.split_dataset(
            sample_coco_data,
            strategy='random',
            random_seed=42
        )

        splits2 = DatasetSplitter.split_dataset(
            sample_coco_data,
            strategy='random',
            random_seed=42
        )

        # Same seed should produce same splits
        train_ids1 = {img['id'] for img in splits1['train']['images']}
        train_ids2 = {img['id'] for img in splits2['train']['images']}
        assert train_ids1 == train_ids2

    def test_different_seeds_produce_different_splits(self, sample_coco_data):
        """Test that different seeds produce different splits."""
        splits1 = DatasetSplitter.split_dataset(
            sample_coco_data,
            strategy='random',
            random_seed=42
        )

        splits2 = DatasetSplitter.split_dataset(
            sample_coco_data,
            strategy='random',
            random_seed=123
        )

        # Different seeds should (very likely) produce different splits
        train_ids1 = {img['id'] for img in splits1['train']['images']}
        train_ids2 = {img['id'] for img in splits2['train']['images']}
        # At least some difference expected
        assert train_ids1 != train_ids2

    def test_get_split_statistics(self, sample_coco_data):
        """Test split statistics calculation."""
        splits = DatasetSplitter.split_dataset(sample_coco_data, strategy='random')
        stats = DatasetSplitter.get_split_statistics(splits)

        assert 'train' in stats
        assert 'val' in stats
        assert 'test' in stats

        for split_name in ['train', 'val', 'test']:
            assert 'num_images' in stats[split_name]
            assert 'num_annotations' in stats[split_name]
            assert 'class_distribution' in stats[split_name]


@pytest.mark.unit
class TestKFoldConfig:
    """Tests for KFoldConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = KFoldConfig()

        assert config.n_folds == 5
        assert config.stratified is True
        assert config.random_seed == 42
        assert config.shuffle is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = KFoldConfig(n_folds=10, stratified=False, random_seed=123)

        assert config.n_folds == 10
        assert config.stratified is False
        assert config.random_seed == 123

    def test_invalid_folds_raises_error(self):
        """Test that n_folds < 2 raises error."""
        with pytest.raises(ValueError):
            KFoldConfig(n_folds=1)

        with pytest.raises(ValueError):
            KFoldConfig(n_folds=0)

        with pytest.raises(ValueError):
            KFoldConfig(n_folds=-1)


@pytest.mark.unit
class TestFoldResult:
    """Tests for FoldResult dataclass."""

    def test_fold_result_creation(self):
        """Test FoldResult creation and auto-calculated fields."""
        result = FoldResult(
            fold_number=1,
            train_indices=[0, 1, 2, 3],
            val_indices=[4, 5],
            train_images=[{'id': i} for i in range(4)],
            val_images=[{'id': i} for i in range(4, 6)]
        )

        assert result.fold_number == 1
        assert result.n_train == 4
        assert result.n_val == 2

    def test_fold_result_summary(self):
        """Test FoldResult summary generation."""
        result = FoldResult(
            fold_number=3,
            train_indices=[0, 1, 2],
            val_indices=[3],
            train_images=[{'id': i} for i in range(3)],
            val_images=[{'id': 3}]
        )

        summary = result.get_summary()
        assert "Fold 3" in summary
        assert "train=3" in summary
        assert "val=1" in summary


@pytest.mark.unit
class TestKFoldGenerator:
    """Tests for KFoldGenerator class."""

    def test_generator_default_init(self):
        """Test generator with default initialization."""
        generator = KFoldGenerator()
        assert generator.config.n_folds == 5

    def test_generator_custom_init(self):
        """Test generator with custom configuration."""
        config = KFoldConfig(n_folds=3)
        generator = KFoldGenerator(config=config)
        assert generator.config.n_folds == 3

    def test_generator_n_folds_init(self):
        """Test generator with n_folds parameter."""
        generator = KFoldGenerator(n_folds=10)
        assert generator.config.n_folds == 10

    def test_generate_folds_count(self, sample_coco_data):
        """Test that correct number of folds are generated."""
        generator = KFoldGenerator(n_folds=5)
        folds = list(generator.generate_folds(
            sample_coco_data['images'],
            sample_coco_data['annotations']
        ))

        assert len(folds) == 5

    def test_generate_folds_coverage(self, sample_coco_data):
        """Test that all images appear in validation exactly once."""
        generator = KFoldGenerator(n_folds=5)
        folds = generator.get_all_folds(
            sample_coco_data['images'],
            sample_coco_data['annotations']
        )

        val_indices = []
        for fold in folds:
            val_indices.extend(fold.val_indices)

        # Each index should appear exactly once as validation
        assert len(val_indices) == len(sample_coco_data['images'])
        assert len(set(val_indices)) == len(sample_coco_data['images'])

    def test_generate_folds_no_overlap(self, sample_coco_data):
        """Test that train and val don't overlap in each fold."""
        generator = KFoldGenerator(n_folds=5)
        folds = generator.get_all_folds(
            sample_coco_data['images'],
            sample_coco_data['annotations']
        )

        for fold in folds:
            train_set = set(fold.train_indices)
            val_set = set(fold.val_indices)
            assert train_set.isdisjoint(val_set), \
                f"Fold {fold.fold_number} has overlapping train/val"

    def test_export_fold(self, sample_coco_data):
        """Test exporting a fold to COCO format."""
        generator = KFoldGenerator(n_folds=3)
        folds = generator.get_all_folds(
            sample_coco_data['images'],
            sample_coco_data['annotations']
        )

        train_data, val_data = generator.export_fold(folds[0], sample_coco_data)

        # Check structure
        assert 'images' in train_data
        assert 'annotations' in train_data
        assert 'categories' in train_data

        assert 'images' in val_data
        assert 'annotations' in val_data
        assert 'categories' in val_data

        # Check counts match fold
        assert len(train_data['images']) == folds[0].n_train
        assert len(val_data['images']) == folds[0].n_val

    def test_get_fold_statistics(self, sample_coco_data):
        """Test fold statistics calculation."""
        generator = KFoldGenerator(n_folds=3)
        folds = generator.get_all_folds(
            sample_coco_data['images'],
            sample_coco_data['annotations']
        )

        stats = generator.get_fold_statistics(
            folds,
            sample_coco_data['annotations'],
            sample_coco_data['categories']
        )

        assert stats['n_folds'] == 3
        assert 'folds' in stats
        assert 'avg_train_size' in stats
        assert 'avg_val_size' in stats

    def test_stratified_folds(self, sample_coco_data):
        """Test stratified K-Fold generation."""
        config = KFoldConfig(n_folds=5, stratified=True)
        generator = KFoldGenerator(config=config)

        folds = generator.get_all_folds(
            sample_coco_data['images'],
            sample_coco_data['annotations']
        )

        # Should still generate 5 folds
        assert len(folds) == 5

    def test_non_stratified_folds(self, sample_coco_data):
        """Test non-stratified K-Fold generation."""
        config = KFoldConfig(n_folds=5, stratified=False)
        generator = KFoldGenerator(config=config)

        folds = generator.get_all_folds(
            sample_coco_data['images'],
            sample_coco_data['annotations']
        )

        assert len(folds) == 5

    def test_too_few_samples_raises_error(self):
        """Test that too few samples raises an error."""
        generator = KFoldGenerator(n_folds=5)
        small_images = [{'id': i} for i in range(3)]  # Only 3 samples
        small_annotations = [{'id': 1, 'image_id': 0, 'category_id': 1}]

        with pytest.raises(ValueError):
            list(generator.generate_folds(small_images, small_annotations))
