"""
Unit tests for Class Balancer and Weights Calculator.

These tests verify the class balancing and weight calculation functionality.
"""
import pytest
import sys
from pathlib import Path

# Add frontend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "frontend"))

from app.utils.balancer import (
    ClassBalancer,
    ClassWeightsCalculator,
    BalancingConfig,
    BalancingResult
)


@pytest.mark.unit
class TestBalancingConfig:
    """Tests for BalancingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BalancingConfig()

        assert config.target_count is None
        assert config.strategy == 'oversample'
        assert config.random_seed == 42
        assert config.max_oversample_ratio == 5.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = BalancingConfig(
            target_count=100,
            strategy='undersample',
            random_seed=123,
            max_oversample_ratio=3.0
        )

        assert config.target_count == 100
        assert config.strategy == 'undersample'
        assert config.random_seed == 123
        assert config.max_oversample_ratio == 3.0


@pytest.mark.unit
class TestBalancingResult:
    """Tests for BalancingResult dataclass."""

    def test_result_creation(self):
        """Test BalancingResult creation."""
        result = BalancingResult(
            original_counts={1: 10, 2: 100},
            balanced_counts={1: 50, 2: 100},
            strategy_used='oversample',
            samples_added=40,
            samples_removed=0,
            target_count=100
        )

        assert result.original_counts[1] == 10
        assert result.balanced_counts[1] == 50
        assert result.samples_added == 40
        assert result.samples_removed == 0


@pytest.mark.unit
class TestClassBalancer:
    """Tests for ClassBalancer class."""

    def test_available_strategies(self):
        """Test that all expected strategies are available."""
        balancer = ClassBalancer()

        assert 'oversample' in balancer.STRATEGIES
        assert 'undersample' in balancer.STRATEGIES
        assert 'hybrid' in balancer.STRATEGIES

    def test_oversample_increases_minority(self, imbalanced_coco_data):
        """Test oversampling increases minority class count."""
        balancer = ClassBalancer()

        balanced_data, result = balancer.balance(
            imbalanced_coco_data,
            strategy='oversample'
        )

        # Rare class (id=1) should be increased
        assert result.balanced_counts.get(1, 0) >= result.original_counts.get(1, 0)

    def test_undersample_decreases_majority(self, imbalanced_coco_data):
        """Test undersampling decreases majority class count."""
        balancer = ClassBalancer()

        balanced_data, result = balancer.balance(
            imbalanced_coco_data,
            strategy='undersample'
        )

        # Common class (id=2) should be decreased
        assert result.balanced_counts.get(2, 0) <= result.original_counts.get(2, 0)

    def test_hybrid_balancing(self, imbalanced_coco_data):
        """Test hybrid balancing strategy."""
        balancer = ClassBalancer()

        balanced_data, result = balancer.balance(
            imbalanced_coco_data,
            strategy='hybrid'
        )

        # Should have performed some balancing
        assert result.strategy_used == 'hybrid'

    def test_balance_preserves_categories(self, imbalanced_coco_data):
        """Test that balancing preserves category information."""
        balancer = ClassBalancer()

        balanced_data, _ = balancer.balance(
            imbalanced_coco_data,
            strategy='oversample'
        )

        assert balanced_data['categories'] == imbalanced_coco_data['categories']

    def test_balance_with_target_count(self, imbalanced_coco_data):
        """Test balancing with specific target count."""
        balancer = ClassBalancer()

        balanced_data, result = balancer.balance(
            imbalanced_coco_data,
            strategy='oversample',
            target_count=50
        )

        assert result.target_count == 50

    def test_annotations_reference_valid_images(self, imbalanced_coco_data):
        """Test that balanced annotations reference valid images."""
        balancer = ClassBalancer()

        balanced_data, _ = balancer.balance(
            imbalanced_coco_data,
            strategy='oversample'
        )

        image_ids = {img['id'] for img in balanced_data['images']}
        for ann in balanced_data['annotations']:
            assert ann['image_id'] in image_ids

    def test_empty_dataset_handling(self):
        """Test handling of empty dataset."""
        empty_data = {
            'categories': [{'id': 1, 'name': 'test'}],
            'images': [],
            'annotations': []
        }

        balancer = ClassBalancer()
        balanced_data, result = balancer.balance(empty_data, strategy='oversample')

        assert len(balanced_data['images']) == 0
        assert len(balanced_data['annotations']) == 0

    def test_already_balanced_dataset(self, sample_coco_data):
        """Test handling of already balanced dataset."""
        balancer = ClassBalancer()

        balanced_data, result = balancer.balance(
            sample_coco_data,
            strategy='oversample'
        )

        # Should not fail on relatively balanced data
        assert 'images' in balanced_data
        assert 'annotations' in balanced_data

    def test_invalid_strategy_raises_error(self, imbalanced_coco_data):
        """Test that invalid strategy raises error."""
        balancer = ClassBalancer()

        with pytest.raises((ValueError, KeyError)):
            balancer.balance(imbalanced_coco_data, strategy='invalid_strategy')

    def test_reproducibility_with_seed(self, imbalanced_coco_data):
        """Test that same seed produces same results."""
        balancer = ClassBalancer()

        config1 = BalancingConfig(random_seed=42)
        config2 = BalancingConfig(random_seed=42)

        balanced1, _ = balancer.balance(
            imbalanced_coco_data,
            strategy='oversample',
            config=config1
        )

        balanced2, _ = balancer.balance(
            imbalanced_coco_data,
            strategy='oversample',
            config=config2
        )

        # Same seed should produce same number of samples
        assert len(balanced1['annotations']) == len(balanced2['annotations'])


@pytest.mark.unit
class TestClassWeightsCalculator:
    """Tests for ClassWeightsCalculator class."""

    def test_inverse_frequency_weights(self):
        """Test inverse frequency weight calculation."""
        class_counts = {1: 10, 2: 100}

        weights = ClassWeightsCalculator.compute_inverse_frequency(class_counts)

        # Rare class should have higher weight
        assert weights[1] > weights[2]

    def test_inverse_frequency_normalized(self):
        """Test that normalized weights sum to n_classes."""
        class_counts = {1: 10, 2: 50, 3: 100}

        weights = ClassWeightsCalculator.compute_inverse_frequency(
            class_counts,
            normalize=True
        )

        # Normalized weights should sum to number of classes
        total = sum(weights.values())
        assert abs(total - len(class_counts)) < 0.01

    def test_effective_samples_weights(self):
        """Test effective samples weight calculation."""
        class_counts = {1: 10, 2: 100}

        weights = ClassWeightsCalculator.compute_effective_samples(class_counts)

        # Rare class should have higher weight
        assert weights[1] > weights[2]

    def test_effective_samples_beta_parameter(self):
        """Test effective samples with different beta values."""
        class_counts = {1: 10, 2: 100}

        weights_high_beta = ClassWeightsCalculator.compute_effective_samples(
            class_counts,
            beta=0.9999
        )
        weights_low_beta = ClassWeightsCalculator.compute_effective_samples(
            class_counts,
            beta=0.9
        )

        # Different betas should produce different weights
        assert weights_high_beta[1] != weights_low_beta[1]

    def test_focal_weights(self):
        """Test focal loss weight calculation."""
        class_counts = {1: 10, 2: 100}

        weights = ClassWeightsCalculator.compute_focal_weights(class_counts)

        # Rare class should have higher weight
        assert weights[1] > weights[2]

    def test_focal_weights_gamma_parameter(self):
        """Test focal weights with different gamma values."""
        class_counts = {1: 10, 2: 100}

        weights_high_gamma = ClassWeightsCalculator.compute_focal_weights(
            class_counts,
            gamma=3.0
        )
        weights_low_gamma = ClassWeightsCalculator.compute_focal_weights(
            class_counts,
            gamma=1.0
        )

        # Higher gamma should increase the difference
        ratio_high = weights_high_gamma[1] / weights_high_gamma[2]
        ratio_low = weights_low_gamma[1] / weights_low_gamma[2]
        assert ratio_high > ratio_low

    def test_sqrt_inverse_weights(self):
        """Test square root inverse weight calculation."""
        class_counts = {1: 10, 2: 100}

        weights = ClassWeightsCalculator.compute_sqrt_inverse(class_counts)

        # Rare class should have higher weight
        assert weights[1] > weights[2]

    def test_from_coco_data(self, imbalanced_coco_data):
        """Test weight calculation from COCO data."""
        weights = ClassWeightsCalculator.from_coco_data(
            imbalanced_coco_data,
            method='inverse_frequency'
        )

        # Should have weights for both classes
        assert 1 in weights  # rare_class
        assert 2 in weights  # common_class

        # Rare class should have higher weight
        assert weights[1] > weights[2]

    def test_from_coco_data_all_methods(self, imbalanced_coco_data):
        """Test all weight calculation methods from COCO data."""
        methods = ['inverse_frequency', 'effective_samples', 'focal', 'sqrt_inverse']

        for method in methods:
            weights = ClassWeightsCalculator.from_coco_data(
                imbalanced_coco_data,
                method=method
            )

            assert len(weights) == 2, f"Method {method} should return 2 weights"
            assert all(w > 0 for w in weights.values()), \
                f"Method {method} should return positive weights"

    def test_empty_counts_handling(self):
        """Test handling of empty class counts."""
        empty_counts = {}

        weights = ClassWeightsCalculator.compute_inverse_frequency(empty_counts)
        assert weights == {}

    def test_single_class_handling(self):
        """Test handling of single class."""
        single_class = {1: 100}

        weights = ClassWeightsCalculator.compute_inverse_frequency(single_class)

        assert len(weights) == 1
        assert weights[1] > 0

    def test_weights_are_positive(self):
        """Test that all weight calculations return positive values."""
        class_counts = {1: 1, 2: 10, 3: 100, 4: 1000}

        for method_name in ['compute_inverse_frequency', 'compute_effective_samples',
                            'compute_focal_weights', 'compute_sqrt_inverse']:
            method = getattr(ClassWeightsCalculator, method_name)
            weights = method(class_counts)

            assert all(w > 0 for w in weights.values()), \
                f"{method_name} should return all positive weights"
