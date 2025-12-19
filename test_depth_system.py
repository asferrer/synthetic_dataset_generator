"""
Test script for depth-aware augmentation system.
Tests the DepthEstimator and depth-aware scaling functionality.
"""

import os
import sys
import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.augmentation.depth_engine import DepthEstimator

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

def create_test_image():
    """Create a simple test image with gradient depth."""
    # Create a 640x480 test image with depth gradient
    height, width = 480, 640
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Create a blue-green gradient (simulating underwater scene)
    for y in range(height):
        # Darker at top (far), lighter at bottom (near)
        intensity = int(50 + (y / height) * 150)
        img[y, :] = [intensity, intensity - 20, 100]  # BGR

    # Add some texture
    noise = np.random.randint(-20, 20, (height, width, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img

def test_depth_estimator():
    """Test DepthEstimator initialization and inference."""
    print("\n" + "="*80)
    print("TEST 1: DepthEstimator Initialization")
    print("="*80)

    try:
        # Test initialization with small model
        estimator = DepthEstimator(model_size='small', device='cuda', cache_dir='checkpoints')
        print("✓ DepthEstimator initialized successfully")
        print(f"  Model: {estimator.model_size}")
        print(f"  Device: {estimator.device}")
        print(f"  Cache dir: {estimator.cache_dir}")

        return estimator
    except Exception as e:
        print(f"✗ Failed to initialize DepthEstimator: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_depth_estimation(estimator):
    """Test depth map estimation."""
    print("\n" + "="*80)
    print("TEST 2: Depth Map Estimation")
    print("="*80)

    if estimator is None:
        print("✗ Skipping (no estimator)")
        return None

    try:
        # Create test image
        test_img = create_test_image()
        print(f"✓ Created test image: {test_img.shape}")

        # Estimate depth
        depth_map = estimator.estimate_depth(test_img, normalize=True)
        print(f"✓ Depth map estimated: {depth_map.shape}")
        print(f"  Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
        print(f"  Depth mean: {depth_map.mean():.3f}")
        print(f"  Depth std: {depth_map.std():.3f}")

        return test_img, depth_map
    except Exception as e:
        print(f"✗ Failed to estimate depth: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_depth_zones(estimator, depth_map):
    """Test depth zone classification."""
    print("\n" + "="*80)
    print("TEST 3: Depth Zone Classification")
    print("="*80)

    if estimator is None or depth_map is None:
        print("✗ Skipping (no depth map)")
        return

    try:
        zones_mask, depth_ranges = estimator.classify_depth_zones(depth_map, num_zones=3)
        print(f"✓ Depth zones classified: {zones_mask.shape}")
        print(f"  Zones: {np.unique(zones_mask)}")
        print(f"  Depth ranges:")
        for i, (min_d, max_d) in enumerate(depth_ranges):
            pixels = np.sum(zones_mask == i)
            percentage = (pixels / zones_mask.size) * 100
            print(f"    Zone {i} (near->far): [{min_d:.3f}, {max_d:.3f}] - {pixels} pixels ({percentage:.1f}%)")

        return zones_mask, depth_ranges
    except Exception as e:
        print(f"✗ Failed to classify zones: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_depth_aware_augmentor():
    """Test SyntheticDataAugmentor with depth-aware enabled."""
    print("\n" + "="*80)
    print("TEST 4: Depth-Aware Augmentor Integration")
    print("="*80)

    try:
        from src.augmentation.augmentor import SyntheticDataAugmentor

        # Initialize with depth-aware enabled
        augmentor = SyntheticDataAugmentor(
            output_dir="/app/test_output",
            depth_aware=True,
            depth_model_size='small',
            depth_cache_dir='checkpoints'
        )
        print("✓ SyntheticDataAugmentor initialized with depth-aware=True")
        print(f"  Depth estimator active: {augmentor.depth_estimator is not None}")

        # Test calculate_depth_aware_scale
        test_img = create_test_image()
        obj_dims = (100, 100)  # height, width

        scale_factor, pos, depth_value, depth_map = augmentor.calculate_depth_aware_scale(
            obj_dims, test_img
        )

        print(f"✓ Depth-aware scale calculation successful:")
        print(f"  Scale factor: {scale_factor:.2f}")
        print(f"  Position: {pos}")
        print(f"  Depth at position: {depth_value:.3f}")

        return augmentor
    except Exception as e:
        print(f"✗ Failed to test augmentor: {e}")
        import traceback
        traceback.print_exc()
        return None

def visualize_results(test_img, depth_map, zones_mask):
    """Create visualization of test results."""
    print("\n" + "="*80)
    print("TEST 5: Visualization")
    print("="*80)

    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Test Image')
        axes[0].axis('off')

        # Depth map
        im1 = axes[1].imshow(depth_map, cmap='viridis')
        axes[1].set_title('Depth Map (Depth Anything V2)')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], label='Depth (normalized)')

        # Depth zones
        im2 = axes[2].imshow(zones_mask, cmap='tab10')
        axes[2].set_title('Depth Zones (Near/Mid/Far)')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], label='Zone')

        output_path = 'test_depth_visualization.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to: {output_path}")

    except Exception as e:
        print(f"✗ Failed to create visualization: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("DEPTH-AWARE AUGMENTATION SYSTEM TEST SUITE")
    print("="*80)
    print("\nThis script tests the depth estimation and depth-aware scaling system.")
    print("Requirements: CUDA-capable GPU, PyTorch with CUDA support, Depth Anything V2")
    print("\n" + "="*80 + "\n")

    # Test 1: Initialize depth estimator
    estimator = test_depth_estimator()

    # Test 2: Estimate depth map
    test_img, depth_map = test_depth_estimation(estimator)

    # Test 3: Classify depth zones
    zones_mask, depth_ranges = test_depth_zones(estimator, depth_map)

    # Test 4: Test augmentor integration
    augmentor = test_depth_aware_augmentor()

    # Test 5: Create visualization
    if test_img is not None and depth_map is not None and zones_mask is not None:
        visualize_results(test_img, depth_map, zones_mask)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"✓ Depth Estimator: {'PASS' if estimator is not None else 'FAIL'}")
    print(f"✓ Depth Estimation: {'PASS' if depth_map is not None else 'FAIL'}")
    print(f"✓ Zone Classification: {'PASS' if zones_mask is not None else 'FAIL'}")
    print(f"✓ Augmentor Integration: {'PASS' if augmentor is not None else 'FAIL'}")
    print("="*80 + "\n")

    if all([estimator, depth_map, zones_mask, augmentor]):
        print("✓ ALL TESTS PASSED - Depth-aware system ready!")
        return 0
    else:
        print("✗ SOME TESTS FAILED - Check errors above")
        return 1

if __name__ == "__main__":
    exit(main())
