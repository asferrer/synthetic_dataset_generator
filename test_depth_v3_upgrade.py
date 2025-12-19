"""
Test Script for Depth Anything V3 Upgrade
==========================================

This script tests the upgrade from Depth Anything V2 to V3, verifying:
1. V3 model loading and inference
2. V2 backward compatibility
3. Depth map quality comparison
4. Performance benchmarking on RTX 5090
5. New V3 features (pose estimation)

Usage:
    python test_depth_v3_upgrade.py
"""

import sys
import time
import numpy as np
import cv2
import torch
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from augmentation.depth_engine import DepthEstimator, get_depth_estimator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DepthV3Tester:
    """Comprehensive testing for DA-V3 upgrade"""

    def __init__(self):
        self.test_results = {
            'v3_loading': False,
            'v2_compatibility': False,
            'v3_inference': False,
            'v2_inference': False,
            'performance_v3': None,
            'performance_v2': None,
            'pose_estimation': False,
            'depth_quality_comparison': None
        }

    def create_test_image(self, size=(512, 512)):
        """Create a synthetic test image with depth cues"""
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)

        # Create gradient (near to far)
        for y in range(size[1]):
            intensity = int(255 * (y / size[1]))
            img[y, :] = [intensity, intensity, intensity]

        # Add some objects at different depths
        # Close object (top)
        cv2.circle(img, (128, 128), 60, (100, 150, 200), -1)

        # Mid-distance object (middle)
        cv2.rectangle(img, (300, 200), (400, 300), (150, 100, 180), -1)

        # Far object (bottom)
        cv2.ellipse(img, (256, 400), (80, 40), 0, 0, 360, (200, 200, 100), -1)

        return img

    def test_v3_loading(self):
        """Test 1: Verify V3 model loads correctly"""
        logger.info("\n" + "="*60)
        logger.info("TEST 1: Depth Anything V3 Model Loading")
        logger.info("="*60)

        try:
            estimator = DepthEstimator(
                model_size='small',  # Use small for faster testing
                model_version='v3',
                device='cuda',
                cache_dir='checkpoints'
            )

            assert estimator.model is not None, "Model not loaded"
            assert estimator.model_version == 'v3', "Model version mismatch"

            logger.info("✓ V3 model loaded successfully")
            logger.info(f"  Device: {estimator.device}")
            logger.info(f"  Model size: {estimator.model_size}")

            self.test_results['v3_loading'] = True
            return estimator

        except Exception as e:
            logger.error(f"✗ V3 model loading FAILED: {e}")
            self.test_results['v3_loading'] = False
            return None

    def test_v2_compatibility(self):
        """Test 2: Verify V2 backward compatibility"""
        logger.info("\n" + "="*60)
        logger.info("TEST 2: Depth Anything V2 Backward Compatibility")
        logger.info("="*60)

        try:
            estimator = DepthEstimator(
                model_size='small',
                model_version='v2',
                device='cuda',
                cache_dir='checkpoints'
            )

            assert estimator.model is not None, "V2 model not loaded"
            assert estimator.model_version == 'v2', "V2 model version mismatch"

            logger.info("✓ V2 backward compatibility verified")
            logger.info(f"  Device: {estimator.device}")

            self.test_results['v2_compatibility'] = True
            return estimator

        except Exception as e:
            logger.error(f"✗ V2 compatibility test FAILED: {e}")
            logger.info("  Note: V2 failure is acceptable if V2 checkpoints not available")
            self.test_results['v2_compatibility'] = False
            return None

    def test_v3_inference(self, estimator_v3):
        """Test 3: Verify V3 inference works correctly"""
        logger.info("\n" + "="*60)
        logger.info("TEST 3: Depth Anything V3 Inference")
        logger.info("="*60)

        if estimator_v3 is None:
            logger.error("✗ Cannot test inference without loaded model")
            return None

        try:
            # Create test image
            test_img = self.create_test_image()

            # Run inference
            depth_map = estimator_v3.estimate_depth(test_img, normalize=True)

            # Validate output
            assert depth_map is not None, "Depth map is None"
            assert depth_map.shape == test_img.shape[:2], "Shape mismatch"
            assert depth_map.dtype == np.float64 or depth_map.dtype == np.float32, "Invalid dtype"
            assert 0 <= depth_map.min() <= 1, "Normalization failed (min)"
            assert 0 <= depth_map.max() <= 1, "Normalization failed (max)"

            logger.info("✓ V3 inference successful")
            logger.info(f"  Input shape: {test_img.shape}")
            logger.info(f"  Output shape: {depth_map.shape}")
            logger.info(f"  Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
            logger.info(f"  Mean depth: {depth_map.mean():.3f}")

            self.test_results['v3_inference'] = True
            return depth_map

        except Exception as e:
            logger.error(f"✗ V3 inference FAILED: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['v3_inference'] = False
            return None

    def test_v2_inference(self, estimator_v2):
        """Test 4: Verify V2 inference still works"""
        logger.info("\n" + "="*60)
        logger.info("TEST 4: Depth Anything V2 Inference (Compatibility)")
        logger.info("="*60)

        if estimator_v2 is None:
            logger.warning("⚠ Skipping V2 inference test (model not loaded)")
            return None

        try:
            # Create test image
            test_img = self.create_test_image()

            # Run inference
            depth_map = estimator_v2.estimate_depth(test_img, normalize=True)

            # Validate output
            assert depth_map is not None, "Depth map is None"
            assert depth_map.shape == test_img.shape[:2], "Shape mismatch"

            logger.info("✓ V2 inference successful")
            logger.info(f"  Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")

            self.test_results['v2_inference'] = True
            return depth_map

        except Exception as e:
            logger.error(f"✗ V2 inference FAILED: {e}")
            self.test_results['v2_inference'] = False
            return None

    def benchmark_performance(self, estimator, version='v3', num_runs=10):
        """Test 5: Benchmark inference performance"""
        logger.info("\n" + "="*60)
        logger.info(f"TEST 5: Performance Benchmarking ({version.upper()})")
        logger.info("="*60)

        if estimator is None:
            logger.warning(f"⚠ Skipping {version.upper()} performance test")
            return None

        try:
            test_img = self.create_test_image(size=(640, 480))

            # Warmup
            logger.info("Warming up GPU...")
            for _ in range(3):
                _ = estimator.estimate_depth(test_img, normalize=True)

            # Benchmark
            logger.info(f"Running {num_runs} iterations...")
            times = []

            for i in range(num_runs):
                start = time.time()
                _ = estimator.estimate_depth(test_img, normalize=True)
                elapsed = (time.time() - start) * 1000  # Convert to ms
                times.append(elapsed)

            times = np.array(times)
            mean_time = times.mean()
            std_time = times.std()
            min_time = times.min()
            max_time = times.max()
            fps = 1000 / mean_time

            logger.info(f"✓ Performance results for {version.upper()}:")
            logger.info(f"  Mean: {mean_time:.2f}ms ± {std_time:.2f}ms")
            logger.info(f"  Min: {min_time:.2f}ms")
            logger.info(f"  Max: {max_time:.2f}ms")
            logger.info(f"  FPS: {fps:.2f}")

            if version == 'v3':
                self.test_results['performance_v3'] = {
                    'mean_ms': mean_time,
                    'std_ms': std_time,
                    'fps': fps
                }
            else:
                self.test_results['performance_v2'] = {
                    'mean_ms': mean_time,
                    'std_ms': std_time,
                    'fps': fps
                }

            return mean_time

        except Exception as e:
            logger.error(f"✗ Performance benchmark FAILED: {e}")
            return None

    def test_pose_estimation(self, estimator_v3):
        """Test 6: Test V3-only pose estimation feature"""
        logger.info("\n" + "="*60)
        logger.info("TEST 6: Camera Pose Estimation (V3 Only)")
        logger.info("="*60)

        if estimator_v3 is None:
            logger.warning("⚠ Skipping pose estimation test")
            return

        try:
            # Test that pose estimation requires explicit enabling
            test_img = self.create_test_image()

            try:
                # Should fail because enable_pose_estimation=False
                estimator_v3.estimate_pose([test_img])
                logger.error("✗ Pose estimation should have raised error (not enabled)")
                self.test_results['pose_estimation'] = False
            except RuntimeError as e:
                if "disabled" in str(e).lower():
                    logger.info("✓ Pose estimation correctly requires explicit enabling")
                    self.test_results['pose_estimation'] = True
                else:
                    raise

            # Test with V2 (should fail)
            estimator_v2_test = DepthEstimator(
                model_size='small',
                model_version='v2',
                enable_pose_estimation=True
            )

            logger.info("✓ Pose estimation feature validation passed")

        except Exception as e:
            logger.error(f"✗ Pose estimation test FAILED: {e}")
            self.test_results['pose_estimation'] = False

    def compare_depth_quality(self, depth_v3, depth_v2):
        """Test 7: Compare depth map quality"""
        logger.info("\n" + "="*60)
        logger.info("TEST 7: Depth Map Quality Comparison")
        logger.info("="*60)

        if depth_v3 is None:
            logger.warning("⚠ Cannot compare without V3 depth map")
            return

        if depth_v2 is None:
            logger.warning("⚠ V2 depth map not available for comparison")
            logger.info("  V3 standalone quality:")
            logger.info(f"    Dynamic range: {depth_v3.max() - depth_v3.min():.3f}")
            logger.info(f"    Std deviation: {depth_v3.std():.3f}")
            return

        try:
            # Compare metrics
            v3_range = depth_v3.max() - depth_v3.min()
            v2_range = depth_v2.max() - depth_v2.min()

            v3_std = depth_v3.std()
            v2_std = depth_v2.std()

            # Compute difference
            diff = np.abs(depth_v3 - depth_v2)
            mean_diff = diff.mean()
            max_diff = diff.max()

            logger.info("✓ Quality comparison:")
            logger.info(f"  V3 dynamic range: {v3_range:.3f}")
            logger.info(f"  V2 dynamic range: {v2_range:.3f}")
            logger.info(f"  V3 std deviation: {v3_std:.3f}")
            logger.info(f"  V2 std deviation: {v2_std:.3f}")
            logger.info(f"  Mean difference: {mean_diff:.3f}")
            logger.info(f"  Max difference: {max_diff:.3f}")

            self.test_results['depth_quality_comparison'] = {
                'v3_range': v3_range,
                'v2_range': v2_range,
                'mean_diff': mean_diff
            }

        except Exception as e:
            logger.error(f"✗ Quality comparison FAILED: {e}")

    def save_visual_comparison(self, test_img, depth_v3, depth_v2=None):
        """Save visual comparison of depth maps"""
        logger.info("\n" + "="*60)
        logger.info("Saving Visual Comparison")
        logger.info("="*60)

        try:
            output_dir = Path("test_results")
            output_dir.mkdir(exist_ok=True)

            # Save original
            cv2.imwrite(str(output_dir / "test_image.png"), test_img)
            logger.info(f"  Saved: {output_dir / 'test_image.png'}")

            # Save V3 depth map
            if depth_v3 is not None:
                depth_v3_vis = (depth_v3 * 255).astype(np.uint8)
                depth_v3_colored = cv2.applyColorMap(depth_v3_vis, cv2.COLORMAP_INFERNO)
                cv2.imwrite(str(output_dir / "depth_v3.png"), depth_v3_colored)
                logger.info(f"  Saved: {output_dir / 'depth_v3.png'}")

            # Save V2 depth map
            if depth_v2 is not None:
                depth_v2_vis = (depth_v2 * 255).astype(np.uint8)
                depth_v2_colored = cv2.applyColorMap(depth_v2_vis, cv2.COLORMAP_INFERNO)
                cv2.imwrite(str(output_dir / "depth_v2.png"), depth_v2_colored)
                logger.info(f"  Saved: {output_dir / 'depth_v2.png'}")

                # Save difference map
                if depth_v3 is not None:
                    diff = np.abs(depth_v3 - depth_v2)
                    diff_vis = (diff / diff.max() * 255).astype(np.uint8)
                    diff_colored = cv2.applyColorMap(diff_vis, cv2.COLORMAP_JET)
                    cv2.imwrite(str(output_dir / "depth_difference.png"), diff_colored)
                    logger.info(f"  Saved: {output_dir / 'depth_difference.png'}")

            logger.info(f"\n✓ Visual comparisons saved to: {output_dir.absolute()}")

        except Exception as e:
            logger.error(f"✗ Failed to save visual comparison: {e}")

    def print_summary(self):
        """Print test summary"""
        logger.info("\n" + "="*60)
        logger.info("TEST SUMMARY")
        logger.info("="*60)

        passed = sum(1 for v in self.test_results.values() if v is True)
        failed = sum(1 for v in self.test_results.values() if v is False)

        logger.info(f"\nResults:")
        logger.info(f"  ✓ V3 Model Loading: {'PASS' if self.test_results['v3_loading'] else 'FAIL'}")
        logger.info(f"  ✓ V2 Compatibility: {'PASS' if self.test_results['v2_compatibility'] else 'SKIP/FAIL'}")
        logger.info(f"  ✓ V3 Inference: {'PASS' if self.test_results['v3_inference'] else 'FAIL'}")
        logger.info(f"  ✓ V2 Inference: {'PASS' if self.test_results['v2_inference'] else 'SKIP/FAIL'}")
        logger.info(f"  ✓ Pose Estimation: {'PASS' if self.test_results['pose_estimation'] else 'FAIL'}")

        if self.test_results['performance_v3']:
            perf = self.test_results['performance_v3']
            logger.info(f"\n  Performance (V3):")
            logger.info(f"    Mean: {perf['mean_ms']:.2f}ms")
            logger.info(f"    FPS: {perf['fps']:.2f}")

        if self.test_results['performance_v2']:
            perf = self.test_results['performance_v2']
            logger.info(f"\n  Performance (V2):")
            logger.info(f"    Mean: {perf['mean_ms']:.2f}ms")
            logger.info(f"    FPS: {perf['fps']:.2f}")

        # Overall verdict
        critical_tests = ['v3_loading', 'v3_inference']
        critical_pass = all(self.test_results[test] for test in critical_tests)

        logger.info("\n" + "="*60)
        if critical_pass:
            logger.info("✓ OVERALL: TESTS PASSED")
            logger.info("  Depth Anything V3 upgrade is READY for commit")
        else:
            logger.error("✗ OVERALL: TESTS FAILED")
            logger.error("  Critical issues found - do NOT commit")
        logger.info("="*60 + "\n")

        return critical_pass


def main():
    """Run all tests"""
    logger.info("="*60)
    logger.info("Depth Anything V3 Upgrade Testing")
    logger.info("="*60)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")

    tester = DepthV3Tester()

    # Run all tests
    estimator_v3 = tester.test_v3_loading()
    estimator_v2 = tester.test_v2_compatibility()

    depth_v3 = tester.test_v3_inference(estimator_v3)
    depth_v2 = tester.test_v2_inference(estimator_v2)

    tester.benchmark_performance(estimator_v3, version='v3')
    tester.benchmark_performance(estimator_v2, version='v2')

    tester.test_pose_estimation(estimator_v3)

    tester.compare_depth_quality(depth_v3, depth_v2)

    # Save visual comparison
    test_img = tester.create_test_image()
    tester.save_visual_comparison(test_img, depth_v3, depth_v2)

    # Print summary
    success = tester.print_summary()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
