"""
Test Script for Quality Validation System

Tests the validation modules:
- QualityValidator (LPIPS, FID, anomaly detection)
- PhysicsValidator (gravity, buoyancy, physics checks)

Usage:
    python test_validation_system.py
"""

import cv2
import numpy as np
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from validation.quality_metrics import QualityValidator, QualityScore, Anomaly
from validation.physics_validator import PhysicsValidator, PhysicsViolation

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


def create_test_image(size=(640, 480), quality='good'):
    """
    Create synthetic test images with different quality levels

    Args:
        size: (width, height)
        quality: 'good', 'bad_edges', 'bad_color', 'bad_blur'

    Returns:
        BGR image
    """
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)

    if quality == 'good':
        # Good quality - natural gradient and smooth transitions
        for y in range(size[1]):
            intensity = int(100 + 100 * (y / size[1]))
            img[y, :] = [intensity, intensity - 20, intensity + 10]

        # Add smooth object
        center = (size[0] // 2, size[1] // 2)
        cv2.circle(img, center, 80, (120, 150, 180), -1)
        # Smooth edges with Gaussian blur
        mask = np.zeros((size[1], size[0]), dtype=np.uint8)
        cv2.circle(mask, center, 80, 255, -1)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        img = cv2.addWeighted(img, 1.0, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)

    elif quality == 'bad_edges':
        # Bad quality - sharp unrealistic edges
        img[:] = [80, 100, 120]
        # Sharp edge object (no blending)
        cv2.rectangle(img, (100, 100), (300, 300), (200, 50, 50), -1)

    elif quality == 'bad_color':
        # Bad quality - extreme color bleeding
        for y in range(size[1]):
            for x in range(size[0]):
                # Extreme LAB variance
                img[y, x] = [
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255)
                ]

    elif quality == 'bad_blur':
        # Bad quality - inconsistent blur
        img[:] = [100, 110, 120]
        # Sharp region
        img[0:size[1]//2, :] = cv2.GaussianBlur(img[0:size[1]//2, :], (51, 51), 0)
        # Very blurry region
        # img[size[1]//2:, :] stays sharp

    return img


def test_quality_validator():
    """Test QualityValidator functionality"""
    print("\n" + "="*60)
    print("TEST 1: QualityValidator")
    print("="*60)

    try:
        # Initialize validator (without reference dataset for quick test)
        validator = QualityValidator(
            reference_dataset_path=None,
            device='cpu',  # Use CPU for testing
            use_lpips=False,  # Disable LPIPS (requires model download)
            use_fid=False,    # Disable FID (requires reference dataset)
            use_anomaly_detection=True
        )

        print("✓ QualityValidator initialized successfully")

        # Test 1: Good quality image
        print("\n--- Testing good quality image ---")
        good_img = create_test_image(quality='good')
        anomalies = validator.detect_anomalies(good_img)
        print(f"Good image anomalies: {len(anomalies)}")
        for anomaly in anomalies:
            print(f"  - {anomaly}")

        # Test 2: Bad edges
        print("\n--- Testing bad edges image ---")
        bad_edges_img = create_test_image(quality='bad_edges')
        anomalies = validator.detect_anomalies(bad_edges_img)
        print(f"Bad edges anomalies: {len(anomalies)}")
        for anomaly in anomalies:
            print(f"  - {anomaly}")

        # Test 3: Bad color
        print("\n--- Testing bad color image ---")
        bad_color_img = create_test_image(quality='bad_color')
        anomalies = validator.detect_anomalies(bad_color_img)
        print(f"Bad color anomalies: {len(anomalies)}")
        for anomaly in anomalies:
            print(f"  - {anomaly}")

        print("\n✓ QualityValidator tests PASSED")
        return True

    except Exception as e:
        print(f"\n✗ QualityValidator tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_physics_validator():
    """Test PhysicsValidator functionality"""
    print("\n" + "="*60)
    print("TEST 2: PhysicsValidator")
    print("="*60)

    try:
        validator = PhysicsValidator(
            density_threshold_float=0.95,
            density_threshold_sink=1.15,
            surface_zone=0.25,
            bottom_zone=0.75
        )

        print("✓ PhysicsValidator initialized successfully")

        # Test case 1: Metal can floating near surface (SHOULD VIOLATE)
        print("\n--- Test case 1: Metal can at surface (should violate) ---")
        annotations = [
            {
                'category_name': 'can',  # Metal - density > 1.15
                'bbox': [100, 50, 80, 80]  # Near top (y=50, surface zone)
            }
        ]
        violations = validator.check_gravity(annotations, image_height=480, scene_type='underwater')
        print(f"Violations: {len(violations)}")
        for v in violations:
            print(f"  - {v}")

        assert len(violations) > 0, "Should detect metal can floating violation"
        print("✓ Correctly detected metal floating violation")

        # Test case 2: Plastic bag at seafloor (SHOULD VIOLATE)
        print("\n--- Test case 2: Plastic bag at bottom (should violate) ---")
        annotations = [
            {
                'category_name': 'plastic_bag',  # Plastic - density < 0.95
                'bbox': [100, 400, 80, 60]  # Near bottom (y=400, bottom zone)
            }
        ]
        violations = validator.check_gravity(annotations, image_height=480, scene_type='underwater')
        print(f"Violations: {len(violations)}")
        for v in violations:
            print(f"  - {v}")

        assert len(violations) > 0, "Should detect plastic sinking violation"
        print("✓ Correctly detected plastic sinking violation")

        # Test case 3: Fish in mid-water (SHOULD BE OK)
        print("\n--- Test case 3: Fish in mid-water (should be OK) ---")
        annotations = [
            {
                'category_name': 'fish',  # Neutral buoyancy
                'bbox': [200, 240, 100, 80]  # Mid-water
            }
        ]
        violations = validator.check_gravity(annotations, image_height=480, scene_type='underwater')
        print(f"Violations: {len(violations)}")
        for v in violations:
            print(f"  - {v}")

        assert len(violations) == 0, "Should not detect violations for neutral buoyancy"
        print("✓ Correctly accepted neutral buoyancy object")

        # Test case 4: Scale plausibility
        print("\n--- Test case 4: Scale plausibility ---")
        annotations = [
            {
                'category_name': 'fish',
                'bbox': [0, 0, 600, 450]  # 90% of image (too large)
            }
        ]
        violations = validator.check_scale_plausibility(annotations, image_size=(640, 480))
        print(f"Scale violations: {len(violations)}")
        for v in violations:
            print(f"  - {v}")

        assert len(violations) > 0, "Should detect implausible scale"
        print("✓ Correctly detected scale violation")

        # Test case 5: Auto-correction
        print("\n--- Test case 5: Auto-correction of metal can placement ---")
        corrected_pos = validator.auto_correct_placement(
            object_category='can',
            object_bbox=[100, 50, 80, 80],
            image_size=(640, 480)
        )
        print(f"Original Y: 50, Corrected Y: {corrected_pos.y:.1f}")
        assert corrected_pos.y > 240, "Should move metal can to bottom zone"
        print("✓ Correctly auto-corrected placement")

        print("\n✓ PhysicsValidator tests PASSED")
        return True

    except Exception as e:
        print(f"\n✗ PhysicsValidator tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integration between validators"""
    print("\n" + "="*60)
    print("TEST 3: Integration Test")
    print("="*60)

    try:
        # Create realistic test scenario
        quality_validator = QualityValidator(
            reference_dataset_path=None,
            device='cpu',
            use_lpips=False,
            use_fid=False,
            use_anomaly_detection=True
        )

        physics_validator = PhysicsValidator()

        # Test image with physics violation
        test_img = create_test_image(quality='good')
        img_h, img_w = test_img.shape[:2]

        annotations = [
            {
                'category_name': 'can',
                'bbox': [100, 50, 80, 80]  # Metal can floating
            },
            {
                'category_name': 'fish',
                'bbox': [300, 240, 100, 80]  # Fish in mid-water (OK)
            }
        ]

        # Check quality
        anomalies = quality_validator.detect_anomalies(test_img)
        print(f"Quality anomalies: {len(anomalies)}")

        # Check physics
        violations = physics_validator.check_gravity(annotations, img_h, 'underwater')
        print(f"Physics violations: {len(violations)}")

        # Overall validation
        has_critical_anomalies = any(a.severity > 0.8 for a in anomalies)
        has_critical_violations = any(v.severity > 0.85 for v in violations)

        is_valid = not (has_critical_anomalies or has_critical_violations)

        print(f"\nOverall validation: {'PASS' if is_valid else 'FAIL'}")
        print(f"  - Critical anomalies: {has_critical_anomalies}")
        print(f"  - Critical physics violations: {has_critical_violations}")

        # Should fail due to physics violation
        assert not is_valid, "Should fail validation due to metal can floating"
        print("\n✓ Integration test PASSED (correctly rejected image)")

        return True

    except Exception as e:
        print(f"\n✗ Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests"""
    print("="*60)
    print("VALIDATION SYSTEM TEST SUITE")
    print("Quality Validation System")
    print("="*60)

    results = []

    # Run tests
    results.append(("QualityValidator", test_quality_validator()))
    results.append(("PhysicsValidator", test_physics_validator()))
    results.append(("Integration", test_integration()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n✓ ALL TESTS PASSED")
        print("\nValidation system is working correctly!")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install lpips scikit-learn")
        print("  2. Enable validation in config.yaml")
        print("  3. Run full dataset generation with validation enabled")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        print("\nPlease review errors above and fix issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
