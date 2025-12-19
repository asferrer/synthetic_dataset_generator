"""
Test Script for Advanced Lighting System

Tests the advanced lighting estimation system:
- Multi-light source detection
- HDR environment map estimation
- Multi-source shadow generation
- Underwater light attenuation

Usage:
    python test_lighting_system.py
"""

import cv2
import numpy as np
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from augmentation.lighting_engine import (
    AdvancedLightingEstimator,
    generate_multi_source_shadows,
    apply_underwater_light_attenuation,
    LightSource
)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


def create_test_underwater_image(size=(640, 480)):
    """Create a synthetic underwater scene for testing"""
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)

    # Blue gradient (simulating underwater)
    for y in range(size[1]):
        depth_factor = y / size[1]
        blue = int(80 + 100 * (1 - depth_factor))
        green = int(60 + 80 * (1 - depth_factor))
        red = int(20 + 40 * (1 - depth_factor))
        img[y, :] = [blue, green, red]

    # Add bright region (sunlight from above)
    cv2.circle(img, (size[0] // 2, 100), 80, (180, 160, 140), -1)
    cv2.circle(img, (size[0] // 2, 100), 80, (200, 180, 160), 30)

    # Add some ambient light variation
    noise = np.random.randint(-10, 10, (size[1], size[0], 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def test_lighting_estimator():
    """Test AdvancedLightingEstimator"""
    print("\n" + "="*60)
    print("TEST 1: Advanced Lighting Estimator")
    print("="*60)

    try:
        # Create estimator
        estimator = AdvancedLightingEstimator(
            max_light_sources=3,
            intensity_threshold=0.5,
            use_hdr_estimation=False
        )
        print("✓ Lighting estimator initialized")

        # Create test image
        test_img = create_test_underwater_image()
        print("✓ Test underwater image created")

        # Estimate lighting
        lighting_map = estimator.estimate_lighting(test_img)
        print(f"✓ Lighting estimated successfully")
        print(f"  - Detected {len(lighting_map.light_sources)} light sources")
        print(f"  - Dominant direction: {lighting_map.dominant_direction}")
        print(f"  - Color temperature: {lighting_map.color_temperature:.0f}K")
        print(f"  - Ambient intensity: {lighting_map.ambient_intensity:.2f}")

        # Check light sources
        assert len(lighting_map.light_sources) > 0, "Should detect at least one light source"

        for i, light in enumerate(lighting_map.light_sources):
            print(f"\n  Light {i+1}: {light}")
            assert 0 <= light.intensity <= 1, "Light intensity should be in range [0, 1]"
            assert len(light.color) == 3, "Light color should have 3 channels"

        print("\n✓ Lighting estimator tests PASSED")
        return True

    except Exception as e:
        print(f"\n✗ Lighting estimator tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_source_shadows():
    """Test multi-source shadow generation"""
    print("\n" + "="*60)
    print("TEST 2: Multi-Source Shadow Generation")
    print("="*60)

    try:
        # Create test setup
        image_size = (480, 640)  # h, w

        # Create object mask (circular object in center)
        object_mask = np.zeros(image_size, dtype=np.uint8)
        cv2.circle(object_mask, (320, 240), 80, 255, -1)

        # Object bounding box
        object_bbox = (240, 160, 160, 160)  # x, y, w, h

        # Create test light sources
        light_sources = [
            LightSource(
                position=(np.pi/4, np.pi/3),  # 45° azimuth, 60° elevation
                intensity=0.8,
                color=[255, 255, 240],
                light_type='directional',
                angular_size=0.0
            ),
            LightSource(
                position=(-np.pi/6, np.pi/4),  # -30° azimuth, 45° elevation
                intensity=0.5,
                color=[240, 250, 255],
                light_type='area',
                angular_size=0.05
            )
        ]

        print(f"✓ Test setup created: {len(light_sources)} light sources")

        # Generate shadows
        shadow_mask = generate_multi_source_shadows(
            object_mask=object_mask,
            object_bbox=object_bbox,
            light_sources=light_sources,
            image_size=image_size,
            object_height_estimate=0.2,
            max_shadow_intensity=0.7
        )

        print("✓ Multi-source shadows generated")
        assert shadow_mask.shape == image_size, "Shadow mask shape mismatch"
        assert shadow_mask.dtype == np.float32, "Shadow mask should be float32"
        assert np.min(shadow_mask) >= 0 and np.max(shadow_mask) <= 1, "Shadow values out of range"

        # Check that shadows exist
        shadow_pixels = np.sum(shadow_mask > 0.1)
        print(f"  - Shadow coverage: {shadow_pixels} pixels ({100*shadow_pixels/np.prod(image_size):.1f}%)")
        assert shadow_pixels > 100, "Shadows should be visible"

        # Check that object area has no shadow
        object_pixels = np.sum(object_mask > 0)
        shadow_under_object = np.sum(shadow_mask[object_mask > 0] > 0.1)
        print(f"  - Shadow under object: {shadow_under_object}/{object_pixels} pixels")
        assert shadow_under_object < object_pixels * 0.1, "Object should not be shadowed"

        print("\n✓ Multi-source shadow tests PASSED")
        return True

    except Exception as e:
        print(f"\n✗ Multi-source shadow tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_underwater_attenuation():
    """Test underwater light attenuation"""
    print("\n" + "="*60)
    print("TEST 3: Underwater Light Attenuation")
    print("="*60)

    try:
        # Create test light sources
        light_sources = [
            LightSource(
                position=(0, np.pi/3),
                intensity=1.0,
                color=[255, 200, 150],  # Warm light
                light_type='directional',
                angular_size=0.0
            )
        ]

        print("✓ Original light source created")
        print(f"  - Intensity: {light_sources[0].intensity}")
        print(f"  - Color: {light_sources[0].color}")

        # Test shallow water
        shallow_lights = apply_underwater_light_attenuation(
            light_sources,
            depth_category='shallow',
            water_clarity='clear'
        )

        print("\n  Shallow water (clear):")
        print(f"  - Intensity: {shallow_lights[0].intensity:.2f}")
        print(f"  - Color: {shallow_lights[0].color}")
        assert shallow_lights[0].intensity < light_sources[0].intensity, "Should attenuate"
        assert shallow_lights[0].intensity > 0.5, "Shallow water should preserve intensity"

        # Test deep water
        deep_lights = apply_underwater_light_attenuation(
            light_sources,
            depth_category='deep',
            water_clarity='clear'
        )

        print("\n  Deep water (clear):")
        print(f"  - Intensity: {deep_lights[0].intensity:.2f}")
        print(f"  - Color: {deep_lights[0].color}")
        assert deep_lights[0].intensity < shallow_lights[0].intensity, "Deep should attenuate more"

        # Check blue shift
        assert deep_lights[0].color[0] > light_sources[0].color[0], "Should increase blue"
        print("  - ✓ Blue shift detected")

        # Test murky water
        murky_lights = apply_underwater_light_attenuation(
            light_sources,
            depth_category='mid',
            water_clarity='murky'
        )

        print("\n  Mid water (murky):")
        print(f"  - Intensity: {murky_lights[0].intensity:.2f}")
        assert murky_lights[0].intensity < 0.6, "Murky water should attenuate significantly"

        print("\n✓ Underwater attenuation tests PASSED")
        return True

    except Exception as e:
        print(f"\n✗ Underwater attenuation tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test full lighting pipeline"""
    print("\n" + "="*60)
    print("TEST 4: Integration Test")
    print("="*60)

    try:
        # Create estimator
        estimator = AdvancedLightingEstimator(max_light_sources=3)

        # Create test scene
        background = create_test_underwater_image()

        # Estimate lighting
        lighting_map = estimator.estimate_lighting(background)
        print(f"✓ Detected {len(lighting_map.light_sources)} light sources")

        # Apply underwater attenuation
        adjusted_lights = apply_underwater_light_attenuation(
            lighting_map.light_sources,
            depth_category='mid',
            water_clarity='clear'
        )
        print(f"✓ Applied underwater attenuation")

        # Create object mask
        object_mask = np.zeros(background.shape[:2], dtype=np.uint8)
        cv2.circle(object_mask, (320, 300), 60, 255, -1)

        # Generate shadows
        shadow_mask = generate_multi_source_shadows(
            object_mask=object_mask,
            object_bbox=(260, 240, 120, 120),
            light_sources=adjusted_lights,
            image_size=(background.shape[0], background.shape[1]),
            object_height_estimate=0.15,
            max_shadow_intensity=0.6
        )
        print(f"✓ Generated multi-source shadows")

        # Apply shadows to background
        shadow_3ch = np.stack([shadow_mask] * 3, axis=-1)
        result = background.astype(np.float32) * (1.0 - shadow_3ch)
        result = np.clip(result, 0, 255).astype(np.uint8)

        print("✓ Applied shadows to background")

        # Verify result
        assert result.shape == background.shape, "Result shape mismatch"
        assert np.min(result) >= 0 and np.max(result) <= 255, "Result values out of range"

        # Check that shadows made the image darker
        avg_brightness_before = np.mean(background)
        avg_brightness_after = np.mean(result)
        print(f"\n  Average brightness: {avg_brightness_before:.1f} → {avg_brightness_after:.1f}")
        assert avg_brightness_after < avg_brightness_before, "Shadows should darken image"

        print("\n✓ Integration test PASSED")
        return True

    except Exception as e:
        print(f"\n✗ Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all lighting system tests"""
    print("="*60)
    print("ADVANCED LIGHTING SYSTEM TEST SUITE")
    print("Multi-Light Source Estimation & Shadow Generation")
    print("="*60)

    results = []

    # Run tests
    results.append(("Lighting Estimator", test_lighting_estimator()))
    results.append(("Multi-Source Shadows", test_multi_source_shadows()))
    results.append(("Underwater Attenuation", test_underwater_attenuation()))
    results.append(("Integration", test_integration()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:25s}: {status}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n✓ ALL TESTS PASSED")
        print("\nAdvanced lighting system is working correctly!")
        print("\nNext steps:")
        print("  1. Enable advanced_lighting in config.yaml")
        print("  2. Run dataset generation with multi-source shadows")
        print("  3. Compare shadow quality vs baseline")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        print("\nPlease review errors above and fix issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
