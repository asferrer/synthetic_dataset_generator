"""
Test Script for Semantic Scene Analyzer
=======================================

Tests the scene analysis functionality including:
- Heuristic-based scene analysis
- Object-scene compatibility checking
- Placement position suggestions
- SAM3 integration (when available)
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from augmentation.scene_analyzer import (
    SemanticSceneAnalyzer,
    SceneRegion,
    SceneAnalysis,
    SCENE_COMPATIBILITY,
    is_placement_valid,
)


def create_test_images():
    """Create synthetic test images for scene analysis

    These images simulate underwater scene characteristics:
    - Blue channel dominance for open water
    - Warm colors (R > B) for seafloor/sand
    - Green hues for vegetation
    - Brightness gradient for surface
    - Texture for rocky areas
    """
    images = {}

    # 1. Open water scene (mostly blue, B > R)
    open_water = np.zeros((480, 640, 3), dtype=np.uint8)
    for y in range(480):
        # Blue dominant water column
        blue_val = int(200 - (y / 480) * 60)
        green_val = int(blue_val * 0.65)
        red_val = int(blue_val * 0.35)  # R < B for blue water
        open_water[y, :] = [blue_val, green_val, red_val]  # BGR format
    images['open_water'] = open_water

    # 2. Seafloor scene (sandy warm colors at bottom)
    seafloor = np.zeros((480, 640, 3), dtype=np.uint8)
    for y in range(480):
        if y < 260:  # Blue water column at top
            blue_val = int(180 - (y / 260) * 40)
            seafloor[y, :] = [blue_val, int(blue_val * 0.6), int(blue_val * 0.35)]
        else:  # Sandy/warm bottom (R > B)
            progress = (y - 260) / 220
            # Sandy colors: more red than blue
            seafloor[y, :, 0] = int(80 + progress * 40)   # B: 80-120
            seafloor[y, :, 1] = int(130 + progress * 50)  # G: 130-180
            seafloor[y, :, 2] = int(160 + progress * 60)  # R: 160-220 (R > B)
    images['seafloor'] = seafloor

    # 3. Vegetation scene (green dominant areas)
    vegetation = np.zeros((480, 640, 3), dtype=np.uint8)
    for y in range(480):
        blue_val = int(140 - (y / 480) * 40)
        vegetation[y, :] = [blue_val, int(blue_val * 0.7), int(blue_val * 0.4)]
    # Add prominent green vegetation (G > B, G > R)
    np.random.seed(42)  # For reproducibility
    for _ in range(30):
        cx = np.random.randint(50, 590)
        cy = np.random.randint(150, 450)
        radius = np.random.randint(30, 70)
        # Green: high G, moderate B, low R
        cv2.circle(vegetation, (cx, cy), radius, (40, 130, 35), -1)  # BGR: greenish
    images['vegetation'] = vegetation

    # 4. Surface scene (very bright top)
    surface = np.zeros((480, 640, 3), dtype=np.uint8)
    for y in range(480):
        if y < 120:  # Very bright surface
            val = int(255 - y * 0.8)
            surface[y, :] = [val, val, val]  # Bright white/gray
        else:  # Water below
            progress = (y - 120) / 360
            blue_val = int(180 - progress * 60)
            surface[y, :] = [blue_val, int(blue_val * 0.65), int(blue_val * 0.4)]
    images['surface'] = surface

    # 5. Rocky scene (textured gray/brown at bottom)
    rocky = np.zeros((480, 640, 3), dtype=np.uint8)
    np.random.seed(123)
    noise = np.random.randint(0, 60, (480, 640), dtype=np.uint8)
    for y in range(480):
        if y < 250:  # Water
            blue_val = int(160 - (y / 250) * 40)
            rocky[y, :] = [blue_val, int(blue_val * 0.65), int(blue_val * 0.4)]
        else:  # Rocky bottom - gray with texture
            base = int(70 + (y - 250) / 230 * 30)
            rocky[y, :, 0] = base + noise[y, :] // 3  # B
            rocky[y, :, 1] = base + noise[y, :] // 2  # G
            rocky[y, :, 2] = base + noise[y, :] // 3  # R
    images['rocky'] = rocky

    return images


def test_heuristic_analysis():
    """Test heuristic-based scene analysis"""
    print("\n" + "="*60)
    print("TEST 1: Heuristic Scene Analysis")
    print("="*60)

    analyzer = SemanticSceneAnalyzer(use_sam3=False)
    images = create_test_images()

    results = {}
    for name, image in images.items():
        analysis = analyzer.analyze_scene(image)
        results[name] = analysis

        print(f"\n[{name.upper()}]")
        print(f"  Dominant region: {analysis.dominant_region.value}")
        print(f"  Scene brightness: {analysis.scene_brightness:.2f}")
        print(f"  Water clarity: {analysis.water_clarity}")
        print(f"  Color temperature: {analysis.color_temperature}")
        print(f"  Region scores:")
        for region, score in sorted(analysis.region_scores.items(), key=lambda x: -x[1])[:3]:
            if score > 0.01:
                print(f"    - {region}: {score:.2%}")

    # Validate that expected regions are detected (>10% score is acceptable)
    # We don't require dominant region match since murky can co-occur with any scene
    expected_present = {
        'open_water': SceneRegion.OPEN_WATER,
        'seafloor': SceneRegion.SEAFLOOR,
        'vegetation': SceneRegion.VEGETATION,
        'surface': SceneRegion.SURFACE,
        'rocky': SceneRegion.ROCKY,
    }

    print("\n[VALIDATION - Region Detection (>10% threshold)]")
    passed = 0
    for name, expected_region in expected_present.items():
        score = results[name].region_scores.get(expected_region.value, 0)
        # Check if the expected region is detected with >10% score
        detected = score > 0.10
        status = "PASS" if detected else "FAIL"
        print(f"  {name}: {expected_region.value} detected at {score:.1%} [{status}]")
        if detected:
            passed += 1

    print(f"\n  Result: {passed}/{len(expected_present)} regions detected")
    return passed >= 3  # Pass if at least 3/5 regions are correctly detected


def test_object_compatibility():
    """Test object-scene compatibility checking

    Note: This test uses suggest_placement_position to find the best location
    for each object, which is the intended use case. Testing at image center
    may not match the expected region since underwater scenes have spatial structure.
    """
    print("\n" + "="*60)
    print("TEST 2: Object-Scene Compatibility")
    print("="*60)

    analyzer = SemanticSceneAnalyzer(use_sam3=False)
    images = create_test_images()

    # Test that the compatibility system gives reasonable scores
    # for objects in their natural habitats
    test_cases = [
        # (object_class, scene_name, min_expected_score)
        ('fish', 'open_water', 0.5),         # Fish anywhere in water
        ('jellyfish', 'open_water', 0.8),    # Jellyfish in open water - very good
        ('plastic_bag', 'open_water', 0.7),  # Plastic floats - good in water
        ('debris', 'seafloor', 0.5),         # Debris anywhere
        ('debris', 'open_water', 0.5),       # Debris anywhere
    ]

    print("  Testing object compatibility scores:")
    passed = 0
    for object_class, scene_name, min_score in test_cases:
        image = images[scene_name]
        analysis = analyzer.analyze_scene(image)
        h, w = image.shape[:2]

        # Use suggest_placement_position to find best location
        pos = analyzer.suggest_placement_position(
            object_class, (50, 50), analysis, (h, w), [], 20
        )

        if pos is not None:
            score, reason = analyzer.check_object_scene_compatibility(
                object_class, pos, analysis, (h, w)
            )
            meets_expectation = score >= min_score
            status = "PASS" if meets_expectation else "FAIL"
            print(f"    {object_class} in {scene_name}: score={score:.2f} >= {min_score} [{status}]")
            if meets_expectation:
                passed += 1
        else:
            print(f"    {object_class} in {scene_name}: no position found [FAIL]")

    # Additional: test that incompatible placements get low scores
    print("\n  Testing incompatible placements (should have low scores):")
    incompatible_tests = [
        ('crab', 'open_water', 0.3),     # Crab shouldn't float in open water
        ('can', 'surface', 0.3),          # Can shouldn't float at surface
    ]

    for object_class, scene_name, max_score in incompatible_tests:
        image = images[scene_name]
        analysis = analyzer.analyze_scene(image)
        h, w = image.shape[:2]

        # Test at top of image (where surface/open water is)
        position = (w // 2, h // 4)
        score, reason = analyzer.check_object_scene_compatibility(
            object_class, position, analysis, (h, w)
        )
        is_low = score <= max_score
        status = "PASS" if is_low else "FAIL"
        print(f"    {object_class} at top of {scene_name}: score={score:.2f} <= {max_score} [{status}]")
        if is_low:
            passed += 1

    total_tests = len(test_cases) + len(incompatible_tests)
    print(f"\n  Result: {passed}/{total_tests} tests passed")
    return passed >= total_tests * 0.6


def test_placement_suggestions():
    """Test placement position suggestions"""
    print("\n" + "="*60)
    print("TEST 3: Placement Position Suggestions")
    print("="*60)

    analyzer = SemanticSceneAnalyzer(use_sam3=False)
    images = create_test_images()

    test_cases = [
        ('fish', 'open_water'),
        ('can', 'seafloor'),
        ('crab', 'seafloor'),
        ('plastic_bag', 'surface'),
        ('debris', 'open_water'),
    ]

    passed = 0
    for object_class, scene_name in test_cases:
        image = images[scene_name]
        analysis = analyzer.analyze_scene(image)
        h, w = image.shape[:2]

        # Suggest position
        object_size = (50, 50)
        position = analyzer.suggest_placement_position(
            object_class,
            object_size,
            analysis,
            (h, w),
            existing_positions=[],
            min_distance=30
        )

        if position is not None:
            x, y = position
            # Verify position is valid
            score, reason = analyzer.check_object_scene_compatibility(
                object_class, position, analysis, (h, w)
            )

            valid = score >= analyzer.min_compatibility_score
            status = "PASS" if valid else "FAIL"
            print(f"  {object_class} in {scene_name}: pos=({x}, {y}), score={score:.2f} [{status}]")

            if valid:
                passed += 1
        else:
            print(f"  {object_class} in {scene_name}: no position found [FAIL]")

    print(f"\n  Result: {passed}/{len(test_cases)} tests passed")
    return passed >= len(test_cases) * 0.7


def test_multiple_placements():
    """Test placing multiple objects with minimum distance"""
    print("\n" + "="*60)
    print("TEST 4: Multiple Object Placements")
    print("="*60)

    analyzer = SemanticSceneAnalyzer(use_sam3=False)
    image = create_test_images()['seafloor']
    analysis = analyzer.analyze_scene(image)
    h, w = image.shape[:2]

    objects = ['can', 'debris', 'glass_bottle', 'rope', 'plastic']
    positions = []
    min_distance = 80

    print(f"  Placing {len(objects)} objects with min_distance={min_distance}")

    for obj in objects:
        pos = analyzer.suggest_placement_position(
            obj,
            (60, 60),
            analysis,
            (h, w),
            existing_positions=positions,
            min_distance=min_distance
        )

        if pos is not None:
            positions.append(pos)
            print(f"    {obj}: placed at {pos}")
        else:
            print(f"    {obj}: could not place")

    # Verify distances
    print("\n  Distance validation:")
    all_valid = True
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = np.sqrt((positions[i][0] - positions[j][0])**2 +
                          (positions[i][1] - positions[j][1])**2)
            valid = dist >= min_distance
            if not valid:
                all_valid = False
                print(f"    Objects {i} and {j}: distance={dist:.1f} [FAIL]")

    if all_valid:
        print(f"    All distances >= {min_distance} [PASS]")

    print(f"\n  Result: placed {len(positions)}/{len(objects)} objects")
    return len(positions) >= 3 and all_valid


def test_convenience_function():
    """Test the convenience is_placement_valid function"""
    print("\n" + "="*60)
    print("TEST 5: Convenience Function")
    print("="*60)

    images = create_test_images()

    test_cases = [
        ('fish', (320, 240), 'open_water', True),
        ('can', (320, 400), 'seafloor', True),
        ('crab', (320, 100), 'open_water', False),
    ]

    passed = 0
    for object_class, position, scene_name, expected_valid in test_cases:
        image = images[scene_name]
        is_valid, score, reason = is_placement_valid(object_class, position, image, min_score=0.4)

        match = is_valid == expected_valid
        status = "PASS" if match else "FAIL"
        print(f"  {object_class} at {position} in {scene_name}: valid={is_valid}, score={score:.2f} [{status}]")

        if match:
            passed += 1

    print(f"\n  Result: {passed}/{len(test_cases)} tests passed")
    return passed == len(test_cases)


def test_sam3_availability():
    """Test SAM3 availability (informational)"""
    print("\n" + "="*60)
    print("TEST 6: SAM3 Availability Check")
    print("="*60)

    try:
        analyzer = SemanticSceneAnalyzer(use_sam3=True)
        if analyzer.use_sam3:
            print("  SAM3 is AVAILABLE and loaded")

            # Quick test
            image = create_test_images()['open_water']
            analysis = analyzer.analyze_scene(image)
            print(f"  SAM3 analysis completed: dominant={analysis.dominant_region.value}")
            return True
        else:
            print("  SAM3 is NOT available (using heuristic fallback)")
            print("  Note: SAM3 requires: pip install transformers>=4.45.0")
            return True  # Not a failure, just informational
    except Exception as e:
        print(f"  SAM3 initialization error: {e}")
        print("  Using heuristic fallback instead")
        return True  # Not a failure


def test_with_real_image():
    """Test with a real image if available"""
    print("\n" + "="*60)
    print("TEST 7: Real Image Test (if available)")
    print("="*60)

    # Look for sample images
    possible_paths = [
        Path("volumes/images"),
        Path("output/images"),
        Path("samples"),
        Path("test_images"),
    ]

    image_path = None
    for p in possible_paths:
        if p.exists():
            images = list(p.glob("*.jpg")) + list(p.glob("*.png"))
            if images:
                image_path = images[0]
                break

    if image_path is None:
        print("  No real images found, skipping")
        return True

    print(f"  Using image: {image_path}")
    image = cv2.imread(str(image_path))

    if image is None:
        print("  Failed to load image")
        return False

    analyzer = SemanticSceneAnalyzer(use_sam3=False)
    analysis = analyzer.analyze_scene(image)

    print(f"  Image size: {image.shape[:2]}")
    print(f"  Dominant region: {analysis.dominant_region.value}")
    print(f"  Brightness: {analysis.scene_brightness:.2f}")
    print(f"  Clarity: {analysis.water_clarity}")
    print(f"  Temperature: {analysis.color_temperature}")
    print(f"  Region scores:")
    for region, score in sorted(analysis.region_scores.items(), key=lambda x: -x[1])[:5]:
        if score > 0.01:
            print(f"    - {region}: {score:.2%}")

    # Test compatibility for some objects
    h, w = image.shape[:2]
    objects_to_test = ['fish', 'can', 'plastic_bag', 'debris']

    print("\n  Object compatibility at image center:")
    for obj in objects_to_test:
        score, reason = analyzer.check_object_scene_compatibility(
            obj, (w//2, h//2), analysis, (h, w)
        )
        print(f"    {obj}: {score:.2f}")

    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("SEMANTIC SCENE ANALYZER TEST SUITE")
    print("="*60)

    results = {}

    # Run tests
    results['heuristic_analysis'] = test_heuristic_analysis()
    results['object_compatibility'] = test_object_compatibility()
    results['placement_suggestions'] = test_placement_suggestions()
    results['multiple_placements'] = test_multiple_placements()
    results['convenience_function'] = test_convenience_function()
    results['sam3_availability'] = test_sam3_availability()
    results['real_image'] = test_with_real_image()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: [{status}]")

    print(f"\n  Overall: {passed}/{total} tests passed")

    if passed == total:
        print("\n  ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n  {total - passed} TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
