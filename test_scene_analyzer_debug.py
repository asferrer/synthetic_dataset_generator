"""
SAM3 Scene Analyzer Debug Test
==============================

Tests the debug and explainability features of the SemanticSceneAnalyzer.
Generates visualizations and reports for understanding SAM3/heuristic behavior.
"""

import sys
import os
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.augmentation.scene_analyzer import (
    SemanticSceneAnalyzer,
    SceneRegion,
    DebugInfo,
    PlacementDecision,
)


def create_test_underwater_image():
    """Create a realistic underwater test image"""
    h, w = 480, 640
    image = np.zeros((h, w, 3), dtype=np.uint8)

    # Create gradient background (blue water)
    for y in range(h):
        blue_val = int(200 - (y / h) * 80)
        green_val = int(blue_val * 0.65)
        red_val = int(blue_val * 0.35)
        image[y, :] = [blue_val, green_val, red_val]

    # Add bright surface at top
    for y in range(int(h * 0.15)):
        brightness = 1.0 - (y / (h * 0.15)) * 0.4
        image[y, :] = (image[y, :] * brightness + np.array([255, 255, 240]) * (1 - brightness) * 0.5).astype(np.uint8)

    # Add sandy seafloor at bottom
    for y in range(int(h * 0.7), h):
        progress = (y - int(h * 0.7)) / (h * 0.3)
        image[y, :, 0] = int(80 + progress * 40)   # B
        image[y, :, 1] = int(130 + progress * 50)  # G
        image[y, :, 2] = int(160 + progress * 60)  # R

    # Add some vegetation (green patches)
    np.random.seed(42)
    for _ in range(5):
        cx = np.random.randint(100, w - 100)
        cy = np.random.randint(int(h * 0.5), int(h * 0.8))
        radius = np.random.randint(30, 60)
        cv2.circle(image, (cx, cy), radius, (40, 130, 35), -1)

    # Add rocky texture
    noise = np.random.randint(0, 30, (h, w), dtype=np.uint8)
    for y in range(int(h * 0.75), h):
        if np.random.random() > 0.7:
            image[y, :, :] = np.clip(image[y, :, :].astype(int) + noise[y, :, np.newaxis], 0, 255).astype(np.uint8)

    return image


def test_debug_analysis():
    """Test debug analysis functionality"""
    print("\n" + "=" * 60)
    print("TEST: Debug Analysis with Visualization")
    print("=" * 60)

    # Create analyzer with debug enabled
    debug_dir = "./debug_output_test"
    analyzer = SemanticSceneAnalyzer(
        use_sam3=False,  # Use heuristic for local testing
        debug=True,
        debug_output_dir=debug_dir,
    )

    # Create test image
    image = create_test_underwater_image()
    print(f"\n[1] Created test image: {image.shape}")

    # Run debug analysis
    print("\n[2] Running analyze_scene_with_debug()...")
    analysis, debug_info = analyzer.analyze_scene_with_debug(
        image,
        save_visualization=True,
        image_id="test_underwater_scene"
    )

    # Print debug info
    print(f"\n[3] Analysis Results:")
    print(f"    Method: {debug_info.analysis_method}")
    print(f"    Processing time: {debug_info.processing_time_ms:.1f}ms")
    print(f"    Dominant region: {analysis.dominant_region.value}")
    print(f"    Visualization: {debug_info.visualization_path}")

    print(f"\n[4] Region Scores:")
    for region, score in sorted(analysis.region_scores.items(), key=lambda x: -x[1]):
        if score > 0.01:
            print(f"    - {region}: {score:.2%}")

    print(f"\n[5] Decision Log:")
    for entry in debug_info.decision_log:
        print(f"    {entry}")

    print(f"\n[6] Region Masks Generated:")
    for region_name, mask in debug_info.region_masks.items():
        coverage = (mask > 0).sum() / mask.size * 100
        print(f"    - {region_name}: {coverage:.1f}% coverage")

    return True


def test_placement_debug():
    """Test placement decisions with debug info"""
    print("\n" + "=" * 60)
    print("TEST: Placement Decision Debug")
    print("=" * 60)

    debug_dir = "./debug_output_test"
    analyzer = SemanticSceneAnalyzer(
        use_sam3=False,
        debug=True,
        debug_output_dir=debug_dir,
    )

    image = create_test_underwater_image()
    h, w = image.shape[:2]

    # First analyze the scene
    analysis, _ = analyzer.analyze_scene_with_debug(
        image,
        save_visualization=True,
        image_id="placement_test"
    )

    # Test various placements with debug
    test_cases = [
        ("fish", (w // 2, h // 4)),      # Fish in upper water
        ("can", (w // 2, h // 4)),        # Can in upper water (should be bad)
        ("crab", (w // 2, int(h * 0.85))), # Crab at seafloor (should be good)
        ("plastic_bag", (w // 2, 50)),    # Plastic at surface
        ("starfish", (w // 2, h // 2)),   # Starfish in mid-water (should be bad)
    ]

    print("\n[1] Testing placement decisions:")
    for object_class, position in test_cases:
        score, reason, decision = analyzer.check_object_scene_compatibility_with_debug(
            object_class, position, analysis, (h, w)
        )

        print(f"\n    Object: {object_class}")
        print(f"    Position: {position}")
        print(f"    Region: {decision.region_at_position}")
        print(f"    Score: {score:.2f}")
        print(f"    Decision: {decision.decision}")
        print(f"    Reason: {reason}")

        if decision.alternative_positions:
            print(f"    Alternatives: {decision.alternative_positions[:2]}")

    # Export report
    print("\n[2] Exporting debug report...")
    report = analyzer.export_debug_report("placement_test")
    print(f"    Report saved to: {report.get('report_path', 'N/A')}")
    print(f"    Total placement decisions: {len(report.get('placement_decisions', []))}")

    return True


def test_visualization_output():
    """Test that visualizations are generated correctly"""
    print("\n" + "=" * 60)
    print("TEST: Visualization Output")
    print("=" * 60)

    debug_dir = "./debug_output_test"
    analyzer = SemanticSceneAnalyzer(
        use_sam3=False,
        debug=True,
        debug_output_dir=debug_dir,
    )

    image = create_test_underwater_image()

    # Run analysis
    analysis, debug_info = analyzer.analyze_scene_with_debug(
        image,
        save_visualization=True,
        image_id="viz_test"
    )

    # Check files exist
    print(f"\n[1] Checking output files...")

    viz_exists = os.path.exists(debug_info.visualization_path) if debug_info.visualization_path else False
    print(f"    Visualization: {'EXISTS' if viz_exists else 'MISSING'}")

    masks_dir = os.path.join(debug_dir, "viz_test_masks")
    masks_exist = os.path.exists(masks_dir)
    print(f"    Masks directory: {'EXISTS' if masks_exist else 'MISSING'}")

    if masks_exist:
        mask_files = os.listdir(masks_dir)
        print(f"    Mask files: {len(mask_files)}")
        for mf in mask_files:
            print(f"      - {mf}")

    # Load and verify visualization
    if viz_exists:
        viz = cv2.imread(debug_info.visualization_path)
        print(f"\n[2] Visualization dimensions: {viz.shape}")
        print(f"    Expected: {image.shape[0]} x {image.shape[1] * 3} x 3")

    return viz_exists and masks_exist


def test_sam3_mode_detection():
    """Test that SAM3 mode is correctly detected"""
    print("\n" + "=" * 60)
    print("TEST: SAM3 Mode Detection")
    print("=" * 60)

    # Test with SAM3 disabled
    analyzer_heuristic = SemanticSceneAnalyzer(use_sam3=False, debug=True)
    print(f"\n[1] Heuristic mode:")
    print(f"    use_sam3: {analyzer_heuristic.use_sam3}")
    print(f"    SAM3 available: {SemanticSceneAnalyzer.is_sam3_available()}")

    # Test with SAM3 enabled (will fall back if not available)
    analyzer_sam3 = SemanticSceneAnalyzer(use_sam3=True, debug=True)
    print(f"\n[2] SAM3 mode (requested):")
    print(f"    use_sam3 (actual): {analyzer_sam3.use_sam3}")
    print(f"    Model loaded: {analyzer_sam3._sam3_model is not None}")

    return True


def run_all_tests():
    """Run all debug tests"""
    print("\n" + "=" * 60)
    print("SAM3 SCENE ANALYZER DEBUG TEST SUITE")
    print("=" * 60)

    results = {}

    # Test 1: Debug analysis
    try:
        results['debug_analysis'] = test_debug_analysis()
        print("\n[PASS] Debug Analysis")
    except Exception as e:
        results['debug_analysis'] = False
        print(f"\n[FAIL] Debug Analysis: {e}")

    # Test 2: Placement debug
    try:
        results['placement_debug'] = test_placement_debug()
        print("\n[PASS] Placement Debug")
    except Exception as e:
        results['placement_debug'] = False
        print(f"\n[FAIL] Placement Debug: {e}")

    # Test 3: Visualization output
    try:
        results['visualization'] = test_visualization_output()
        print("\n[PASS] Visualization Output")
    except Exception as e:
        results['visualization'] = False
        print(f"\n[FAIL] Visualization Output: {e}")

    # Test 4: SAM3 detection
    try:
        results['sam3_detection'] = test_sam3_mode_detection()
        print("\n[PASS] SAM3 Mode Detection")
    except Exception as e:
        results['sam3_detection'] = False
        print(f"\n[FAIL] SAM3 Mode Detection: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {test_name}: {status}")

    print(f"\n  Overall: {passed}/{total} tests passed")

    if passed == total:
        print("\n  ALL TESTS PASSED!")
    else:
        print("\n  SOME TESTS FAILED")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
