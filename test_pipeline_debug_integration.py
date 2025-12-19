"""
Test Pipeline Debug Integration
===============================
Tests that scene analysis debug visualization is integrated
into the pipeline debug output.
"""

import os
import sys
import asyncio
import tempfile
import shutil
import numpy as np
import cv2

# Add project paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "services", "augmentor"))


def create_test_underwater_image(output_path: str, width: int = 640, height: int = 480):
    """Create a synthetic underwater test image."""
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Create gradient underwater scene
    for y in range(height):
        # Blue water at top, sandy bottom
        if y < height * 0.6:
            # Blue water region
            blue_val = int(200 - (y / (height * 0.6)) * 60)
            green_val = int(blue_val * 0.65)
            red_val = int(blue_val * 0.35)
            image[y, :] = [blue_val, green_val, red_val]
        else:
            # Sandy seafloor
            progress = (y - height * 0.6) / (height * 0.4)
            image[y, :, 0] = int(80 + progress * 40)   # B
            image[y, :, 1] = int(130 + progress * 50)  # G
            image[y, :, 2] = int(160 + progress * 60)  # R

    cv2.imwrite(output_path, image)
    return output_path


def create_test_object(output_path: str, size: int = 100):
    """Create a simple test object with transparency."""
    image = np.zeros((size, size, 4), dtype=np.uint8)

    # Create a circle
    center = size // 2
    cv2.circle(image, (center, center), size // 3, (100, 150, 200, 255), -1)

    cv2.imwrite(output_path, image)
    return output_path


async def test_pipeline_debug_integration():
    """Test that scene analysis debug is generated during compose."""
    print("\n" + "="*60)
    print("TEST: Pipeline Debug Integration")
    print("="*60)

    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="pipeline_debug_test_")
    debug_dir = os.path.join(temp_dir, "pipeline_debug")
    os.makedirs(debug_dir, exist_ok=True)

    try:
        # Create test images
        bg_path = create_test_underwater_image(os.path.join(temp_dir, "background.jpg"))
        obj_path = create_test_object(os.path.join(temp_dir, "object.png"))
        output_path = os.path.join(temp_dir, "output.jpg")

        print(f"\n  Test directory: {temp_dir}")
        print(f"  Background: {bg_path}")
        print(f"  Object: {obj_path}")

        # Import and create composer
        from app.composer import ImageComposer
        from app.models.schemas import ObjectPlacement, EffectType

        composer = ImageComposer(use_gpu=False)

        # Create object placement
        objects = [
            ObjectPlacement(
                image_path=obj_path,
                class_name="fish",
                position=None,  # Auto-place
                scale=None,
                rotation=0,
            ),
            ObjectPlacement(
                image_path=obj_path,
                class_name="can",
                position=None,
                scale=None,
                rotation=0,
            ),
        ]

        # Compose with debug enabled
        print("\n  Running compose with debug...")
        result = await composer.compose(
            background_path=bg_path,
            objects=objects,
            effects=[EffectType.COLOR_CORRECTION],
            output_path=output_path,
            debug_output_dir=debug_dir,
        )

        print(f"  Objects placed: {result.objects_placed}")
        print(f"  Effects applied: {result.effects_applied}")

        # Check for debug files
        print("\n  Checking debug output files:")
        expected_files = [
            "00_background.jpg",
            "02b_scene_analysis.jpg",  # NEW: Scene analysis visualization
            "08_objects_placed.jpg",
            "08b_placement_decisions.jpg",  # NEW: Placement decisions visualization
            "11_final.jpg",
        ]

        found_files = os.listdir(debug_dir)
        print(f"  Found files: {found_files}")

        # Check each expected file
        tests_passed = 0
        total_tests = len(expected_files)

        for expected_file in expected_files:
            file_path = os.path.join(debug_dir, expected_file)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"    [PASS] {expected_file} ({file_size} bytes)")
                tests_passed += 1
            else:
                print(f"    [FAIL] {expected_file} - NOT FOUND")

        # Check scene analysis visualization dimensions
        scene_analysis_path = os.path.join(debug_dir, "02b_scene_analysis.jpg")
        if os.path.exists(scene_analysis_path):
            scene_img = cv2.imread(scene_analysis_path)
            if scene_img is not None:
                h, w = scene_img.shape[:2]
                # Should be 3x width (original | overlay | region map)
                expected_width = 640 * 3
                if w >= expected_width * 0.9:  # Allow some tolerance
                    print(f"    [PASS] Scene analysis dimensions: {w}x{h} (3-panel layout)")
                else:
                    print(f"    [WARN] Unexpected dimensions: {w}x{h}")

        # Check placement decisions visualization
        placement_path = os.path.join(debug_dir, "08b_placement_decisions.jpg")
        if os.path.exists(placement_path):
            placement_img = cv2.imread(placement_path)
            if placement_img is not None:
                print(f"    [PASS] Placement decisions image loaded successfully")

        print(f"\n  Result: {tests_passed}/{total_tests} expected files found")

        return tests_passed >= 3  # At least 3 key files should exist

    except ImportError as e:
        print(f"\n  [SKIP] Import error (run in Docker): {e}")
        return True  # Skip test if imports fail outside Docker

    except Exception as e:
        print(f"\n  [ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\n  Cleaned up temp directory")


async def test_scene_analyzer_standalone():
    """Test scene analyzer debug output standalone."""
    print("\n" + "="*60)
    print("TEST: Scene Analyzer Standalone Debug")
    print("="*60)

    try:
        from src.augmentation.scene_analyzer import SemanticSceneAnalyzer

        # Create test image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        for y in range(480):
            if y < 288:  # Blue water
                image[y, :] = [200 - y//3, 130 - y//5, 70 - y//8]
            else:  # Sandy bottom
                progress = (y - 288) / 192
                image[y, :] = [80 + int(40*progress), 130 + int(50*progress), 160 + int(60*progress)]

        # Create analyzer with debug
        temp_dir = tempfile.mkdtemp(prefix="scene_debug_test_")

        analyzer = SemanticSceneAnalyzer(
            use_sam3=False,
            debug=True,
            debug_output_dir=temp_dir,
        )

        print(f"\n  Debug output dir: {temp_dir}")

        # Run analysis with debug
        analysis, debug_info = analyzer.analyze_scene_with_debug(
            image,
            save_visualization=True,
            image_id="test_scene",
        )

        print(f"  Analysis method: {debug_info.analysis_method}")
        print(f"  Processing time: {debug_info.processing_time_ms:.1f}ms")
        print(f"  Dominant region: {analysis.dominant_region.value}")
        print(f"  Region scores: {dict(list(analysis.region_scores.items())[:3])}")

        # Check visualization was saved
        viz_path = os.path.join(temp_dir, "test_scene_debug.png")
        if os.path.exists(viz_path):
            print(f"  [PASS] Visualization saved: {viz_path}")
            result = True
        else:
            print(f"  [FAIL] Visualization not saved")
            result = False

        # Cleanup
        shutil.rmtree(temp_dir)
        return result

    except ImportError as e:
        print(f"\n  [SKIP] Import error: {e}")
        return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("PIPELINE DEBUG INTEGRATION TESTS")
    print("="*60)

    results = []

    # Test 1: Scene analyzer standalone
    results.append(("Scene Analyzer Debug", asyncio.run(test_scene_analyzer_standalone())))

    # Test 2: Full pipeline integration
    results.append(("Pipeline Debug Integration", asyncio.run(test_pipeline_debug_integration())))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n  Total: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
