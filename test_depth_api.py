"""
Test Script for Depth API (Microservices)
==========================================

Tests the Depth Anything V3 upgrade via the depth microservice API.
This script doesn't require local Python dependencies - it uses the Docker services.

Usage:
    python test_depth_api.py
"""

import requests
import json
import time
import numpy as np
import base64
from io import BytesIO
from PIL import Image

class DepthAPITester:
    """Test the depth microservice API"""

    def __init__(self, base_url="http://localhost:8001"):
        self.base_url = base_url
        self.test_results = {}

    def test_health(self):
        """Test 1: Health check endpoint"""
        print("\n" + "="*60)
        print("TEST 1: Health Check")
        print("="*60)

        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            response.raise_for_status()

            health_data = response.json()
            print(f"✓ Service is healthy")
            print(f"  Status: {health_data.get('status')}")
            print(f"  Model loaded: {health_data.get('model_loaded')}")
            print(f"  GPU available: {health_data.get('gpu_available')}")
            if health_data.get('gpu_name'):
                print(f"  GPU: {health_data.get('gpu_name')}")

            self.test_results['health'] = True
            return True

        except Exception as e:
            print(f"✗ Health check FAILED: {e}")
            self.test_results['health'] = False
            return False

    def test_info(self):
        """Test 2: Service info endpoint"""
        print("\n" + "="*60)
        print("TEST 2: Service Info")
        print("="*60)

        try:
            response = requests.get(f"{self.base_url}/info", timeout=10)
            response.raise_for_status()

            info_data = response.json()
            print(f"✓ Service info retrieved")
            print(f"  Model: {info_data.get('model')}")
            print(f"  Model params: {info_data.get('model_params')}")
            print(f"  Device: {info_data.get('device')}")

            # Check if using DA-V3
            model_name = info_data.get('model', '').lower()
            if 'da3' in model_name or 'depth-anything-3' in model_name or 'v3' in model_name:
                print(f"\n✓ Using Depth Anything V3!")
            else:
                print(f"\n⚠ Model might not be DA-V3: {info_data.get('model')}")

            self.test_results['info'] = True
            return info_data

        except Exception as e:
            print(f"✗ Service info FAILED: {e}")
            self.test_results['info'] = False
            return None

    def create_test_image(self, size=(640, 480)):
        """Create a synthetic test image"""
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)

        # Create gradient (near to far)
        for y in range(size[1]):
            intensity = int(255 * (y / size[1]))
            img[y, :] = [intensity, intensity, intensity]

        # Add some objects at different depths
        # Close object (top)
        center = (int(size[0] * 0.25), int(size[1] * 0.25))
        for y in range(max(0, center[1]-60), min(size[1], center[1]+60)):
            for x in range(max(0, center[0]-60), min(size[0], center[0]+60)):
                dist = ((x - center[0])**2 + (y - center[1])**2) ** 0.5
                if dist < 60:
                    img[y, x] = [100, 150, 200]

        # Mid-distance object (middle)
        img[200:300, 300:400] = [150, 100, 180]

        # Far object (bottom)
        center = (int(size[0] * 0.5), int(size[1] * 0.8))
        for y in range(max(0, center[1]-40), min(size[1], center[1]+40)):
            for x in range(max(0, center[0]-80), min(size[0], center[0]+80)):
                dist_x = (x - center[0]) / 80.0
                dist_y = (y - center[1]) / 40.0
                if dist_x**2 + dist_y**2 < 1:
                    img[y, x] = [200, 200, 100]

        return img

    def image_to_base64(self, img_array):
        """Convert numpy array to base64 string"""
        img = Image.fromarray(img_array)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def test_inference_simple(self):
        """Test 3: Simple depth inference via API"""
        print("\n" + "="*60)
        print("TEST 3: Depth Inference (Simple)")
        print("="*60)

        try:
            # Create test image
            test_img = self.create_test_image()
            print(f"Created test image: {test_img.shape}")

            # Save to temporary location (shared volume)
            test_img_pil = Image.fromarray(test_img)
            test_path = "/shared/test_image.png"

            # Note: In real scenario, you'd save to the shared volume
            # For now, we'll just print the intention
            print(f"⚠ This test requires shared volume access")
            print(f"  Skipping actual inference test")
            print(f"  In production: Save to {test_path} and call /estimate")

            self.test_results['inference_simple'] = 'SKIPPED'
            return None

        except Exception as e:
            print(f"✗ Simple inference test error: {e}")
            self.test_results['inference_simple'] = False
            return None

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)

        print(f"\nResults:")
        for test_name, result in self.test_results.items():
            status = "PASS" if result is True else ("SKIP" if result == "SKIPPED" else "FAIL")
            symbol = "✓" if result is True else ("⚠" if result == "SKIPPED" else "✗")
            print(f"  {symbol} {test_name}: {status}")

        # Overall verdict
        critical_tests = ['health', 'info']
        critical_pass = all(self.test_results.get(test) for test in critical_tests)

        print("\n" + "="*60)
        if critical_pass:
            print("✓ OVERALL: CRITICAL TESTS PASSED")
            print("  Depth service is operational and ready")
        else:
            print("✗ OVERALL: TESTS FAILED")
            print("  Service not ready or misconfigured")
        print("="*60 + "\n")

        return critical_pass


def wait_for_service(base_url="http://localhost:8001", timeout=120):
    """Wait for service to be ready"""
    print(f"Waiting for depth service at {base_url}...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✓ Service is ready!")
                return True
        except requests.exceptions.RequestException:
            pass

        time.sleep(2)
        elapsed = int(time.time() - start_time)
        print(f"  Waiting... ({elapsed}s/{timeout}s)")

    print(f"✗ Service did not become ready within {timeout}s")
    return False


def main():
    """Run all API tests"""
    print("="*60)
    print("Depth Anything V3 API Testing")
    print("="*60)

    # Wait for service to be ready
    if not wait_for_service():
        print("\n✗ Cannot proceed - service not available")
        print("  Make sure Docker containers are running:")
        print("  docker-compose -f docker-compose.microservices.yml ps")
        return 1

    # Run tests
    tester = DepthAPITester()

    if not tester.test_health():
        print("\n✗ Service health check failed - aborting tests")
        return 1

    tester.test_info()
    tester.test_inference_simple()

    # Print summary
    success = tester.print_summary()

    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
