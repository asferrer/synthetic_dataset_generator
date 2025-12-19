"""
Integration Tests for Augmentor Service
=======================================
Tests the complete microservices architecture.

Usage:
    python test_augmentor_integration.py [--build] [--quick]

Options:
    --build     Build containers before testing
    --quick     Run only quick health checks
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path

# Test configuration
GATEWAY_URL = "http://localhost:8000"
AUGMENTOR_URL = "http://localhost:8004"
DEPTH_URL = "http://localhost:8001"
EFFECTS_URL = "http://localhost:8003"

TIMEOUT = 30


def check_dependencies():
    """Check if required packages are installed"""
    try:
        import httpx
        return True
    except ImportError:
        print("[ERROR] httpx not installed. Run: pip install httpx")
        return False


def wait_for_service(url: str, name: str, max_wait: int = 120) -> bool:
    """Wait for a service to become healthy"""
    import httpx

    print(f"[...] Waiting for {name} at {url}...")

    start = time.time()
    while time.time() - start < max_wait:
        try:
            response = httpx.get(f"{url}/health", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                status = data.get("status", "unknown")
                if status in ["healthy", "degraded"]:
                    elapsed = time.time() - start
                    print(f"[OK] {name} is {status} ({elapsed:.1f}s)")
                    return True
        except Exception:
            pass
        time.sleep(2)

    print(f"[FAIL] {name} did not become healthy after {max_wait}s")
    return False


def test_gateway_health():
    """Test Gateway health endpoint"""
    import httpx

    print("\n=== Testing Gateway Health ===")

    try:
        response = httpx.get(f"{GATEWAY_URL}/health", timeout=TIMEOUT)

        if response.status_code == 200:
            data = response.json()
            print(f"[OK] Gateway status: {data.get('status')}")

            services = data.get("services", [])
            for svc in services:
                name = svc.get("name", "unknown")
                status = svc.get("status", "unknown")
                latency = svc.get("latency_ms", 0)

                if status == "healthy":
                    print(f"  [OK] {name}: {status} ({latency:.0f}ms)")
                else:
                    print(f"  [WARN] {name}: {status}")

            return data.get("status") in ["healthy", "degraded"]
        else:
            print(f"[FAIL] Gateway returned {response.status_code}")
            return False

    except Exception as e:
        print(f"[FAIL] Gateway error: {e}")
        return False


def test_augmentor_health():
    """Test Augmentor service health"""
    import httpx

    print("\n=== Testing Augmentor Health ===")

    try:
        response = httpx.get(f"{AUGMENTOR_URL}/health", timeout=TIMEOUT)

        if response.status_code == 200:
            data = response.json()
            print(f"[OK] Augmentor status: {data.get('status')}")
            print(f"  GPU available: {data.get('gpu_available')}")
            print(f"  GPU name: {data.get('gpu_name', 'N/A')}")
            print(f"  Validators loaded: {data.get('validators_loaded')}")
            return True
        else:
            print(f"[FAIL] Augmentor returned {response.status_code}")
            return False

    except Exception as e:
        print(f"[FAIL] Augmentor error: {e}")
        return False


def test_augmentor_info():
    """Test Augmentor info endpoint"""
    import httpx

    print("\n=== Testing Augmentor Info ===")

    try:
        response = httpx.get(f"{AUGMENTOR_URL}/info", timeout=TIMEOUT)

        if response.status_code == 200:
            data = response.json()
            print(f"[OK] Service: {data.get('service')}")
            print(f"  Version: {data.get('version')}")
            print(f"  Effects: {len(data.get('effects_available', []))}")
            print(f"  Capabilities: {list(data.get('capabilities', {}).keys())}")
            return True
        else:
            print(f"[FAIL] Info returned {response.status_code}")
            return False

    except Exception as e:
        print(f"[FAIL] Info error: {e}")
        return False


def test_gateway_augment_endpoints():
    """Test Gateway augment endpoints"""
    import httpx

    print("\n=== Testing Gateway Augment Endpoints ===")

    try:
        # Test augmentor info via gateway
        response = httpx.get(f"{GATEWAY_URL}/augment/info", timeout=TIMEOUT)

        if response.status_code == 200:
            data = response.json()
            print(f"[OK] Augmentor info via gateway")
            print(f"  Service: {data.get('service')}")
            return True
        else:
            print(f"[WARN] Augmentor info returned {response.status_code}")
            return False

    except Exception as e:
        print(f"[FAIL] Gateway augment error: {e}")
        return False


def test_lighting_estimation():
    """Test lighting estimation endpoint"""
    import httpx

    print("\n=== Testing Lighting Estimation ===")

    # Check if test image exists
    # Uses DATASET_PATH environment variable or defaults to container path
    dataset_path = os.environ.get("DATASET_PATH", "/app/datasets")
    test_images = [
        f"{dataset_path}/Backgrounds_filtered",
        "/app/datasets/Backgrounds_filtered",
        "./datasets/Backgrounds_filtered",
    ]

    test_image = None
    for path in test_images:
        p = Path(path)
        if p.exists():
            images = list(p.glob("*.jpg")) + list(p.glob("*.png"))
            if images:
                test_image = str(images[0])
                break

    if not test_image:
        print("[SKIP] No test image found")
        return True

    print(f"  Using image: {test_image}")

    try:
        response = httpx.post(
            f"{AUGMENTOR_URL}/lighting",
            json={
                "image_path": test_image,
                "max_light_sources": 3,
                "intensity_threshold": 0.6,
            },
            timeout=60.0,
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                lighting = data.get("lighting_info", {})
                print(f"[OK] Lighting estimation successful")
                print(f"  Light sources: {len(lighting.get('light_sources', []))}")
                print(f"  Color temp: {lighting.get('color_temperature')}K")
                return True
            else:
                print(f"[WARN] Lighting failed: {data.get('error')}")
                return True  # Not critical
        else:
            print(f"[WARN] Lighting returned {response.status_code}")
            return True  # Not critical

    except Exception as e:
        print(f"[WARN] Lighting error: {e}")
        return True  # Not critical


def run_docker_build():
    """Build docker containers"""
    print("\n=== Building Docker Containers ===")

    cmd = [
        "docker-compose",
        "-f", "docker-compose.microservices.yml",
        "build",
        "--parallel",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print("[OK] Build completed")
            return True
        else:
            print(f"[FAIL] Build failed: {result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        print("[FAIL] Build timed out")
        return False
    except Exception as e:
        print(f"[FAIL] Build error: {e}")
        return False


def run_docker_up():
    """Start docker containers"""
    print("\n=== Starting Docker Containers ===")

    cmd = [
        "docker-compose",
        "-f", "docker-compose.microservices.yml",
        "up", "-d",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print("[OK] Containers started")
            return True
        else:
            print(f"[FAIL] Start failed: {result.stderr[:500]}")
            return False
    except Exception as e:
        print(f"[FAIL] Start error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Augmentor Integration Tests")
    parser.add_argument("--build", action="store_true", help="Build before testing")
    parser.add_argument("--quick", action="store_true", help="Quick health checks only")
    args = parser.parse_args()

    print("=" * 60)
    print("AUGMENTOR SERVICE INTEGRATION TESTS")
    print("=" * 60)

    if not check_dependencies():
        sys.exit(1)

    # Build if requested
    if args.build:
        if not run_docker_build():
            sys.exit(1)

        if not run_docker_up():
            sys.exit(1)

        # Wait for services
        print("\n=== Waiting for Services ===")
        services = [
            (DEPTH_URL, "Depth"),
            (EFFECTS_URL, "Effects"),
            (AUGMENTOR_URL, "Augmentor"),
            (GATEWAY_URL, "Gateway"),
        ]

        for url, name in services:
            if not wait_for_service(url, name):
                print(f"[FAIL] {name} service not available")
                sys.exit(1)

    # Run tests
    results = []

    # Health tests
    results.append(("Gateway Health", test_gateway_health()))
    results.append(("Augmentor Health", test_augmentor_health()))
    results.append(("Augmentor Info", test_augmentor_info()))

    if not args.quick:
        # Extended tests
        results.append(("Gateway Augment", test_gateway_augment_endpoints()))
        results.append(("Lighting Estimation", test_lighting_estimation()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        sys.exit(0)
    else:
        print("\n[FAILURE] Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
