"""
Shared fixtures for all tests.

This module provides common fixtures used across unit, integration, and e2e tests.
"""
import pytest
import base64
import json
import subprocess
import time
import sys
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "frontend"))


# =============================================================================
# Image Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def sample_image_array():
    """Generate a random sample image as numpy array."""
    np.random.seed(42)
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def sample_image_pil(sample_image_array):
    """Generate a sample PIL Image."""
    return Image.fromarray(sample_image_array)


@pytest.fixture(scope="session")
def sample_image_base64(sample_image_pil):
    """Generate a sample image in base64 format."""
    buffer = BytesIO()
    sample_image_pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture(scope="session")
def sample_rgba_image():
    """Generate a sample RGBA image (with alpha channel)."""
    np.random.seed(42)
    img_array = np.random.randint(0, 255, (200, 200, 4), dtype=np.uint8)
    return Image.fromarray(img_array, mode="RGBA")


@pytest.fixture(scope="session")
def sample_rgba_base64(sample_rgba_image):
    """Generate a sample RGBA image in base64 format."""
    buffer = BytesIO()
    sample_rgba_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture
def sample_images(tmp_path):
    """Create temporary sample images for testing."""
    np.random.seed(42)

    # Background image (larger)
    bg_array = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    bg_img = Image.fromarray(bg_array)
    bg_path = tmp_path / "background.jpg"
    bg_img.save(bg_path, format="JPEG")

    # Object image with alpha channel
    obj_array = np.random.randint(0, 255, (200, 200, 4), dtype=np.uint8)
    obj_img = Image.fromarray(obj_array, mode="RGBA")
    obj_path = tmp_path / "object.png"
    obj_img.save(obj_path, format="PNG")

    return bg_path, obj_path


# =============================================================================
# COCO Data Fixtures
# =============================================================================

@pytest.fixture
def sample_coco_data():
    """Generate sample COCO format dataset."""
    return {
        "info": {
            "description": "Test Dataset",
            "version": "1.0",
            "year": 2024
        },
        "licenses": [],
        "categories": [
            {"id": 1, "name": "class_a", "supercategory": "object"},
            {"id": 2, "name": "class_b", "supercategory": "object"},
            {"id": 3, "name": "class_c", "supercategory": "object"}
        ],
        "images": [
            {"id": i, "file_name": f"image_{i:04d}.jpg", "width": 640, "height": 480}
            for i in range(1, 101)
        ],
        "annotations": [
            {
                "id": i,
                "image_id": (i % 100) + 1,
                "category_id": (i % 3) + 1,
                "bbox": [10 + i % 50, 10 + i % 50, 100, 100],
                "area": 10000,
                "iscrowd": 0
            }
            for i in range(1, 201)
        ]
    }


@pytest.fixture
def small_coco_data():
    """Generate a small COCO dataset for quick tests."""
    return {
        "info": {"description": "Small Test Dataset"},
        "licenses": [],
        "categories": [
            {"id": 1, "name": "cat"},
            {"id": 2, "name": "dog"}
        ],
        "images": [
            {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "img2.jpg", "width": 640, "height": 480},
            {"id": 3, "file_name": "img3.jpg", "width": 640, "height": 480}
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 100, 100], "area": 10000, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [150, 10, 80, 80], "area": 6400, "iscrowd": 0},
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [20, 20, 90, 90], "area": 8100, "iscrowd": 0},
            {"id": 4, "image_id": 3, "category_id": 2, "bbox": [30, 30, 110, 110], "area": 12100, "iscrowd": 0}
        ]
    }


@pytest.fixture
def imbalanced_coco_data():
    """Generate an imbalanced COCO dataset for balancing tests."""
    return {
        "info": {"description": "Imbalanced Dataset"},
        "licenses": [],
        "categories": [
            {"id": 1, "name": "rare_class"},
            {"id": 2, "name": "common_class"}
        ],
        "images": [
            {"id": i, "file_name": f"img_{i}.jpg", "width": 640, "height": 480}
            for i in range(1, 111)
        ],
        "annotations": (
            # 10 annotations for rare class
            [
                {"id": i, "image_id": i, "category_id": 1, "bbox": [10, 10, 50, 50], "area": 2500, "iscrowd": 0}
                for i in range(1, 11)
            ] +
            # 100 annotations for common class
            [
                {"id": i + 10, "image_id": i + 10, "category_id": 2, "bbox": [10, 10, 50, 50], "area": 2500, "iscrowd": 0}
                for i in range(1, 101)
            ]
        )
    }


@pytest.fixture
def coco_with_segmentation():
    """Generate COCO dataset with segmentation annotations."""
    return {
        "info": {"description": "Dataset with Segmentation"},
        "licenses": [],
        "categories": [
            {"id": 1, "name": "object"}
        ],
        "images": [
            {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 100, 200, 150],
                "area": 30000,
                "segmentation": [[100, 100, 300, 100, 300, 250, 100, 250]],
                "iscrowd": 0
            }
        ]
    }


# =============================================================================
# Docker Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def docker_compose_file():
    """Get path to docker-compose file."""
    return PROJECT_ROOT / "docker-compose.microservices.yml"


@pytest.fixture(scope="module")
def docker_services(docker_compose_file):
    """
    Start and stop Docker services for integration tests.

    This fixture starts all services before tests and stops them after.
    Use with integration and e2e tests only.
    """
    compose_file = str(docker_compose_file)

    # Build images first
    build_result = subprocess.run(
        ["docker", "compose", "-f", compose_file, "build"],
        capture_output=True,
        text=True,
        timeout=1800  # 30 minutes for build
    )

    if build_result.returncode != 0:
        pytest.fail(f"Docker build failed: {build_result.stderr}")

    # Start services
    up_result = subprocess.run(
        ["docker", "compose", "-f", compose_file, "up", "-d"],
        capture_output=True,
        text=True,
        timeout=120
    )

    if up_result.returncode != 0:
        pytest.fail(f"Docker compose up failed: {up_result.stderr}")

    # Wait for services to be healthy
    _wait_for_services()

    yield

    # Cleanup: stop services
    subprocess.run(
        ["docker", "compose", "-f", compose_file, "down"],
        capture_output=True,
        text=True,
        timeout=120
    )


def _wait_for_services(timeout: int = 300, interval: int = 5):
    """Wait for all services to be healthy."""
    import requests

    services = {
        "gateway": "http://localhost:8000/health",
        "depth": "http://localhost:8001/health",
        "segmentation": "http://localhost:8002/health",
        "effects": "http://localhost:8003/health",
        "augmentor": "http://localhost:8004/health",
    }

    start_time = time.time()

    for name, url in services.items():
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(interval)
        else:
            pytest.fail(f"Service {name} did not become healthy within {timeout}s")


# =============================================================================
# Service URL Fixtures
# =============================================================================

@pytest.fixture
def gateway_url():
    """Gateway service URL."""
    return "http://localhost:8000"


@pytest.fixture
def depth_url():
    """Depth service URL."""
    return "http://localhost:8001"


@pytest.fixture
def segmentation_url():
    """Segmentation service URL."""
    return "http://localhost:8002"


@pytest.fixture
def effects_url():
    """Effects service URL."""
    return "http://localhost:8003"


@pytest.fixture
def augmentor_url():
    """Augmentor service URL."""
    return "http://localhost:8004"


@pytest.fixture
def frontend_url():
    """Frontend service URL."""
    return "http://localhost:8501"


# =============================================================================
# File System Fixtures
# =============================================================================

@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def temp_dataset_dir(tmp_path, sample_coco_data):
    """Create a temporary dataset directory with COCO data."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    # Save COCO JSON
    coco_file = dataset_dir / "annotations.json"
    with open(coco_file, "w") as f:
        json.dump(sample_coco_data, f)

    # Create dummy images
    images_dir = dataset_dir / "images"
    images_dir.mkdir()

    for img in sample_coco_data["images"]:
        img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img_pil = Image.fromarray(img_array)
        img_pil.save(images_dir / img["file_name"])

    return dataset_dir


# =============================================================================
# Batch Configuration Fixtures
# =============================================================================

@pytest.fixture
def sample_batch_config(tmp_path):
    """Generate sample batch generation configuration."""
    return {
        "backgrounds_dir": str(tmp_path / "backgrounds"),
        "objects_dir": str(tmp_path / "objects"),
        "output_dir": str(tmp_path / "output"),
        "num_images": 5,
        "config": {
            "effects": {
                "enabled": ["color_transfer", "blur_match"],
                "intensity": 0.5
            },
            "placement": {
                "strategy": "depth_aware",
                "max_objects": 3
            }
        }
    }
