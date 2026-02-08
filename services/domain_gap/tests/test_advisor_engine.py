"""
Tests for AdvisorEngine - domain gap analysis and parameter suggestions.

Tests verify:
1. Engine initialization (CPU-only, no GPU required)
2. Full analysis pipeline with synthetic image directories
3. Color/brightness analysis detection
4. Edge/texture/frequency analysis detection
5. Issue-to-suggestion mapping with current_config
6. Error handling for empty and non-existent directories
7. Issue and suggestion schema correctness
8. Severity ordering of results

All tests run inside Docker during the build stage.
Image directories are created via tmp_path with numpy/cv2.
"""

import pytest
import numpy as np
import cv2

from app.engines.advisor_engine import AdvisorEngine
from app.models.schemas import (
    GapIssue,
    ParameterSuggestion,
    IssueCategory,
    IssueSeverity,
    ImpactLevel,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def engine():
    """Create an AdvisorEngine instance."""
    return AdvisorEngine()


@pytest.fixture
def bright_images_dir(tmp_path):
    """Create a directory of bright images (mean ~220)."""
    d = tmp_path / "bright"
    d.mkdir()
    np.random.seed(42)
    for i in range(5):
        img = np.full((100, 100, 3), 220, dtype=np.uint8)
        noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(str(d / f"bright_{i}.jpg"), img)
    return str(d)


@pytest.fixture
def dark_images_dir(tmp_path):
    """Create a directory of dark images (mean ~40)."""
    d = tmp_path / "dark"
    d.mkdir()
    np.random.seed(123)
    for i in range(5):
        img = np.full((100, 100, 3), 40, dtype=np.uint8)
        noise = np.random.randint(-15, 15, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(str(d / f"dark_{i}.jpg"), img)
    return str(d)


@pytest.fixture
def random_images_dir(tmp_path):
    """Create a directory of uniform random images."""
    d = tmp_path / "random"
    d.mkdir()
    np.random.seed(99)
    for i in range(5):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(d / f"random_{i}.jpg"), img)
    return str(d)


@pytest.fixture
def sharp_images_dir(tmp_path):
    """Create a directory of images with sharp edges (rectangles + lines)."""
    d = tmp_path / "sharp"
    d.mkdir()
    for i in range(5):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img, (10 + i, 10 + i), (90 - i, 90 - i), (255, 255, 255), 2)
        cv2.line(img, (0, 50), (100, 50), (128, 128, 128), 1)
        cv2.imwrite(str(d / f"sharp_{i}.jpg"), img)
    return str(d)


@pytest.fixture
def blurry_images_dir(tmp_path):
    """Create a directory of heavily blurred images."""
    d = tmp_path / "blurry"
    d.mkdir()
    for i in range(5):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img, (10, 10), (90, 90), (255, 255, 255), 2)
        img = cv2.GaussianBlur(img, (21, 21), 5.0)
        cv2.imwrite(str(d / f"blurry_{i}.jpg"), img)
    return str(d)
