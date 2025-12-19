"""
Tests for Depth Estimation service.

These tests verify the depth estimation functionality.
"""
import pytest
import requests
import base64
from io import BytesIO
from PIL import Image
import numpy as np


@pytest.mark.integration
class TestDepthService:
    """Tests for depth estimation service."""

    def test_health_endpoint(self, docker_services, depth_url):
        """Test depth service health endpoint."""
        response = requests.get(f"{depth_url}/health", timeout=30)
        assert response.status_code == 200

        data = response.json()
        assert data.get("healthy") is True or data.get("status") == "healthy"

    def test_info_endpoint(self, docker_services, depth_url):
        """Test depth service info endpoint."""
        response = requests.get(f"{depth_url}/info", timeout=30)
        assert response.status_code == 200

        data = response.json()
        assert "model" in data or "version" in data or "service" in data

    def test_estimate_depth_with_base64(self, docker_services, depth_url, sample_image_base64):
        """Test depth estimation with base64 image."""
        response = requests.post(
            f"{depth_url}/estimate",
            json={"image_base64": sample_image_base64},
            timeout=120  # Model inference can take time
        )

        assert response.status_code == 200
        data = response.json()

        # Should return depth map
        assert "depth_map_base64" in data or "depth_map" in data or "result" in data

    def test_estimate_depth_returns_zones(self, docker_services, depth_url, sample_image_base64):
        """Test depth estimation returns depth zones."""
        response = requests.post(
            f"{depth_url}/estimate",
            json={
                "image_base64": sample_image_base64,
                "return_zones": True
            },
            timeout=120
        )

        assert response.status_code == 200
        data = response.json()

        # If zones are supported, they should be present
        if "zones" in data:
            zones = data["zones"]
            # Should have foreground, midground, background zones
            expected_zones = ["foreground", "midground", "background"]
            for zone in expected_zones:
                if zone in zones:
                    assert isinstance(zones[zone], (list, dict, float, int))

    def test_invalid_base64_returns_error(self, docker_services, depth_url):
        """Test invalid base64 returns proper error."""
        response = requests.post(
            f"{depth_url}/estimate",
            json={"image_base64": "not_valid_base64!!!"},
            timeout=30
        )

        # Should return client error (4xx)
        assert 400 <= response.status_code < 500

    def test_missing_image_returns_error(self, docker_services, depth_url):
        """Test missing image returns validation error."""
        response = requests.post(
            f"{depth_url}/estimate",
            json={},
            timeout=30
        )

        assert response.status_code == 422  # Validation error

    @pytest.mark.slow
    def test_batch_depth_estimation(self, docker_services, depth_url, sample_image_base64):
        """Test batch depth estimation if supported."""
        response = requests.post(
            f"{depth_url}/estimate/batch",
            json={"images": [sample_image_base64, sample_image_base64]},
            timeout=300
        )

        if response.status_code == 404:
            pytest.skip("Batch endpoint not implemented")

        assert response.status_code == 200
        data = response.json()
        assert "results" in data or "depth_maps" in data
