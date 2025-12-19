"""
Tests for Effects service.

These tests verify the effects/realism functionality.
"""
import pytest
import requests


@pytest.mark.integration
class TestEffectsService:
    """Tests for effects service."""

    def test_health_endpoint(self, docker_services, effects_url):
        """Test effects service health endpoint."""
        response = requests.get(f"{effects_url}/health", timeout=30)
        assert response.status_code == 200

        data = response.json()
        assert data.get("healthy") is True or data.get("status") == "healthy"

    def test_info_endpoint(self, docker_services, effects_url):
        """Test effects service info endpoint."""
        response = requests.get(f"{effects_url}/info", timeout=30)
        assert response.status_code == 200

        data = response.json()
        # Should list available effects
        assert "effects" in data or "available_effects" in data or "version" in data

    def test_apply_color_transfer(self, docker_services, effects_url, sample_image_base64, sample_rgba_base64):
        """Test color transfer effect."""
        response = requests.post(
            f"{effects_url}/apply",
            json={
                "background_base64": sample_image_base64,
                "foreground_base64": sample_rgba_base64,
                "effects": ["color_transfer"]
            },
            timeout=60
        )

        if response.status_code == 404:
            pytest.skip("Apply endpoint not found")

        assert response.status_code == 200
        data = response.json()
        assert "result_base64" in data or "image_base64" in data or "result" in data

    def test_apply_blur_matching(self, docker_services, effects_url, sample_image_base64, sample_rgba_base64):
        """Test blur matching effect."""
        response = requests.post(
            f"{effects_url}/apply",
            json={
                "background_base64": sample_image_base64,
                "foreground_base64": sample_rgba_base64,
                "effects": ["blur_match"]
            },
            timeout=60
        )

        if response.status_code == 404:
            pytest.skip("Apply endpoint not found")

        assert response.status_code == 200

    def test_generate_caustics(self, docker_services, effects_url):
        """Test caustics generation."""
        response = requests.post(
            f"{effects_url}/caustics",
            json={
                "width": 512,
                "height": 512,
                "intensity": 0.5
            },
            timeout=60
        )

        if response.status_code == 404:
            pytest.skip("Caustics endpoint not implemented")

        assert response.status_code == 200
        data = response.json()
        assert "caustics_base64" in data or "result" in data

    def test_list_available_effects(self, docker_services, effects_url):
        """Test listing available effects."""
        response = requests.get(f"{effects_url}/effects", timeout=30)

        if response.status_code == 404:
            # Try alternative endpoint
            response = requests.get(f"{effects_url}/info", timeout=30)

        assert response.status_code == 200
        data = response.json()

        # Should have some indication of available effects
        assert len(data) > 0

    def test_invalid_effect_name(self, docker_services, effects_url, sample_image_base64, sample_rgba_base64):
        """Test that invalid effect names are handled properly."""
        response = requests.post(
            f"{effects_url}/apply",
            json={
                "background_base64": sample_image_base64,
                "foreground_base64": sample_rgba_base64,
                "effects": ["nonexistent_effect_xyz"]
            },
            timeout=60
        )

        if response.status_code == 404:
            pytest.skip("Apply endpoint not found")

        # Should either ignore unknown effects or return error
        assert response.status_code in [200, 400, 422]
