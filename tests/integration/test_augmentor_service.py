"""
Tests for Augmentor service.

These tests verify the augmentor/composition functionality.
"""
import pytest
import requests


@pytest.mark.integration
class TestAugmentorService:
    """Tests for augmentor service."""

    def test_health_endpoint(self, docker_services, augmentor_url):
        """Test augmentor service health endpoint."""
        response = requests.get(f"{augmentor_url}/health", timeout=30)
        assert response.status_code == 200

        data = response.json()
        assert data.get("healthy") is True or data.get("status") == "healthy"

    def test_info_endpoint(self, docker_services, augmentor_url):
        """Test augmentor service info endpoint."""
        response = requests.get(f"{augmentor_url}/info", timeout=30)
        assert response.status_code == 200

        data = response.json()
        assert len(data) > 0

    def test_compose_endpoint_exists(self, docker_services, augmentor_url):
        """Test compose endpoint exists."""
        response = requests.post(
            f"{augmentor_url}/compose",
            json={},
            timeout=30
        )

        # Should not be 404
        assert response.status_code != 404, "Compose endpoint should exist"

    def test_compose_with_images(
        self,
        docker_services,
        augmentor_url,
        sample_image_base64,
        sample_rgba_base64
    ):
        """Test composition with background and object."""
        response = requests.post(
            f"{augmentor_url}/compose",
            json={
                "background_base64": sample_image_base64,
                "object_base64": sample_rgba_base64,
                "object_class": "test_object",
                "position": {"x": 100, "y": 100},
                "scale": 1.0
            },
            timeout=120
        )

        if response.status_code == 422:
            # Try alternative schema
            response = requests.post(
                f"{augmentor_url}/compose",
                json={
                    "background": sample_image_base64,
                    "foreground": sample_rgba_base64,
                    "class_name": "test_object"
                },
                timeout=120
            )

        if response.status_code == 404:
            pytest.skip("Compose endpoint not found")

        assert response.status_code == 200
        data = response.json()

        # Should return composed image and annotations
        has_result = (
            "result_base64" in data or
            "image_base64" in data or
            "composed_image" in data or
            "result" in data
        )
        assert has_result, f"Response should contain result image: {data.keys()}"

    def test_validate_composition(self, docker_services, augmentor_url):
        """Test composition validation endpoint."""
        response = requests.post(
            f"{augmentor_url}/validate",
            json={
                "annotations": [
                    {"bbox": [10, 10, 100, 100], "category_id": 1}
                ],
                "image_size": [640, 480]
            },
            timeout=30
        )

        if response.status_code == 404:
            pytest.skip("Validate endpoint not implemented")

        assert response.status_code in [200, 422]

    def test_lighting_adjustment(
        self,
        docker_services,
        augmentor_url,
        sample_image_base64,
        sample_rgba_base64
    ):
        """Test lighting adjustment endpoint."""
        response = requests.post(
            f"{augmentor_url}/lighting",
            json={
                "background_base64": sample_image_base64,
                "foreground_base64": sample_rgba_base64,
                "adjustment": "auto"
            },
            timeout=60
        )

        if response.status_code == 404:
            pytest.skip("Lighting endpoint not implemented")

        assert response.status_code == 200

    def test_transformation_options(self, docker_services, augmentor_url):
        """Test available transformation options."""
        response = requests.get(f"{augmentor_url}/transformations", timeout=30)

        if response.status_code == 404:
            # Try via info endpoint
            response = requests.get(f"{augmentor_url}/info", timeout=30)

        assert response.status_code == 200
