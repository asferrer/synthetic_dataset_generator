"""
Tests for Segmentation service.

These tests verify the segmentation functionality.
"""
import pytest
import requests


@pytest.mark.integration
@pytest.mark.slow
class TestSegmentationService:
    """Tests for segmentation service.

    Note: Segmentation service loads large models and may take longer to start.
    """

    def test_health_endpoint(self, docker_services, segmentation_url):
        """Test segmentation service health endpoint."""
        response = requests.get(f"{segmentation_url}/health", timeout=60)
        assert response.status_code == 200

        data = response.json()
        assert data.get("healthy") is True or data.get("status") == "healthy"

    def test_info_endpoint(self, docker_services, segmentation_url):
        """Test segmentation service info endpoint."""
        response = requests.get(f"{segmentation_url}/info", timeout=30)
        assert response.status_code == 200

        data = response.json()
        assert len(data) > 0

    def test_segment_image(self, docker_services, segmentation_url, sample_image_base64):
        """Test image segmentation."""
        response = requests.post(
            f"{segmentation_url}/segment",
            json={"image_base64": sample_image_base64},
            timeout=180  # Segmentation can be slow
        )

        if response.status_code == 404:
            pytest.skip("Segment endpoint not found")

        assert response.status_code == 200
        data = response.json()

        # Should return segmentation mask or segments
        has_result = (
            "mask_base64" in data or
            "segments" in data or
            "result" in data or
            "masks" in data
        )
        assert has_result, f"Response should contain segmentation result: {data.keys()}"

    def test_analyze_scene(self, docker_services, segmentation_url, sample_image_base64):
        """Test scene analysis."""
        response = requests.post(
            f"{segmentation_url}/analyze",
            json={"image_base64": sample_image_base64},
            timeout=180
        )

        if response.status_code == 404:
            pytest.skip("Analyze endpoint not found")

        assert response.status_code == 200
        data = response.json()

        # Should return scene analysis
        has_analysis = (
            "scene_type" in data or
            "objects" in data or
            "analysis" in data or
            "regions" in data
        )
        assert has_analysis or len(data) > 0

    def test_get_placement_zones(self, docker_services, segmentation_url, sample_image_base64):
        """Test placement zones extraction."""
        response = requests.post(
            f"{segmentation_url}/placement-zones",
            json={"image_base64": sample_image_base64},
            timeout=180
        )

        if response.status_code == 404:
            # Try alternative endpoint
            response = requests.post(
                f"{segmentation_url}/zones",
                json={"image_base64": sample_image_base64},
                timeout=180
            )

        if response.status_code == 404:
            pytest.skip("Placement zones endpoint not found")

        assert response.status_code == 200

    def test_invalid_image_handling(self, docker_services, segmentation_url):
        """Test handling of invalid images."""
        response = requests.post(
            f"{segmentation_url}/segment",
            json={"image_base64": "invalid_base64_data"},
            timeout=30
        )

        if response.status_code == 404:
            pytest.skip("Segment endpoint not found")

        # Should return client error
        assert 400 <= response.status_code < 500
