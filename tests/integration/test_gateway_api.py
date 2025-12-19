"""
Tests for Gateway API endpoints.

These tests verify the API Gateway functionality.
"""
import pytest
import requests
import base64
from io import BytesIO
from PIL import Image
import numpy as np


@pytest.mark.integration
class TestGatewayAPI:
    """Tests for Gateway API endpoints."""

    def test_root_endpoint(self, docker_services, gateway_url):
        """Test root endpoint returns service info."""
        response = requests.get(f"{gateway_url}/", timeout=30)
        assert response.status_code == 200

        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "endpoints" in data

    def test_info_endpoint(self, docker_services, gateway_url):
        """Test GET /info endpoint."""
        response = requests.get(f"{gateway_url}/info", timeout=30)
        assert response.status_code == 200

        data = response.json()
        assert "available_services" in data
        assert "endpoints" in data

        # Check expected services are listed
        services = data["available_services"]
        assert "depth" in services
        assert "effects" in services
        assert "augmentor" in services

    def test_health_endpoint_structure(self, docker_services, gateway_url):
        """Test GET /health endpoint returns expected structure."""
        response = requests.get(f"{gateway_url}/health", timeout=30)
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "services" in data
        assert "all_healthy" in data

    def test_depth_service_info(self, docker_services, gateway_url):
        """Test GET /services/depth endpoint."""
        response = requests.get(f"{gateway_url}/services/depth", timeout=30)
        assert response.status_code == 200

        data = response.json()
        assert data["service"] == "depth"
        assert "health" in data

    def test_effects_service_info(self, docker_services, gateway_url):
        """Test GET /services/effects endpoint."""
        response = requests.get(f"{gateway_url}/services/effects", timeout=30)
        assert response.status_code == 200

        data = response.json()
        assert data["service"] == "effects"
        assert "health" in data

    def test_augmentor_service_info(self, docker_services, gateway_url):
        """Test GET /services/augmentor endpoint."""
        response = requests.get(f"{gateway_url}/services/augmentor", timeout=30)
        assert response.status_code == 200

        data = response.json()
        assert data["service"] == "augmentor"
        assert "health" in data


@pytest.mark.integration
class TestGatewayGeneration:
    """Tests for Gateway image generation endpoints."""

    def test_generate_image_endpoint_exists(self, docker_services, gateway_url):
        """Test POST /generate/image endpoint exists."""
        # Send minimal request to check endpoint exists
        response = requests.post(
            f"{gateway_url}/generate/image",
            json={},
            timeout=30
        )

        # Should get 422 (validation error) not 404
        assert response.status_code != 404, "Endpoint /generate/image should exist"

    def test_generate_batch_endpoint_exists(self, docker_services, gateway_url):
        """Test POST /generate/batch endpoint exists."""
        response = requests.post(
            f"{gateway_url}/generate/batch",
            json={},
            timeout=30
        )

        # Should get 422 (validation error) not 404
        assert response.status_code != 404, "Endpoint /generate/batch should exist"

    def test_invalid_request_returns_validation_error(self, docker_services, gateway_url):
        """Test that invalid requests return proper validation errors."""
        response = requests.post(
            f"{gateway_url}/generate/image",
            json={"invalid_field": "value"},
            timeout=30
        )

        assert response.status_code == 422, "Should return validation error"
        data = response.json()
        assert "detail" in data


@pytest.mark.integration
class TestGatewayAugmentRouter:
    """Tests for Gateway augment router endpoints."""

    def test_augment_compose_endpoint_exists(self, docker_services, gateway_url):
        """Test POST /augment/compose endpoint exists."""
        response = requests.post(
            f"{gateway_url}/augment/compose",
            json={},
            timeout=30
        )

        # Should not be 404
        assert response.status_code != 404, "Endpoint /augment/compose should exist"

    def test_augment_validate_endpoint_exists(self, docker_services, gateway_url):
        """Test POST /augment/validate endpoint exists."""
        response = requests.post(
            f"{gateway_url}/augment/validate",
            json={},
            timeout=30
        )

        assert response.status_code != 404, "Endpoint /augment/validate should exist"

    def test_augment_lighting_endpoint_exists(self, docker_services, gateway_url):
        """Test POST /augment/lighting endpoint exists."""
        response = requests.post(
            f"{gateway_url}/augment/lighting",
            json={},
            timeout=30
        )

        assert response.status_code != 404, "Endpoint /augment/lighting should exist"


@pytest.mark.integration
class TestGatewayErrorHandling:
    """Tests for Gateway error handling."""

    def test_not_found_returns_404(self, docker_services, gateway_url):
        """Test non-existent endpoint returns 404."""
        response = requests.get(f"{gateway_url}/nonexistent", timeout=30)
        assert response.status_code == 404

    def test_method_not_allowed_returns_405(self, docker_services, gateway_url):
        """Test wrong HTTP method returns 405."""
        # POST to an endpoint that only accepts GET
        response = requests.post(f"{gateway_url}/info", timeout=30)
        assert response.status_code == 405

    def test_malformed_json_returns_422(self, docker_services, gateway_url):
        """Test malformed JSON returns 422."""
        response = requests.post(
            f"{gateway_url}/generate/image",
            data="not valid json",
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        assert response.status_code == 422
