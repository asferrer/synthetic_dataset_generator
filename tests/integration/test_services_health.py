"""
Tests for service health checks.

These tests verify that all services respond correctly to health checks.
"""
import pytest
import requests
import time


# Service configuration
SERVICES_CONFIG = {
    "gateway": {
        "url": "http://localhost:8000/health",
        "startup_timeout": 30,
        "name": "Gateway"
    },
    "depth": {
        "url": "http://localhost:8001/health",
        "startup_timeout": 180,  # Model loading takes time
        "name": "Depth Estimation"
    },
    "segmentation": {
        "url": "http://localhost:8002/health",
        "startup_timeout": 240,  # Largest model
        "name": "Segmentation"
    },
    "effects": {
        "url": "http://localhost:8003/health",
        "startup_timeout": 30,
        "name": "Effects"
    },
    "augmentor": {
        "url": "http://localhost:8004/health",
        "startup_timeout": 60,
        "name": "Augmentor"
    },
}


@pytest.mark.integration
class TestServicesHealth:
    """Tests for service health endpoints."""

    @pytest.mark.parametrize("service_key,config", SERVICES_CONFIG.items())
    def test_service_health_endpoint(self, docker_services, service_key: str, config: dict):
        """Verify each service responds to health check."""
        url = config["url"]
        max_retries = 30
        retry_interval = config["startup_timeout"] / max_retries

        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    # Check for healthy status (different services may use different fields)
                    is_healthy = (
                        data.get("status") == "healthy" or
                        data.get("healthy") is True or
                        data.get("status") == "ok"
                    )
                    assert is_healthy, f"Service {service_key} not healthy: {data}"
                    return
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    pytest.fail(f"Service {service_key} not reachable after {max_retries} attempts: {e}")
            time.sleep(retry_interval)

        pytest.fail(f"Service {service_key} did not become healthy")

    def test_gateway_aggregated_health(self, docker_services, gateway_url):
        """Verify gateway reports aggregated health of all services."""
        response = requests.get(f"{gateway_url}/health", timeout=30)
        assert response.status_code == 200

        data = response.json()

        # Gateway should report overall status
        assert "status" in data, "Gateway health should include overall status"

        # Gateway should report downstream service statuses
        if "services" in data:
            services = data["services"]
            # Verify structure of service health reports
            for service in services:
                if isinstance(service, dict):
                    assert "name" in service or "service" in service, "Service health should include name"

    def test_all_services_reachable_from_gateway(self, docker_services, gateway_url):
        """Verify gateway can communicate with all downstream services."""
        # Check depth service via gateway
        response = requests.get(f"{gateway_url}/services/depth", timeout=30)
        assert response.status_code == 200
        data = response.json()
        assert "service" in data and data["service"] == "depth"

        # Check effects service via gateway
        response = requests.get(f"{gateway_url}/services/effects", timeout=30)
        assert response.status_code == 200
        data = response.json()
        assert "service" in data and data["service"] == "effects"

        # Check augmentor service via gateway
        response = requests.get(f"{gateway_url}/services/augmentor", timeout=30)
        assert response.status_code == 200
        data = response.json()
        assert "service" in data and data["service"] == "augmentor"

    @pytest.mark.parametrize("service_key,config", SERVICES_CONFIG.items())
    def test_service_returns_json(self, docker_services, service_key: str, config: dict):
        """Verify health endpoints return valid JSON."""
        url = config["url"]

        response = requests.get(url, timeout=30)
        assert response.status_code == 200

        # Should return JSON
        content_type = response.headers.get("content-type", "")
        assert "application/json" in content_type, f"Expected JSON, got {content_type}"

        # Should be valid JSON
        try:
            data = response.json()
            assert isinstance(data, dict), "Response should be a JSON object"
        except ValueError as e:
            pytest.fail(f"Invalid JSON response from {service_key}: {e}")

    def test_services_restart_gracefully(self, docker_services, gateway_url):
        """Verify services handle restart gracefully."""
        import subprocess
        from pathlib import Path

        compose_file = Path(__file__).parent.parent.parent / "docker-compose.microservices.yml"

        # Restart effects service (quick to restart)
        subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "restart", "effects"],
            capture_output=True,
            timeout=60
        )

        # Wait for service to be back
        max_retries = 20
        for _ in range(max_retries):
            try:
                response = requests.get("http://localhost:8003/health", timeout=5)
                if response.status_code == 200:
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)

        pytest.fail("Effects service did not recover after restart")
