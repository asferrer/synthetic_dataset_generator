"""
Tests for Docker image building.

These tests verify that all Docker images build successfully.
"""
import subprocess
import pytest
from pathlib import Path

# Services to test
SERVICES = ["gateway", "depth", "segmentation", "effects", "augmentor", "frontend"]

# Path to docker-compose file
COMPOSE_FILE = Path(__file__).parent.parent.parent / "docker-compose.microservices.yml"


@pytest.mark.integration
class TestDockerBuild:
    """Tests for Docker image building."""

    @pytest.mark.slow
    def test_build_all_services(self):
        """Verify that docker-compose build completes without errors."""
        result = subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "build"],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )

        assert result.returncode == 0, f"Build failed:\nSTDERR: {result.stderr}\nSTDOUT: {result.stdout}"

    @pytest.mark.parametrize("service", SERVICES)
    def test_build_individual_service(self, service: str):
        """Verify each service builds individually."""
        result = subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "build", service],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes per service
        )

        assert result.returncode == 0, f"Build {service} failed:\nSTDERR: {result.stderr}"

    def test_compose_config_valid(self):
        """Verify docker-compose configuration is valid."""
        result = subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "config"],
            capture_output=True,
            text=True,
            timeout=30
        )

        assert result.returncode == 0, f"Config invalid:\nSTDERR: {result.stderr}"

    @pytest.mark.parametrize("service", SERVICES)
    def test_dockerfile_exists(self, service: str):
        """Verify Dockerfile exists for each service."""
        if service == "frontend":
            dockerfile_path = COMPOSE_FILE.parent / "frontend" / "Dockerfile"
        else:
            dockerfile_path = COMPOSE_FILE.parent / "services" / service / "Dockerfile"

        assert dockerfile_path.exists(), f"Dockerfile not found for {service} at {dockerfile_path}"

    def test_env_file_exists(self):
        """Verify environment file exists."""
        env_file = COMPOSE_FILE.parent / ".env.microservices"
        env_example = COMPOSE_FILE.parent / ".env.microservices.example"

        # At least the example should exist
        assert env_example.exists() or env_file.exists(), "No environment file found"
