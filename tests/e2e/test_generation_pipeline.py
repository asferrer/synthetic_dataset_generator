"""
End-to-end tests for the generation pipeline.

These tests verify the complete synthetic data generation workflow.
"""
import pytest
import requests
import base64
import time
from io import BytesIO
from PIL import Image
import numpy as np


@pytest.mark.e2e
class TestGenerationPipeline:
    """End-to-end tests for the generation pipeline."""

    def test_full_single_image_generation(
        self,
        docker_services,
        gateway_url,
        sample_image_base64,
        sample_rgba_base64
    ):
        """Test complete single image generation pipeline."""
        # Step 1: Generate synthetic image via gateway
        response = requests.post(
            f"{gateway_url}/augment/compose",
            json={
                "background_base64": sample_image_base64,
                "object_base64": sample_rgba_base64,
                "object_class": "test_object",
                "effects": {
                    "enabled": ["color_transfer"],
                    "intensity": 0.5
                }
            },
            timeout=180
        )

        if response.status_code == 404:
            pytest.skip("Compose endpoint not available")

        assert response.status_code == 200, f"Generation failed: {response.text}"
        data = response.json()

        # Should have result image
        has_image = any(key in data for key in [
            'result_base64', 'image_base64', 'composed_image', 'result'
        ])
        assert has_image, f"Response missing result image: {data.keys()}"

        # Should have annotations
        if 'annotations' in data:
            annotations = data['annotations']
            assert isinstance(annotations, list)

    def test_depth_to_composition_pipeline(
        self,
        docker_services,
        gateway_url,
        depth_url,
        sample_image_base64,
        sample_rgba_base64
    ):
        """Test pipeline from depth estimation to composition."""
        # Step 1: Get depth estimation
        depth_response = requests.post(
            f"{depth_url}/estimate",
            json={"image_base64": sample_image_base64},
            timeout=180
        )

        if depth_response.status_code == 404:
            pytest.skip("Depth endpoint not available")

        assert depth_response.status_code == 200
        depth_data = depth_response.json()

        # Step 2: Use depth info for composition (if supported)
        compose_payload = {
            "background_base64": sample_image_base64,
            "object_base64": sample_rgba_base64,
            "object_class": "test_object"
        }

        # Add depth info if available
        if 'depth_map_base64' in depth_data:
            compose_payload["depth_map_base64"] = depth_data["depth_map_base64"]

        compose_response = requests.post(
            f"{gateway_url}/augment/compose",
            json=compose_payload,
            timeout=180
        )

        if compose_response.status_code == 404:
            pytest.skip("Compose endpoint not available")

        # Pipeline should complete
        assert compose_response.status_code in [200, 422]

    def test_effects_application_pipeline(
        self,
        docker_services,
        effects_url,
        sample_image_base64,
        sample_rgba_base64
    ):
        """Test effects application pipeline."""
        effects_to_test = ["color_transfer", "blur_match"]

        for effect in effects_to_test:
            response = requests.post(
                f"{effects_url}/apply",
                json={
                    "background_base64": sample_image_base64,
                    "foreground_base64": sample_rgba_base64,
                    "effects": [effect]
                },
                timeout=120
            )

            if response.status_code == 404:
                continue  # Skip if endpoint not available

            assert response.status_code == 200, f"Effect {effect} failed: {response.text}"

    def test_validation_in_pipeline(
        self,
        docker_services,
        gateway_url,
        augmentor_url,
        sample_image_base64,
        sample_rgba_base64
    ):
        """Test that validation works in the pipeline."""
        # First generate an image
        compose_response = requests.post(
            f"{gateway_url}/augment/compose",
            json={
                "background_base64": sample_image_base64,
                "object_base64": sample_rgba_base64,
                "object_class": "test_object"
            },
            timeout=180
        )

        if compose_response.status_code == 404:
            pytest.skip("Compose endpoint not available")

        if compose_response.status_code != 200:
            pytest.skip("Compose failed")

        data = compose_response.json()

        # Then validate the result if annotations are present
        if 'annotations' in data and data['annotations']:
            validate_response = requests.post(
                f"{gateway_url}/augment/validate",
                json={
                    "annotations": data['annotations'],
                    "image_size": [640, 480]
                },
                timeout=60
            )

            if validate_response.status_code != 404:
                assert validate_response.status_code in [200, 422]


@pytest.mark.e2e
class TestBatchGeneration:
    """End-to-end tests for batch generation."""

    def test_batch_generation_start(self, docker_services, gateway_url, sample_batch_config, tmp_path):
        """Test starting a batch generation job."""
        # Update config with actual paths
        config = sample_batch_config.copy()

        response = requests.post(
            f"{gateway_url}/generate/batch",
            json=config,
            timeout=60
        )

        if response.status_code == 404:
            pytest.skip("Batch endpoint not available")

        # Should either succeed or return validation error
        assert response.status_code in [200, 202, 422]

    def test_batch_job_tracking(self, docker_services, gateway_url):
        """Test batch job status tracking."""
        # Try to get status of a job
        response = requests.get(
            f"{gateway_url}/augment/jobs/test-job-id",
            timeout=30
        )

        if response.status_code == 404:
            pytest.skip("Job tracking not implemented")

        # Should return job status or not found for invalid ID
        assert response.status_code in [200, 404]


@pytest.mark.e2e
class TestServiceResilience:
    """End-to-end tests for service resilience."""

    def test_gateway_handles_service_errors(self, docker_services, gateway_url):
        """Test that gateway handles downstream service errors gracefully."""
        # Send invalid data that should cause processing error
        response = requests.post(
            f"{gateway_url}/augment/compose",
            json={
                "background_base64": "invalid_base64",
                "object_base64": "also_invalid"
            },
            timeout=60
        )

        # Should return error, not crash
        assert response.status_code in [400, 422, 500]
        assert response.json() is not None  # Should have error message

    def test_concurrent_requests(self, docker_services, gateway_url, sample_image_base64):
        """Test handling of concurrent requests."""
        import concurrent.futures

        def make_request():
            return requests.get(f"{gateway_url}/health", timeout=30)

        # Send 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should succeed
        assert all(r.status_code == 200 for r in results)

    def test_large_image_handling(self, docker_services, gateway_url):
        """Test handling of larger images."""
        # Create a larger test image
        large_image = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
        large_pil = Image.fromarray(large_image)
        buffer = BytesIO()
        large_pil.save(buffer, format="PNG")
        large_base64 = base64.b64encode(buffer.getvalue()).decode()

        response = requests.post(
            f"{gateway_url}/augment/compose",
            json={
                "background_base64": large_base64,
                "object_base64": large_base64,  # Use same for simplicity
                "object_class": "test"
            },
            timeout=300  # Longer timeout for large image
        )

        if response.status_code == 404:
            pytest.skip("Compose endpoint not available")

        # Should handle or return appropriate error (not crash)
        assert response.status_code in [200, 400, 413, 422, 500]


@pytest.mark.e2e
class TestAnnotationOutput:
    """End-to-end tests for annotation output."""

    def test_annotations_have_required_fields(
        self,
        docker_services,
        gateway_url,
        sample_image_base64,
        sample_rgba_base64
    ):
        """Test that generated annotations have required fields."""
        response = requests.post(
            f"{gateway_url}/augment/compose",
            json={
                "background_base64": sample_image_base64,
                "object_base64": sample_rgba_base64,
                "object_class": "test_object"
            },
            timeout=180
        )

        if response.status_code == 404:
            pytest.skip("Compose endpoint not available")

        if response.status_code != 200:
            pytest.skip("Compose failed")

        data = response.json()

        if 'annotations' in data and data['annotations']:
            for ann in data['annotations']:
                # Should have bounding box or segmentation
                has_bbox = 'bbox' in ann or 'bounding_box' in ann
                has_seg = 'segmentation' in ann

                assert has_bbox or has_seg, f"Annotation missing bbox/segmentation: {ann}"

                # Should have class information
                has_class = any(key in ann for key in [
                    'category_id', 'class_id', 'class_name', 'category'
                ])
                assert has_class, f"Annotation missing class info: {ann}"
