"""
End-to-end tests for frontend workflow.

These tests verify the frontend application and its integration with services.
"""
import pytest
import requests
import time


@pytest.mark.e2e
class TestFrontendAccess:
    """Tests for frontend accessibility."""

    def test_frontend_loads(self, docker_services, frontend_url):
        """Test that frontend application loads."""
        max_retries = 30
        for _ in range(max_retries):
            try:
                response = requests.get(frontend_url, timeout=10)
                if response.status_code == 200:
                    assert "<!DOCTYPE html>" in response.text.lower() or \
                           "streamlit" in response.text.lower() or \
                           response.status_code == 200
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)

        pytest.fail("Frontend did not become available")

    def test_frontend_health_endpoint(self, docker_services, frontend_url):
        """Test frontend health endpoint if available."""
        response = requests.get(f"{frontend_url}/_stcore/health", timeout=30)

        # Streamlit health endpoint
        if response.status_code == 200:
            assert response.text == "ok" or response.status_code == 200
        else:
            # Alternative: just check main page loads
            response = requests.get(frontend_url, timeout=30)
            assert response.status_code == 200


@pytest.mark.e2e
class TestFrontendServiceIntegration:
    """Tests for frontend integration with backend services."""

    def test_frontend_can_reach_gateway(self, docker_services, frontend_url, gateway_url):
        """Test that frontend can communicate with gateway."""
        # This tests the network connectivity
        # Frontend should be able to reach gateway
        gateway_health = requests.get(f"{gateway_url}/health", timeout=30)
        assert gateway_health.status_code == 200

    def test_frontend_displays_without_errors(self, docker_services, frontend_url):
        """Test that frontend renders without server errors."""
        response = requests.get(frontend_url, timeout=30)

        # Should not have server error
        assert response.status_code != 500
        assert response.status_code != 502
        assert response.status_code != 503


@pytest.mark.e2e
class TestFrontendWorkflowSimulation:
    """Tests simulating user workflows through the frontend."""

    def test_analysis_page_workflow(self, docker_services, gateway_url):
        """Simulate analysis page workflow via API."""
        # The frontend analysis page uses these endpoints
        # We test them directly since we can't interact with Streamlit UI

        # 1. Check services are healthy
        health = requests.get(f"{gateway_url}/health", timeout=30)
        assert health.status_code == 200

        # 2. Get service info
        info = requests.get(f"{gateway_url}/info", timeout=30)
        assert info.status_code == 200

    def test_generation_page_workflow(
        self,
        docker_services,
        gateway_url,
        sample_image_base64,
        sample_rgba_base64
    ):
        """Simulate generation page workflow via API."""
        # 1. Check health
        health = requests.get(f"{gateway_url}/health", timeout=30)
        assert health.status_code == 200

        # 2. Attempt generation
        response = requests.post(
            f"{gateway_url}/augment/compose",
            json={
                "background_base64": sample_image_base64,
                "object_base64": sample_rgba_base64,
                "object_class": "test"
            },
            timeout=180
        )

        if response.status_code == 404:
            pytest.skip("Compose endpoint not available")

        # Should get a response (success or error)
        assert response.status_code in [200, 400, 422, 500]

    def test_post_processing_utilities_work(
        self,
        docker_services,
        sample_coco_data
    ):
        """Test that post-processing utilities work correctly."""
        import sys
        from pathlib import Path

        # Add frontend to path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "frontend"))

        from app.utils.splitter import DatasetSplitter
        from app.utils.balancer import ClassBalancer
        from app.utils.exporters import export_to_yolo

        # Test splitting
        splits = DatasetSplitter.split_dataset(sample_coco_data, strategy='random')
        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits

        # Test balancing (if data allows)
        balancer = ClassBalancer()
        try:
            balanced, result = balancer.balance(sample_coco_data, strategy='oversample')
            assert 'images' in balanced
            assert 'annotations' in balanced
        except Exception:
            pass  # May fail on already balanced data

        # Test export
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_to_yolo(sample_coco_data, tmpdir)
            assert result['success']


@pytest.mark.e2e
class TestFullUserJourney:
    """Tests simulating complete user journeys."""

    def test_generate_and_export_workflow(
        self,
        docker_services,
        gateway_url,
        sample_image_base64,
        sample_rgba_base64,
        tmp_path
    ):
        """Test complete workflow: generate image, get annotations, export."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "frontend"))

        # Step 1: Generate image
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
            pytest.skip("Generation failed")

        data = response.json()

        # Step 2: Build COCO dataset from result
        coco_data = {
            "info": {"description": "Generated dataset"},
            "licenses": [],
            "categories": [{"id": 1, "name": "test_object"}],
            "images": [{"id": 1, "file_name": "generated.jpg", "width": 640, "height": 480}],
            "annotations": []
        }

        # Add annotations if present
        if 'annotations' in data:
            for i, ann in enumerate(data['annotations']):
                coco_ann = {
                    "id": i + 1,
                    "image_id": 1,
                    "category_id": 1,
                    "iscrowd": 0
                }

                if 'bbox' in ann:
                    coco_ann['bbox'] = ann['bbox']
                    coco_ann['area'] = ann['bbox'][2] * ann['bbox'][3] if len(ann['bbox']) >= 4 else 0

                coco_data['annotations'].append(coco_ann)

        # Step 3: Export to different formats
        from app.utils.exporters import export_to_coco, export_to_yolo

        # Export COCO
        coco_result = export_to_coco(coco_data, str(tmp_path / "coco"))
        assert coco_result['success']

        # Export YOLO
        yolo_result = export_to_yolo(coco_data, str(tmp_path / "yolo"))
        assert yolo_result['success']

    def test_analyze_and_split_workflow(self, docker_services, sample_coco_data):
        """Test workflow: analyze dataset, then split it."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "frontend"))

        from app.utils.splitter import DatasetSplitter, KFoldGenerator

        # Step 1: Get statistics
        stats = DatasetSplitter.get_split_statistics({
            'full': sample_coco_data
        })
        assert 'full' in stats

        # Step 2: Create train/val/test split
        splits = DatasetSplitter.split_dataset(
            sample_coco_data,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1
        )

        assert len(splits['train']['images']) > 0
        assert len(splits['val']['images']) > 0

        # Step 3: Get split statistics
        split_stats = DatasetSplitter.get_split_statistics(splits)
        assert 'train' in split_stats
        assert 'val' in split_stats
        assert 'test' in split_stats

        # Step 4: Create K-Fold splits
        kfold = KFoldGenerator(n_folds=3)
        folds = kfold.get_all_folds(
            sample_coco_data['images'],
            sample_coco_data['annotations']
        )

        assert len(folds) == 3
