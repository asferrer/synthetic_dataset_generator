"""
API Client for Gateway Communication
=====================================
Centralized HTTP client for all Gateway API calls.
"""

import os
import json
import httpx
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8000")


@dataclass
class ServiceHealth:
    """Service health status"""
    name: str
    status: str
    url: str
    latency_ms: Optional[float] = None
    healthy: bool = False


class APIClient:
    """Client for Gateway API communication"""

    def __init__(self, base_url: str = None):
        self.base_url = base_url or GATEWAY_URL
        self.timeout = httpx.Timeout(120.0, connect=10.0)

    # =========================================================================
    # Health & Info
    # =========================================================================

    def get_health(self) -> Dict[str, Any]:
        """Get health status of all services"""
        try:
            response = httpx.get(
                f"{self.base_url}/health",
                timeout=10.0
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            return {"error": str(e), "status": "unhealthy", "services": []}
        return {"status": "unhealthy", "services": []}

    def get_service_info(self, service: str) -> Dict[str, Any]:
        """Get info for a specific service"""
        try:
            response = httpx.get(
                f"{self.base_url}/services/{service}",
                timeout=10.0
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            return {"error": str(e)}
        return {}

    # =========================================================================
    # Augmentor Endpoints
    # =========================================================================

    def compose_image(
        self,
        background_path: str,
        objects: List[Dict],
        effects: List[str],
        effects_config: Dict,
        output_path: str,
        validate_quality: bool = False,
        validate_physics: bool = False,
    ) -> Dict[str, Any]:
        """Compose a single synthetic image"""
        try:
            response = httpx.post(
                f"{self.base_url}/augment/compose",
                json={
                    "background_path": background_path,
                    "objects": objects,
                    "effects": effects,
                    "effects_config": effects_config,
                    "output_path": output_path,
                    "validate_quality": validate_quality,
                    "validate_physics": validate_physics,
                    "save_annotations": True,
                },
                timeout=self.timeout,
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def compose_batch(
        self,
        backgrounds_dir: str,
        objects_dir: str,
        output_dir: str,
        num_images: int,
        targets_per_class: Optional[Dict[str, int]] = None,
        max_objects_per_image: int = 5,
        effects: List[str] = None,
        effects_config: Dict = None,
        depth_aware: bool = True,
        validate_quality: bool = False,
        validate_physics: bool = False,
        save_pipeline_debug: bool = False,
    ) -> Dict[str, Any]:
        """Start a batch composition job"""
        try:
            response = httpx.post(
                f"{self.base_url}/augment/compose-batch",
                json={
                    "backgrounds_dir": backgrounds_dir,
                    "objects_dir": objects_dir,
                    "output_dir": output_dir,
                    "num_images": num_images,
                    "targets_per_class": targets_per_class,
                    "max_objects_per_image": max_objects_per_image,
                    "effects": effects or ["color_correction", "blur_matching", "caustics"],
                    "effects_config": effects_config or {},
                    "depth_aware": depth_aware,
                    "validate_quality": validate_quality,
                    "validate_physics": validate_physics,
                    "reject_invalid": True,
                    "save_pipeline_debug": save_pipeline_debug,
                },
                timeout=self.timeout,
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a batch job"""
        try:
            response = httpx.get(
                f"{self.base_url}/augment/jobs/{job_id}",
                timeout=10.0,
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            return {"error": str(e)}
        return {}

    def list_jobs(self) -> Dict[str, Any]:
        """List all batch jobs"""
        try:
            response = httpx.get(
                f"{self.base_url}/augment/jobs",
                timeout=10.0,
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"jobs": [], "total": 0, "error": f"Status {response.status_code}"}
        except Exception as e:
            return {"jobs": [], "total": 0, "error": str(e)}

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel a running batch job"""
        try:
            response = httpx.delete(
                f"{self.base_url}/augment/jobs/{job_id}",
                timeout=10.0,
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": f"Status {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def validate_image(
        self,
        image_path: str,
        annotations: List[Dict] = None,
        check_quality: bool = True,
        check_physics: bool = True,
    ) -> Dict[str, Any]:
        """Validate an image"""
        try:
            response = httpx.post(
                f"{self.base_url}/augment/validate",
                json={
                    "image_path": image_path,
                    "annotations": annotations or [],
                    "check_quality": check_quality,
                    "check_anomalies": True,
                    "check_physics": check_physics,
                },
                timeout=60.0,
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            return {"error": str(e)}
        return {}

    def estimate_lighting(self, image_path: str) -> Dict[str, Any]:
        """Estimate lighting from an image"""
        try:
            response = httpx.post(
                f"{self.base_url}/augment/lighting",
                json={
                    "image_path": image_path,
                    "max_light_sources": 3,
                    "intensity_threshold": 0.6,
                },
                timeout=30.0,
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            return {"error": str(e)}
        return {}

    # =========================================================================
    # Legacy Generate Endpoints (for backward compatibility)
    # =========================================================================

    def generate_image(
        self,
        background_path: str,
        objects: List[Dict],
        config: Dict,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a synthetic image (legacy endpoint)"""
        try:
            response = httpx.post(
                f"{self.base_url}/generate/image",
                json={
                    "background_path": background_path,
                    "objects": objects,
                    "config": config,
                    "output_path": output_path,
                },
                timeout=self.timeout,
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}


# Global client instance
_client: Optional[APIClient] = None


def get_api_client() -> APIClient:
    """Get singleton API client"""
    global _client
    if _client is None:
        _client = APIClient()
    return _client
