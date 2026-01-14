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
SEGMENTATION_URL = os.environ.get("SEGMENTATION_SERVICE_URL", "http://localhost:8002")


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
        overlap_threshold: float = 0.1,
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
                    "overlap_threshold": overlap_threshold,
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

    def resume_job(self, job_id: str) -> Dict[str, Any]:
        """Resume an interrupted job from checkpoint"""
        try:
            response = httpx.post(
                f"{self.base_url}/augment/jobs/{job_id}/resume",
                json={},
                timeout=30.0,
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def retry_job(self, job_id: str) -> Dict[str, Any]:
        """Retry a failed job from scratch (creates new job)"""
        try:
            response = httpx.post(
                f"{self.base_url}/augment/jobs/{job_id}/retry",
                json={},
                timeout=30.0,
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_job_logs(self, job_id: str, level: str = None, limit: int = 100) -> Dict[str, Any]:
        """Get logs for a specific job"""
        try:
            params = {"limit": limit}
            if level:
                params["level"] = level

            response = httpx.get(
                f"{self.base_url}/augment/jobs/{job_id}/logs",
                params=params,
                timeout=30.0,
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"logs": [], "error": f"Status {response.status_code}"}
        except Exception as e:
            return {"logs": [], "error": str(e)}

    def list_datasets(
        self,
        dataset_type: Optional[str] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """List all available datasets with metadata"""
        try:
            params = {"limit": limit}
            if dataset_type:
                params["dataset_type"] = dataset_type

            response = httpx.get(
                f"{self.base_url}/augment/datasets",
                params=params,
                timeout=10.0
            )
            if response.status_code == 200:
                return response.json()
            return {"datasets": [], "total": 0, "error": response.text}
        except Exception as e:
            return {"datasets": [], "total": 0, "error": str(e)}

    def get_dataset_metadata(self, job_id: str) -> Dict[str, Any]:
        """Get metadata for a specific dataset"""
        try:
            response = httpx.get(
                f"{self.base_url}/augment/datasets/{job_id}",
                timeout=10.0
            )
            if response.status_code == 200:
                return response.json()
            return {"error": response.text}
        except Exception as e:
            return {"error": str(e)}

    def load_dataset_coco(self, job_id: str) -> Dict[str, Any]:
        """Load full COCO JSON for a dataset"""
        try:
            response = httpx.get(
                f"{self.base_url}/augment/datasets/{job_id}/coco",
                timeout=30.0  # Larger timeout for big files
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            return {"success": False, "error": response.text}
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


    # =========================================================================
    # Object Extraction Endpoints (Segmentation Service)
    # =========================================================================

    def analyze_dataset_annotations(
        self,
        coco_data: Optional[Dict[str, Any]] = None,
        coco_json_path: Optional[str] = None,
        images_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze a COCO dataset to determine annotation types"""
        try:
            response = httpx.post(
                f"{SEGMENTATION_URL}/extract/analyze-dataset",
                json={
                    "coco_data": coco_data,
                    "coco_json_path": coco_json_path,
                    "images_dir": images_dir,
                },
                timeout=120.0,  # Increased for large datasets
            )
            if response.status_code == 200:
                return response.json()
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text[:200]}"}
        except httpx.TimeoutException:
            return {"success": False, "error": f"Timeout analizando dataset en {SEGMENTATION_URL}"}
        except httpx.ConnectError:
            return {"success": False, "error": f"No se puede conectar a {SEGMENTATION_URL}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def extract_objects(
        self,
        images_dir: str,
        output_dir: str,
        coco_data: Optional[Dict[str, Any]] = None,
        coco_json_path: Optional[str] = None,
        categories_to_extract: Optional[List[str]] = None,
        use_sam3_for_bbox: bool = True,
        force_bbox_only: bool = False,
        force_sam3_resegmentation: bool = False,
        force_sam3_text_prompt: bool = False,
        padding: int = 5,
        min_object_area: int = 100,
        save_individual_coco: bool = True,
        deduplication: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract objects from a COCO dataset as transparent PNGs with deduplication support"""
        try:
            payload = {
                "coco_data": coco_data,
                "coco_json_path": coco_json_path,
                "images_dir": images_dir,
                "output_dir": output_dir,
                "categories_to_extract": categories_to_extract or [],
                "use_sam3_for_bbox": use_sam3_for_bbox,
                "force_bbox_only": force_bbox_only,
                "force_sam3_resegmentation": force_sam3_resegmentation,
                "force_sam3_text_prompt": force_sam3_text_prompt,
                "padding": padding,
                "min_object_area": min_object_area,
                "save_individual_coco": save_individual_coco,
            }

            # Add deduplication config if provided
            if deduplication is not None:
                payload["deduplication"] = deduplication

            response = httpx.post(
                f"{SEGMENTATION_URL}/extract/objects",
                json=payload,
                timeout=60.0,  # Should return quickly since it's async
            )
            if response.status_code == 200:
                return response.json()
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text[:200]}"}
        except httpx.TimeoutException:
            return {"success": False, "error": f"Timeout iniciando extraccion en {SEGMENTATION_URL}"}
        except httpx.ConnectError:
            return {"success": False, "error": f"No se puede conectar a {SEGMENTATION_URL}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def extract_custom_objects(
        self,
        images_dir: str,
        output_dir: str,
        object_names: List[str],
        padding: int = 5,
        min_object_area: int = 100,
        save_individual_coco: bool = True,
        deduplication: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract custom objects using text prompts (no COCO JSON required).

        Args:
            images_dir: Directory containing images
            output_dir: Output directory for extracted objects
            object_names: List of object names to segment
            padding: Pixels of padding around objects
            min_object_area: Minimum object area in pixels
            save_individual_coco: Save individual COCO JSON per object
            deduplication: Deduplication configuration dict

        Returns:
            Response with job_id and status
        """
        try:
            payload = {
                "images_dir": images_dir,
                "output_dir": output_dir,
                "object_names": object_names,
                "padding": padding,
                "min_object_area": min_object_area,
                "save_individual_coco": save_individual_coco,
            }

            # Add deduplication config if provided
            if deduplication is not None:
                payload["deduplication"] = deduplication

            response = httpx.post(
                f"{SEGMENTATION_URL}/extract/custom-objects",
                json=payload,
                timeout=60.0,  # Should return quickly since it's async
            )
            if response.status_code == 200:
                return response.json()
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text[:200]}"}
        except httpx.TimeoutException:
            return {"success": False, "error": f"Timeout iniciando extraccion custom en {SEGMENTATION_URL}"}
        except httpx.ConnectError:
            return {"success": False, "error": f"No se puede conectar a {SEGMENTATION_URL}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_extraction_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of an extraction job"""
        try:
            response = httpx.get(
                f"{SEGMENTATION_URL}/extract/jobs/{job_id}",
                timeout=10.0,
            )
            if response.status_code == 200:
                return response.json()
            return {"error": response.text}
        except Exception as e:
            return {"error": str(e)}

    def extract_single_object(
        self,
        annotation: Dict[str, Any],
        category_name: str,
        image_path: Optional[str] = None,
        image_base64: Optional[str] = None,
        use_sam3: bool = False,
        padding: int = 5,
        force_bbox_only: bool = False,
        force_sam3_resegmentation: bool = False,
        force_sam3_text_prompt: bool = False,
    ) -> Dict[str, Any]:
        """Extract a single object for preview"""
        try:
            response = httpx.post(
                f"{SEGMENTATION_URL}/extract/single-object",
                json={
                    "image_path": image_path,
                    "image_base64": image_base64,
                    "annotation": annotation,
                    "category_name": category_name,
                    "use_sam3": use_sam3,
                    "padding": padding,
                    "force_bbox_only": force_bbox_only,
                    "force_sam3_resegmentation": force_sam3_resegmentation,
                    "force_sam3_text_prompt": force_sam3_text_prompt,
                },
                timeout=60.0,
            )
            if response.status_code == 200:
                return response.json()
            return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def extract_from_imagenet(
        self,
        root_dir: str,
        output_dir: str,
        padding: int = 5,
        min_object_area: int = 100,
        max_objects_per_class: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Extract objects from ImageNet-style directory structure.

        Args:
            root_dir: Root directory with class subdirectories
            output_dir: Output directory for extracted objects
            padding: Padding around extracted objects
            min_object_area: Minimum object area filter
            max_objects_per_class: Limit objects per class (None=all)

        Returns:
            Dictionary with success status, job_id, and message
        """
        try:
            response = httpx.post(
                f"{SEGMENTATION_URL}/extract/imagenet",
                json={
                    "root_dir": root_dir,
                    "output_dir": output_dir,
                    "padding": padding,
                    "min_object_area": min_object_area,
                    "max_objects_per_class": max_objects_per_class,
                },
                timeout=60.0,  # Should return quickly since it's async
            )
            if response.status_code == 200:
                return response.json()
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text[:200]}"}
        except httpx.TimeoutException:
            return {"success": False, "error": f"Timeout iniciando extracción ImageNet en {SEGMENTATION_URL}"}
        except httpx.ConnectError:
            return {"success": False, "error": f"No se puede conectar a {SEGMENTATION_URL}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # SAM3 Tool Endpoints (Segmentation Service)
    # =========================================================================

    def sam3_segment_image(
        self,
        bbox: Optional[List[float]] = None,
        point: Optional[List[int]] = None,
        text_prompt: Optional[str] = None,
        image_path: Optional[str] = None,
        image_base64: Optional[str] = None,
        return_polygon: bool = True,
        return_mask: bool = True,
        simplify_polygon: bool = True,
        simplify_tolerance: float = 2.0,
    ) -> Dict[str, Any]:
        """Segment an image using SAM3 with box, point, or text prompt"""
        try:
            response = httpx.post(
                f"{SEGMENTATION_URL}/sam3/segment-image",
                json={
                    "image_path": image_path,
                    "image_base64": image_base64,
                    "bbox": bbox,
                    "point": point,
                    "text_prompt": text_prompt,
                    "return_polygon": return_polygon,
                    "return_mask": return_mask,
                    "simplify_polygon": simplify_polygon,
                    "simplify_tolerance": simplify_tolerance,
                },
                timeout=60.0,
            )
            if response.status_code == 200:
                return response.json()
            return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def sam3_convert_dataset(
        self,
        images_dir: str,
        output_path: str,
        coco_data: Optional[Dict[str, Any]] = None,
        coco_json_path: Optional[str] = None,
        categories_to_convert: Optional[List[str]] = None,
        overwrite_existing: bool = False,
        simplify_polygons: bool = True,
        simplify_tolerance: float = 2.0,
    ) -> Dict[str, Any]:
        """Convert bbox annotations to segmentations using SAM3"""
        try:
            response = httpx.post(
                f"{SEGMENTATION_URL}/sam3/convert-dataset",
                json={
                    "coco_data": coco_data,
                    "coco_json_path": coco_json_path,
                    "images_dir": images_dir,
                    "output_path": output_path,
                    "categories_to_convert": categories_to_convert or [],
                    "overwrite_existing": overwrite_existing,
                    "simplify_polygons": simplify_polygons,
                    "simplify_tolerance": simplify_tolerance,
                },
                timeout=60.0,  # Should return quickly since it's async
            )
            if response.status_code == 200:
                return response.json()
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text[:200]}"}
        except httpx.TimeoutException:
            return {"success": False, "error": f"Timeout iniciando conversion en {SEGMENTATION_URL}"}
        except httpx.ConnectError:
            return {"success": False, "error": f"No se puede conectar a {SEGMENTATION_URL}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_sam3_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a SAM3 conversion job"""
        try:
            response = httpx.get(
                f"{SEGMENTATION_URL}/sam3/jobs/{job_id}",
                timeout=10.0,
            )
            if response.status_code == 200:
                return response.json()
            return {"error": response.text}
        except Exception as e:
            return {"error": str(e)}

    def get_segmentation_health(self) -> Dict[str, Any]:
        """Get health status of segmentation service"""
        try:
            response = httpx.get(
                f"{SEGMENTATION_URL}/health",
                timeout=30.0  # Increased timeout for busy service
            )
            if response.status_code == 200:
                return response.json()
            return {"error": f"HTTP {response.status_code}: {response.text[:100]}", "status": "unhealthy"}
        except httpx.TimeoutException:
            return {"error": f"Timeout conectando a {SEGMENTATION_URL}", "status": "unhealthy"}
        except httpx.ConnectError:
            return {"error": f"No se puede conectar a {SEGMENTATION_URL}", "status": "unhealthy"}
        except Exception as e:
            return {"error": str(e), "status": "unhealthy"}

    def list_extraction_jobs(self) -> Dict[str, Any]:
        """List all extraction jobs from segmentation service"""
        try:
            response = httpx.get(
                f"{SEGMENTATION_URL}/extract/jobs",
                timeout=10.0
            )
            if response.status_code == 200:
                return response.json()
            return {"jobs": [], "total": 0, "error": response.text}
        except Exception as e:
            return {"jobs": [], "total": 0, "error": str(e)}

    def list_sam3_jobs(self) -> Dict[str, Any]:
        """List all SAM3 conversion jobs from segmentation service"""
        try:
            response = httpx.get(
                f"{SEGMENTATION_URL}/sam3/jobs",
                timeout=10.0
            )
            if response.status_code == 200:
                return response.json()
            return {"jobs": [], "total": 0, "error": response.text}
        except Exception as e:
            return {"jobs": [], "total": 0, "error": str(e)}

    # =========================================================================
    # Object Size Configuration Methods
    # =========================================================================

    def get_object_sizes(self) -> Dict[str, Any]:
        """Get all configured object sizes"""
        try:
            response = httpx.get(
                f"{GATEWAY_URL}/config/object-sizes",
                timeout=10.0
            )
            if response.status_code == 200:
                return response.json()
            return {"sizes": {}, "error": response.text}
        except Exception as e:
            return {"sizes": {}, "error": str(e)}

    def get_object_size(self, class_name: str) -> Dict[str, Any]:
        """Get size for a specific object class"""
        try:
            response = httpx.get(
                f"{GATEWAY_URL}/config/object-sizes/{class_name}",
                timeout=10.0
            )
            if response.status_code == 200:
                return response.json()
            return {"error": response.text}
        except Exception as e:
            return {"error": str(e)}

    def update_object_size(self, class_name: str, size: float) -> Dict[str, Any]:
        """Update size for a specific object class"""
        try:
            response = httpx.put(
                f"{GATEWAY_URL}/config/object-sizes/{class_name}",
                params={"size": size},
                timeout=10.0
            )
            if response.status_code == 200:
                return response.json()
            return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def update_multiple_object_sizes(self, sizes: Dict[str, float]) -> Dict[str, Any]:
        """Update multiple object sizes at once"""
        try:
            response = httpx.post(
                f"{GATEWAY_URL}/config/object-sizes/batch",
                json=sizes,
                timeout=10.0
            )
            if response.status_code == 200:
                return response.json()
            return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def delete_object_size(self, class_name: str) -> Dict[str, Any]:
        """Delete size configuration for an object class"""
        try:
            response = httpx.delete(
                f"{GATEWAY_URL}/config/object-sizes/{class_name}",
                timeout=10.0
            )
            if response.status_code == 200:
                return response.json()
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
