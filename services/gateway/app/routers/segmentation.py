"""
Segmentation Router
===================
Endpoints for object extraction, labeling, and SAM3 operations.
Proxies requests to the Segmentation service.
"""

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.client import get_service_registry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/segmentation", tags=["Segmentation"])


# =============================================================================
# Request/Response Models
# =============================================================================

class ExtractObjectsRequest(BaseModel):
    """Request to extract objects from a COCO dataset"""
    coco_json_path: str = Field(..., description="Path to COCO annotations JSON")
    images_dir: str = Field(..., description="Directory containing source images")
    output_dir: str = Field(..., description="Output directory for extracted objects")
    categories: Optional[List[str]] = Field(None, description="Categories to extract (None = all)")
    min_size: int = Field(32, description="Minimum object size in pixels")
    include_masks: bool = Field(True, description="Include segmentation masks")
    use_sam3: bool = Field(False, description="Use SAM3 for improved segmentation")
    padding: int = Field(10, description="Padding around objects")
    deduplicate: bool = Field(False, description="Remove duplicate objects")


class ExtractSingleObjectRequest(BaseModel):
    """Request to extract a single object"""
    image_path: str = Field(..., description="Path to source image")
    bbox: List[int] = Field(..., description="Bounding box [x, y, width, height]")
    output_path: str = Field(..., description="Output path for extracted object")
    use_sam3: bool = Field(False, description="Use SAM3 for segmentation")
    padding: int = Field(10, description="Padding around object")


class LabelingStartRequest(BaseModel):
    """Request to start a labeling job"""
    images_dir: str = Field(..., description="Directory containing images to label")
    output_dir: str = Field(..., description="Output directory for labels")
    categories: List[str] = Field(..., description="Categories to label")
    model: str = Field("auto", description="Detection model to use")
    confidence_threshold: float = Field(0.5, ge=0, le=1)
    use_sam3: bool = Field(True, description="Use SAM3 for segmentation")
    generate_masks: bool = Field(True, description="Generate segmentation masks")
    min_area: int = Field(100, description="Minimum annotation area")
    max_overlap: float = Field(0.5, ge=0, le=1, description="Maximum allowed overlap")


class RelabelingRequest(BaseModel):
    """Request to relabel existing annotations"""
    coco_json_path: str = Field(..., description="Path to existing COCO JSON")
    images_dir: str = Field(..., description="Directory containing images")
    output_dir: str = Field(..., description="Output directory")
    categories: Optional[List[str]] = Field(None, description="Categories to relabel")
    confidence_threshold: float = Field(0.5, ge=0, le=1)
    use_sam3: bool = Field(True)


class SAM3SegmentRequest(BaseModel):
    """Request to segment an image using SAM3"""
    image_path: str = Field(..., description="Path to image")
    points: Optional[List[List[int]]] = Field(None, description="Point prompts [[x,y], ...]")
    point_labels: Optional[List[int]] = Field(None, description="Point labels (1=foreground, 0=background)")
    boxes: Optional[List[List[int]]] = Field(None, description="Box prompts [[x1,y1,x2,y2], ...]")
    text_prompt: Optional[str] = Field(None, description="Text prompt for segmentation")
    output_path: Optional[str] = Field(None, description="Output path for mask")


class SAM3ConvertDatasetRequest(BaseModel):
    """Request to convert dataset annotations using SAM3"""
    coco_json_path: str = Field(..., description="Path to COCO JSON")
    images_dir: str = Field(..., description="Directory containing images")
    output_dir: str = Field(..., description="Output directory")
    min_area: int = Field(100, description="Minimum mask area")
    confidence_threshold: float = Field(0.8, ge=0, le=1)


# =============================================================================
# Object Extraction Endpoints
# =============================================================================

@router.post("/extract/objects")
async def extract_objects(request: ExtractObjectsRequest):
    """
    Extract objects from a COCO dataset.

    Creates individual object images with transparent backgrounds.
    Returns a job_id for tracking progress.
    """
    logger.info(f"Extract objects request: {request.coco_json_path}")

    try:
        registry = get_service_registry()
        result = await registry.segmentation.post("/extract/objects", request.model_dump())
        return result
    except Exception as e:
        logger.error(f"Extract objects failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract/single-object")
async def extract_single_object(request: ExtractSingleObjectRequest):
    """Extract a single object from an image."""
    logger.info(f"Extract single object: {request.image_path}")

    try:
        registry = get_service_registry()
        result = await registry.segmentation.post("/extract/single-object", request.model_dump())
        return result
    except Exception as e:
        logger.error(f"Extract single object failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract/analyze-dataset")
async def analyze_extraction_dataset(coco_json_path: str, images_dir: str):
    """Analyze a dataset before extraction."""
    logger.info(f"Analyze dataset for extraction: {coco_json_path}")

    try:
        registry = get_service_registry()
        result = await registry.segmentation.post("/extract/analyze-dataset", {
            "coco_json_path": coco_json_path,
            "images_dir": images_dir
        })
        return result
    except Exception as e:
        logger.error(f"Analyze dataset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/extract/jobs")
async def list_extraction_jobs(
    status: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200)
):
    """List all extraction jobs."""
    logger.info(f"List extraction jobs - status: {status}")

    try:
        registry = get_service_registry()
        params = {"limit": limit}
        if status:
            params["status"] = status
        result = await registry.segmentation.get("/extract/jobs", params=params)
        return result
    except Exception as e:
        logger.error(f"List extraction jobs failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/extract/jobs/{job_id}")
async def get_extraction_job(job_id: str):
    """Get status of an extraction job."""
    logger.info(f"Get extraction job: {job_id}")

    try:
        registry = get_service_registry()
        result = await registry.segmentation.get(f"/extract/jobs/{job_id}")
        return result
    except Exception as e:
        logger.error(f"Get extraction job failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/extract/jobs/{job_id}")
async def cancel_extraction_job(job_id: str):
    """Cancel a running extraction job."""
    logger.info(f"Cancel extraction job: {job_id}")

    try:
        registry = get_service_registry()
        result = await registry.segmentation.delete(f"/extract/jobs/{job_id}")
        return result
    except Exception as e:
        logger.error(f"Cancel extraction job failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Labeling Endpoints
# =============================================================================

@router.post("/labeling/start")
async def start_labeling(request: LabelingStartRequest):
    """
    Start an automatic labeling job.

    Uses detection models to automatically annotate images.
    Returns a job_id for tracking progress.
    """
    logger.info(f"Start labeling: {request.images_dir}")

    try:
        registry = get_service_registry()
        result = await registry.segmentation.post("/labeling/start", request.model_dump())
        return result
    except Exception as e:
        logger.error(f"Start labeling failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/labeling/relabel")
async def relabel_dataset(request: RelabelingRequest):
    """
    Relabel an existing annotated dataset.

    Updates annotations using improved models or parameters.
    """
    logger.info(f"Relabel dataset: {request.coco_json_path}")

    try:
        registry = get_service_registry()
        result = await registry.segmentation.post("/labeling/relabel", request.model_dump())
        return result
    except Exception as e:
        logger.error(f"Relabel failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/labeling/jobs")
async def list_labeling_jobs(
    status: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200)
):
    """List all labeling jobs."""
    logger.info(f"List labeling jobs - status: {status}")

    try:
        registry = get_service_registry()
        params = {"limit": limit}
        if status:
            params["status"] = status
        result = await registry.segmentation.get("/labeling/jobs", params=params)
        return result
    except Exception as e:
        logger.error(f"List labeling jobs failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/labeling/jobs/active")
async def get_active_labeling_jobs():
    """Get all active (running/queued) labeling jobs."""
    logger.info("Get active labeling jobs")

    try:
        registry = get_service_registry()
        result = await registry.segmentation.get("/labeling/jobs", params={"status": "running"})
        # Also get pending jobs
        pending = await registry.segmentation.get("/labeling/jobs", params={"status": "pending"})

        jobs = result.get("jobs", []) + pending.get("jobs", [])
        return jobs
    except Exception as e:
        logger.error(f"Get active labeling jobs failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/labeling/jobs/{job_id}")
async def get_labeling_job(job_id: str):
    """Get status of a labeling job."""
    logger.info(f"Get labeling job: {job_id}")

    try:
        registry = get_service_registry()
        result = await registry.segmentation.get(f"/labeling/jobs/{job_id}")
        return result
    except Exception as e:
        logger.error(f"Get labeling job failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/labeling/jobs/{job_id}/result")
async def get_labeling_result(job_id: str):
    """Get the result of a completed labeling job."""
    logger.info(f"Get labeling result: {job_id}")

    try:
        registry = get_service_registry()
        result = await registry.segmentation.get(f"/labeling/jobs/{job_id}/result")
        return result
    except Exception as e:
        logger.error(f"Get labeling result failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/labeling/jobs/{job_id}/previews")
async def get_labeling_previews(job_id: str, limit: int = 10):
    """Get preview images from a labeling job."""
    logger.info(f"Get labeling previews: {job_id}")

    try:
        registry = get_service_registry()
        result = await registry.segmentation.get(
            f"/labeling/jobs/{job_id}/previews",
            params={"limit": limit}
        )
        return result
    except Exception as e:
        logger.error(f"Get labeling previews failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/labeling/jobs/{job_id}/resume")
async def resume_labeling_job(job_id: str):
    """Resume an interrupted labeling job."""
    logger.info(f"Resume labeling job: {job_id}")

    try:
        registry = get_service_registry()
        result = await registry.segmentation.post(f"/labeling/jobs/{job_id}/resume", {})
        return result
    except Exception as e:
        logger.error(f"Resume labeling job failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/labeling/jobs/{job_id}")
async def cancel_labeling_job(job_id: str):
    """Cancel a running labeling job."""
    logger.info(f"Cancel labeling job: {job_id}")

    try:
        registry = get_service_registry()
        result = await registry.segmentation.delete(f"/labeling/jobs/{job_id}")
        return result
    except Exception as e:
        logger.error(f"Cancel labeling job failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/labeling/jobs/{job_id}/delete")
async def delete_labeling_job(job_id: str):
    """Permanently delete a labeling job."""
    logger.info(f"Delete labeling job: {job_id}")

    try:
        registry = get_service_registry()
        result = await registry.segmentation.post(f"/labeling/jobs/{job_id}/delete", {})
        return result
    except Exception as e:
        logger.error(f"Delete labeling job failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SAM3 Tool Endpoints
# =============================================================================

@router.post("/sam3/segment-image")
async def sam3_segment_image(request: SAM3SegmentRequest):
    """
    Segment an image using SAM3 with various prompts.

    Supports point prompts, box prompts, and text prompts.
    """
    logger.info(f"SAM3 segment image: {request.image_path}")

    try:
        registry = get_service_registry()
        result = await registry.segmentation.post("/sam3/segment-image", request.model_dump())
        return result
    except Exception as e:
        logger.error(f"SAM3 segment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sam3/convert-dataset")
async def sam3_convert_dataset(request: SAM3ConvertDatasetRequest):
    """
    Convert dataset annotations to SAM3 segmentation masks.

    Improves existing bbox annotations with precise segmentation masks.
    """
    logger.info(f"SAM3 convert dataset: {request.coco_json_path}")

    try:
        registry = get_service_registry()
        result = await registry.segmentation.post("/sam3/convert-dataset", request.model_dump())
        return result
    except Exception as e:
        logger.error(f"SAM3 convert dataset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sam3/jobs")
async def list_sam3_jobs(
    status: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200)
):
    """List all SAM3 conversion jobs."""
    logger.info(f"List SAM3 jobs - status: {status}")

    try:
        registry = get_service_registry()
        params = {"limit": limit}
        if status:
            params["status"] = status
        result = await registry.segmentation.get("/sam3/jobs", params=params)
        return result
    except Exception as e:
        logger.error(f"List SAM3 jobs failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sam3/jobs/{job_id}")
async def get_sam3_job(job_id: str):
    """Get status of a SAM3 conversion job."""
    logger.info(f"Get SAM3 job: {job_id}")

    try:
        registry = get_service_registry()
        result = await registry.segmentation.get(f"/sam3/jobs/{job_id}")
        return result
    except Exception as e:
        logger.error(f"Get SAM3 job failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sam3/jobs/{job_id}")
async def cancel_sam3_job(job_id: str):
    """Cancel a running SAM3 conversion job."""
    logger.info(f"Cancel SAM3 job: {job_id}")

    try:
        registry = get_service_registry()
        result = await registry.segmentation.delete(f"/sam3/jobs/{job_id}")
        return result
    except Exception as e:
        logger.error(f"Cancel SAM3 job failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Text-based Segmentation
# =============================================================================

@router.post("/segment-text")
async def segment_with_text(image_path: str, text_prompt: str, output_path: Optional[str] = None):
    """
    Segment an image using a text prompt.

    Uses text-guided segmentation models.
    """
    logger.info(f"Text segmentation: {image_path} - '{text_prompt}'")

    try:
        registry = get_service_registry()
        result = await registry.segmentation.post("/segment-text", {
            "image_path": image_path,
            "text_prompt": text_prompt,
            "output_path": output_path
        })
        return result
    except Exception as e:
        logger.error(f"Text segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
