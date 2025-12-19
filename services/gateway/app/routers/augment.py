"""
Augment Router
==============
Endpoints for synthetic image composition and validation.
Proxies requests to the Augmentor service.
"""

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from enum import Enum

from app.services.client import get_service_registry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/augment", tags=["Augmentation"])


# =============================================================================
# Request/Response Models
# =============================================================================

class EffectType(str, Enum):
    """Available realism effects"""
    COLOR_CORRECTION = "color_correction"
    BLUR_MATCHING = "blur_matching"
    LIGHTING = "lighting"
    UNDERWATER = "underwater"
    MOTION_BLUR = "motion_blur"
    SHADOWS = "shadows"
    CAUSTICS = "caustics"
    POISSON_BLEND = "poisson_blend"
    EDGE_SMOOTHING = "edge_smoothing"


class WaterClarity(str, Enum):
    """Water clarity levels"""
    CLEAR = "clear"
    MURKY = "murky"
    VERY_MURKY = "very_murky"


class EffectsConfig(BaseModel):
    """Effects configuration"""
    color_intensity: float = Field(0.4, ge=0, le=1, description="Color correction intensity (lower preserves object colors better)")
    blur_strength: float = Field(1.0, ge=0, le=3)
    underwater_intensity: float = Field(0.25, ge=0, le=1)
    caustics_intensity: float = Field(0.15, ge=0, le=0.5)
    shadow_opacity: float = Field(0.4, ge=0, le=1)
    lighting_type: str = Field("ambient")
    lighting_intensity: float = Field(0.5, ge=0, le=1)
    motion_blur_probability: float = Field(0.2, ge=0, le=1)
    water_clarity: WaterClarity = Field(WaterClarity.CLEAR)


class ObjectPlacement(BaseModel):
    """Object to place in composition"""
    image_path: str = Field(..., description="Path to object image")
    class_name: str = Field(..., description="Object class name")
    position: Optional[tuple] = Field(None, description="(x, y) position")
    scale: Optional[float] = Field(None, ge=0.1, le=5.0)
    rotation: Optional[float] = Field(None, ge=-180, le=180)
    material: Optional[str] = Field("plastic")


class ComposeRequest(BaseModel):
    """Request to compose a synthetic image"""
    background_path: str = Field(..., description="Path to background image")
    objects: List[ObjectPlacement] = Field(..., min_length=1)
    depth_map_path: Optional[str] = Field(None)
    effects: List[EffectType] = Field(
        default=[EffectType.COLOR_CORRECTION, EffectType.BLUR_MATCHING]
    )
    effects_config: EffectsConfig = Field(default_factory=EffectsConfig)
    validate_quality: bool = Field(False)
    validate_physics: bool = Field(False)
    output_path: str = Field(..., description="Output path for composed image")
    save_annotations: bool = Field(True)


class AnnotationBox(BaseModel):
    """Bounding box annotation"""
    x: int
    y: int
    width: int
    height: int
    class_name: str
    confidence: float = 1.0
    area: int = 0


class QualityScoreInfo(BaseModel):
    """Quality validation scores"""
    perceptual_quality: float
    distribution_match: float = 1.0
    anomaly_score: float
    composition_score: float
    overall_score: float
    overall_pass: bool


class PhysicsViolationInfo(BaseModel):
    """Physics violation info"""
    violation_type: str
    object_class: str
    severity: str
    description: str
    suggested_fix: Optional[str] = None


class ComposeResponse(BaseModel):
    """Response from compose endpoint"""
    success: bool
    output_path: str
    annotations: List[AnnotationBox] = []
    objects_placed: int
    depth_used: bool
    effects_applied: List[str]
    quality_score: Optional[QualityScoreInfo] = None
    physics_violations: List[PhysicsViolationInfo] = []
    is_valid: bool = True
    rejection_reason: Optional[str] = None
    processing_time_ms: float
    error: Optional[str] = None


class ComposeBatchRequest(BaseModel):
    """Request for batch composition"""
    backgrounds_dir: str = Field(..., description="Directory with backgrounds")
    objects_dir: str = Field(..., description="Directory with objects by class")
    output_dir: str = Field(..., description="Output directory")
    num_images: int = Field(..., ge=1, le=100000)
    targets_per_class: Optional[Dict[str, int]] = Field(None)
    max_objects_per_image: int = Field(5, ge=1, le=20)
    effects: List[EffectType] = Field(
        default=[EffectType.COLOR_CORRECTION, EffectType.BLUR_MATCHING, EffectType.CAUSTICS]
    )
    effects_config: EffectsConfig = Field(default_factory=EffectsConfig)
    depth_aware: bool = Field(True)
    validate_quality: bool = Field(False)
    validate_physics: bool = Field(False)
    reject_invalid: bool = Field(True)
    save_pipeline_debug: bool = Field(False, description="Save intermediate pipeline images for first iteration")


class ComposeBatchResponse(BaseModel):
    """Response from batch compose"""
    success: bool
    job_id: str
    status: str
    images_generated: int = 0
    images_rejected: int = 0
    images_pending: int = 0
    synthetic_counts: Dict[str, int] = {}
    output_coco_path: Optional[str] = None
    processing_time_ms: float = 0
    error: Optional[str] = None


class ValidateRequest(BaseModel):
    """Request to validate an image"""
    image_path: str = Field(..., description="Path to image")
    annotations: List[AnnotationBox] = Field(default_factory=list)
    reference_images: Optional[List[str]] = Field(None)
    depth_map_path: Optional[str] = Field(None)
    check_quality: bool = Field(True)
    check_anomalies: bool = Field(True)
    check_physics: bool = Field(True)
    min_perceptual_quality: float = Field(0.7, ge=0, le=1)
    min_anomaly_score: float = Field(0.6, ge=0, le=1)


class ValidateResponse(BaseModel):
    """Response from validation"""
    is_valid: bool
    quality_score: QualityScoreInfo
    anomalies: List[Dict[str, Any]] = []
    physics_violations: List[PhysicsViolationInfo] = []
    processing_time_ms: float
    error: Optional[str] = None


class LightSourceInfo(BaseModel):
    """Detected light source"""
    light_type: str
    position: tuple
    intensity: float
    color: tuple
    shadow_softness: float = 0.5


class LightingInfo(BaseModel):
    """Lighting estimation result"""
    light_sources: List[LightSourceInfo] = []
    dominant_direction: tuple
    color_temperature: float
    ambient_intensity: float


class LightingRequest(BaseModel):
    """Request to estimate lighting"""
    image_path: str = Field(..., description="Path to image")
    max_light_sources: int = Field(3, ge=1, le=5)
    intensity_threshold: float = Field(0.6, ge=0.3, le=0.9)
    estimate_hdr: bool = Field(False)
    apply_water_attenuation: bool = Field(False)


class LightingResponse(BaseModel):
    """Response from lighting estimation"""
    success: bool
    lighting_info: Optional[LightingInfo] = None
    processing_time_ms: float
    error: Optional[str] = None


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/compose", response_model=ComposeResponse)
async def compose_image(request: ComposeRequest):
    """
    Compose a single synthetic image.

    Places objects on background with realistic effects and optional validation.
    """
    logger.info(f"Compose request: {request.background_path} with {len(request.objects)} objects")

    try:
        registry = get_service_registry()

        # Convert to dict for API call
        request_data = request.model_dump()
        # Convert enums to strings
        request_data['effects'] = [e.value if hasattr(e, 'value') else e for e in request_data['effects']]
        if 'effects_config' in request_data and 'water_clarity' in request_data['effects_config']:
            wc = request_data['effects_config']['water_clarity']
            request_data['effects_config']['water_clarity'] = wc.value if hasattr(wc, 'value') else wc

        result = await registry.augmentor.post("/compose", request_data)

        # Convert response
        return ComposeResponse(
            success=result.get("success", False),
            output_path=result.get("output_path", ""),
            annotations=[AnnotationBox(**a) for a in result.get("annotations", [])],
            objects_placed=result.get("objects_placed", 0),
            depth_used=result.get("depth_used", False),
            effects_applied=result.get("effects_applied", []),
            quality_score=QualityScoreInfo(**result["quality_score"]) if result.get("quality_score") else None,
            physics_violations=[PhysicsViolationInfo(**v) for v in result.get("physics_violations", [])],
            is_valid=result.get("is_valid", True),
            rejection_reason=result.get("rejection_reason"),
            processing_time_ms=result.get("processing_time_ms", 0),
            error=result.get("error"),
        )

    except Exception as e:
        logger.error(f"Compose failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compose-batch", response_model=ComposeBatchResponse)
async def compose_batch(request: ComposeBatchRequest):
    """
    Start a batch composition job.

    Creates multiple synthetic images asynchronously.
    Use GET /augment/jobs/{job_id} to check progress.
    """
    logger.info(f"Batch compose request: {request.num_images} images")

    try:
        registry = get_service_registry()

        # Convert to dict
        request_data = request.model_dump()
        request_data['effects'] = [e.value if hasattr(e, 'value') else e for e in request_data['effects']]
        if 'effects_config' in request_data and 'water_clarity' in request_data['effects_config']:
            wc = request_data['effects_config']['water_clarity']
            request_data['effects_config']['water_clarity'] = wc.value if hasattr(wc, 'value') else wc

        result = await registry.augmentor.post("/compose-batch", request_data)

        return ComposeBatchResponse(
            success=result.get("success", False),
            job_id=result.get("job_id", ""),
            status=result.get("status", "unknown"),
            images_generated=result.get("images_generated", 0),
            images_rejected=result.get("images_rejected", 0),
            images_pending=result.get("images_pending", 0),
            synthetic_counts=result.get("synthetic_counts", {}),
            output_coco_path=result.get("output_coco_path"),
            processing_time_ms=result.get("processing_time_ms", 0),
            error=result.get("error"),
        )

    except Exception as e:
        logger.error(f"Batch compose failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}", response_model=ComposeBatchResponse)
async def get_job_status(job_id: str):
    """Get status of a batch composition job."""
    try:
        registry = get_service_registry()
        result = await registry.augmentor.get(f"/jobs/{job_id}")

        return ComposeBatchResponse(
            success=result.get("success", False),
            job_id=job_id,
            status=result.get("status", "unknown"),
            images_generated=result.get("images_generated", 0),
            images_rejected=result.get("images_rejected", 0),
            images_pending=result.get("images_pending", 0),
            synthetic_counts=result.get("synthetic_counts", {}),
            output_coco_path=result.get("output_coco_path"),
            processing_time_ms=result.get("processing_time_ms", 0),
            error=result.get("error"),
        )

    except Exception as e:
        logger.error(f"Get job status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs")
async def list_all_jobs():
    """
    List all batch composition jobs.

    Returns list of jobs with their current status, progress, and output info.
    Jobs are sorted by creation time (most recent first).
    """
    logger.info("Listing all batch composition jobs")

    try:
        registry = get_service_registry()
        result = await registry.augmentor.get("/jobs")

        return {
            "success": True,
            "jobs": result.get("jobs", []),
            "total": result.get("total", 0),
        }

    except Exception as e:
        logger.error(f"List jobs failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a running batch composition job.

    Stops job processing at the next iteration, preserving already generated images.
    """
    logger.info(f"Cancel request for job: {job_id}")

    try:
        registry = get_service_registry()
        result = await registry.augmentor.delete(f"/jobs/{job_id}")

        return result

    except Exception as e:
        logger.error(f"Cancel job failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate", response_model=ValidateResponse)
async def validate_image(request: ValidateRequest):
    """
    Validate a composed image.

    Checks quality (LPIPS), anomalies, and physics plausibility.
    """
    logger.info(f"Validate request: {request.image_path}")

    try:
        registry = get_service_registry()
        result = await registry.augmentor.post("/validate", request.model_dump())

        return ValidateResponse(
            is_valid=result.get("is_valid", False),
            quality_score=QualityScoreInfo(**result["quality_score"]) if result.get("quality_score") else QualityScoreInfo(
                perceptual_quality=1.0,
                anomaly_score=1.0,
                composition_score=1.0,
                overall_score=1.0,
                overall_pass=True,
            ),
            anomalies=result.get("anomalies", []),
            physics_violations=[PhysicsViolationInfo(**v) for v in result.get("physics_violations", [])],
            processing_time_ms=result.get("processing_time_ms", 0),
            error=result.get("error"),
        )

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/lighting", response_model=LightingResponse)
async def estimate_lighting(request: LightingRequest):
    """
    Estimate lighting from a background image.

    Detects light sources, dominant direction, and color temperature.
    """
    logger.info(f"Lighting request: {request.image_path}")

    try:
        registry = get_service_registry()
        result = await registry.augmentor.post("/lighting", request.model_dump())

        lighting_info = None
        if result.get("lighting_info"):
            li = result["lighting_info"]
            lighting_info = LightingInfo(
                light_sources=[LightSourceInfo(**ls) for ls in li.get("light_sources", [])],
                dominant_direction=tuple(li.get("dominant_direction", (0, 45))),
                color_temperature=li.get("color_temperature", 5500),
                ambient_intensity=li.get("ambient_intensity", 0.5),
            )

        return LightingResponse(
            success=result.get("success", False),
            lighting_info=lighting_info,
            processing_time_ms=result.get("processing_time_ms", 0),
            error=result.get("error"),
        )

    except Exception as e:
        logger.error(f"Lighting estimation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info")
async def augmentor_info():
    """Get Augmentor service information."""
    try:
        registry = get_service_registry()
        info = await registry.augmentor.get("/info")
        health = await registry.augmentor.health_check()

        return {
            "service": "augmentor",
            "url": registry.augmentor.base_url,
            "info": info,
            "health": health
        }

    except Exception as e:
        logger.error(f"Failed to get augmentor info: {e}")
        return {
            "service": "augmentor",
            "error": str(e),
            "health": {"healthy": False}
        }
