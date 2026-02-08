"""
Augment Router
==============
Endpoints for synthetic image composition and validation.
Proxies requests to the Augmentor service.
"""

import logging
import os
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
    """Effects configuration (BUG #11 FIX - harmonized with Augmentor defaults)"""
    color_intensity: float = Field(0.12, ge=0, le=1, description="Color correction intensity (0.1-0.15 recommended)")
    blur_strength: float = Field(0.5, ge=0, le=3, description="Blur matching strength")
    underwater_intensity: float = Field(0.15, ge=0, le=1, description="Underwater tint intensity")
    caustics_intensity: float = Field(0.10, ge=0, le=0.5, description="Caustics effect intensity")
    shadow_opacity: float = Field(0.10, ge=0, le=1, description="Shadow darkness")
    lighting_type: str = Field("ambient", description="spotlight|gradient|ambient")
    lighting_intensity: float = Field(0.5, ge=0, le=1, description="Lighting effect intensity")
    motion_blur_probability: float = Field(0.2, ge=0, le=1, description="Probability of motion blur")
    water_clarity: WaterClarity = Field(WaterClarity.CLEAR, description="Water clarity level")


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
    """Request for batch composition.

    Accepts both legacy format (backgrounds_dir + objects_dir + num_images)
    and frontend format (source_dataset + target_counts + nested configs).
    """
    # --- Legacy format (direct augmentor fields) ---
    backgrounds_dir: Optional[str] = Field(None, description="Directory with backgrounds")
    objects_dir: Optional[str] = Field(None, description="Directory with objects by class")
    num_images: Optional[int] = Field(None, ge=1, le=100000)

    # --- Frontend format ---
    source_dataset: Optional[str] = Field(None, description="Dataset root containing Backgrounds_filtered/ and Objects/")
    target_counts: Optional[Dict[str, int]] = Field(None, description="Target count per class")

    # --- Common fields ---
    output_dir: str = Field(..., description="Output directory")
    targets_per_class: Optional[Dict[str, int]] = Field(None)
    max_objects_per_image: int = Field(5, ge=1, le=20)
    effects: List[EffectType] = Field(
        default=[EffectType.COLOR_CORRECTION, EffectType.BLUR_MATCHING, EffectType.CAUSTICS]
    )
    effects_config: Optional[Any] = Field(default=None, description="Effects config (flat or nested)")
    depth_aware: bool = Field(True)
    validate_quality: bool = Field(False)
    validate_physics: bool = Field(False)
    reject_invalid: bool = Field(True)
    save_pipeline_debug: bool = Field(False, description="Save intermediate pipeline images for first iteration")

    # --- Frontend-specific nested configs (accepted but transformed) ---
    placement_config: Optional[Dict[str, Any]] = Field(None)
    validation_config: Optional[Dict[str, Any]] = Field(None)
    batch_config: Optional[Dict[str, Any]] = Field(None)
    lighting_config: Optional[Dict[str, Any]] = Field(None)
    metadata: Optional[Dict[str, Any]] = Field(None)
    use_depth: Optional[bool] = Field(None)
    use_segmentation: Optional[bool] = Field(None)
    depth_aware_placement: Optional[bool] = Field(None)


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
    # Resolve backgrounds_dir and objects_dir from source_dataset if needed
    backgrounds_dir = request.backgrounds_dir
    objects_dir = request.objects_dir
    num_images = request.num_images

    if request.source_dataset:
        if not backgrounds_dir:
            backgrounds_dir = os.path.join(request.source_dataset, "Backgrounds_filtered")
        if not objects_dir:
            objects_dir = os.path.join(request.source_dataset, "Objects")

    if not backgrounds_dir or not objects_dir:
        raise HTTPException(
            status_code=422,
            detail="Either source_dataset or both backgrounds_dir and objects_dir are required"
        )

    # Derive num_images from target_counts if not provided
    targets = request.targets_per_class or request.target_counts
    if num_images is None:
        if targets:
            num_images = sum(targets.values())
        else:
            raise HTTPException(
                status_code=422,
                detail="Either num_images or target_counts is required"
            )

    # Resolve depth_aware from frontend flag
    depth_aware = request.depth_aware
    if request.depth_aware_placement is not None:
        depth_aware = request.depth_aware_placement

    # Resolve validation flags from nested config
    validate_quality = request.validate_quality
    validate_physics = request.validate_physics
    reject_invalid = request.reject_invalid
    if request.validation_config:
        validate_quality = request.validation_config.get("validate_quality", validate_quality)
        validate_physics = request.validation_config.get("validate_physics", validate_physics)
        reject_invalid = request.validation_config.get("reject_invalid", reject_invalid)

    # Resolve batch config
    save_pipeline_debug = request.save_pipeline_debug
    parallel = True
    concurrent_limit = 4
    vram_threshold = 0.7
    if request.batch_config:
        save_pipeline_debug = request.batch_config.get("save_pipeline_debug", save_pipeline_debug)
        parallel = request.batch_config.get("parallel", parallel)
        concurrent_limit = request.batch_config.get("concurrent_limit", concurrent_limit)
        vram_threshold = request.batch_config.get("vram_threshold", vram_threshold)

    # Resolve max_objects_per_image from placement_config
    max_objects = request.max_objects_per_image
    if request.placement_config:
        max_objects = request.placement_config.get("max_objects_per_image", max_objects)

    # Transform effects_config from frontend nested format to flat augmentor format
    effects_config = {}
    ec = request.effects_config
    if isinstance(ec, dict):
        # Frontend sends nested format with color_correction.enabled, etc.
        effects_config = {
            "color_intensity": ec.get("color_correction", {}).get("color_intensity", 0.12) if isinstance(ec.get("color_correction"), dict) else 0.12,
            "blur_strength": ec.get("blur_matching", {}).get("blur_strength", 0.5) if isinstance(ec.get("blur_matching"), dict) else 0.5,
            "underwater_intensity": ec.get("underwater", {}).get("underwater_intensity", 0.15) if isinstance(ec.get("underwater"), dict) else 0.15,
            "caustics_intensity": ec.get("caustics", {}).get("caustics_intensity", 0.10) if isinstance(ec.get("caustics"), dict) else 0.10,
            "shadow_opacity": ec.get("shadows", {}).get("shadow_opacity", 0.10) if isinstance(ec.get("shadows"), dict) else 0.10,
            "lighting_type": ec.get("lighting", {}).get("lighting_type", "ambient") if isinstance(ec.get("lighting"), dict) else "ambient",
            "lighting_intensity": ec.get("lighting", {}).get("lighting_intensity", 0.5) if isinstance(ec.get("lighting"), dict) else 0.5,
            "motion_blur_probability": ec.get("motion_blur", {}).get("motion_blur_probability", 0.2) if isinstance(ec.get("motion_blur"), dict) else 0.2,
            "water_clarity": ec.get("underwater", {}).get("water_clarity", "clear") if isinstance(ec.get("underwater"), dict) else "clear",
        }
        # If it already has flat keys (legacy format), pass through
        if "color_intensity" in ec:
            effects_config = ec
    elif ec is None:
        effects_config = {}

    # Build enabled effects list from nested config
    effects_list = [e.value if hasattr(e, 'value') else e for e in request.effects]
    if isinstance(request.effects_config, dict) and "color_correction" in request.effects_config:
        # Derive effects list from nested config enabled flags
        effects_list = []
        effect_map = {
            "color_correction": "color_correction",
            "blur_matching": "blur_matching",
            "lighting": "lighting",
            "underwater": "underwater",
            "motion_blur": "motion_blur",
            "shadows": "shadows",
            "caustics": "caustics",
            "edge_smoothing": "edge_smoothing",
        }
        for key, effect_name in effect_map.items():
            section = request.effects_config.get(key, {})
            if isinstance(section, dict) and section.get("enabled", False):
                effects_list.append(effect_name)
        # Ensure at least basic effects
        if not effects_list:
            effects_list = ["color_correction", "blur_matching", "caustics"]

    # Build the augmentor-compatible request
    request_data = {
        "backgrounds_dir": backgrounds_dir,
        "objects_dir": objects_dir,
        "output_dir": request.output_dir,
        "num_images": num_images,
        "targets_per_class": targets,
        "max_objects_per_image": max_objects,
        "effects": effects_list,
        "effects_config": effects_config,
        "depth_aware": depth_aware,
        "validate_quality": validate_quality,
        "validate_physics": validate_physics,
        "reject_invalid": reject_invalid,
        "save_pipeline_debug": save_pipeline_debug,
        "parallel": parallel,
        "concurrent_limit": concurrent_limit,
        "vram_threshold": vram_threshold,
    }

    # Add metadata if present
    if request.metadata:
        request_data["metadata"] = request.metadata

    logger.info(f"Batch compose request: {num_images} images, backgrounds={backgrounds_dir}, objects={objects_dir}")

    try:
        registry = get_service_registry()

        # Ensure water_clarity is a string
        if 'effects_config' in request_data and isinstance(request_data['effects_config'], dict):
            wc = request_data['effects_config'].get('water_clarity')
            if hasattr(wc, 'value'):
                request_data['effects_config']['water_clarity'] = wc.value

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


@router.post("/jobs/{job_id}/delete")
async def delete_job(job_id: str):
    """
    Permanently delete a job from the database.

    If the job is active (queued/processing), it will be automatically stopped
    before deletion. Generated images are preserved on disk.

    **Warning**: This action cannot be undone.
    """
    logger.info(f"Delete request for job: {job_id}")

    try:
        registry = get_service_registry()
        result = await registry.augmentor.post(f"/jobs/{job_id}/delete", {})

        return result

    except Exception as e:
        logger.error(f"Delete job failed: {e}")
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


# =============================================================================
# Job Management Endpoints (Resume, Retry, Logs)
# =============================================================================

@router.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str):
    """
    Resume an interrupted job from its checkpoint.

    Only works for jobs with status 'interrupted' that have a saved checkpoint.
    The job will continue generating from where it left off.
    """
    logger.info(f"Resume request for job: {job_id}")

    try:
        registry = get_service_registry()
        result = await registry.augmentor.post(f"/jobs/{job_id}/resume", {})

        return result

    except Exception as e:
        logger.error(f"Resume job failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/retry")
async def retry_job(job_id: str):
    """
    Retry a failed job from scratch.

    Creates a new job with the same parameters as the original.
    The original job is preserved in history.
    """
    logger.info(f"Retry request for job: {job_id}")

    try:
        registry = get_service_registry()
        result = await registry.augmentor.post(f"/jobs/{job_id}/retry", {})

        return result

    except Exception as e:
        logger.error(f"Retry job failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/regenerate-dataset")
async def regenerate_dataset(job_id: str, force: bool = False):
    """
    Regenerate the synthetic_dataset.json (COCO format) from individual annotation files.

    This is useful for:
    - Jobs that were cancelled/failed before COCO JSON was generated
    - Jobs where the COCO JSON was corrupted or deleted
    - Manual regeneration when desired

    Args:
        job_id: The job ID to regenerate dataset for
        force: If True, regenerate even if synthetic_dataset.json already exists
    """
    logger.info(f"Regenerate dataset request for job: {job_id}, force={force}")

    try:
        registry = get_service_registry()
        # Pass force parameter in the URL query string
        force_param = "true" if force else "false"
        result = await registry.augmentor.post(
            f"/jobs/{job_id}/regenerate-dataset?force={force_param}",
            {}
        )

        return result

    except Exception as e:
        logger.error(f"Regenerate dataset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/logs")
async def get_job_logs(job_id: str, level: Optional[str] = None, limit: int = 100):
    """
    Get logs for a specific job.

    Args:
        job_id: The job ID
        level: Filter by log level (INFO, WARNING, ERROR)
        limit: Maximum number of logs to return
    """
    logger.info(f"Get logs request for job: {job_id}")

    try:
        registry = get_service_registry()
        params = {}
        if level:
            params["level"] = level
        if limit:
            params["limit"] = limit

        result = await registry.augmentor.get(f"/jobs/{job_id}/logs", params=params)

        return result

    except Exception as e:
        logger.error(f"Get job logs failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Dataset Management Endpoints
# =============================================================================

@router.get("/datasets")
async def list_datasets(
    dataset_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """List all generated datasets with metadata."""
    logger.info(f"List datasets request - type: {dataset_type}, limit: {limit}")

    try:
        registry = get_service_registry()

        params = f"?limit={limit}&offset={offset}"
        if dataset_type:
            params += f"&dataset_type={dataset_type}"

        result = await registry.augmentor.get(f"/datasets{params}")

        return result

    except Exception as e:
        logger.error(f"List datasets failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/{job_id}")
async def get_dataset_metadata(job_id: str):
    """Get detailed metadata for a specific dataset."""
    logger.info(f"Get dataset metadata request for: {job_id}")

    try:
        registry = get_service_registry()
        result = await registry.augmentor.get(f"/datasets/{job_id}")

        return result

    except Exception as e:
        logger.error(f"Get dataset metadata failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/{job_id}/coco")
async def load_dataset_coco(job_id: str):
    """Load the full COCO JSON for a dataset."""
    logger.info(f"Load dataset COCO request for: {job_id}")

    try:
        registry = get_service_registry()
        result = await registry.augmentor.get(f"/datasets/{job_id}/coco")

        return result

    except Exception as e:
        logger.error(f"Load dataset COCO failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
