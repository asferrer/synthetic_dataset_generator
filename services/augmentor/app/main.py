"""
Augmentor Service - Main FastAPI Application
=============================================
Microservice for synthetic image composition with quality validation.

Endpoints:
- POST /compose          - Compose single synthetic image
- POST /compose-batch    - Compose multiple images (async job)
- POST /validate         - Validate image quality
- POST /lighting         - Estimate lighting from background
- GET  /health           - Health check
- GET  /info             - Service information
"""

import os
import uuid
import asyncio
from datetime import datetime
from typing import Dict, Optional
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.models.schemas import (
    # Requests
    ComposeRequest,
    ComposeBatchRequest,
    ValidateRequest,
    LightingRequest,
    # Responses
    ComposeResponse,
    ComposeBatchResponse,
    ValidateResponse,
    LightingResponse,
    HealthResponse,
    InfoResponse,
    # Data models
    EffectType,
)


# =============================================================================
# Global State
# =============================================================================

class ServiceState:
    """Global state for the Augmentor service"""
    def __init__(self):
        self.composer = None
        self.quality_validator = None
        self.physics_validator = None
        self.gpu_available = False
        self.gpu_name = None
        self.jobs: Dict[str, dict] = {}  # In-memory job queue

    async def initialize(self):
        """Initialize service components"""
        # Check GPU availability
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detected: {self.gpu_name}")
        else:
            logger.warning("No GPU detected, running on CPU")

        # Initialize components (lazy loading)
        try:
            from app.composer import ImageComposer
            self.composer = ImageComposer(use_gpu=self.gpu_available)
            logger.info("ImageComposer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ImageComposer: {e}")

        try:
            from app.validators import QualityValidator, PhysicsValidator
            self.quality_validator = QualityValidator(use_gpu=self.gpu_available)
            self.physics_validator = PhysicsValidator()
            logger.info("Validators initialized")
        except Exception as e:
            logger.error(f"Failed to initialize validators: {e}")

    def get_gpu_memory(self) -> Optional[str]:
        """Get GPU memory usage"""
        if not self.gpu_available:
            return None
        try:
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"{allocated:.1f}GB / {total:.1f}GB"
        except:
            return None


state = ServiceState()


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup service resources"""
    logger.info("Starting Augmentor Service...")
    await state.initialize()
    logger.info("Augmentor Service ready")
    yield
    logger.info("Shutting down Augmentor Service...")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Augmentor Service",
    description="""
    Microservice for synthetic image composition with quality validation.

    ## Features
    - **Object Composition**: Depth-aware object placement with realistic scaling
    - **Multi-Light Shadows**: Photorealistic shadows from detected light sources
    - **Quality Validation**: LPIPS perceptual quality and anomaly detection
    - **Physics Validation**: Gravity, buoyancy, and occlusion checks
    - **11+ Realism Effects**: Color correction, blur matching, caustics, etc.
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health and Info Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.

    Returns service health status including GPU availability and component status.
    """
    validators_loaded = (
        state.quality_validator is not None and
        state.physics_validator is not None
    )

    status = "healthy"
    if not state.composer:
        status = "degraded"
    if not validators_loaded and not state.composer:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        gpu_available=state.gpu_available,
        gpu_name=state.gpu_name,
        gpu_memory_used=state.get_gpu_memory(),
        validators_loaded=validators_loaded,
        timestamp=datetime.now(),
    )


@app.get("/info", response_model=InfoResponse, tags=["System"])
async def service_info():
    """
    Get service information and capabilities.
    """
    return InfoResponse(
        service="augmentor",
        version="1.0.0",
        description="Synthetic image composition with quality validation",
        endpoints=[
            "POST /compose",
            "POST /compose-batch",
            "POST /validate",
            "POST /lighting",
            "GET /health",
            "GET /info",
        ],
        capabilities={
            "gpu_acceleration": state.gpu_available,
            "quality_validation": state.quality_validator is not None,
            "physics_validation": state.physics_validator is not None,
            "depth_aware_placement": True,
            "multi_light_shadows": True,
            "caustics_effects": True,
        },
        effects_available=[e.value for e in EffectType],
    )


# =============================================================================
# Compose Endpoints
# =============================================================================

@app.post("/compose", response_model=ComposeResponse, tags=["Composition"])
async def compose_image(request: ComposeRequest):
    """
    Compose a single synthetic image.

    Takes a background image and places objects with specified effects.
    Optionally validates quality and physics plausibility.

    ## Parameters
    - **background_path**: Path to background image
    - **objects**: List of objects to place (image_path, class_name, position, scale, etc.)
    - **effects**: List of effects to apply (color_correction, blur_matching, shadows, etc.)
    - **effects_config**: Configuration for effect intensities
    - **validate_quality**: Run LPIPS quality validation
    - **validate_physics**: Run physics plausibility checks
    - **output_path**: Where to save the composed image

    ## Returns
    - Composed image path
    - Annotations (bounding boxes)
    - Validation results if requested
    """
    import time
    start_time = time.time()

    if not state.composer:
        raise HTTPException(status_code=503, detail="Composer not initialized")

    try:
        # Compose the image
        result = await state.composer.compose(
            background_path=request.background_path,
            objects=request.objects,
            depth_map_path=request.depth_map_path,
            lighting_info=request.lighting_info,
            effects=request.effects,
            effects_config=request.effects_config,
            output_path=request.output_path,
            save_annotations=request.save_annotations,
        )

        # Quality validation if requested
        quality_score = None
        if request.validate_quality and state.quality_validator:
            quality_score = await state.quality_validator.validate(
                image_path=request.output_path,
                annotations=result.annotations,
            )

        # Physics validation if requested
        physics_violations = []
        if request.validate_physics and state.physics_validator:
            physics_violations = await state.physics_validator.validate(
                annotations=result.annotations,
                depth_map_path=request.depth_map_path,
            )

        # Determine if valid
        is_valid = True
        rejection_reason = None

        if quality_score and not quality_score.overall_pass:
            is_valid = False
            rejection_reason = f"Quality score {quality_score.overall_score:.2f} below threshold"

        if physics_violations:
            high_severity = [v for v in physics_violations if v.severity == "high"]
            if high_severity:
                is_valid = False
                rejection_reason = f"Physics violations: {len(high_severity)} high severity"

        processing_time = (time.time() - start_time) * 1000

        return ComposeResponse(
            success=True,
            output_path=request.output_path,
            annotations=result.annotations,
            objects_placed=result.objects_placed,
            depth_used=result.depth_used,
            effects_applied=result.effects_applied,
            quality_score=quality_score,
            physics_violations=physics_violations,
            is_valid=is_valid,
            rejection_reason=rejection_reason,
            processing_time_ms=processing_time,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Composition failed: {e}")
        return ComposeResponse(
            success=False,
            output_path=request.output_path,
            objects_placed=0,
            depth_used=False,
            effects_applied=[],
            is_valid=False,
            processing_time_ms=(time.time() - start_time) * 1000,
            error=str(e),
        )


@app.post("/compose-batch", response_model=ComposeBatchResponse, tags=["Composition"])
async def compose_batch(request: ComposeBatchRequest, background_tasks: BackgroundTasks):
    """
    Start a batch composition job.

    Creates multiple synthetic images asynchronously. Use GET /jobs/{job_id}
    to check progress.

    ## Parameters
    - **backgrounds_dir**: Directory containing background images
    - **objects_dir**: Directory with object images organized by class
    - **output_dir**: Output directory for generated images
    - **num_images**: Number of images to generate
    - **targets_per_class**: Optional dict specifying target count per class
    - **effects**: Effects to apply to all images

    ## Returns
    - **job_id**: Use this to track progress
    - **status**: "queued" initially
    """
    import time
    start_time = time.time()

    if not state.composer:
        raise HTTPException(status_code=503, detail="Composer not initialized")

    # Create job with unique output directory
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    job_output_dir = os.path.join(request.output_dir, job_id)
    job = {
        "id": job_id,
        "status": "queued",
        "request": request.model_dump(),
        "output_dir": job_output_dir,
        "images_generated": 0,
        "images_rejected": 0,
        "images_pending": request.num_images,
        "synthetic_counts": {},
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
        "error": None,
        "cancelled": False,  # Flag for cancellation
    }
    state.jobs[job_id] = job

    # Start background task
    background_tasks.add_task(run_batch_composition, job_id, request)

    return ComposeBatchResponse(
        success=True,
        job_id=job_id,
        status="queued",
        images_pending=request.num_images,
        processing_time_ms=(time.time() - start_time) * 1000,
    )


@app.get("/jobs/{job_id}", response_model=ComposeBatchResponse, tags=["Composition"])
async def get_job_status(job_id: str):
    """
    Get the status of a batch composition job.
    """
    if job_id not in state.jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = state.jobs[job_id]

    return ComposeBatchResponse(
        success=job["error"] is None,
        job_id=job_id,
        status=job["status"],
        images_generated=job["images_generated"],
        images_rejected=job["images_rejected"],
        images_pending=job["images_pending"],
        synthetic_counts=job["synthetic_counts"],
        output_coco_path=job.get("output_coco_path"),
        output_dir=job.get("output_dir"),
        processing_time_ms=0,
        error=job["error"],
    )


@app.get("/jobs", tags=["Composition"])
async def list_jobs():
    """
    List all batch composition jobs.

    Returns a list of all jobs with their current status, progress, and output location.
    Jobs are sorted by creation time (most recent first).
    """
    jobs_list = []
    for job_id, job in state.jobs.items():
        jobs_list.append({
            "job_id": job_id,
            "status": job["status"],
            "images_generated": job["images_generated"],
            "images_rejected": job["images_rejected"],
            "images_pending": job["images_pending"],
            "synthetic_counts": job["synthetic_counts"],
            "created_at": job["created_at"],
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
            "output_dir": job.get("output_dir"),
            "output_coco_path": job.get("output_coco_path"),
            "error": job.get("error"),
        })

    # Sort by creation time (most recent first)
    jobs_list.sort(key=lambda x: x["created_at"], reverse=True)

    return {"jobs": jobs_list, "total": len(jobs_list)}


@app.delete("/jobs/{job_id}", tags=["Composition"])
async def cancel_job(job_id: str):
    """
    Cancel a running batch composition job.

    Sets the cancelled flag which stops the job at the next iteration.
    Already generated images are preserved.
    """
    if job_id not in state.jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = state.jobs[job_id]

    # Check if job can be cancelled
    if job["status"] in ["completed", "failed", "cancelled"]:
        return {
            "success": False,
            "job_id": job_id,
            "message": f"Job already {job['status']}, cannot cancel",
            "status": job["status"],
        }

    # Set cancellation flag
    job["cancelled"] = True
    logger.info(f"Cancellation requested for job {job_id}")

    # If job is still queued, mark as cancelled immediately
    if job["status"] == "queued":
        job["status"] = "cancelled"
        job["completed_at"] = datetime.now().isoformat()
        return {
            "success": True,
            "job_id": job_id,
            "message": "Job cancelled before starting",
            "status": "cancelled",
        }

    return {
        "success": True,
        "job_id": job_id,
        "message": "Cancellation requested, job will stop after current image",
        "status": "cancelling",
    }


async def run_batch_composition(job_id: str, request: ComposeBatchRequest):
    """Background task for batch composition"""
    job = state.jobs[job_id]

    # Check if already cancelled before starting
    if job.get("cancelled", False):
        job["status"] = "cancelled"
        job["completed_at"] = datetime.now().isoformat()
        logger.info(f"Job {job_id} was cancelled before starting")
        return

    job["status"] = "processing"
    job["started_at"] = datetime.now().isoformat()

    # Use the job-specific output directory
    job_output_dir = job["output_dir"]

    # Cancellation check callback
    def check_cancelled() -> bool:
        return job.get("cancelled", False)

    try:
        result = await state.composer.compose_batch(
            backgrounds_dir=request.backgrounds_dir,
            objects_dir=request.objects_dir,
            output_dir=job_output_dir,  # Use job-specific output dir
            num_images=request.num_images,
            targets_per_class=request.targets_per_class,
            max_objects_per_image=request.max_objects_per_image,
            effects=request.effects,
            effects_config=request.effects_config,
            depth_aware=request.depth_aware,
            depth_service_url=request.depth_service_url,
            validate_quality=request.validate_quality,
            validate_physics=request.validate_physics,
            reject_invalid=request.reject_invalid,
            save_pipeline_debug=request.save_pipeline_debug,  # NEW: save pipeline steps
            progress_callback=lambda p: update_job_progress(job_id, p),
            cancel_check=check_cancelled,  # NEW: cancellation check
        )

        # Check if job was cancelled
        if job.get("cancelled", False):
            job["status"] = "cancelled"
            logger.info(f"Job {job_id} was cancelled after {result.images_generated} images")
        else:
            job["status"] = "completed"

        job["images_generated"] = result.images_generated
        job["images_rejected"] = result.images_rejected
        job["images_pending"] = 0
        job["synthetic_counts"] = result.synthetic_counts
        job["output_coco_path"] = result.output_coco_path
        job["completed_at"] = datetime.now().isoformat()

    except Exception as e:
        logger.exception(f"Batch job {job_id} failed: {e}")
        job["status"] = "failed"
        job["error"] = str(e)
        job["completed_at"] = datetime.now().isoformat()


def update_job_progress(job_id: str, progress: dict):
    """Update job progress from callback"""
    if job_id in state.jobs:
        job = state.jobs[job_id]
        job["images_generated"] = progress.get("generated", 0)
        job["images_rejected"] = progress.get("rejected", 0)
        job["images_pending"] = progress.get("pending", 0)
        job["synthetic_counts"] = progress.get("counts", {})


# =============================================================================
# Validation Endpoint
# =============================================================================

@app.post("/validate", response_model=ValidateResponse, tags=["Validation"])
async def validate_image(request: ValidateRequest):
    """
    Validate a composed image for quality and physics plausibility.

    ## Checks performed:
    - **LPIPS Quality**: Perceptual quality score (0-1)
    - **Anomaly Detection**: Isolation Forest anomaly score
    - **Physics Validation**: Gravity, buoyancy, occlusion checks

    ## Parameters
    - **image_path**: Path to image to validate
    - **annotations**: Object bounding boxes
    - **check_quality**: Enable LPIPS check
    - **check_anomalies**: Enable anomaly detection
    - **check_physics**: Enable physics validation
    """
    import time
    start_time = time.time()

    quality_validator = state.quality_validator
    physics_validator = state.physics_validator

    if not quality_validator:
        raise HTTPException(status_code=503, detail="Quality validator not initialized")

    try:
        # Run quality validation
        quality_score = await quality_validator.validate(
            image_path=request.image_path,
            annotations=request.annotations,
            reference_images=request.reference_images,
            check_quality=request.check_quality,
            check_anomalies=request.check_anomalies,
            min_perceptual_quality=request.min_perceptual_quality,
            min_anomaly_score=request.min_anomaly_score,
        )

        # Run physics validation if requested
        physics_violations = []
        anomalies = []

        if request.check_physics and physics_validator:
            physics_violations = await physics_validator.validate(
                annotations=request.annotations,
                depth_map_path=request.depth_map_path,
            )

        # Check for anomalies
        if not quality_score.overall_pass:
            anomalies.append({
                "type": "quality_failure",
                "score": quality_score.overall_score,
                "details": "Image failed quality thresholds",
            })

        is_valid = quality_score.overall_pass and len(physics_violations) == 0

        return ValidateResponse(
            is_valid=is_valid,
            quality_score=quality_score,
            anomalies=anomalies,
            physics_violations=physics_violations,
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Lighting Endpoint
# =============================================================================

@app.post("/lighting", response_model=LightingResponse, tags=["Lighting"])
async def estimate_lighting(request: LightingRequest):
    """
    Estimate lighting conditions from a background image.

    Detects light sources, estimates dominant direction, color temperature,
    and ambient intensity. Can apply underwater light attenuation.

    ## Parameters
    - **image_path**: Path to background image
    - **max_light_sources**: Maximum light sources to detect (1-5)
    - **intensity_threshold**: Minimum intensity for light detection
    - **apply_water_attenuation**: Apply underwater light falloff
    - **depth_category**: Water depth (near/mid/far)
    - **water_clarity**: Water clarity (clear/murky/very_murky)

    ## Returns
    - List of detected light sources with position, intensity, color
    - Dominant light direction
    - Color temperature in Kelvin
    - Ambient intensity
    """
    import time
    start_time = time.time()

    if not state.composer:
        raise HTTPException(status_code=503, detail="Composer not initialized")

    try:
        lighting_info = await state.composer.estimate_lighting(
            image_path=request.image_path,
            max_light_sources=request.max_light_sources,
            intensity_threshold=request.intensity_threshold,
            estimate_hdr=request.estimate_hdr,
            apply_water_attenuation=request.apply_water_attenuation,
            depth_category=request.depth_category,
            water_clarity=request.water_clarity,
        )

        return LightingResponse(
            success=True,
            lighting_info=lighting_info,
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Lighting estimation failed: {e}")
        return LightingResponse(
            success=False,
            lighting_info=None,
            processing_time_ms=(time.time() - start_time) * 1000,
            error=str(e),
        )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8004,
        reload=True,
        log_level="info",
    )
