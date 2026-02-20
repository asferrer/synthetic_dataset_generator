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
import sys
import json
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Add shared module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.job_database import JobDatabase, get_job_db
from shared.job_logger import JobLogger

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
        self.db: Optional[JobDatabase] = None  # Database for job persistence
        # In-memory cache for active jobs (for quick access during processing)
        self._active_jobs_cache: Dict[str, dict] = {}

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
        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {e}")
            return None


state = ServiceState()


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup service resources"""
    logger.info("Starting Augmentor Service...")

    # Initialize database
    try:
        state.db = get_job_db()
        logger.info("Job database initialized")

        # Recover orphaned queued jobs
        state.db.mark_orphaned_jobs("augmentor")

        # For running jobs, check if they have checkpoints for resume
        processing_jobs = state.db.list_jobs(service="augmentor", status="interrupted")
        for job in processing_jobs:
            output_path = job.get("output_path", "")
            progress_file = Path(output_path) / "progress.json" if output_path else None

            if progress_file and progress_file.exists():
                logger.info(f"Job {job['id']} interrupted with checkpoint - can be resumed")
            else:
                state.db.complete_job(
                    job["id"],
                    "failed",
                    error_message="Service restarted without checkpoint"
                )
                logger.warning(f"Job {job['id']} marked as failed - no checkpoint found")

    except Exception as e:
        logger.error(f"Failed to initialize job database: {e}")

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
    - **status**: "pending" initially
    """
    import time
    start_time = time.time()

    if not state.composer:
        raise HTTPException(status_code=503, detail="Composer not initialized")

    if not state.db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    # Create job with unique output directory
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    job_output_dir = os.path.join(request.output_dir, job_id)

    # Create job in database
    state.db.create_job(
        job_id=job_id,
        job_type="generation",
        service="augmentor",
        request_params=request.model_dump(),
        total_items=request.num_images,
        output_path=job_output_dir
    )

    # Cache in memory for quick access during processing
    state._active_jobs_cache[job_id] = {
        "cancelled": False,
        "output_dir": job_output_dir,
    }

    # Log job creation
    job_logger = JobLogger(job_id, state.db)
    job_logger.info("Job created", {
        "num_images": request.num_images,
        "targets_per_class": request.targets_per_class,
        "output_dir": job_output_dir
    })

    # Start background task
    background_tasks.add_task(run_batch_composition, job_id, request, job_output_dir)

    return ComposeBatchResponse(
        success=True,
        job_id=job_id,
        status="pending",
        images_pending=request.num_images,
        processing_time_ms=(time.time() - start_time) * 1000,
    )


@app.get("/jobs/{job_id}", response_model=ComposeBatchResponse, tags=["Composition"])
async def get_job_status(job_id: str):
    """
    Get the status of a batch composition job.
    """
    if not state.db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    job = state.db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Extract progress details
    progress = job.get("progress_details") or {}
    result = job.get("result_summary") or {}

    total_items = job.get("total_items", 0)
    processed = job.get("processed_items", 0)

    # Use effective total_target if available (considers per-class targets)
    effective_total = progress.get("total_target") or total_items
    # Use pending from progress if available (already calculated correctly)
    effective_pending = progress.get("pending") if progress.get("pending") is not None else max(0, effective_total - processed)

    return ComposeBatchResponse(
        success=job["error_message"] is None,
        job_id=job_id,
        status=job["status"],
        images_generated=processed,
        images_rejected=job.get("failed_items", 0),
        images_pending=effective_pending,
        total_items=effective_total,
        synthetic_counts=progress.get("synthetic_counts", {}),
        output_coco_path=result.get("output_coco_path"),
        output_dir=job.get("output_path"),
        processing_time_ms=job.get("processing_time_ms", 0),
        error=job.get("error_message"),
    )


@app.get("/jobs", tags=["Composition"])
async def list_jobs():
    """
    List all batch composition jobs.

    Returns a list of all jobs with their current status, progress, and output location.
    Jobs are sorted by creation time (most recent first).
    """
    if not state.db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    jobs = state.db.list_jobs(service="augmentor", job_type="generation")

    jobs_list = []
    for job in jobs:
        progress = job.get("progress_details") or {}
        result = job.get("result_summary") or {}
        request_params = job.get("request_params") or {}

        total_items = job.get("total_items", 0)
        processed = job.get("processed_items", 0)

        # Use effective total_target if available (considers per-class targets)
        effective_total = progress.get("total_target") or total_items
        # Use pending from progress if available (already calculated correctly)
        effective_pending = progress.get("pending") if progress.get("pending") is not None else max(0, effective_total - processed)

        # Get per-class data for monitoring
        synthetic_counts = progress.get("synthetic_counts", {})
        targets_per_class = request_params.get("targets_per_class", {})

        # Calculate progress percentage
        progress_pct = round((processed / effective_total * 100), 1) if effective_total > 0 else 0.0

        jobs_list.append({
            "job_id": job["id"],
            "type": "generation",  # Frontend expects 'type' field
            "status": job["status"],
            "progress": progress_pct,  # Add progress percentage for JobMonitor
            "images_generated": processed,
            "images_rejected": job.get("failed_items", 0),
            "images_pending": effective_pending,
            "total_items": effective_total,
            "synthetic_counts": synthetic_counts,
            "targets_per_class": targets_per_class,
            "generated_per_class": synthetic_counts,  # Alias for frontend
            "current_category": progress.get("current_category", ""),
            "created_at": job.get("created_at"),
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
            "output_dir": job.get("output_path"),
            "output_coco_path": result.get("output_coco_path"),
            "error": job.get("error_message"),
        })

    return {"jobs": jobs_list, "total": len(jobs_list)}


@app.delete("/jobs/{job_id}", tags=["Composition"])
async def cancel_job(job_id: str):
    """
    Cancel a running batch composition job.

    Sets the cancelled flag which stops the job at the next iteration.
    Already generated images are preserved.
    """
    if not state.db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    job = state.db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Check if job can be cancelled
    if job["status"] in ["completed", "failed", "cancelled"]:
        return {
            "success": False,
            "job_id": job_id,
            "message": f"Job already {job['status']}, cannot cancel",
            "status": job["status"],
        }

    # Set cancellation flag in cache
    if job_id in state._active_jobs_cache:
        state._active_jobs_cache[job_id]["cancelled"] = True
    logger.info(f"Cancellation requested for job {job_id}")

    # Log cancellation request
    job_logger = JobLogger(job_id, state.db)
    job_logger.info("Cancellation requested")

    # If job is still pending, mark as cancelled immediately
    if job["status"] == "pending":
        state.db.complete_job(job_id, "cancelled")
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


@app.post("/jobs/{job_id}/delete", tags=["Composition"])
async def delete_job(job_id: str):
    """
    Permanently delete a job from the database.

    If the job is active (pending/running), it will be stopped first,
    then deleted from the database. Generated images are preserved on disk.

    **Important**: This action cannot be undone.

    Args:
        job_id: The job ID to delete

    Returns:
        Success status and message
    """
    if not state.db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    # Get job info
    job = state.db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job_status = job["status"]

    # Step 1: Stop active jobs first
    if job_status in ["pending", "running"]:
        # Set cancellation flag if in cache
        if job_id in state._active_jobs_cache:
            state._active_jobs_cache[job_id]["cancelled"] = True
            logger.info(f"Set cancellation flag for job {job_id} before deletion")

        # If pending, mark as cancelled in DB
        if job_status == "pending":
            state.db.complete_job(job_id, "cancelled")
            logger.info(f"Cancelled pending job {job_id}")

        # If running, wait briefly for graceful stop
        elif job_status == "running":
            # Give it a moment to check the cancellation flag
            await asyncio.sleep(1.0)

            # Update status to cancelled
            state.db.complete_job(job_id, "cancelled")
            logger.info(f"Stopped processing job {job_id}")

    # Step 2: Delete from database
    try:
        deleted = state.db.delete_job(job_id)

        if deleted:
            # Clean up cache if present
            if job_id in state._active_jobs_cache:
                del state._active_jobs_cache[job_id]

            logger.info(f"Job {job_id} permanently deleted from database")

            return {
                "success": True,
                "job_id": job_id,
                "message": f"Job deleted (was {job_status})",
                "status": "deleted",
                "note": "Generated images preserved on disk"
            }
        else:
            # This shouldn't happen if job exists, but handle it
            return {
                "success": False,
                "job_id": job_id,
                "message": "Job could not be deleted",
                "error": "Database delete returned false"
            }

    except Exception as e:
        logger.exception(f"Failed to delete job {job_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete job: {str(e)}"
        )


@app.post("/jobs/{job_id}/resume", tags=["Composition"])
async def resume_job(job_id: str, background_tasks: BackgroundTasks):
    """
    Resume an interrupted job from its checkpoint.

    Only works for jobs with status 'interrupted' that have a saved checkpoint.
    The job will continue from where it left off.
    """
    if not state.db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    if not state.composer:
        raise HTTPException(status_code=503, detail="Composer not initialized")

    job = state.db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Check if job can be resumed
    if job["status"] not in ["interrupted", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Job status is '{job['status']}', only 'interrupted' or 'failed' jobs can be resumed"
        )

    output_dir = job.get("output_path")
    if not output_dir:
        raise HTTPException(status_code=400, detail="Job has no output directory")

    # Load checkpoint
    checkpoint = load_checkpoint(output_dir)
    if not checkpoint:
        raise HTTPException(
            status_code=400,
            detail="No checkpoint found. Use /jobs/{job_id}/retry to restart from scratch"
        )

    # Get original request params
    request_params = job.get("request_params")
    if not request_params:
        raise HTTPException(status_code=400, detail="Original request parameters not found")

    # Create request object from stored params
    request = ComposeBatchRequest(**request_params)

    # Update job status
    state.db.update_job_status(job_id, "pending")

    # Cache for cancellation
    state._active_jobs_cache[job_id] = {
        "cancelled": False,
        "output_dir": output_dir,
    }

    # Log resume
    job_logger = JobLogger(job_id, state.db)
    job_logger.info("Job resumed from checkpoint", {
        "checkpoint_generated": checkpoint.get("generated", 0),
        "checkpoint_rejected": checkpoint.get("rejected", 0),
    })

    # Start background task with resume info
    background_tasks.add_task(
        run_batch_composition_resume,
        job_id,
        request,
        output_dir,
        checkpoint
    )

    return {
        "success": True,
        "job_id": job_id,
        "message": f"Job resumed from checkpoint ({checkpoint.get('generated', 0)} images already generated)",
        "status": "pending",
        "checkpoint": checkpoint,
    }


@app.post("/jobs/{job_id}/retry", tags=["Composition"])
async def retry_job(job_id: str, background_tasks: BackgroundTasks):
    """
    Retry a failed job from scratch.

    Creates a new job with the same parameters as the original.
    The original job is preserved in history.
    """
    if not state.db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    if not state.composer:
        raise HTTPException(status_code=503, detail="Composer not initialized")

    job = state.db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Check if job can be retried
    if job["status"] not in ["failed", "cancelled"]:
        raise HTTPException(
            status_code=400,
            detail=f"Job status is '{job['status']}', only 'failed' or 'cancelled' jobs can be retried"
        )

    # Get original request params
    request_params = job.get("request_params")
    if not request_params:
        raise HTTPException(status_code=400, detail="Original request parameters not found")

    # Create request object from stored params
    request = ComposeBatchRequest(**request_params)

    # Create NEW job with new ID
    new_job_id = f"job_{uuid.uuid4().hex[:12]}"
    new_output_dir = os.path.join(request.output_dir, new_job_id)

    # Create job in database
    state.db.create_job(
        job_id=new_job_id,
        job_type="generation",
        service="augmentor",
        request_params=request_params,
        total_items=request.num_images,
        output_path=new_output_dir
    )

    # Cache for cancellation
    state._active_jobs_cache[new_job_id] = {
        "cancelled": False,
        "output_dir": new_output_dir,
    }

    # Log retry
    job_logger = JobLogger(new_job_id, state.db)
    job_logger.info("Job created as retry", {"original_job_id": job_id})

    # Start background task
    background_tasks.add_task(run_batch_composition, new_job_id, request, new_output_dir)

    return {
        "success": True,
        "job_id": new_job_id,
        "original_job_id": job_id,
        "message": "New job created as retry of original",
        "status": "pending",
    }


@app.post("/jobs/{job_id}/regenerate-dataset", tags=["Composition"])
async def regenerate_dataset(job_id: str, force: bool = False):
    """
    Regenerate the synthetic_dataset.json (COCO format) from individual annotation files.

    This is useful for:
    - Jobs that were cancelled/failed before COCO JSON was generated
    - Jobs where the COCO JSON was corrupted or deleted
    - Manual regeneration when desired

    The function scans the job's images directory for annotation files (*_annotations.json)
    and reconstructs the complete COCO JSON from them.

    ## Parameters
    - **job_id**: The job ID to regenerate dataset for
    - **force**: If True, regenerate even if synthetic_dataset.json already exists

    ## Returns
    - Success status and path to generated COCO JSON
    - Statistics about images and annotations found
    """
    if not state.db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    job = state.db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    output_dir = job.get("output_path")
    if not output_dir:
        raise HTTPException(status_code=400, detail="Job has no output directory")

    images_dir = os.path.join(output_dir, "images")
    if not os.path.exists(images_dir):
        raise HTTPException(status_code=400, detail=f"Images directory not found: {images_dir}")

    coco_path = os.path.join(output_dir, "synthetic_dataset.json")

    # Check if COCO already exists
    if os.path.exists(coco_path) and not force:
        return {
            "success": False,
            "job_id": job_id,
            "message": "synthetic_dataset.json already exists. Use force=true to regenerate.",
            "coco_path": coco_path,
        }

    # Regenerate COCO from individual annotations
    try:
        result = _regenerate_coco_from_annotations(images_dir, coco_path)

        # Update job result_summary with new COCO path
        current_result = job.get("result_summary") or {}
        current_result["output_coco_path"] = coco_path
        state.db.complete_job(
            job_id,
            job["status"],  # Keep current status
            result_summary=current_result,
            processing_time_ms=job.get("processing_time_ms", 0)
        )

        # Also update/create dataset metadata
        if result["num_images"] > 0:
            try:
                request_params = job.get("request_params") or {}
                request = ComposeBatchRequest(**request_params) if request_params else None
                metadata = _extract_dataset_metadata(
                    job_id=job_id,
                    output_dir=output_dir,
                    coco_path=coco_path,
                    request=request
                )
                # Try to update existing or create new
                existing_metadata = state.db.get_dataset_metadata(job_id)
                if existing_metadata:
                    # Delete and recreate (simpler than update)
                    state.db.delete_dataset_metadata(job_id)
                state.db.create_dataset_metadata(**metadata)
                logger.info(f"Dataset metadata updated for job {job_id}")
            except Exception as e:
                logger.warning(f"Failed to update dataset metadata: {e}")

        logger.info(f"Regenerated COCO JSON for job {job_id}: {result['num_images']} images, {result['num_annotations']} annotations")

        return {
            "success": True,
            "job_id": job_id,
            "message": "COCO JSON regenerated successfully",
            "coco_path": coco_path,
            "num_images": result["num_images"],
            "num_annotations": result["num_annotations"],
            "num_categories": result["num_categories"],
            "categories": result["categories"],
        }

    except Exception as e:
        logger.exception(f"Failed to regenerate COCO for job {job_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to regenerate COCO JSON: {str(e)}"
        )


def _regenerate_coco_from_annotations(images_dir: str, output_coco_path: str) -> dict:
    """
    Regenerate COCO JSON from individual annotation files.

    Scans the images directory for image files and their corresponding annotation files,
    then constructs a complete COCO format JSON.

    Args:
        images_dir: Directory containing images and *_annotations.json files
        output_coco_path: Path where to save the regenerated COCO JSON

    Returns:
        Dictionary with statistics about regeneration
    """
    import json
    import glob
    from datetime import datetime
    from PIL import Image

    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))

    # Sort by name for consistent ordering
    image_files.sort()

    # Collect all annotations and categories
    categories_set = set()
    images_list = []
    annotations_list = []
    annotation_id = 1

    for img_idx, img_path in enumerate(image_files):
        img_filename = os.path.basename(img_path)
        base_name = os.path.splitext(img_filename)[0]
        ann_path = os.path.join(images_dir, f"{base_name}_annotations.json")

        # Get image dimensions
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception as e:
            logger.warning(f"Could not read image {img_path}: {e}")
            continue

        # Add image entry
        images_list.append({
            "id": img_idx,
            "file_name": img_filename,
            "width": width,
            "height": height
        })

        # Load annotations if they exist
        if os.path.exists(ann_path):
            try:
                with open(ann_path, 'r') as f:
                    annotations = json.load(f)

                for ann in annotations:
                    class_name = ann.get("class_name", "Unknown")
                    categories_set.add(class_name)

                    # Convert to COCO format
                    x = ann.get("x", 0)
                    y = ann.get("y", 0)
                    w = ann.get("width", 0)
                    h = ann.get("height", 0)
                    area = ann.get("area", w * h)

                    annotations_list.append({
                        "id": annotation_id,
                        "image_id": img_idx,
                        "category_name": class_name,  # Temporary, will be replaced with ID
                        "bbox": [x, y, w, h],
                        "area": area,
                        "segmentation": [],
                        "iscrowd": 0
                    })
                    annotation_id += 1
            except Exception as e:
                logger.warning(f"Could not read annotations {ann_path}: {e}")

    # Create category list with IDs
    categories_list = sorted(list(categories_set))
    category_to_id = {name: idx for idx, name in enumerate(categories_list)}

    categories_coco = [
        {"id": idx, "name": name, "supercategory": "object"}
        for idx, name in enumerate(categories_list)
    ]

    # Update annotations with category IDs
    for ann in annotations_list:
        cat_name = ann.pop("category_name")
        ann["category_id"] = category_to_id.get(cat_name, 0)

    # Create COCO structure
    coco_data = {
        "info": {
            "description": "Synthetic Dataset (Regenerated)",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [],
        "images": images_list,
        "annotations": annotations_list,
        "categories": categories_coco
    }

    # Save COCO JSON
    with open(output_coco_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    return {
        "num_images": len(images_list),
        "num_annotations": len(annotations_list),
        "num_categories": len(categories_coco),
        "categories": categories_list
    }


@app.get("/jobs/{job_id}/logs", tags=["Composition"])
async def get_job_logs(job_id: str, level: Optional[str] = None, limit: int = 100):
    """
    Get logs for a specific job.

    ## Parameters
    - **level**: Filter by log level (INFO, WARNING, ERROR)
    - **limit**: Maximum number of logs to return
    """
    if not state.db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    job = state.db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    logs = state.db.get_job_logs(job_id, level=level, limit=limit)

    return {
        "job_id": job_id,
        "logs": logs,
        "total": len(logs),
    }


# =============================================================================
# Dataset Management Endpoints
# =============================================================================

@app.get("/datasets", tags=["Datasets"])
async def list_datasets(
    dataset_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """
    List all datasets with metadata.

    Returns list of datasets with statistics, timestamps, and preview info.
    """
    if not state.db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        datasets = state.db.list_datasets(
            dataset_type=dataset_type,
            limit=limit,
            offset=offset
        )

        return {"datasets": datasets, "total": len(datasets)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {e}")


@app.get("/datasets/{job_id}", tags=["Datasets"])
async def get_dataset_metadata(job_id: str):
    """
    Get detailed metadata for a specific dataset.

    Returns full metadata including class distribution, preview images, and config.
    """
    if not state.db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    metadata = state.db.get_dataset_metadata(job_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Dataset {job_id} not found")

    return metadata


@app.get("/datasets/{job_id}/coco", tags=["Datasets"])
async def load_dataset_coco(job_id: str):
    """
    Load the full COCO JSON for a dataset.

    Returns the complete COCO dataset from disk.
    Useful for resuming workflow with existing datasets.
    """
    import json
    from pathlib import Path

    if not state.db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    metadata = state.db.get_dataset_metadata(job_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Dataset {job_id} not found")

    coco_path = metadata.get("coco_json_path")
    if not coco_path or not Path(coco_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"COCO file not found at {coco_path}"
        )

    try:
        with open(coco_path) as f:
            coco_data = json.load(f)
        return coco_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load COCO: {e}")


# =============================================================================
# Background Task Functions
# =============================================================================

async def run_batch_composition_resume(
    job_id: str,
    request: ComposeBatchRequest,
    job_output_dir: str,
    checkpoint: dict
):
    """Background task for resuming batch composition from checkpoint"""
    import time
    start_time = time.time()

    job_logger = JobLogger(job_id, state.db)

    # Check if already cancelled before starting
    cache = state._active_jobs_cache.get(job_id, {})
    if cache.get("cancelled", False):
        state.db.complete_job(job_id, "cancelled")
        job_logger.info("Job cancelled before resuming")
        return

    # Update status to running
    state.db.update_job_status(job_id, "running", started_at=datetime.now())
    job_logger.start("Resuming batch composition from checkpoint")

    # Cancellation check callback
    def check_cancelled() -> bool:
        cache = state._active_jobs_cache.get(job_id, {})
        return cache.get("cancelled", False)

    try:
        # Calculate remaining work
        already_generated = checkpoint.get("generated", 0)
        remaining_images = request.num_images - already_generated

        if remaining_images <= 0:
            # Job was already complete
            state.db.complete_job(
                job_id,
                "completed",
                result_summary={
                    "images_generated": already_generated,
                    "images_rejected": checkpoint.get("rejected", 0),
                    "synthetic_counts": checkpoint.get("synthetic_counts", {}),
                },
                processing_time_ms=0
            )
            job_logger.complete("Job already completed (no remaining images)")
            return

        # Calculate remaining targets per class
        existing_counts = checkpoint.get("synthetic_counts", {})
        original_targets = checkpoint.get("targets_per_class") or request.targets_per_class
        remaining_targets = None

        if original_targets:
            remaining_targets = {}
            for cls, target in original_targets.items():
                current = existing_counts.get(cls, 0)
                remaining = max(0, target - current)
                if remaining > 0:
                    remaining_targets[cls] = remaining

        # Run composition for remaining images
        result = await state.composer.compose_batch(
            backgrounds_dir=request.backgrounds_dir,
            objects_dir=request.objects_dir,
            output_dir=job_output_dir,
            num_images=remaining_images,
            targets_per_class=remaining_targets,
            max_objects_per_image=request.max_objects_per_image,
            effects=request.effects,
            effects_config=request.effects_config,
            depth_aware=request.depth_aware,
            depth_service_url=request.depth_service_url,
            validate_quality=request.validate_quality,
            validate_physics=request.validate_physics,
            reject_invalid=request.reject_invalid,
            save_pipeline_debug=request.save_pipeline_debug,
            progress_callback=lambda p: update_job_progress(
                job_id,
                {
                    "generated": already_generated + p.get("generated", 0),
                    "rejected": checkpoint.get("rejected", 0) + p.get("rejected", 0),
                    "pending": p.get("pending", 0),
                    "counts": merge_counts(existing_counts, p.get("counts", {})),
                    "current_class": p.get("current_class", ""),
                },
                job_output_dir,
                request
            ),
            cancel_check=check_cancelled,
            resume_from_index=checkpoint.get("last_image_index", -1) + 1,
        )

        processing_time = (time.time() - start_time) * 1000

        # Merge final counts
        final_counts = merge_counts(existing_counts, result.synthetic_counts)
        total_generated = already_generated + result.images_generated
        total_rejected = checkpoint.get("rejected", 0) + result.images_rejected

        # Check if job was cancelled
        if check_cancelled():
            state.db.complete_job(
                job_id,
                "cancelled",
                result_summary={
                    "images_generated": total_generated,
                    "images_rejected": total_rejected,
                    "synthetic_counts": final_counts,
                    "output_coco_path": result.output_coco_path,
                },
                processing_time_ms=processing_time
            )
            job_logger.info(f"Job cancelled after {total_generated} total images")
        else:
            state.db.complete_job(
                job_id,
                "completed",
                result_summary={
                    "images_generated": total_generated,
                    "images_rejected": total_rejected,
                    "synthetic_counts": final_counts,
                    "output_coco_path": result.output_coco_path,
                },
                processing_time_ms=processing_time
            )

            # Extract and save dataset metadata
            try:
                metadata = _extract_dataset_metadata(
                    job_id=job_id,
                    output_dir=job_output_dir,
                    coco_path=result.output_coco_path,
                    request=request
                )
                state.db.create_dataset_metadata(**metadata)
                job_logger.info(f"Dataset metadata saved: {metadata['num_images']} images, {metadata['num_annotations']} annotations")
            except Exception as e:
                job_logger.warning(f"Failed to save dataset metadata: {e}")
                # Don't fail the job if metadata save fails

            job_logger.complete("Batch composition completed (resumed)", {
                "images_generated": total_generated,
                "new_images": result.images_generated,
            })

        # Update final progress in DB
        state.db.update_job_progress(
            job_id,
            processed_items=total_generated,
            failed_items=total_rejected,
            progress_details={"synthetic_counts": final_counts}
        )

    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        logger.exception(f"Resumed batch job {job_id} failed: {e}")
        state.db.complete_job(
            job_id,
            "failed",
            error_message=str(e),
            processing_time_ms=processing_time
        )
        job_logger.fail(f"Resumed batch composition failed: {e}", e)

    finally:
        # Clean up cache
        if job_id in state._active_jobs_cache:
            del state._active_jobs_cache[job_id]


def merge_counts(existing: dict, new: dict) -> dict:
    """Merge two count dictionaries"""
    result = existing.copy()
    for key, value in new.items():
        result[key] = result.get(key, 0) + value
    return result


async def run_batch_composition(job_id: str, request: ComposeBatchRequest, job_output_dir: str):
    """Background task for batch composition"""
    import time
    start_time = time.time()

    job_logger = JobLogger(job_id, state.db)

    # Check if already cancelled before starting
    cache = state._active_jobs_cache.get(job_id, {})
    if cache.get("cancelled", False):
        state.db.complete_job(job_id, "cancelled")
        job_logger.info("Job cancelled before starting")
        return

    # Update status to running
    state.db.update_job_status(job_id, "running", started_at=datetime.now())
    job_logger.start("Starting batch composition")

    # Cancellation check callback
    def check_cancelled() -> bool:
        cache = state._active_jobs_cache.get(job_id, {})
        return cache.get("cancelled", False)

    try:
        # Choose parallel or sequential processing based on request
        # Prepare dataset_info dict if provided
        dataset_info_dict = None
        if request.dataset_info:
            dataset_info_dict = request.dataset_info.model_dump()

        if request.parallel:
            job_logger.info(f"Starting PARALLEL batch (concurrent_limit={request.concurrent_limit})")
            result = await state.composer.compose_batch_parallel(
                backgrounds_dir=request.backgrounds_dir,
                objects_dir=request.objects_dir,
                output_dir=job_output_dir,
                num_images=request.num_images,
                targets_per_class=request.targets_per_class,
                max_objects_per_image=request.max_objects_per_image,
                concurrent_limit=request.concurrent_limit,
                vram_threshold=request.vram_threshold,
                effects=request.effects,
                effects_config=request.effects_config,
                depth_aware=request.depth_aware,
                save_pipeline_debug=request.save_pipeline_debug,
                progress_callback=lambda p: update_job_progress(job_id, p, job_output_dir, request),
                cancel_check=check_cancelled,
                dataset_info=dataset_info_dict,
            )
        else:
            job_logger.info("Starting SEQUENTIAL batch (legacy mode)")
            result = await state.composer.compose_batch(
                backgrounds_dir=request.backgrounds_dir,
                objects_dir=request.objects_dir,
                output_dir=job_output_dir,
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
                save_pipeline_debug=request.save_pipeline_debug,
                progress_callback=lambda p: update_job_progress(job_id, p, job_output_dir, request),
                cancel_check=check_cancelled,
                dataset_info=dataset_info_dict,
            )

        processing_time = (time.time() - start_time) * 1000

        # Check if job was cancelled
        was_cancelled = check_cancelled()
        if was_cancelled:
            state.db.complete_job(
                job_id,
                "cancelled",
                result_summary={
                    "images_generated": result.images_generated,
                    "images_rejected": result.images_rejected,
                    "synthetic_counts": result.synthetic_counts,
                    "output_coco_path": result.output_coco_path,
                },
                processing_time_ms=processing_time
            )
            job_logger.info(f"Job cancelled after {result.images_generated} images")
        else:
            state.db.complete_job(
                job_id,
                "completed",
                result_summary={
                    "images_generated": result.images_generated,
                    "images_rejected": result.images_rejected,
                    "synthetic_counts": result.synthetic_counts,
                    "output_coco_path": result.output_coco_path,
                },
                processing_time_ms=processing_time
            )
            job_logger.complete("Batch composition completed", {
                "images_generated": result.images_generated,
                "images_rejected": result.images_rejected,
            })

        # Extract and save dataset metadata for BOTH completed and cancelled jobs
        # This ensures partial datasets from cancelled jobs are also usable
        if result.images_generated > 0 and result.output_coco_path:
            try:
                metadata = _extract_dataset_metadata(
                    job_id=job_id,
                    output_dir=job_output_dir,
                    coco_path=result.output_coco_path,
                    request=request
                )
                state.db.create_dataset_metadata(**metadata)
                status_note = " (partial - job was cancelled)" if was_cancelled else ""
                job_logger.info(f"Dataset metadata saved: {metadata['num_images']} images, {metadata['num_annotations']} annotations{status_note}")
            except Exception as e:
                job_logger.warning(f"Failed to save dataset metadata: {e}")
                # Don't fail the job if metadata save fails

        # Update final progress in DB
        state.db.update_job_progress(
            job_id,
            processed_items=result.images_generated,
            failed_items=result.images_rejected,
            progress_details={"synthetic_counts": result.synthetic_counts}
        )

    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        logger.exception(f"Batch job {job_id} failed: {e}")
        state.db.complete_job(
            job_id,
            "failed",
            error_message=str(e),
            processing_time_ms=processing_time
        )
        job_logger.fail(f"Batch composition failed: {e}", e)

    finally:
        # Clean up cache
        if job_id in state._active_jobs_cache:
            del state._active_jobs_cache[job_id]


def update_job_progress(job_id: str, progress: dict, output_dir: str = None, request: ComposeBatchRequest = None):
    """Update job progress from callback and save checkpoint"""
    generated = progress.get("generated", 0)
    rejected = progress.get("rejected", 0)
    pending = progress.get("pending", 0)
    synthetic_counts = progress.get("counts", {})
    total_target = progress.get("total_target")  # Effective target based on per-class targets

    # Update database
    if state.db:
        progress_details = {
            "synthetic_counts": synthetic_counts,
            "pending": pending
        }
        # Include total_target if provided (for correct progress calculation)
        if total_target is not None:
            progress_details["total_target"] = total_target

        state.db.update_job_progress(
            job_id,
            processed_items=generated,
            failed_items=rejected,
            current_item=progress.get("current_class", ""),
            progress_details=progress_details
        )

    # Save checkpoint every 10 images
    if output_dir and generated > 0 and generated % 10 == 0:
        save_checkpoint(job_id, output_dir, progress, request)


def save_checkpoint(job_id: str, output_dir: str, progress: dict, request: ComposeBatchRequest = None):
    """Save progress checkpoint to file for resume capability"""
    try:
        checkpoint_path = Path(output_dir) / "progress.json"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "job_id": job_id,
            "generated": progress.get("generated", 0),
            "rejected": progress.get("rejected", 0),
            "synthetic_counts": progress.get("counts", {}),
            "targets_per_class": request.targets_per_class if request else None,
            "num_images": request.num_images if request else None,
            "last_image_index": progress.get("generated", 0) - 1,
            "timestamp": datetime.now().isoformat()
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    except Exception as e:
        logger.warning(f"Failed to save checkpoint for job {job_id}: {e}")


def load_checkpoint(output_dir: str) -> Optional[dict]:
    """Load progress checkpoint from file"""
    try:
        checkpoint_path = Path(output_dir) / "progress.json"
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load checkpoint from {output_dir}: {e}")
    return None


def _extract_dataset_metadata(
    job_id: str,
    output_dir: str,
    coco_path: str,
    request: ComposeBatchRequest
) -> Dict[str, Any]:
    """Extract metadata from completed generation job."""
    import json
    from pathlib import Path

    # Load COCO file to extract stats
    with open(coco_path) as f:
        coco_data = json.load(f)

    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])
    categories = coco_data.get("categories", [])

    # Calculate class distribution
    class_dist = {}
    for ann in annotations:
        cat_id = ann.get("category_id")
        cat_name = next((c["name"] for c in categories if c["id"] == cat_id), "Unknown")
        class_dist[cat_name] = class_dist.get(cat_name, 0) + 1

    # Get preview images (first 5)
    images_dir = Path(output_dir) / "images"
    preview_paths = []
    for img in images[:5]:
        img_path = images_dir / img["file_name"]
        if img_path.exists():
            preview_paths.append(str(img_path))

    # Load generation config if available
    effects_config_path = Path(output_dir) / "effects_preset.json"
    generation_config = None
    if effects_config_path.exists():
        with open(effects_config_path) as f:
            generation_config = json.load(f)

    # Calculate file size
    file_size_mb = Path(coco_path).stat().st_size / (1024 * 1024)

    # Generate dataset name from timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    dataset_name = f"Synthetic_{timestamp}_{job_id[-8:]}"

    return {
        "job_id": job_id,
        "dataset_name": dataset_name,
        "dataset_type": "generation",
        "coco_json_path": coco_path,
        "images_dir": str(images_dir),
        "effects_config_path": str(effects_config_path) if effects_config_path.exists() else None,
        "num_images": len(images),
        "num_annotations": len(annotations),
        "num_categories": len(categories),
        "class_distribution": class_dist,
        "categories": categories,
        "preview_images": preview_paths,
        "generation_config": generation_config,
        "file_size_mb": file_size_mb
    }


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
# Object Size Configuration Endpoints
# =============================================================================

@app.get("/config/object-sizes", tags=["Configuration"])
async def get_object_sizes():
    """
    Get all configured object sizes.

    Returns dictionary of {class_name: size_in_meters}.
    """
    try:
        from app.config_manager import get_object_size_config
        config = get_object_size_config()
        return {
            "sizes": config.get_all_sizes(),
            "reference_distance": config.get_reference_distance()
        }
    except Exception as e:
        logger.exception(f"Failed to get object sizes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config/object-sizes/{class_name}", tags=["Configuration"])
async def get_object_size(class_name: str):
    """
    Get the size for a specific object class.

    Args:
        class_name: Object class name

    Returns:
        Size in meters
    """
    try:
        from app.config_manager import get_object_size_config
        config = get_object_size_config()
        size = config.get_size(class_name)
        return {
            "class_name": class_name,
            "size": size
        }
    except Exception as e:
        logger.exception(f"Failed to get object size for {class_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/config/object-sizes/{class_name}", tags=["Configuration"])
async def update_object_size(class_name: str, size: float):
    """
    Update the size for a specific object class.

    Args:
        class_name: Object class name
        size: Size in meters (must be > 0)

    Returns:
        Success status
    """
    try:
        if size <= 0:
            raise HTTPException(status_code=400, detail="Size must be positive")

        from app.config_manager import get_object_size_config
        config = get_object_size_config()
        config.set_size(class_name, size)

        logger.info(f"Updated size for {class_name}: {size}m")
        return {
            "success": True,
            "class_name": class_name,
            "size": size
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to update object size for {class_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/config/object-sizes/batch", tags=["Configuration"])
async def update_multiple_object_sizes(sizes: Dict[str, float]):
    """
    Update multiple object sizes at once.

    Args:
        sizes: Dictionary of {class_name: size_in_meters}

    Returns:
        Success status with updated count
    """
    try:
        # Validate all sizes
        for class_name, size in sizes.items():
            if size <= 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Size for {class_name} must be positive"
                )

        from app.config_manager import get_object_size_config
        config = get_object_size_config()
        config.update_sizes(sizes)

        logger.info(f"Updated {len(sizes)} object sizes")
        return {
            "success": True,
            "updated_count": len(sizes),
            "classes": list(sizes.keys())
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to update object sizes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/config/object-sizes/{class_name}", tags=["Configuration"])
async def delete_object_size(class_name: str):
    """
    Delete the size configuration for an object class.

    Args:
        class_name: Object class name to remove

    Returns:
        Success status
    """
    try:
        from app.config_manager import get_object_size_config
        config = get_object_size_config()
        config.delete_size(class_name)

        logger.info(f"Deleted size configuration for {class_name}")
        return {
            "success": True,
            "class_name": class_name
        }
    except Exception as e:
        logger.exception(f"Failed to delete object size for {class_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
