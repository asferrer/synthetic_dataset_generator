"""
Domain Gap Reduction Service - Main FastAPI Application
========================================================
Microservice for measuring and reducing the domain gap between
synthetic and real image datasets.

Endpoints:
- POST /references/upload       - Upload real reference images
- GET  /references              - List reference sets
- GET  /references/{set_id}     - Get reference set details
- DELETE /references/{set_id}   - Delete reference set
- POST /metrics/compute         - Compute FID/KID/color metrics
- POST /metrics/compare         - Compare before/after metrics
- POST /analyze                 - Full gap analysis with suggestions
- POST /randomize/apply         - Apply domain randomization (single)
- POST /randomize/apply-batch   - Apply domain randomization (batch, async)
- GET  /jobs                    - List async jobs
- GET  /jobs/{job_id}           - Get job status
- DELETE /jobs/{job_id}         - Cancel job
- GET  /health                  - Health check
- GET  /info                    - Service info
"""

import os
import sys
import time
import uuid
import asyncio
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Add shared module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.job_database import JobDatabase, get_job_db

from app.models.schemas import (
    # Reference
    ReferenceUploadResponse,
    ReferenceListResponse,
    ReferenceImageSet,
    # Metrics
    MetricsRequest,
    MetricsResult,
    MetricsCompareRequest,
    MetricsCompareResponse,
    # Analysis
    AnalyzeRequest,
    GapAnalysis,
    # Randomization
    RandomizationApplyRequest,
    RandomizationBatchRequest,
    RandomizationResponse,
    # Jobs
    JobStatusResponse,
    JobListResponse,
    # System
    HealthResponse,
    InfoResponse,
    DGRTechnique,
)


# =============================================================================
# Global State
# =============================================================================

class ServiceState:
    """Global state for the Domain Gap service."""

    def __init__(self):
        self.metrics_engine = None
        self.advisor_engine = None
        self.randomization_engine = None
        self.reference_manager = None
        self.gpu_available = False
        self.gpu_name = None
        self.db: Optional[JobDatabase] = None
        self._active_jobs_cache: Dict[str, dict] = {}

    async def initialize(self):
        """Initialize service components."""
        # Check GPU
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detected: {self.gpu_name}")
        else:
            logger.warning("No GPU detected, running on CPU")

        # Initialize Reference Manager
        try:
            from app.reference_manager import ReferenceManager
            base_dir = os.environ.get("REFERENCES_DIR", "/shared/references")
            self.reference_manager = ReferenceManager(db=self.db, base_dir=base_dir)
            logger.info("ReferenceManager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ReferenceManager: {e}")

        # Initialize Metrics Engine (GPU-accelerated)
        try:
            from app.engines.metrics_engine import MetricsEngine
            self.metrics_engine = MetricsEngine(use_gpu=self.gpu_available)
            logger.info("MetricsEngine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize MetricsEngine: {e}")

        # Initialize Advisor Engine (CPU-only)
        try:
            from app.engines.advisor_engine import AdvisorEngine
            self.advisor_engine = AdvisorEngine()
            logger.info("AdvisorEngine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize AdvisorEngine: {e}")

        # Initialize Randomization Engine (CPU-only)
        try:
            from app.engines.randomization_engine import RandomizationEngine
            self.randomization_engine = RandomizationEngine()
            logger.info("RandomizationEngine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize RandomizationEngine: {e}")

    def get_gpu_memory(self) -> Optional[str]:
        """Get GPU memory usage."""
        if not self.gpu_available:
            return None
        try:
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"{allocated:.1f}GB / {total:.1f}GB"
        except Exception:
            return None


state = ServiceState()


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup service resources."""
    logger.info("Starting Domain Gap Reduction Service...")

    # Initialize database
    try:
        state.db = get_job_db()
        logger.info("Job database initialized")

        # Recover interrupted jobs
        processing_jobs = state.db.list_jobs(service="domain_gap", status="running")
        for job in processing_jobs:
            state.db.complete_job(
                job["id"], "failed",
                error_message="Service restarted before job completed"
            )
            logger.warning(f"Job {job['id']} marked as failed after restart")
    except Exception as e:
        logger.error(f"Failed to initialize job database: {e}")

    await state.initialize()
    logger.info("Domain Gap Reduction Service ready")
    yield
    logger.info("Shutting down Domain Gap Reduction Service...")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Domain Gap Reduction Service",
    description="""
    Microservice for measuring and reducing the domain gap between
    synthetic and real image datasets.

    ## Features
    - **Reference Management**: Upload and manage real image reference sets
    - **FID/KID Metrics**: Inception-based distribution comparison
    - **Color Analysis**: LAB color distribution with Earth Mover's Distance
    - **Auto-Suggestions**: Parameter adjustment recommendations
    - **Domain Randomization**: CPU-based image augmentation for diversity
    """,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health & Info
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check with component status."""
    engines_loaded = {
        "metrics": state.metrics_engine is not None,
        "advisor": state.advisor_engine is not None,
        "randomization": state.randomization_engine is not None,
        "reference_manager": state.reference_manager is not None,
    }

    all_loaded = all(engines_loaded.values())
    any_loaded = any(engines_loaded.values())

    if all_loaded:
        status = "healthy"
    elif any_loaded:
        status = "degraded"
    else:
        status = "unhealthy"

    ref_count = 0
    if state.reference_manager:
        try:
            ref_count = len(state.reference_manager.list_sets())
        except Exception:
            pass

    return HealthResponse(
        status=status,
        gpu_available=state.gpu_available,
        gpu_name=state.gpu_name,
        gpu_memory=state.get_gpu_memory(),
        engines_loaded=engines_loaded,
        reference_sets_count=ref_count,
        timestamp=datetime.now(),
    )


@app.get("/info", response_model=InfoResponse, tags=["System"])
async def service_info():
    """Service capabilities."""
    return InfoResponse(
        service="domain_gap",
        version="1.0.0",
        techniques=[t.value for t in DGRTechnique],
        metrics_available=["fid", "kid", "color_distribution", "edge_analysis"],
        gpu_available=state.gpu_available,
    )


# =============================================================================
# Reference Management
# =============================================================================

@app.post("/references/upload", response_model=ReferenceUploadResponse, tags=["References"])
async def upload_references(
    files: List[UploadFile] = File(..., description="Reference images to upload"),
    name: str = Form(..., description="Name for this reference set"),
    description: str = Form("", description="Optional description"),
    domain_id: str = Form("default", description="Associated domain ID"),
):
    """
    Upload real reference images to create a new reference set.

    The images are stored and their statistics (LAB/RGB distribution,
    edge sharpness, brightness) are pre-computed.
    """
    if not state.reference_manager:
        raise HTTPException(status_code=503, detail="Reference manager not initialized")

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # Save uploaded files to a temporary directory
    temp_dir = tempfile.mkdtemp(prefix="dgr_upload_")
    saved_paths = []
    try:
        for f in files:
            if not f.filename:
                continue
            file_path = os.path.join(temp_dir, f.filename)
            content = await f.read()
            with open(file_path, "wb") as out:
                out.write(content)
            saved_paths.append(file_path)

        if not saved_paths:
            raise HTTPException(status_code=400, detail="No valid files received")

        # Create reference set (copies images, computes stats)
        ref_set = state.reference_manager.create_set(
            name=name,
            description=description,
            domain_id=domain_id,
            image_paths=saved_paths,
        )

        return ReferenceUploadResponse(
            success=True,
            set_id=ref_set.set_id,
            name=ref_set.name,
            image_count=ref_set.image_count,
            stats=ref_set.stats,
            message=f"Reference set created with {ref_set.image_count} images",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to create reference set: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp files
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.get("/references", response_model=ReferenceListResponse, tags=["References"])
async def list_references(domain_id: Optional[str] = None):
    """List all reference sets, optionally filtered by domain."""
    if not state.reference_manager:
        raise HTTPException(status_code=503, detail="Reference manager not initialized")

    sets = state.reference_manager.list_sets(domain_id=domain_id)
    return ReferenceListResponse(sets=sets, total=len(sets))


@app.get("/references/{set_id}", response_model=ReferenceImageSet, tags=["References"])
async def get_reference(set_id: str):
    """Get reference set details with pre-computed statistics."""
    if not state.reference_manager:
        raise HTTPException(status_code=503, detail="Reference manager not initialized")

    ref_set = state.reference_manager.get_set(set_id)
    if not ref_set:
        raise HTTPException(status_code=404, detail=f"Reference set {set_id} not found")
    return ref_set


@app.delete("/references/{set_id}", tags=["References"])
async def delete_reference(set_id: str):
    """Delete a reference set and its images."""
    if not state.reference_manager:
        raise HTTPException(status_code=503, detail="Reference manager not initialized")

    deleted = state.reference_manager.delete_set(set_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Reference set {set_id} not found")

    return {"success": True, "message": f"Reference set {set_id} deleted"}


# =============================================================================
# Metrics
# =============================================================================

@app.post("/metrics/compute", response_model=MetricsResult, tags=["Metrics"])
async def compute_metrics(request: MetricsRequest):
    """
    Compute domain gap metrics between synthetic images and a reference set.

    Returns FID, KID, color distribution EMD, and an overall gap score (0-100).
    """
    if not state.metrics_engine:
        raise HTTPException(status_code=503, detail="Metrics engine not initialized")
    if not state.reference_manager:
        raise HTTPException(status_code=503, detail="Reference manager not initialized")

    # Resolve reference set to directory
    ref_set = state.reference_manager.get_set(request.reference_set_id)
    if not ref_set:
        raise HTTPException(
            status_code=404,
            detail=f"Reference set {request.reference_set_id} not found"
        )

    if not os.path.isdir(request.synthetic_dir):
        raise HTTPException(
            status_code=404,
            detail=f"Synthetic directory not found: {request.synthetic_dir}"
        )

    try:
        result = state.metrics_engine.compute_metrics(
            synthetic_dir=request.synthetic_dir,
            real_dir=ref_set.image_dir,
            max_images=request.max_images,
            compute_fid=request.compute_fid,
            compute_kid=request.compute_kid,
            compute_color=request.compute_color_distribution,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Metrics computation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/metrics/compare", response_model=MetricsCompareResponse, tags=["Metrics"])
async def compare_metrics(request: MetricsCompareRequest):
    """
    Compare domain gap metrics before and after processing.

    Computes metrics for both original and processed synthetic images
    against the same reference set, and calculates improvement percentages.
    """
    if not state.metrics_engine:
        raise HTTPException(status_code=503, detail="Metrics engine not initialized")
    if not state.reference_manager:
        raise HTTPException(status_code=503, detail="Reference manager not initialized")

    ref_set = state.reference_manager.get_set(request.reference_set_id)
    if not ref_set:
        raise HTTPException(
            status_code=404,
            detail=f"Reference set {request.reference_set_id} not found"
        )

    start_time = time.time()

    try:
        before = state.metrics_engine.compute_metrics(
            synthetic_dir=request.original_synthetic_dir,
            real_dir=ref_set.image_dir,
            max_images=request.max_images,
        )
        after = state.metrics_engine.compute_metrics(
            synthetic_dir=request.processed_synthetic_dir,
            real_dir=ref_set.image_dir,
            max_images=request.max_images,
        )

        # Calculate improvement percentages (positive = better)
        improvement = {}
        if before.fid_score and after.fid_score and before.fid_score > 0:
            improvement["fid"] = round(
                (before.fid_score - after.fid_score) / before.fid_score * 100, 2
            )
        if before.kid_score and after.kid_score and before.kid_score > 0:
            improvement["kid"] = round(
                (before.kid_score - after.kid_score) / before.kid_score * 100, 2
            )
        if before.overall_gap_score > 0:
            improvement["overall"] = round(
                (before.overall_gap_score - after.overall_gap_score)
                / before.overall_gap_score * 100, 2
            )

        return MetricsCompareResponse(
            success=True,
            before=before,
            after=after,
            improvement_pct=improvement,
            processing_time_ms=(time.time() - start_time) * 1000,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Metrics comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Full Analysis (Metrics + Issues + Suggestions)
# =============================================================================

@app.post("/analyze", response_model=GapAnalysis, tags=["Analysis"])
async def analyze_gap(request: AnalyzeRequest):
    """
    Full domain gap analysis: compute metrics, detect issues, generate suggestions.

    This is the main validation endpoint. It combines FID/KID/color metrics
    with edge, frequency, and texture analysis to produce actionable
    parameter adjustment suggestions.
    """
    if not state.metrics_engine:
        raise HTTPException(status_code=503, detail="Metrics engine not initialized")
    if not state.advisor_engine:
        raise HTTPException(status_code=503, detail="Advisor engine not initialized")
    if not state.reference_manager:
        raise HTTPException(status_code=503, detail="Reference manager not initialized")

    ref_set = state.reference_manager.get_set(request.reference_set_id)
    if not ref_set:
        raise HTTPException(
            status_code=404,
            detail=f"Reference set {request.reference_set_id} not found"
        )

    if not os.path.isdir(request.synthetic_dir):
        raise HTTPException(
            status_code=404,
            detail=f"Synthetic directory not found: {request.synthetic_dir}"
        )

    start_time = time.time()

    try:
        # 1. Compute quantitative metrics
        metrics = state.metrics_engine.compute_metrics(
            synthetic_dir=request.synthetic_dir,
            real_dir=ref_set.image_dir,
            max_images=request.max_images,
        )

        # 2. Run advisor analysis (color, edges, frequency, texture)
        issues, suggestions = state.advisor_engine.analyze(
            synthetic_dir=request.synthetic_dir,
            real_dir=ref_set.image_dir,
            max_images=request.max_images,
            current_config=request.current_config,
        )

        # 3. Collect sample paths for visual comparison
        syn_samples = _get_sample_paths(request.synthetic_dir, max_count=5)
        real_samples = _get_sample_paths(ref_set.image_dir, max_count=5)

        processing_time = (time.time() - start_time) * 1000

        return GapAnalysis(
            gap_score=metrics.overall_gap_score,
            gap_level=metrics.gap_level,
            metrics=metrics,
            issues=issues,
            suggestions=suggestions,
            sample_synthetic_paths=syn_samples,
            sample_real_paths=real_samples,
            processing_time_ms=processing_time,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Gap analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Domain Randomization
# =============================================================================

@app.post("/randomize/apply", response_model=RandomizationResponse, tags=["Randomization"])
async def randomize_single(request: RandomizationApplyRequest):
    """
    Apply domain randomization to a single image (synchronous).

    Generates N variants of the input image with randomized color,
    brightness, contrast, noise, and blur.
    """
    if not state.randomization_engine:
        raise HTTPException(status_code=503, detail="Randomization engine not initialized")

    start_time = time.time()
    config = request.config

    # Get reference stats for histogram matching if reference_set_id provided
    ref_stats = None
    if config.reference_set_id and state.reference_manager:
        stats = state.reference_manager.get_set_stats(config.reference_set_id)
        if stats:
            ref_stats = {
                "channel_means_lab": stats.channel_means_lab,
                "channel_stds_lab": stats.channel_stds_lab,
            }

    try:
        variants = state.randomization_engine.apply_single(
            image_path=request.image_path,
            output_dir=request.output_dir,
            num_variants=config.num_variants,
            intensity=config.intensity,
            color_jitter=config.color_jitter,
            brightness_range=config.brightness_range,
            contrast_range=config.contrast_range,
            saturation_range=config.saturation_range,
            noise_intensity=config.noise_intensity,
            blur_range=config.blur_range,
            reference_stats=ref_stats,
            histogram_match_strength=config.histogram_match_strength,
            annotations_path=request.annotations_path if config.preserve_annotations else None,
        )

        return RandomizationResponse(
            success=True,
            variants_created=len(variants),
            output_dir=request.output_dir,
            processing_time_ms=(time.time() - start_time) * 1000,
        )
    except Exception as e:
        logger.exception(f"Randomization failed: {e}")
        return RandomizationResponse(
            success=False,
            variants_created=0,
            output_dir=request.output_dir,
            processing_time_ms=(time.time() - start_time) * 1000,
            error=str(e),
        )


@app.post("/randomize/apply-batch", tags=["Randomization"])
async def randomize_batch(
    request: RandomizationBatchRequest,
    background_tasks: BackgroundTasks,
):
    """
    Apply domain randomization to all images in a directory (async job).

    Returns a job_id to track progress via GET /jobs/{job_id}.
    """
    if not state.randomization_engine:
        raise HTTPException(status_code=503, detail="Randomization engine not initialized")
    if not state.db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    if not os.path.isdir(request.images_dir):
        raise HTTPException(
            status_code=404,
            detail=f"Images directory not found: {request.images_dir}"
        )

    job_id = f"dgr_{uuid.uuid4().hex[:12]}"

    state.db.create_job(
        job_id=job_id,
        job_type="domain_randomization",
        service="domain_gap",
        request_params={
            "images_dir": request.images_dir,
            "output_dir": request.output_dir,
            "config": request.config.model_dump(),
        },
        total_items=0,
        output_path=request.output_dir,
    )

    state._active_jobs_cache[job_id] = {"cancelled": False}

    background_tasks.add_task(
        _run_randomization_batch, job_id, request
    )

    return {
        "success": True,
        "job_id": job_id,
        "status": "pending",
        "message": "Domain randomization batch job started",
    }


# =============================================================================
# Job Management
# =============================================================================

@app.get("/jobs", tags=["Jobs"])
async def list_jobs():
    """List all domain gap jobs."""
    if not state.db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    jobs = state.db.list_jobs(service="domain_gap")
    jobs_list = []
    for job in jobs:
        progress = job.get("progress_details") or {}
        total = job.get("total_items", 0)
        processed = job.get("processed_items", 0)
        progress_pct = round((processed / total * 100), 1) if total > 0 else 0.0

        jobs_list.append({
            "job_id": job["id"],
            "type": job.get("job_type", "unknown"),
            "status": job["status"],
            "progress": progress_pct,
            "total_items": total,
            "processed_items": processed,
            "failed_items": job.get("failed_items", 0),
            "result": job.get("result_summary"),
            "error": job.get("error_message"),
            "created_at": job.get("created_at"),
            "output_dir": job.get("output_path"),
        })

    return {"jobs": jobs_list, "total": len(jobs_list)}


@app.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["Jobs"])
async def get_job_status(job_id: str):
    """Get status of an async job."""
    if not state.db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    job = state.db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    total = job.get("total_items", 0)
    processed = job.get("processed_items", 0)
    progress_pct = round((processed / total * 100), 1) if total > 0 else 0.0

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=progress_pct,
        total_items=total,
        processed_items=processed,
        failed_items=job.get("failed_items", 0),
        result=job.get("result_summary"),
        error=job.get("error_message"),
        created_at=job.get("created_at"),
    )


@app.delete("/jobs/{job_id}", tags=["Jobs"])
async def cancel_job(job_id: str):
    """Cancel a running job."""
    if not state.db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    job = state.db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job["status"] in ["completed", "failed", "cancelled"]:
        return {
            "success": False,
            "job_id": job_id,
            "message": f"Job already {job['status']}",
        }

    if job_id in state._active_jobs_cache:
        state._active_jobs_cache[job_id]["cancelled"] = True

    if job["status"] == "pending":
        state.db.complete_job(job_id, "cancelled")
        return {"success": True, "job_id": job_id, "message": "Job cancelled"}

    return {
        "success": True,
        "job_id": job_id,
        "message": "Cancellation requested",
    }


# =============================================================================
# Background Tasks
# =============================================================================

async def _run_randomization_batch(job_id: str, request: RandomizationBatchRequest):
    """Background task for batch domain randomization."""
    start_time = time.time()
    config = request.config

    # Update status to running
    state.db.update_job_status(job_id, "running", started_at=datetime.now())

    # Get reference stats for histogram matching
    ref_stats = None
    if config.reference_set_id and state.reference_manager:
        stats = state.reference_manager.get_set_stats(config.reference_set_id)
        if stats:
            ref_stats = {
                "channel_means_lab": stats.channel_means_lab,
                "channel_stds_lab": stats.channel_stds_lab,
            }

    def progress_callback(processed, total):
        cache = state._active_jobs_cache.get(job_id, {})
        if cache.get("cancelled", False):
            return
        state.db.update_job_progress(
            job_id,
            processed_items=processed,
            progress_details={"total": total},
        )
        # Update total_items on first callback
        if processed == 1:
            state.db.update_job_progress(job_id, processed_items=1)

    try:
        result = state.randomization_engine.apply_batch(
            images_dir=request.images_dir,
            output_dir=request.output_dir,
            num_variants=config.num_variants,
            intensity=config.intensity,
            color_jitter=config.color_jitter,
            brightness_range=config.brightness_range,
            contrast_range=config.contrast_range,
            saturation_range=config.saturation_range,
            noise_intensity=config.noise_intensity,
            blur_range=config.blur_range,
            reference_stats=ref_stats,
            histogram_match_strength=config.histogram_match_strength,
            annotations_dir=request.annotations_dir if config.preserve_annotations else None,
            progress_callback=progress_callback,
        )

        processing_time = (time.time() - start_time) * 1000

        cache = state._active_jobs_cache.get(job_id, {})
        final_status = "cancelled" if cache.get("cancelled") else "completed"

        state.db.complete_job(
            job_id,
            final_status,
            result_summary=result,
            processing_time_ms=processing_time,
        )
        state.db.update_job_progress(
            job_id,
            processed_items=result["total_images"],
            failed_items=result["failed"],
        )

        logger.info(f"Randomization batch {job_id} {final_status}: {result}")

    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        logger.exception(f"Randomization batch {job_id} failed: {e}")
        state.db.complete_job(
            job_id, "failed",
            error_message=str(e),
            processing_time_ms=processing_time,
        )
    finally:
        if job_id in state._active_jobs_cache:
            del state._active_jobs_cache[job_id]


# =============================================================================
# Helpers
# =============================================================================

def _get_sample_paths(directory: str, max_count: int = 5) -> List[str]:
    """Get sample image paths from a directory for visual comparison."""
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    try:
        dir_path = Path(directory)
        images = sorted([
            str(p) for p in dir_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ])
        return images[:max_count]
    except Exception:
        return []


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8005,
        reload=True,
        log_level="info",
    )
