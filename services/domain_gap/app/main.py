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
    ReferenceCreateResponse,
    ReferenceBatchResponse,
    ReferenceFinalizeResponse,
    ReferenceFromDirectoryRequest,
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
    # Style Transfer
    StyleTransferApplyRequest,
    StyleTransferBatchRequest,
    StyleTransferResponse,
    # Optimization
    OptimizeRequest,
    # Jobs
    JobStatusResponse,
    JobListResponse,
    # System
    HealthResponse,
    InfoResponse,
    DGRTechnique,
)

# Import routers
from app.routers import ml_optimizer


# =============================================================================
# Global State
# =============================================================================

class ServiceState:
    """Global state for the Domain Gap service."""

    def __init__(self):
        self.metrics_engine = None
        self.advisor_engine = None
        self.randomization_engine = None
        self.style_transfer_engine = None
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

        # Initialize Style Transfer Engine (GPU-accelerated)
        try:
            from app.engines.style_transfer_engine import StyleTransferEngine
            self.style_transfer_engine = StyleTransferEngine(use_gpu=self.gpu_available)
            logger.info("StyleTransferEngine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize StyleTransferEngine: {e}")

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

# Include routers
app.include_router(ml_optimizer.router)


# =============================================================================
# Health & Info
# =============================================================================

@app.get("/ping", tags=["System"])
async def ping():
    """Lightweight liveness probe for Docker health checks.

    Returns immediately even when heavy computations (FID/KID) are
    running on the event loop, because this endpoint does zero I/O.
    Use /health for full component status.
    """
    return {"status": "ok"}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check with component status."""
    engines_loaded = {
        "metrics": state.metrics_engine is not None,
        "advisor": state.advisor_engine is not None,
        "randomization": state.randomization_engine is not None,
        "style_transfer": state.style_transfer_engine is not None,
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
        version="2.1.0",
        techniques=[t.value for t in DGRTechnique],
        metrics_available=[
            "radio_mmd", "fd_radio",
            "fid", "kid", "cmmd", "precision", "recall",
            "density", "coverage", "color_distribution", "edge_analysis",
        ],
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


# --- Chunked upload protocol ---

@app.post("/references/create", response_model=ReferenceCreateResponse, tags=["References"])
async def create_reference_set(
    name: str = Form(..., description="Name for this reference set"),
    description: str = Form("", description="Optional description"),
    domain_id: str = Form("default", description="Associated domain ID"),
):
    """Phase 1 of chunked upload: create an empty reference set shell."""
    if not state.reference_manager:
        raise HTTPException(status_code=503, detail="Reference manager not initialized")

    try:
        set_id, image_dir = state.reference_manager.create_empty_set(name, description, domain_id)
        return ReferenceCreateResponse(
            success=True,
            set_id=set_id,
            name=name,
            image_dir=image_dir,
            message="Reference set created, ready for image batches",
        )
    except Exception as e:
        logger.exception(f"Failed to create reference set shell: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/references/{set_id}/add-batch", response_model=ReferenceBatchResponse, tags=["References"])
async def add_reference_batch(
    set_id: str,
    files: List[UploadFile] = File(..., description="Batch of reference images"),
):
    """Phase 2 of chunked upload: add a batch of images to an existing set."""
    if not state.reference_manager:
        raise HTTPException(status_code=503, detail="Reference manager not initialized")

    image_dir = state.reference_manager.get_image_dir(set_id)
    if not image_dir:
        raise HTTPException(status_code=404, detail=f"Reference set {set_id} not found")

    image_dir_path = Path(image_dir)
    saved_paths = []

    for f in files:
        if not f.filename:
            continue
        dst = image_dir_path / f.filename
        # Avoid name collisions
        if dst.exists():
            stem = dst.stem
            suffix = dst.suffix
            counter = 1
            while dst.exists():
                dst = image_dir_path / f"{stem}_{counter}{suffix}"
                counter += 1

        # Write file in chunks to avoid memory spikes
        with open(str(dst), "wb") as out:
            while chunk := await f.read(1024 * 1024):  # 1MB chunks
                out.write(chunk)
        saved_paths.append(str(dst))

    if not saved_paths:
        raise HTTPException(status_code=400, detail="No valid files in this batch")

    total = state.reference_manager.add_images_to_set(set_id, saved_paths)

    return ReferenceBatchResponse(
        success=True,
        set_id=set_id,
        images_added=len(saved_paths),
        total_images=total,
        message=f"Added {len(saved_paths)} images (total: {total})",
    )


@app.post("/references/{set_id}/finalize", response_model=ReferenceFinalizeResponse, tags=["References"])
async def finalize_reference_set(set_id: str):
    """Phase 3 of chunked upload: compute statistics and finalize the set."""
    if not state.reference_manager:
        raise HTTPException(status_code=503, detail="Reference manager not initialized")

    try:
        ref_set = state.reference_manager.finalize_set(set_id)
        return ReferenceFinalizeResponse(
            success=True,
            set_id=ref_set.set_id,
            name=ref_set.name,
            image_count=ref_set.image_count,
            stats=ref_set.stats,
            message=f"Finalized with {ref_set.image_count} images",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to finalize reference set: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/references/from-directory", response_model=ReferenceUploadResponse, tags=["References"])
async def create_from_directory(request: ReferenceFromDirectoryRequest):
    """Create a reference set from images in a server-side directory."""
    if not state.reference_manager:
        raise HTTPException(status_code=503, detail="Reference manager not initialized")

    try:
        ref_set = state.reference_manager.create_from_directory(
            name=request.name,
            description=request.description,
            domain_id=request.domain_id,
            source_dir=request.directory_path,
        )
        return ReferenceUploadResponse(
            success=True,
            set_id=ref_set.set_id,
            name=ref_set.name,
            image_count=ref_set.image_count,
            stats=ref_set.stats,
            message=f"Created from directory with {ref_set.image_count} images",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to create reference set from directory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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

# GPU semaphore: prevents concurrent RADIO extractions that would OOM the GPU.
# Only one heavy GPU metrics computation at a time; subsequent requests queue.
_gpu_metrics_semaphore = asyncio.Semaphore(1)

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
        async with _gpu_metrics_semaphore:
            logger.info("GPU semaphore acquired for metrics computation")
            result = await asyncio.to_thread(
                state.metrics_engine.compute_metrics,
                synthetic_dir=request.synthetic_dir,
                real_dir=ref_set.image_dir,
                max_images=request.max_images,
                compute_radio_mmd=request.compute_radio_mmd,
                compute_fd_radio=request.compute_fd_radio,
                compute_fid=request.compute_fid,
                compute_kid=request.compute_kid,
                compute_color=request.compute_color_distribution,
                compute_cmmd=request.compute_cmmd,
                compute_prdc=request.compute_prdc,
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
        before = await asyncio.to_thread(
            state.metrics_engine.compute_metrics,
            synthetic_dir=request.original_synthetic_dir,
            real_dir=ref_set.image_dir,
            max_images=request.max_images,
        )
        after = await asyncio.to_thread(
            state.metrics_engine.compute_metrics,
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

_ANALYSIS_PHASES = {
    # RADIO backbone phases (primary)
    "extracting_radio_synthetic": (0, 12),
    "extracting_radio_real": (12, 24),
    "computing_radio_mmd": (24, 28),
    "computing_fd_radio": (28, 32),
    "computing_prdc": (32, 36),
    # CLIP phases (optional legacy)
    "extracting_clip_synthetic": (36, 40),
    "extracting_clip_real": (40, 44),
    "computing_cmmd": (44, 46),
    # Inception phases (optional legacy)
    "extracting_features_synthetic": (46, 52),
    "extracting_features_real": (52, 58),
    "computing_fid": (58, 62),
    "computing_kid": (62, 66),
    # Color analysis
    "computing_color": (66, 72),
    # Advisor phases
    "analyzing_color": (72, 78),
    "analyzing_edges": (78, 83),
    "analyzing_frequency": (83, 88),
    "analyzing_texture": (88, 92),
    "generating_suggestions": (92, 96),
    "completing": (96, 100),
}


@app.post("/analyze", tags=["Analysis"])
async def analyze_gap(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Full domain gap analysis: compute metrics, detect issues, generate suggestions.

    Returns a job_id immediately. Poll GET /jobs/{job_id} for progress and results.
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

    job_id = f"gap_{uuid.uuid4().hex[:12]}"

    state.db.create_job(
        job_id=job_id,
        job_type="gap_analysis",
        service="domain_gap",
        request_params={
            "synthetic_dir": request.synthetic_dir,
            "reference_set_id": request.reference_set_id,
            "max_images": request.max_images,
        },
        total_items=100,
    )

    state._active_jobs_cache[job_id] = {"cancelled": False}

    background_tasks.add_task(
        _run_gap_analysis, job_id, request, ref_set.image_dir
    )

    return {
        "success": True,
        "job_id": job_id,
        "status": "pending",
        "message": "Gap analysis started",
    }


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
# Style Transfer
# =============================================================================

@app.post("/style-transfer/apply", response_model=StyleTransferResponse, tags=["Style Transfer"])
async def style_transfer_single(request: StyleTransferApplyRequest):
    """
    Apply neural style transfer to a single image (synchronous).

    Transfers the visual style from a reference set to the input image
    using DA-WCT (Depth-Aware Whitening and Coloring Transform).
    """
    if not state.style_transfer_engine:
        raise HTTPException(status_code=503, detail="Style transfer engine not initialized")
    if not state.reference_manager:
        raise HTTPException(status_code=503, detail="Reference manager not initialized")

    # Resolve reference set to directory for style source
    ref_set = state.reference_manager.get_set(request.config.reference_set_id)
    if not ref_set:
        raise HTTPException(
            status_code=404,
            detail=f"Reference set {request.config.reference_set_id} not found"
        )

    if not os.path.isfile(request.image_path):
        raise HTTPException(
            status_code=404,
            detail=f"Image not found: {request.image_path}"
        )

    start_time = time.time()
    try:
        output_path = await asyncio.to_thread(
            state.style_transfer_engine.apply_single,
            content_path=request.image_path,
            style_dir=ref_set.image_dir,
            output_path=request.output_path,
            style_weight=request.config.style_weight,
            content_weight=request.config.content_weight,
            preserve_structure=request.config.preserve_structure,
            color_only=request.config.color_only,
            depth_guided=request.config.depth_guided,
        )
        return StyleTransferResponse(
            success=True,
            output_path=output_path,
            processing_time_ms=(time.time() - start_time) * 1000,
        )
    except Exception as e:
        logger.exception(f"Style transfer failed: {e}")
        return StyleTransferResponse(
            success=False,
            output_path=request.output_path,
            processing_time_ms=(time.time() - start_time) * 1000,
            error=str(e),
        )


@app.post("/style-transfer/apply-batch", tags=["Style Transfer"])
async def style_transfer_batch(
    request: StyleTransferBatchRequest,
    background_tasks: BackgroundTasks,
):
    """
    Apply style transfer to all images in a directory (async job).

    Returns a job_id to track progress via GET /jobs/{job_id}.
    """
    if not state.style_transfer_engine:
        raise HTTPException(status_code=503, detail="Style transfer engine not initialized")
    if not state.reference_manager:
        raise HTTPException(status_code=503, detail="Reference manager not initialized")
    if not state.db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    ref_set = state.reference_manager.get_set(request.config.reference_set_id)
    if not ref_set:
        raise HTTPException(
            status_code=404,
            detail=f"Reference set {request.config.reference_set_id} not found"
        )

    if not os.path.isdir(request.images_dir):
        raise HTTPException(
            status_code=404,
            detail=f"Images directory not found: {request.images_dir}"
        )

    job_id = f"st_{uuid.uuid4().hex[:12]}"

    state.db.create_job(
        job_id=job_id,
        job_type="style_transfer",
        service="domain_gap",
        request_params={
            "images_dir": request.images_dir,
            "output_dir": request.output_dir,
            "reference_set_id": request.config.reference_set_id,
            "style_weight": request.config.style_weight,
            "color_only": request.config.color_only,
        },
        total_items=0,
        output_path=request.output_dir,
    )

    state._active_jobs_cache[job_id] = {"cancelled": False}

    background_tasks.add_task(
        _run_style_transfer_batch, job_id, request, ref_set.image_dir
    )

    return {
        "success": True,
        "job_id": job_id,
        "status": "pending",
        "message": "Style transfer batch job started",
    }


# =============================================================================
# Automatic Optimization
# =============================================================================

@app.post("/optimize", tags=["Optimization"])
async def optimize_gap(request: OptimizeRequest, background_tasks: BackgroundTasks):
    """
    Automatic iterative domain gap optimization.

    Runs a loop: compute metrics → analyze issues → apply technique → repeat
    until the target gap score is reached or max iterations exhausted.
    Returns a job_id to track progress via GET /jobs/{job_id}.
    """
    if not state.metrics_engine:
        raise HTTPException(status_code=503, detail="Metrics engine not initialized")
    if not state.advisor_engine:
        raise HTTPException(status_code=503, detail="Advisor engine not initialized")
    if not state.reference_manager:
        raise HTTPException(status_code=503, detail="Reference manager not initialized")
    if not state.db:
        raise HTTPException(status_code=503, detail="Database not initialized")

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

    job_id = f"opt_{uuid.uuid4().hex[:12]}"

    state.db.create_job(
        job_id=job_id,
        job_type="optimization",
        service="domain_gap",
        request_params={
            "synthetic_dir": request.synthetic_dir,
            "output_dir": request.output_dir,
            "reference_set_id": request.reference_set_id,
            "target_gap_score": request.target_gap_score,
            "max_iterations": request.max_iterations,
            "techniques": request.techniques,
        },
        total_items=request.max_iterations,
        output_path=request.output_dir,
    )

    state._active_jobs_cache[job_id] = {"cancelled": False}

    background_tasks.add_task(
        _run_optimization, job_id, request, ref_set.image_dir
    )

    return {
        "success": True,
        "job_id": job_id,
        "status": "pending",
        "message": "Optimization started",
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
        progress_details=job.get("progress_details"),
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


async def _run_optimization(job_id: str, request: OptimizeRequest, real_dir: str):
    """Background task for automatic domain gap optimization."""
    start_time = time.time()

    state.db.update_job_status(job_id, "running", started_at=datetime.now())

    def progress_callback(iteration, max_iters, gap_score, phase):
        cache = state._active_jobs_cache.get(job_id, {})
        if cache.get("cancelled", False):
            return
        state.db.update_job_progress(
            job_id,
            processed_items=iteration,
            progress_details={
                "iteration": iteration,
                "max_iterations": max_iters,
                "gap_score": gap_score,
                "phase": phase,
                "global_progress": round((iteration / max_iters) * 100, 1),
            },
        )

    try:
        from app.engines.optimizer_engine import OptimizerEngine

        optimizer = OptimizerEngine(
            metrics_engine=state.metrics_engine,
            advisor_engine=state.advisor_engine,
            randomization_engine=state.randomization_engine,
            style_transfer_engine=state.style_transfer_engine,
        )

        result = await asyncio.to_thread(
            optimizer.optimize,
            synthetic_dir=request.synthetic_dir,
            real_dir=real_dir,
            output_dir=request.output_dir,
            current_config=request.current_config,
            target_gap_score=request.target_gap_score,
            max_iterations=request.max_iterations,
            max_images=request.max_images,
            techniques=request.techniques,
            reference_set_id=request.reference_set_id,
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

        logger.info(f"Optimization {job_id} {final_status}: {result.get('improvement_pct', 0)}% improvement")

    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        logger.exception(f"Optimization {job_id} failed: {e}")
        state.db.complete_job(
            job_id, "failed",
            error_message=str(e),
            processing_time_ms=processing_time,
        )
    finally:
        if job_id in state._active_jobs_cache:
            del state._active_jobs_cache[job_id]


async def _run_style_transfer_batch(
    job_id: str, request: StyleTransferBatchRequest, style_dir: str
):
    """Background task for batch style transfer."""
    start_time = time.time()
    config = request.config

    state.db.update_job_status(job_id, "running", started_at=datetime.now())

    def progress_callback(processed, total):
        cache = state._active_jobs_cache.get(job_id, {})
        if cache.get("cancelled", False):
            return
        state.db.update_job_progress(
            job_id,
            processed_items=processed,
            progress_details={"total": total, "processed": processed},
        )

    try:
        result = state.style_transfer_engine.apply_batch(
            images_dir=request.images_dir,
            style_dir=style_dir,
            output_dir=request.output_dir,
            style_weight=config.style_weight,
            content_weight=config.content_weight,
            preserve_structure=config.preserve_structure,
            color_only=config.color_only,
            depth_guided=config.depth_guided,
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

        logger.info(f"Style transfer batch {job_id} {final_status}: {result}")

    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        logger.exception(f"Style transfer batch {job_id} failed: {e}")
        state.db.complete_job(
            job_id, "failed",
            error_message=str(e),
            processing_time_ms=processing_time,
        )
    finally:
        if job_id in state._active_jobs_cache:
            del state._active_jobs_cache[job_id]


async def _run_gap_analysis(job_id: str, request: AnalyzeRequest, real_dir: str):
    """Background task for full domain gap analysis with progress reporting."""
    start_time = time.time()

    state.db.update_job_status(job_id, "running", started_at=datetime.now())

    def _progress_callback(phase: str, fraction: float):
        """Map (phase, fraction) to global 0-100 percentage and write to DB."""
        cache = state._active_jobs_cache.get(job_id, {})
        if cache.get("cancelled", False):
            return

        phase_range = _ANALYSIS_PHASES.get(phase)
        if phase_range:
            start_pct, end_pct = phase_range
            global_pct = start_pct + (end_pct - start_pct) * fraction
        else:
            global_pct = 0.0

        state.db.update_job_progress(
            job_id,
            processed_items=int(global_pct),
            current_item=phase,
            progress_details={
                "phase": phase,
                "phase_progress": round(fraction * 100, 1),
                "global_progress": round(global_pct, 1),
            },
        )

    try:
        # 1. Compute quantitative metrics (RADIO-primary by default)
        metrics = await asyncio.to_thread(
            state.metrics_engine.compute_metrics,
            synthetic_dir=request.synthetic_dir,
            real_dir=real_dir,
            max_images=request.max_images,
            compute_radio_mmd=True,
            compute_fd_radio=True,
            compute_fid=False,
            compute_kid=False,
            compute_color=True,
            compute_cmmd=False,
            compute_prdc=True,
            progress_callback=_progress_callback,
        )

        # Check cancellation
        cache = state._active_jobs_cache.get(job_id, {})
        if cache.get("cancelled", False):
            state.db.complete_job(job_id, "cancelled")
            return

        # 2. Run advisor analysis
        issues, suggestions = await asyncio.to_thread(
            state.advisor_engine.analyze,
            synthetic_dir=request.synthetic_dir,
            real_dir=real_dir,
            max_images=request.max_images,
            current_config=request.current_config,
            progress_callback=_progress_callback,
        )

        # 3. Collect sample paths
        _progress_callback("completing", 0.0)
        syn_samples = _get_sample_paths(request.synthetic_dir, max_count=5)
        real_samples = _get_sample_paths(real_dir, max_count=5)

        processing_time = (time.time() - start_time) * 1000

        result = GapAnalysis(
            gap_score=metrics.overall_gap_score,
            gap_level=metrics.gap_level,
            metrics=metrics,
            issues=issues,
            suggestions=suggestions,
            sample_synthetic_paths=syn_samples,
            sample_real_paths=real_samples,
            processing_time_ms=processing_time,
        )

        _progress_callback("completing", 1.0)

        cache = state._active_jobs_cache.get(job_id, {})
        final_status = "cancelled" if cache.get("cancelled") else "completed"

        state.db.complete_job(
            job_id,
            final_status,
            result_summary=result.model_dump(mode="json"),
            processing_time_ms=processing_time,
        )

        logger.info(f"Gap analysis {job_id} {final_status} in {processing_time:.0f}ms")

    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        logger.exception(f"Gap analysis {job_id} failed: {e}")
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
