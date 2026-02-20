"""
Auto-Tune Router
=================
Closed-loop auto-tuning: generates probe batches, measures domain gap,
adjusts effects_config, and repeats until convergence.

Orchestrates calls to both the Augmentor and Domain Gap services.
"""

import asyncio
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from app.services.client import get_service_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auto-tune", tags=["Auto-Tune"])

# Timeouts for downstream calls
COMPOSE_TIMEOUT = 300.0   # 5 min for probe batch generation
METRICS_TIMEOUT = 2700.0  # 45 min for RADIO metrics (batch_size=2 on 8GB GPUs can take 30+ min)
ANALYZE_TIMEOUT = 300.0   # 5 min for advisor analysis
POLL_INTERVAL = 3.0       # seconds between job polls
MAX_POLL_SECONDS = 1800.0  # 30 min global timeout per polling loop


# =============================================================================
# Request / Response Models
# =============================================================================

class AutoTuneRequest(BaseModel):
    """Request to start closed-loop auto-tuning.

    Accepts both legacy format (backgrounds_dir + objects_dir + num_images)
    and frontend format (source_dataset + target_counts + nested configs),
    mirroring the ComposeBatchRequest schema.
    """
    # --- Legacy format (direct augmentor fields) ---
    backgrounds_dir: Optional[str] = Field(None, description="Directory with backgrounds")
    objects_dir: Optional[str] = Field(None, description="Directory with objects by class")
    num_images: Optional[int] = Field(None, ge=10, le=100000, description="Total images for full generation")

    # --- Frontend format ---
    source_dataset: Optional[str] = Field(None, description="Dataset root containing Backgrounds_filtered/ and Objects/")
    target_counts: Optional[Dict[str, int]] = Field(None, description="Target count per class")

    # --- Common fields ---
    output_dir: str = Field(..., description="Output directory for final generation")
    targets_per_class: Optional[Dict[str, int]] = Field(None)
    max_objects_per_image: int = Field(5, ge=1, le=20)
    effects: List[str] = Field(
        default=["color_correction", "blur_matching", "caustics"],
        description="Effects to apply during generation",
    )
    effects_config: Optional[Any] = Field(
        default=None,
        description="Effects config (flat dict or nested frontend format)",
    )
    depth_aware: bool = True

    # --- Frontend-specific nested configs (accepted, transformed for augmentor) ---
    placement_config: Optional[Dict[str, Any]] = Field(None)
    validation_config: Optional[Dict[str, Any]] = Field(None)
    batch_config: Optional[Dict[str, Any]] = Field(None)
    lighting_config: Optional[Dict[str, Any]] = Field(None)
    metadata: Optional[Dict[str, Any]] = Field(None)
    use_depth: Optional[bool] = Field(None)
    use_segmentation: Optional[bool] = Field(None)
    depth_aware_placement: Optional[bool] = Field(None)

    # --- Auto-tune parameters ---
    reference_set_id: str = Field(..., description="Reference set for gap measurement")
    target_gap_score: float = Field(25.0, ge=0, le=100, description="Stop when gap score falls below this")
    max_tune_iterations: int = Field(5, ge=1, le=15, description="Max tuning iterations")
    probe_size: int = Field(50, ge=10, le=200, description="Images per probe batch (>=50 recommended for reliable metrics)")
    auto_start_full: bool = Field(True, description="Automatically start full generation after convergence")


class AutoTuneIterationResult(BaseModel):
    """Result of a single auto-tune iteration."""
    iteration: int
    gap_score: float
    gap_level: str
    technique_applied: str
    suggestions_applied: int = 0
    effects_config_snapshot: Dict[str, Any]


class AutoTuneStatusResponse(BaseModel):
    """Status of an auto-tune job."""
    job_id: str
    status: str  # pending, tuning, generating, completed, failed, cancelled
    iteration: int = 0
    max_iterations: int = 0
    current_gap_score: Optional[float] = None
    target_gap_score: float = 25.0
    phase: str = ""  # probe_generating, computing_metrics, analyzing, adjusting, full_generating
    history: List[AutoTuneIterationResult] = []
    tuned_effects_config: Optional[Dict[str, Any]] = None
    full_generation_job_id: Optional[str] = None
    improvement_pct: Optional[float] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


# In-memory job store (gateway doesn't have a shared DB)
_auto_tune_jobs: Dict[str, AutoTuneStatusResponse] = {}


# =============================================================================
# Format Conversion Helpers
# =============================================================================

def _flatten_effects_config(ec: Any) -> Dict[str, Any]:
    """Convert nested frontend EffectsConfig to flat augmentor format.

    Frontend sends: {color_correction: {enabled: true, color_intensity: 0.12}, ...}
    Augmentor expects: {color_intensity: 0.12, blur_strength: 0.5, ...}
    """
    default_flat = {
        "color_intensity": 0.12,
        "blur_strength": 0.5,
        "underwater_intensity": 0.15,
        "caustics_intensity": 0.10,
        "shadow_opacity": 0.10,
        "lighting_type": "ambient",
        "lighting_intensity": 0.5,
        "motion_blur_probability": 0.2,
        "water_clarity": "clear",
    }

    if ec is None:
        return default_flat

    if not isinstance(ec, dict):
        return default_flat

    # Check if already flat (has top-level keys like color_intensity)
    if "color_intensity" in ec or "blur_strength" in ec:
        merged = dict(default_flat)
        merged.update(ec)
        return merged

    # Nested format from frontend — extract values
    def _get(section: str, key: str, default: Any) -> Any:
        sub = ec.get(section)
        if isinstance(sub, dict):
            return sub.get(key, default)
        return default

    return {
        "color_intensity": _get("color_correction", "color_intensity", 0.12),
        "blur_strength": _get("blur_matching", "blur_strength", 0.5),
        "underwater_intensity": _get("underwater", "underwater_intensity", 0.15),
        "caustics_intensity": _get("caustics", "caustics_intensity", 0.10),
        "shadow_opacity": _get("shadows", "shadow_opacity", 0.10),
        "lighting_type": _get("lighting", "lighting_type", "ambient"),
        "lighting_intensity": _get("lighting", "lighting_intensity", 0.5),
        "motion_blur_probability": _get("motion_blur", "motion_blur_probability", 0.2),
        "water_clarity": _get("underwater", "water_clarity", "clear"),
    }


def _fix_legacy_path(p: Optional[str]) -> Optional[str]:
    """Migrate legacy /data/ paths to Docker Compose mount points."""
    if not p:
        return p
    if p.startswith("/data/output"):
        return p.replace("/data/output", "/app/output", 1)
    if p.startswith("/data/"):
        return p.replace("/data/", "/app/datasets/", 1)
    return p


def _resolve_request(request: AutoTuneRequest) -> Dict[str, Any]:
    """Resolve frontend/legacy fields into a normalized dict for the augmentor.

    Mirrors the transformation logic in augment.py compose_batch().
    """
    backgrounds_dir = _fix_legacy_path(request.backgrounds_dir)
    objects_dir = _fix_legacy_path(request.objects_dir)

    if request.source_dataset:
        if not backgrounds_dir:
            backgrounds_dir = os.path.join(request.source_dataset, "Backgrounds_filtered")
        if not objects_dir:
            objects_dir = os.path.join(request.source_dataset, "Objects")

    if not backgrounds_dir or not objects_dir:
        raise ValueError("Either source_dataset or both backgrounds_dir and objects_dir are required")

    # Resolve num_images from target_counts
    targets = request.targets_per_class or request.target_counts
    num_images = request.num_images
    if num_images is None:
        if targets:
            num_images = sum(targets.values())
        else:
            num_images = 200

    # Resolve depth_aware from frontend flag
    depth_aware = request.depth_aware
    if request.depth_aware_placement is not None:
        depth_aware = request.depth_aware_placement

    # Resolve max_objects from placement_config
    max_objects = request.max_objects_per_image
    if request.placement_config:
        max_objects = request.placement_config.get("max_objects_per_image", max_objects)

    return {
        "backgrounds_dir": backgrounds_dir,
        "objects_dir": objects_dir,
        "num_images": num_images,
        "targets_per_class": targets,
        "max_objects_per_image": max_objects,
        "depth_aware": depth_aware,
        "effects": request.effects,
    }


# =============================================================================
# Advisor → EffectsConfig Mapping
# =============================================================================

_ADVISOR_TO_EFFECTS_MAP = {
    "effects.lighting.intensity": "lighting_intensity",
    "effects.color_correction.intensity": "color_intensity",
    "effects.blur_matching.strength": "blur_strength",
    "effects.shadows.opacity": "shadow_opacity",
    "effects.underwater.intensity": "underwater_intensity",
    "effects.caustics.intensity": "caustics_intensity",
    "effects.edge_smoothing.feather_radius": "blur_strength",  # closest match
    "augmentation.contrast": "color_intensity",  # fallback mapping
    "augmentation.noise": "motion_blur_probability",  # approximate mapping
    "domain_randomization.recommended": None,  # handled separately
}


def _effects_config_to_nested(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Convert flat EffectsConfig to the nested structure the advisor expects.

    The advisor's _extract_config_value uses dot-notation paths like
    "effects.lighting.intensity", so we build the corresponding nested dict.
    """
    return {
        "effects": {
            "lighting": {
                "intensity": flat.get("lighting_intensity", 0.4),
                "type": flat.get("lighting_type", "ambient"),
            },
            "color_correction": {
                "intensity": flat.get("color_intensity", 0.15),
            },
            "blur_matching": {
                "strength": flat.get("blur_strength", 0.4),
            },
            "shadows": {
                "opacity": flat.get("shadow_opacity", 0.08),
            },
            "underwater": {
                "intensity": flat.get("underwater_intensity", 0.2),
            },
            "caustics": {
                "intensity": flat.get("caustics_intensity", 0.12),
            },
            "edge_smoothing": {
                "feather_radius": flat.get("blur_strength", 0.4),
            },
        },
        "augmentation": {
            "contrast": flat.get("color_intensity", 0.15),
            "noise": flat.get("motion_blur_probability", 0.1),
        },
    }


def _map_suggestions_to_effects_config(
    suggestions: List[Dict[str, Any]],
    current_config: Dict[str, Any],
) -> tuple[Dict[str, Any], int]:
    """Apply advisor suggestions to effects config.

    Maps advisor parameter_path to EffectsConfig field names and
    applies suggested values with bounds clamping.

    Returns:
        Tuple of (updated_config, count_of_applied_suggestions).
    """
    updated = dict(current_config)
    applied = 0

    for suggestion in suggestions:
        param_path = suggestion.get("parameter_path", "")
        suggested_value = suggestion.get("suggested_value")

        if suggested_value is None:
            continue

        effects_field = _ADVISOR_TO_EFFECTS_MAP.get(param_path)
        if effects_field is None:
            # Try direct match (e.g., if advisor uses effects_config field names)
            if param_path in current_config:
                effects_field = param_path
            else:
                logger.debug(f"No mapping for advisor path: {param_path}")
                continue

        # Clamp to reasonable bounds
        if effects_field in ("color_intensity", "underwater_intensity", "caustics_intensity",
                             "shadow_opacity", "lighting_intensity", "motion_blur_probability"):
            suggested_value = max(0.0, min(1.0, float(suggested_value)))
        elif effects_field == "blur_strength":
            suggested_value = max(0.0, min(2.0, float(suggested_value)))

        logger.info(f"Auto-tune: {effects_field} {current_config.get(effects_field)} -> {suggested_value}")
        updated[effects_field] = suggested_value
        applied += 1

    return updated, applied


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/start", response_model=AutoTuneStatusResponse)
async def start_auto_tune(request: AutoTuneRequest, background_tasks: BackgroundTasks):
    """Start closed-loop auto-tuning.

    Generates probe batches, measures domain gap with C-RADIOv4 metrics,
    gets advisor suggestions, adjusts effects_config, and repeats until
    the target gap score is reached or max iterations exhausted.

    If auto_start_full is True, automatically launches full generation
    with the tuned config after convergence.
    """
    job_id = f"atune_{uuid.uuid4().hex[:12]}"

    job_status = AutoTuneStatusResponse(
        job_id=job_id,
        status="pending",
        max_iterations=request.max_tune_iterations,
        target_gap_score=request.target_gap_score,
        phase="initializing",
        started_at=time.time(),
    )
    _auto_tune_jobs[job_id] = job_status

    background_tasks.add_task(_run_auto_tune, job_id, request)

    return job_status


@router.get("/jobs/{job_id}", response_model=AutoTuneStatusResponse)
async def get_auto_tune_status(job_id: str):
    """Get auto-tune job status."""
    job = _auto_tune_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Auto-tune job {job_id} not found")
    return job


@router.delete("/jobs/{job_id}")
async def cancel_auto_tune(job_id: str):
    """Cancel a running auto-tune job."""
    job = _auto_tune_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Auto-tune job {job_id} not found")

    if job.status in ("completed", "failed", "cancelled"):
        return {"success": False, "message": f"Job already {job.status}"}

    job.status = "cancelled"
    return {"success": True, "job_id": job_id, "message": "Cancellation requested"}


@router.get("/jobs")
async def list_auto_tune_jobs():
    """List all auto-tune jobs."""
    return {
        "jobs": list(_auto_tune_jobs.values()),
        "total": len(_auto_tune_jobs),
    }


# =============================================================================
# Background Task: Auto-Tune Loop
# =============================================================================

async def _run_auto_tune(job_id: str, request: AutoTuneRequest):
    """Closed-loop auto-tune background task."""
    job = _auto_tune_jobs[job_id]
    job.status = "tuning"
    registry = get_service_registry()

    try:
        # Verify domain_gap service is reachable before starting the loop
        try:
            async with httpx.AsyncClient(timeout=10) as http:
                resp = await http.get(f"{registry.domain_gap.base_url}/ping")
                resp.raise_for_status()
            logger.info(f"Auto-tune {job_id}: domain_gap service is reachable")
        except Exception as e:
            raise RuntimeError(
                f"Domain gap service unreachable at {registry.domain_gap.base_url}: {e}"
            )

        # Fix legacy /data/ paths from persisted frontend state
        if request.output_dir.startswith("/data/"):
            request = request.model_copy(update={
                "output_dir": _fix_legacy_path(request.output_dir),
                "source_dataset": _fix_legacy_path(request.source_dataset),
            })
            logger.info(f"Auto-tune {job_id}: migrated legacy paths -> output_dir={request.output_dir}")

        # Resolve frontend/legacy fields into normalized format
        resolved = _resolve_request(request)
        logger.info(
            f"Auto-tune {job_id}: resolved dirs — "
            f"backgrounds={resolved['backgrounds_dir']}, "
            f"objects={resolved['objects_dir']}, "
            f"output={request.output_dir}, "
            f"num_images={resolved['num_images']}"
        )

        # Flatten effects_config from nested frontend format to flat augmentor format
        current_config = _flatten_effects_config(request.effects_config)

        best_score = float("inf")
        stagnation_count = 0

        for iteration in range(1, request.max_tune_iterations + 1):
            if job.status == "cancelled":
                break

            job.iteration = iteration
            job.phase = "probe_generating"
            logger.info(f"Auto-tune {job_id}: iteration {iteration}/{request.max_tune_iterations}")

            # ---------------------------------------------------
            # Step 1: Generate probe batch via Augmentor
            # ---------------------------------------------------
            probe_output_dir = f"{request.output_dir}/probe_iter_{iteration:02d}"
            probe_request = {
                "backgrounds_dir": resolved["backgrounds_dir"],
                "objects_dir": resolved["objects_dir"],
                "output_dir": probe_output_dir,
                "num_images": request.probe_size,
                "effects": resolved["effects"],
                "effects_config": current_config,
                "max_objects_per_image": resolved["max_objects_per_image"],
                "depth_aware": resolved["depth_aware"],
                "targets_per_class": resolved["targets_per_class"],
            }

            probe_job = await _call_service_post(
                registry.augmentor, "/compose-batch", probe_request, COMPOSE_TIMEOUT
            )
            probe_job_id = probe_job.get("job_id")
            if not probe_job_id:
                raise RuntimeError(f"Augmentor did not return job_id: {probe_job}")

            # Poll until probe generation completes
            probe_result = await _poll_augmentor_job(registry.augmentor, probe_job_id, job)
            if job.status == "cancelled":
                break

            logger.info(f"Auto-tune {job_id}: probe batch completed, result keys={list(probe_result.keys()) if probe_result else 'None'}")

            # ---------------------------------------------------
            # Step 2: Compute domain gap metrics (RADIO-primary)
            # ---------------------------------------------------
            job.phase = "computing_metrics"

            # The augmentor nests output inside a job_id subdirectory:
            #   request.output_dir / {job_id} / images/
            # Extract the actual output_dir from the probe result.
            actual_output_dir = (
                probe_result.get("output_dir")
                or f"{probe_output_dir}/{probe_job_id}"
            )
            synthetic_images_dir = f"{actual_output_dir}/images"
            logger.info(f"Auto-tune {job_id}: synthetic_images_dir={synthetic_images_dir}")
            await _wait_for_directory(synthetic_images_dir, timeout=15.0)
            logger.info(
                f"Auto-tune {job_id}: requesting metrics for "
                f"synthetic_dir={synthetic_images_dir}, "
                f"reference_set_id={request.reference_set_id}"
            )

            metrics_request = {
                "synthetic_dir": synthetic_images_dir,
                "reference_set_id": request.reference_set_id,
                "max_images": max(request.probe_size, 50),
                "compute_radio_mmd": True,
                "compute_fd_radio": True,
                "compute_fid": False,
                "compute_kid": False,
                "compute_color_distribution": True,
                "compute_cmmd": False,
                "compute_prdc": True,
            }

            metrics_result = await _call_service_post(
                registry.domain_gap, "/metrics/compute", metrics_request, METRICS_TIMEOUT
            )
            gap_score = metrics_result.get("overall_gap_score", 100.0)
            gap_level = metrics_result.get("gap_level", "unknown")
            job.current_gap_score = gap_score
            logger.info(f"Auto-tune {job_id}: iter {iteration} gap_score={gap_score:.1f} ({gap_level})")

            iteration_result = AutoTuneIterationResult(
                iteration=iteration,
                gap_score=gap_score,
                gap_level=gap_level,
                technique_applied="none",
                suggestions_applied=0,
                effects_config_snapshot=dict(current_config),
            )

            # ---------------------------------------------------
            # Step 3: Check convergence
            # ---------------------------------------------------
            if gap_score <= request.target_gap_score:
                logger.info(f"Auto-tune {job_id}: target reached at iteration {iteration}")
                iteration_result.technique_applied = "target_reached"
                job.history.append(iteration_result)
                break

            # Check stagnation
            if gap_score >= best_score:
                stagnation_count += 1
                if stagnation_count >= 2:
                    logger.warning(f"Auto-tune {job_id}: early stopping (stagnation)")
                    iteration_result.technique_applied = "early_stopping"
                    job.history.append(iteration_result)
                    break
            else:
                best_score = gap_score
                stagnation_count = 0

            # ---------------------------------------------------
            # Step 4: Get advisor suggestions (non-fatal on failure)
            # ---------------------------------------------------
            suggestions: List[Dict[str, Any]] = []
            if iteration < request.max_tune_iterations:
                job.phase = "analyzing"
                try:
                    analyze_request = {
                        "synthetic_dir": synthetic_images_dir,
                        "reference_set_id": request.reference_set_id,
                        "max_images": min(request.probe_size, 50),
                        "current_config": _effects_config_to_nested(current_config),
                    }

                    analysis_job = await _call_service_post(
                        registry.domain_gap, "/analyze", analyze_request, ANALYZE_TIMEOUT
                    )
                    analysis_job_id = analysis_job.get("job_id")
                    if analysis_job_id:
                        analysis_result = await _poll_domain_gap_job(
                            registry.domain_gap, analysis_job_id, job
                        )
                        suggestions = (analysis_result or {}).get("suggestions", [])
                except Exception as analysis_err:
                    logger.warning(
                        f"Auto-tune {job_id}: advisor analysis failed (non-fatal): {analysis_err}"
                    )
                    suggestions = []

                if job.status == "cancelled":
                    break

                # ---------------------------------------------------
                # Step 5: Adjust effects_config from suggestions
                # ---------------------------------------------------
                job.phase = "adjusting"
                if suggestions:
                    current_config, applied_count = _map_suggestions_to_effects_config(
                        suggestions, current_config
                    )
                    iteration_result.technique_applied = "config_adjusted"
                    iteration_result.suggestions_applied = applied_count
                else:
                    iteration_result.technique_applied = "no_suggestions"

            job.history.append(iteration_result)

        # ---------------------------------------------------
        # Compute improvement
        # ---------------------------------------------------
        job.tuned_effects_config = current_config
        if job.history:
            initial_score = job.history[0].gap_score
            final_score = job.history[-1].gap_score
            if initial_score > 0:
                job.improvement_pct = round(
                    (initial_score - final_score) / initial_score * 100, 2
                )

        # ---------------------------------------------------
        # Step 6: Optionally start full generation
        # ---------------------------------------------------
        if request.auto_start_full and job.status != "cancelled":
            job.status = "generating"
            job.phase = "full_generating"
            logger.info(f"Auto-tune {job_id}: starting full generation with tuned config")

            full_request = {
                "backgrounds_dir": resolved["backgrounds_dir"],
                "objects_dir": resolved["objects_dir"],
                "output_dir": request.output_dir,
                "num_images": resolved["num_images"],
                "effects": resolved["effects"],
                "effects_config": current_config,
                "max_objects_per_image": resolved["max_objects_per_image"],
                "depth_aware": resolved["depth_aware"],
                "targets_per_class": resolved["targets_per_class"],
            }

            full_job = await _call_service_post(
                registry.augmentor, "/compose-batch", full_request, COMPOSE_TIMEOUT
            )
            job.full_generation_job_id = full_job.get("job_id")

            # Poll full generation to completion
            if job.full_generation_job_id:
                await _poll_augmentor_job(
                    registry.augmentor, job.full_generation_job_id, job
                )

        if job.status != "cancelled":
            job.status = "completed"
        job.completed_at = time.time()

        final_score_str = f"{job.history[-1].gap_score:.1f}" if job.history else "N/A"
        logger.info(
            f"Auto-tune {job_id} {job.status}: "
            f"{final_score_str} gap score, "
            f"{job.improvement_pct or 0:.1f}% improvement"
        )

    except httpx.HTTPStatusError as e:
        detail = e.response.text[:500] if e.response else ""
        error_msg = f"{e} — Response body: {detail}"
        logger.exception(f"Auto-tune {job_id} HTTP error: {error_msg}")
        job.status = "failed"
        job.error = error_msg
        job.completed_at = time.time()
    except Exception as e:
        error_msg = str(e) or f"{type(e).__name__}: {e!r}"
        logger.exception(f"Auto-tune {job_id} failed: {error_msg}")
        job.status = "failed"
        job.error = error_msg
        job.completed_at = time.time()


# =============================================================================
# Helper Functions
# =============================================================================

async def _wait_for_directory(path: str, timeout: float = 15.0) -> None:
    """Wait for a directory to appear on the filesystem (bind-mount sync).

    The gateway now mounts ./output:/app/output so it can check directly.
    If the directory never appears, log a warning but don't fail — the
    downstream service (domain_gap) will produce a clear 404 error.
    """
    import os
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.isdir(path):
            files = os.listdir(path)
            if files:
                logger.info(f"Directory ready: {path} ({len(files)} files)")
                return
            logger.debug(f"Directory exists but empty: {path}, waiting...")
        await asyncio.sleep(1.0)
    logger.warning(f"Directory not found after {timeout}s: {path}")

async def _call_service_post(
    client, endpoint: str, data: Dict[str, Any], timeout: float,
    max_retries: int = 3, retry_delay: float = 5.0,
) -> Dict[str, Any]:
    """Make a POST to a downstream service with custom timeout and retries."""
    url = f"{client.base_url}{endpoint}"
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as http:
                response = await http.post(url, json=data)
                if response.status_code >= 400:
                    body = response.text[:500]
                    logger.warning(
                        f"Service call {url} returned {response.status_code} "
                        f"(attempt {attempt}/{max_retries}): {body}"
                    )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            last_error = e
            # Don't retry on 4xx client errors (except 404 which may be transient during startup)
            if e.response.status_code != 404 and 400 <= e.response.status_code < 500:
                raise
            if attempt < max_retries:
                logger.info(f"Retrying {url} in {retry_delay}s (attempt {attempt}/{max_retries})")
                await asyncio.sleep(retry_delay)
        except (httpx.RequestError, httpx.TimeoutException) as e:
            last_error = e
            if attempt < max_retries:
                logger.info(f"Retrying {url} in {retry_delay}s (attempt {attempt}/{max_retries}): {e}")
                await asyncio.sleep(retry_delay)

    raise last_error


async def _poll_augmentor_job(
    client, aug_job_id: str, parent_job: AutoTuneStatusResponse
) -> Dict[str, Any]:
    """Poll an augmentor job until completion (with global timeout)."""
    start_time = time.time()
    while True:
        if parent_job.status == "cancelled":
            return {}

        elapsed = time.time() - start_time
        if elapsed > MAX_POLL_SECONDS:
            raise RuntimeError(
                f"Augmentor job {aug_job_id} timed out after {MAX_POLL_SECONDS}s"
            )

        await asyncio.sleep(POLL_INTERVAL)

        try:
            async with httpx.AsyncClient(timeout=30) as http:
                resp = await http.get(f"{client.base_url}/jobs/{aug_job_id}")
                resp.raise_for_status()
                job_data = resp.json()

            status = job_data.get("status", "unknown")
            if status == "completed":
                return job_data.get("result", job_data)
            elif status in ("failed", "cancelled"):
                raise RuntimeError(
                    f"Augmentor job {aug_job_id} {status}: "
                    f"{job_data.get('error', 'unknown error')}"
                )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # Job might not be registered yet, retry
                continue
            raise


async def _poll_domain_gap_job(
    client, gap_job_id: str, parent_job: AutoTuneStatusResponse
) -> Optional[Dict[str, Any]]:
    """Poll a domain gap job until completion (with global timeout)."""
    start_time = time.time()
    while True:
        if parent_job.status == "cancelled":
            return None

        elapsed = time.time() - start_time
        if elapsed > MAX_POLL_SECONDS:
            logger.warning(f"Domain gap job {gap_job_id} timed out after {MAX_POLL_SECONDS}s")
            return None

        await asyncio.sleep(POLL_INTERVAL)

        try:
            async with httpx.AsyncClient(timeout=30) as http:
                resp = await http.get(f"{client.base_url}/jobs/{gap_job_id}")
                resp.raise_for_status()
                job_data = resp.json()

            status = job_data.get("status", "unknown")
            if status == "completed":
                return job_data.get("result", job_data)
            elif status in ("failed", "cancelled"):
                logger.warning(
                    f"Domain gap job {gap_job_id} {status}: "
                    f"{job_data.get('error', 'unknown error')}"
                )
                return None
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                continue
            raise
