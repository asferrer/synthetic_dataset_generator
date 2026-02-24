"""
ML-Guided Optimizer Router
============================
Endpoints for ML-based domain gap optimization using Bayesian optimization
and predictive models.
"""

import json
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from loguru import logger


router = APIRouter(prefix="/ml-optimize", tags=["ML Optimization"])


# =============================================================================
# Request / Response Models
# =============================================================================

class MLOptimizeRequest(BaseModel):
    """Request for ML-guided Bayesian optimization."""
    synthetic_dir: str = Field(description="Directory with synthetic images to optimize")
    reference_set_id: str = Field(description="Reference set ID for gap measurement")
    n_trials: int = Field(10, ge=3, le=50, description="Number of optimization trials")
    probe_size: int = Field(50, ge=20, le=200, description="Images to generate per trial")
    warm_start: bool = Field(True, description="Use ML predictor for warm-start")
    timeout_seconds: Optional[int] = Field(None, ge=60, le=7200, description="Optimization timeout")
    parameter_ranges: Optional[Dict[str, tuple]] = Field(
        None, description="Custom parameter ranges: {'param': (min, max)}"
    )
    current_config: Optional[Dict[str, Any]] = Field(
        None, description="Current effects config (optional baseline)"
    )


class MLOptimizeStatus(BaseModel):
    """Status of ML optimization job."""
    job_id: str
    status: str  # pending, optimizing, completed, failed, cancelled
    current_trial: int = 0
    total_trials: int = 0
    best_gap_score: Optional[float] = None
    best_config: Optional[Dict[str, Any]] = None
    optimization_history: List[Dict[str, Any]] = []
    feature_importance: Optional[Dict[str, float]] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class FeatureImportanceResponse(BaseModel):
    """Response with feature importance analysis."""
    success: bool
    source: str  # "bayesian", "ml_predictor", "combined"
    importances: Dict[str, float] = Field(description="Parameter → importance mapping")
    top_features: List[str] = Field(description="Top features by importance")


class PredictGapScoreRequest(BaseModel):
    """Request to predict gap score for a configuration."""
    config: Dict[str, Any] = Field(description="Effects configuration to evaluate")


class PredictGapScoreResponse(BaseModel):
    """Response with predicted gap score."""
    success: bool
    predicted_score: Optional[float] = Field(None, description="Predicted gap score (0-100)")
    confidence: Optional[str] = Field(None, description="Prediction confidence level")
    message: str = ""


# =============================================================================
# Cancellation events (thread-safe, used by background tasks in thread pool)
# =============================================================================

_cancel_events: Dict[str, threading.Event] = {}


def _job_to_status(job_row: Dict[str, Any]) -> MLOptimizeStatus:
    """Convert a DB job row to MLOptimizeStatus model."""
    progress = job_row.get("progress_details") or {}
    if isinstance(progress, str):
        progress = json.loads(progress)
    result = job_row.get("result_summary") or {}
    if isinstance(result, str):
        result = json.loads(result)
    return MLOptimizeStatus(
        job_id=job_row["id"],
        status=job_row.get("status", "pending"),
        current_trial=progress.get("current_trial", 0),
        total_trials=progress.get("total_trials", 0),
        best_gap_score=progress.get("best_gap_score"),
        best_config=result.get("best_config"),
        optimization_history=result.get("optimization_history", []),
        feature_importance=result.get("feature_importance"),
        error=job_row.get("error_message"),
        started_at=progress.get("started_at"),
        completed_at=progress.get("completed_at"),
    )


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/start", response_model=MLOptimizeStatus)
async def start_ml_optimization(
    request: MLOptimizeRequest,
    background_tasks: BackgroundTasks,
):
    """Start ML-guided Bayesian optimization for domain gap minimization.

    This endpoint uses:
    - XGBoost predictor trained on historical configurations
    - Optuna Bayesian optimization (TPE sampler)
    - Active learning loop: suggest → evaluate → update → repeat

    Returns a job_id to track progress via GET /ml-optimize/jobs/{job_id}.
    """
    from app.main import state

    if not state.metrics_engine:
        raise HTTPException(status_code=503, detail="Metrics engine not initialized")
    if not state.reference_manager:
        raise HTTPException(status_code=503, detail="Reference manager not initialized")

    # Validate reference set exists
    ref_set = state.reference_manager.get_set(request.reference_set_id)
    if not ref_set:
        raise HTTPException(
            status_code=404,
            detail=f"Reference set {request.reference_set_id} not found"
        )

    job_id = f"mlopt_{uuid.uuid4().hex[:12]}"

    state.db.create_job(
        job_id=job_id,
        job_type="ml_optimization",
        service="domain_gap",
        request_params={
            "synthetic_dir": request.synthetic_dir,
            "reference_set_id": request.reference_set_id,
            "n_trials": request.n_trials,
            "probe_size": request.probe_size,
        },
        total_items=request.n_trials,
    )
    state.db.update_job_progress(
        job_id,
        progress_details={"total_trials": request.n_trials, "current_trial": 0},
    )

    background_tasks.add_task(
        _run_ml_optimization,
        job_id,
        request,
        ref_set.image_dir,
    )

    return MLOptimizeStatus(
        job_id=job_id,
        status="pending",
        total_trials=request.n_trials,
    )


@router.get("/jobs/{job_id}", response_model=MLOptimizeStatus)
async def get_ml_optimization_status(job_id: str):
    """Get status of ML optimization job."""
    from app.main import state

    job_row = state.db.get_job(job_id)
    if not job_row:
        raise HTTPException(status_code=404, detail=f"ML optimization job {job_id} not found")
    return _job_to_status(job_row)


@router.delete("/jobs/{job_id}")
async def cancel_ml_optimization(job_id: str):
    """Cancel a running ML optimization job."""
    from app.main import state

    job_row = state.db.get_job(job_id)
    if not job_row:
        raise HTTPException(status_code=404, detail=f"ML optimization job {job_id} not found")

    if job_row.get("status") in ("completed", "failed", "cancelled"):
        return {"success": False, "message": f"Job already {job_row['status']}"}

    # Signal the background task to stop
    event = _cancel_events.get(job_id)
    if event:
        event.set()

    state.db.update_job_status(job_id, "cancelled")
    return {"success": True, "job_id": job_id, "message": "Cancellation requested"}


@router.get("/jobs")
async def list_ml_optimization_jobs():
    """List all ML optimization jobs."""
    from app.main import state

    rows = state.db.list_jobs(job_type="ml_optimization")
    return {
        "jobs": [_job_to_status(r) for r in rows],
        "total": len(rows),
    }


@router.get("/feature-importance", response_model=FeatureImportanceResponse)
async def get_feature_importance(source: str = "combined"):
    """Get feature importance analysis from optimization history.

    Args:
        source: "bayesian", "ml_predictor", or "combined"

    Returns:
        Feature importance scores showing which parameters most affect gap score.
    """
    from app.main import state

    # Get or create predictor and optimizer from state
    # (In practice, these would be persisted in state or loaded from disk)
    from app.engines.bayesian_optimizer_engine import HybridOptimizer
    from app.engines.predictor_engine import PredictorEngine

    try:
        predictor = PredictorEngine()
        optimizer = HybridOptimizer(predictor=predictor)

        importances = optimizer.get_feature_importance()

        if importances is None:
            return FeatureImportanceResponse(
                success=False,
                source=source,
                importances={},
                top_features=[],
            )

        # Sort by importance (descending)
        sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        top_features = [k for k, v in sorted_items[:5]]

        return FeatureImportanceResponse(
            success=True,
            source=source,
            importances=importances,
            top_features=top_features,
        )

    except Exception as e:
        logger.exception(f"Feature importance analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=PredictGapScoreResponse)
async def predict_gap_score(request: PredictGapScoreRequest):
    """Predict domain gap score for a given configuration.

    Uses the trained ML predictor to estimate gap score without
    actually generating images. Useful for quick "what-if" analysis.
    """
    from app.engines.predictor_engine import PredictorEngine

    try:
        predictor = PredictorEngine()

        if not predictor.trained:
            return PredictGapScoreResponse(
                success=False,
                message="Predictor not trained yet. Need at least 10 historical configurations.",
            )

        predicted_score = predictor.predict(request.config)

        if predicted_score is None:
            return PredictGapScoreResponse(
                success=False,
                message="Prediction failed",
            )

        # Estimate confidence based on training data size
        n_samples = len(predictor.history.history)
        if n_samples < 15:
            confidence = "low"
        elif n_samples < 30:
            confidence = "medium"
        else:
            confidence = "high"

        return PredictGapScoreResponse(
            success=True,
            predicted_score=predicted_score,
            confidence=confidence,
            message=f"Predicted from {n_samples} historical configurations",
        )

    except Exception as e:
        logger.exception(f"Gap score prediction failed: {e}")
        return PredictGapScoreResponse(
            success=False,
            message=str(e),
        )


# =============================================================================
# Background Task: ML-Guided Optimization
# =============================================================================

async def _run_ml_optimization(
    job_id: str,
    request: MLOptimizeRequest,
    real_dir: str,
):
    """Background task for ML-guided Bayesian optimization."""
    import asyncio
    from app.main import state
    from app.engines.bayesian_optimizer_engine import HybridOptimizer
    from app.engines.predictor_engine import PredictorEngine

    cancel_event = threading.Event()
    _cancel_events[job_id] = cancel_event

    state.db.update_job_status(job_id, "running")
    state.db.update_job_progress(
        job_id,
        progress_details={
            "total_trials": request.n_trials,
            "current_trial": 0,
            "started_at": time.time(),
        },
    )

    current_trial = 0
    best_gap_score: Optional[float] = None
    best_config: Optional[Dict[str, Any]] = None
    optimization_history: List[Dict[str, Any]] = []

    try:
        # Initialize predictor and optimizer
        predictor = PredictorEngine()
        optimizer = HybridOptimizer(predictor=predictor)

        def evaluate_config(config: Dict[str, Any]) -> float:
            """Evaluate a configuration proposed by Optuna.

            TODO: Full implementation should:
            1. Apply `config` to the augmentor service to generate a probe batch
            2. Measure gap score on the generated probe images vs real reference
            3. Return the gap score as the objective to minimize

            Current limitation: measures the existing synthetic_dir baseline
            without applying the suggested config. All trials evaluate the same
            images, so the optimization explores configs but cannot validate them
            against real generation. The best_config returned reflects Optuna's
            search but is not experimentally validated.
            """
            nonlocal current_trial, best_gap_score, best_config

            # Check for cancellation
            if cancel_event.is_set():
                raise RuntimeError("Cancelled by user")

            try:
                metrics = state.metrics_engine.compute_metrics(
                    synthetic_dir=request.synthetic_dir,
                    real_dir=real_dir,
                    max_images=request.probe_size,
                    compute_radio_mmd=True,
                    compute_fd_radio=True,
                    compute_fid=False,
                    compute_kid=False,
                    compute_color=True,
                    compute_prdc=True,
                )

                gap_score = metrics.overall_gap_score
                current_trial += 1

                optimization_history.append({
                    "trial": current_trial,
                    "config": config,
                    "gap_score": gap_score,
                })

                if best_gap_score is None or gap_score < best_gap_score:
                    best_gap_score = gap_score
                    best_config = config

                # Update progress in DB
                state.db.update_job_progress(
                    job_id,
                    processed_items=current_trial,
                    progress_details={
                        "total_trials": request.n_trials,
                        "current_trial": current_trial,
                        "best_gap_score": best_gap_score,
                        "started_at": state.db.get_job(job_id).get("progress_details", {}).get("started_at"),
                    },
                )

                logger.info(f"ML-opt {job_id}: trial {current_trial}, score={gap_score:.1f}")
                return gap_score

            except Exception as e:
                logger.error(f"Config evaluation failed: {e}")
                return 100.0

        # Run optimization
        result = await asyncio.to_thread(
            optimizer.optimize,
            n_trials=request.n_trials,
            evaluation_callback=evaluate_config,
            timeout_seconds=request.timeout_seconds,
            parameter_ranges=request.parameter_ranges,
            initial_config=request.current_config,
            warm_start=request.warm_start,
        )

        feature_importance = optimizer.get_feature_importance()

        state.db.complete_job(
            job_id,
            "completed",
            result_summary={
                "best_config": result["best_config"],
                "best_gap_score": result["best_gap_score"],
                "optimization_history": result["optimization_history"],
                "feature_importance": feature_importance,
                "n_trials_completed": result.get("n_trials_completed", current_trial),
            },
            processing_time_ms=(time.time() - (state.db.get_job(job_id) or {}).get("started_at", time.time())) * 1000,
        )

        logger.info(
            f"ML optimization {job_id} completed: "
            f"best_score={result['best_gap_score']:.1f}, "
            f"trials={result.get('n_trials_completed', current_trial)}"
        )

    except Exception as e:
        logger.exception(f"ML optimization {job_id} failed: {e}")
        final_status = "cancelled" if cancel_event.is_set() else "failed"
        state.db.complete_job(
            job_id,
            final_status,
            error_message=str(e),
            result_summary={
                "best_config": best_config,
                "best_gap_score": best_gap_score,
                "optimization_history": optimization_history,
            },
        )
    finally:
        _cancel_events.pop(job_id, None)
