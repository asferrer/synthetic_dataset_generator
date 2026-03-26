"""
ML-Guided Optimization Router (Gateway Proxy)
==============================================
Proxies ML optimization requests to the domain_gap service.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.client import get_service_registry


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml-optimize", tags=["ML Optimization"])


# =============================================================================
# Request / Response Models (mirrored from domain_gap service)
# =============================================================================

class MLOptimizeRequest(BaseModel):
    """Request for ML-guided Bayesian optimization."""
    synthetic_dir: str
    reference_set_id: str
    n_trials: int = Field(10, ge=3, le=50)
    probe_size: int = Field(50, ge=20, le=200)
    warm_start: bool = True
    timeout_seconds: Optional[int] = None
    parameter_ranges: Optional[Dict[str, tuple]] = None
    current_config: Optional[Dict[str, Any]] = None


class PredictGapScoreRequest(BaseModel):
    """Request to predict gap score for a configuration."""
    config: Dict[str, Any]


# =============================================================================
# Endpoints (proxies to domain_gap service)
# =============================================================================

@router.post("/start")
async def start_ml_optimization(request: MLOptimizeRequest):
    """Start ML-guided Bayesian optimization.

    Uses:
    - XGBoost predictor trained on historical configurations
    - Optuna Bayesian optimization (TPE sampler)
    - Active learning: suggest → evaluate → update

    Returns a job_id to track progress.
    """
    registry = get_service_registry()

    try:
        response = await registry.domain_gap.post(
            "/ml-optimize/start",
            json=request.model_dump(),
            timeout=30.0,
        )
        return response
    except Exception as e:
        logger.exception(f"ML optimization start failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}")
async def get_ml_optimization_status(job_id: str):
    """Get status of ML optimization job."""
    registry = get_service_registry()

    try:
        response = await registry.domain_gap.get(f"/ml-optimize/jobs/{job_id}")
        return response
    except Exception as e:
        logger.exception(f"Failed to get ML optimization status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/jobs/{job_id}")
async def cancel_ml_optimization(job_id: str):
    """Cancel a running ML optimization job."""
    registry = get_service_registry()

    try:
        response = await registry.domain_gap.delete(f"/ml-optimize/jobs/{job_id}")
        return response
    except Exception as e:
        logger.exception(f"Failed to cancel ML optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs")
async def list_ml_optimization_jobs():
    """List all ML optimization jobs."""
    registry = get_service_registry()

    try:
        response = await registry.domain_gap.get("/ml-optimize/jobs")
        return response
    except Exception as e:
        logger.exception(f"Failed to list ML optimization jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feature-importance")
async def get_feature_importance(source: str = "combined"):
    """Get feature importance analysis.

    Shows which parameters most affect domain gap.

    Args:
        source: "bayesian", "ml_predictor", or "combined"
    """
    registry = get_service_registry()

    try:
        response = await registry.domain_gap.get(
            "/ml-optimize/feature-importance",
            params={"source": source},
        )
        return response
    except Exception as e:
        logger.exception(f"Failed to get feature importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict")
async def predict_gap_score(request: PredictGapScoreRequest):
    """Predict domain gap score for a configuration.

    Uses trained ML predictor for quick "what-if" analysis without
    generating images.
    """
    registry = get_service_registry()

    try:
        response = await registry.domain_gap.post(
            "/ml-optimize/predict",
            json=request.model_dump(),
        )
        return response
    except Exception as e:
        logger.exception(f"Gap score prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
