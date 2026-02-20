"""
Domain Gap Reduction Router
============================
Proxy endpoints for the Domain Gap Reduction service.
Forwards requests from the gateway to the domain_gap microservice.
"""

import logging
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from app.services.client import get_service_registry

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/domain-gap",
    tags=["Domain Gap Reduction"],
)

DOMAIN_GAP_TIMEOUT = 120.0
DOMAIN_GAP_BATCH_TIMEOUT = 300.0    # 5 min per batch of ~50 files
DOMAIN_GAP_COMPUTE_TIMEOUT = 600.0  # 10 min for FID/KID on large image sets
DOMAIN_GAP_FINALIZE_TIMEOUT = 600.0  # 10 min for stats on 1000+ images


def _extract_detail(e: httpx.HTTPStatusError) -> str:
    """Extract error detail from downstream service response."""
    try:
        body = e.response.json()
        return body.get("detail", e.response.text)
    except Exception:
        return e.response.text


def _get_client():
    """Get the domain_gap service client."""
    registry = get_service_registry()
    return registry.domain_gap


# =============================================================================
# References
# =============================================================================

@router.post("/references/upload")
async def upload_references(
    files: List[UploadFile] = File(...),
    name: str = Form(...),
    description: str = Form(""),
    domain_id: str = Form("default"),
):
    """Upload real reference images to create a new reference set."""
    client = _get_client()
    try:
        # Forward multipart upload directly
        async with httpx.AsyncClient(timeout=DOMAIN_GAP_TIMEOUT) as http:
            files_data = []
            for f in files:
                content = await f.read()
                files_data.append(("files", (f.filename, content, f.content_type or "image/jpeg")))

            response = await http.post(
                f"{client.base_url}/references/upload",
                files=files_data,
                data={"name": name, "description": description, "domain_id": domain_id},
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=_extract_detail(e))
    except Exception as e:
        logger.error(f"Failed to upload references: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/references/create")
async def create_reference_set(
    name: str = Form(...),
    description: str = Form(""),
    domain_id: str = Form("default"),
):
    """Create an empty reference set (phase 1 of chunked upload)."""
    client = _get_client()
    try:
        async with httpx.AsyncClient(timeout=DOMAIN_GAP_TIMEOUT) as http:
            response = await http.post(
                f"{client.base_url}/references/create",
                data={"name": name, "description": description, "domain_id": domain_id},
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=_extract_detail(e))
    except Exception as e:
        logger.error(f"Failed to create reference set: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/references/{set_id}/add-batch")
async def add_reference_batch(
    set_id: str,
    files: List[UploadFile] = File(...),
):
    """Add a batch of images to an existing reference set (phase 2)."""
    client = _get_client()
    try:
        async with httpx.AsyncClient(timeout=DOMAIN_GAP_BATCH_TIMEOUT) as http:
            files_data = []
            for f in files:
                content = await f.read()
                files_data.append(("files", (f.filename, content, f.content_type or "image/jpeg")))

            response = await http.post(
                f"{client.base_url}/references/{set_id}/add-batch",
                files=files_data,
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=_extract_detail(e))
    except Exception as e:
        logger.error(f"Failed to add batch to reference set {set_id}: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/references/{set_id}/finalize")
async def finalize_reference_set(set_id: str):
    """Finalize a reference set and compute statistics (phase 3)."""
    client = _get_client()
    try:
        async with httpx.AsyncClient(timeout=DOMAIN_GAP_FINALIZE_TIMEOUT) as http:
            response = await http.post(
                f"{client.base_url}/references/{set_id}/finalize",
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=_extract_detail(e))
    except Exception as e:
        logger.error(f"Failed to finalize reference set {set_id}: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/references/from-directory")
async def create_reference_from_directory(request: Dict[str, Any]):
    """Create a reference set from a server-side directory."""
    client = _get_client()
    try:
        async with httpx.AsyncClient(timeout=DOMAIN_GAP_FINALIZE_TIMEOUT) as http:
            response = await http.post(
                f"{client.base_url}/references/from-directory",
                json=request,
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=_extract_detail(e))
    except Exception as e:
        logger.error(f"Failed to create reference set from directory: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/references")
async def list_references(domain_id: Optional[str] = None):
    """List all reference sets."""
    client = _get_client()
    try:
        params = {}
        if domain_id:
            params["domain_id"] = domain_id
        return await client.get("/references", params=params)
    except Exception as e:
        logger.error(f"Failed to list references: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/references/{set_id}")
async def get_reference(set_id: str):
    """Get reference set details."""
    client = _get_client()
    try:
        return await client.get(f"/references/{set_id}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=_extract_detail(e))
    except Exception as e:
        logger.error(f"Failed to get reference {set_id}: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@router.delete("/references/{set_id}")
async def delete_reference(set_id: str):
    """Delete a reference set."""
    client = _get_client()
    try:
        return await client.delete(f"/references/{set_id}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=_extract_detail(e))
    except Exception as e:
        logger.error(f"Failed to delete reference {set_id}: {e}")
        raise HTTPException(status_code=502, detail=str(e))


# =============================================================================
# Metrics
# =============================================================================

@router.post("/metrics/compute")
async def compute_metrics(request: Dict[str, Any]):
    """Compute domain gap metrics between synthetic and reference images."""
    client = _get_client()
    try:
        async with httpx.AsyncClient(timeout=DOMAIN_GAP_COMPUTE_TIMEOUT) as http:
            response = await http.post(
                f"{client.base_url}/metrics/compute",
                json=request,
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=_extract_detail(e))
    except Exception as e:
        logger.error(f"Failed to compute metrics: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/metrics/compare")
async def compare_metrics(request: Dict[str, Any]):
    """Compare metrics before and after processing."""
    client = _get_client()
    try:
        async with httpx.AsyncClient(timeout=DOMAIN_GAP_COMPUTE_TIMEOUT) as http:
            response = await http.post(
                f"{client.base_url}/metrics/compare",
                json=request,
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=_extract_detail(e))
    except Exception as e:
        logger.error(f"Failed to compare metrics: {e}")
        raise HTTPException(status_code=502, detail=str(e))


# =============================================================================
# Analysis
# =============================================================================

@router.post("/analyze")
async def analyze_gap(request: Dict[str, Any]):
    """Start domain gap analysis (async job). Returns job_id for polling."""
    client = _get_client()
    try:
        return await client.post("/analyze", data=request)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=_extract_detail(e))
    except Exception as e:
        logger.error(f"Failed to start gap analysis: {e}")
        raise HTTPException(status_code=502, detail=str(e))


# =============================================================================
# Randomization
# =============================================================================

@router.post("/randomize/apply")
async def randomize_single(request: Dict[str, Any]):
    """Apply domain randomization to a single image."""
    client = _get_client()
    try:
        return await client.post("/randomize/apply", data=request)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=_extract_detail(e))
    except Exception as e:
        logger.error(f"Failed to apply randomization: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/randomize/apply-batch")
async def randomize_batch(request: Dict[str, Any]):
    """Apply domain randomization to a batch (async job)."""
    client = _get_client()
    try:
        return await client.post("/randomize/apply-batch", data=request)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=_extract_detail(e))
    except Exception as e:
        logger.error(f"Failed to start randomization batch: {e}")
        raise HTTPException(status_code=502, detail=str(e))


# =============================================================================
# Style Transfer
# =============================================================================

@router.post("/style-transfer/apply")
async def style_transfer_single(request: Dict[str, Any]):
    """Apply neural style transfer to a single image."""
    client = _get_client()
    try:
        async with httpx.AsyncClient(timeout=DOMAIN_GAP_COMPUTE_TIMEOUT) as http:
            response = await http.post(
                f"{client.base_url}/style-transfer/apply",
                json=request,
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=_extract_detail(e))
    except Exception as e:
        logger.error(f"Failed to apply style transfer: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/style-transfer/apply-batch")
async def style_transfer_batch(request: Dict[str, Any]):
    """Apply style transfer to a batch (async job)."""
    client = _get_client()
    try:
        return await client.post("/style-transfer/apply-batch", data=request)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=_extract_detail(e))
    except Exception as e:
        logger.error(f"Failed to start style transfer batch: {e}")
        raise HTTPException(status_code=502, detail=str(e))


# =============================================================================
# Optimization
# =============================================================================

@router.post("/optimize")
async def optimize(request: Dict[str, Any]):
    """Start automatic domain gap optimization (async job)."""
    client = _get_client()
    try:
        return await client.post("/optimize", data=request)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=_extract_detail(e))
    except Exception as e:
        logger.error(f"Failed to start optimization: {e}")
        raise HTTPException(status_code=502, detail=str(e))


# =============================================================================
# Jobs
# =============================================================================

@router.get("/jobs")
async def list_jobs():
    """List all domain gap jobs."""
    client = _get_client()
    try:
        return await client.get("/jobs")
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job status."""
    client = _get_client()
    try:
        return await client.get(f"/jobs/{job_id}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=_extract_detail(e))
    except Exception as e:
        logger.error(f"Failed to get job {job_id}: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    client = _get_client()
    try:
        return await client.delete(f"/jobs/{job_id}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=_extract_detail(e))
    except Exception as e:
        logger.error(f"Failed to cancel job {job_id}: {e}")
        raise HTTPException(status_code=502, detail=str(e))


# =============================================================================
# Service Info
# =============================================================================

@router.get("/info")
async def domain_gap_info():
    """Get domain gap service info."""
    client = _get_client()
    try:
        return await client.get("/info")
    except Exception as e:
        logger.error(f"Failed to get domain gap info: {e}")
        raise HTTPException(status_code=502, detail=str(e))
