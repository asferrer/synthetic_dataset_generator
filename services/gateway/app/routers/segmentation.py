"""
Segmentation Router
===================
Endpoints for object extraction and SAM3 operations.
Proxies requests to the Segmentation service.
"""

import logging
from fastapi import APIRouter, HTTPException

from app.services.client import get_service_registry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/segmentation", tags=["Segmentation"])


# =============================================================================
# Job Cancellation Endpoints
# =============================================================================

@router.delete("/extract/jobs/{job_id}")
async def cancel_extraction_job(job_id: str):
    """
    Cancel a running extraction job.

    Stops job processing after current object, preserving already extracted objects.
    """
    logger.info(f"Cancel request for extraction job: {job_id}")

    try:
        registry = get_service_registry()
        result = await registry.segmentation.delete(f"/extract/jobs/{job_id}")
        return result
    except Exception as e:
        logger.error(f"Cancel extraction job failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sam3/jobs/{job_id}")
async def cancel_sam3_job(job_id: str):
    """
    Cancel a running SAM3 conversion job.

    Stops job processing after current annotation, preserving already converted annotations.
    """
    logger.info(f"Cancel request for SAM3 job: {job_id}")

    try:
        registry = get_service_registry()
        result = await registry.segmentation.delete(f"/sam3/jobs/{job_id}")
        return result
    except Exception as e:
        logger.error(f"Cancel SAM3 job failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
