"""
Gateway Service - FastAPI Application

API Gateway that orchestrates synthetic data generation services.
"""
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.services.client import get_service_registry
from app.services.orchestrator import get_orchestrator

# Service URLs
AUGMENTOR_SERVICE_URL = os.environ.get("AUGMENTOR_SERVICE_URL", "http://augmentor:8004")
from app.models.schemas import (
    GenerateImageRequest,
    GenerateImageResponse,
    BatchGenerateRequest,
    BatchGenerateResponse,
    HealthResponse,
    InfoResponse,
    ServiceHealth,
    ServiceStatus,
    AnnotationBox
)
from app.routers import augment, segmentation, datasets, filesystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting Gateway Service...")

    # Initialize service registry
    registry = get_service_registry()
    logger.info("Service registry initialized")

    # Check initial health of services
    try:
        health = await registry.check_all_health()
        for service, status in health.items():
            if status.get("healthy"):
                logger.info(f"Service {service}: healthy")
            else:
                logger.warning(f"Service {service}: {status.get('status', 'unknown')}")
    except Exception as e:
        logger.warning(f"Initial health check failed: {e}")

    yield

    logger.info("Shutting down Gateway Service...")


# Create FastAPI application
app = FastAPI(
    title="Synthetic Data Generation Gateway",
    description="API Gateway for orchestrating synthetic dataset generation",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(augment.router)
app.include_router(segmentation.router)
app.include_router(datasets.router)
app.include_router(filesystem.router)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check health of all services.

    Returns aggregated health status of the gateway and all downstream services.
    """
    import time

    try:
        registry = get_service_registry()

        # Add gateway itself first (if we're responding, we're healthy)
        services = [
            ServiceHealth(
                name="gateway",
                status=ServiceStatus.HEALTHY,
                url="http://gateway:8000",
                latency_ms=0.0,
                details={"version": "1.0.0"}
            )
        ]

        # Check downstream services
        health_results = await registry.check_all_health()
        all_healthy = True

        for service_name, status in health_results.items():
            is_healthy = status.get("healthy", False)
            if not is_healthy:
                all_healthy = False

            service_status = ServiceStatus.HEALTHY if is_healthy else ServiceStatus.UNHEALTHY

            services.append(ServiceHealth(
                name=service_name,
                status=service_status,
                url=getattr(registry, service_name).base_url if hasattr(registry, service_name) else "",
                latency_ms=status.get("latency_ms"),
                details=status.get("details")
            ))

        # Determine overall status
        # Healthy if depth, effects, and augmentor are up (segmentation is optional)
        depth_ok = any(s.name == "depth" and s.status == ServiceStatus.HEALTHY for s in services)
        effects_ok = any(s.name == "effects" and s.status == ServiceStatus.HEALTHY for s in services)
        augmentor_ok = any(s.name == "augmentor" and s.status == ServiceStatus.HEALTHY for s in services)

        if depth_ok and effects_ok and augmentor_ok:
            overall_status = ServiceStatus.HEALTHY
        elif (depth_ok and effects_ok) or (depth_ok and augmentor_ok) or (effects_ok and augmentor_ok):
            overall_status = ServiceStatus.DEGRADED
        else:
            overall_status = ServiceStatus.UNHEALTHY

        return HealthResponse(
            status=overall_status,
            services=services,
            all_healthy=all_healthy
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status=ServiceStatus.UNHEALTHY,
            services=[
                ServiceHealth(
                    name="gateway",
                    status=ServiceStatus.HEALTHY,
                    url="http://gateway:8000",
                    latency_ms=0.0,
                    details={"error": str(e)}
                )
            ],
            all_healthy=False
        )


@app.get("/info", response_model=InfoResponse)
async def service_info():
    """Get gateway service information."""
    return InfoResponse(
        available_services=["depth", "effects", "segmentation", "augmentor"],
        endpoints=[
            "/health",
            "/info",
            "/generate/image",
            "/generate/batch",
            "/augment/compose",
            "/augment/compose-batch",
            "/augment/validate",
            "/augment/lighting",
            "/augment/jobs/{job_id}",
            "/services/depth",
            "/services/effects",
            "/services/augmentor"
        ]
    )


@app.post("/generate/image", response_model=GenerateImageResponse)
async def generate_image(request: GenerateImageRequest):
    """Generate a single synthetic image.

    Orchestrates the pipeline:
    1. Depth estimation (if enabled)
    2. Object placement
    3. Effects application
    4. Annotation generation
    """
    logger.info(f"Generate image request: {request.background_path}")

    try:
        orchestrator = get_orchestrator()
        result = await orchestrator.generate_single_image(
            background_path=request.background_path,
            objects=request.objects,
            config=request.config,
            output_path=request.output_path
        )

        # Convert annotations
        annotations = [
            AnnotationBox(**ann.model_dump()) if hasattr(ann, 'model_dump') else ann
            for ann in result.get("annotations", [])
        ]

        return GenerateImageResponse(
            success=result.get("success", False),
            output_path=result.get("output_path", ""),
            depth_map_path=result.get("depth_map_path"),
            annotations=annotations,
            objects_placed=result.get("objects_placed", 0),
            effects_applied=result.get("effects_applied", []),
            processing_time_ms=result.get("processing_time_ms", 0),
            error=result.get("error")
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/batch", response_model=BatchGenerateResponse)
async def generate_batch(request: BatchGenerateRequest):
    """Generate a batch of synthetic images.

    Creates multiple synthetic images using backgrounds and objects
    from the specified directories.
    """
    logger.info(f"Batch generation request: {request.num_images} images")

    try:
        orchestrator = get_orchestrator()
        result = await orchestrator.generate_batch(
            backgrounds_dir=request.backgrounds_dir,
            objects_dir=request.objects_dir,
            output_dir=request.output_dir,
            num_images=request.num_images,
            config=request.config
        )

        return BatchGenerateResponse(
            success=result.get("success", False),
            job_id=result.get("job_id", ""),
            output_dir=result.get("output_dir", ""),
            total_requested=result.get("total_requested", 0),
            generated=result.get("generated", 0),
            failed=result.get("failed", 0),
            processing_time_ms=result.get("processing_time_ms", 0),
            error=result.get("error")
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/services/depth")
async def depth_service_info():
    """Get depth service information and health."""
    try:
        registry = get_service_registry()
        info = await registry.depth.get("/info")
        health = await registry.depth.health_check()

        return {
            "service": "depth",
            "url": registry.depth.base_url,
            "info": info,
            "health": health
        }
    except Exception as e:
        logger.error(f"Failed to get depth service info: {e}")
        return {
            "service": "depth",
            "error": str(e),
            "health": {"healthy": False}
        }


@app.get("/services/effects")
async def effects_service_info():
    """Get effects service information and health."""
    try:
        registry = get_service_registry()
        info = await registry.effects.get("/info")
        health = await registry.effects.health_check()

        return {
            "service": "effects",
            "url": registry.effects.base_url,
            "info": info,
            "health": health
        }
    except Exception as e:
        logger.error(f"Failed to get effects service info: {e}")
        return {
            "service": "effects",
            "error": str(e),
            "health": {"healthy": False}
        }


@app.get("/services/augmentor")
async def augmentor_service_info():
    """Get augmentor service information and health."""
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
        logger.error(f"Failed to get augmentor service info: {e}")
        return {
            "service": "augmentor",
            "error": str(e),
            "health": {"healthy": False}
        }


# =============================================================================
# Object Size Configuration Proxies
# =============================================================================

@app.get("/config/object-sizes", tags=["Configuration"])
async def get_object_sizes():
    """Get all configured object sizes."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{AUGMENTOR_SERVICE_URL}/config/object-sizes")
        response.raise_for_status()
        return response.json()


@app.get("/config/object-sizes/{class_name}", tags=["Configuration"])
async def get_object_size(class_name: str):
    """Get size for a specific object class."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{AUGMENTOR_SERVICE_URL}/config/object-sizes/{class_name}")
        response.raise_for_status()
        return response.json()


@app.put("/config/object-sizes/{class_name}", tags=["Configuration"])
async def update_object_size(class_name: str, size: float):
    """Update size for a specific object class."""
    async with httpx.AsyncClient() as client:
        response = await client.put(
            f"{AUGMENTOR_SERVICE_URL}/config/object-sizes/{class_name}",
            params={"size": size}
        )
        response.raise_for_status()
        return response.json()


@app.post("/config/object-sizes/batch", tags=["Configuration"])
async def update_multiple_object_sizes(sizes: Dict[str, float]):
    """Update multiple object sizes at once."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{AUGMENTOR_SERVICE_URL}/config/object-sizes/batch",
            json=sizes
        )
        response.raise_for_status()
        return response.json()


@app.delete("/config/object-sizes/{class_name}", tags=["Configuration"])
async def delete_object_size(class_name: str):
    """Delete size configuration for an object class."""
    async with httpx.AsyncClient() as client:
        response = await client.delete(f"{AUGMENTOR_SERVICE_URL}/config/object-sizes/{class_name}")
        response.raise_for_status()
        return response.json()


# =============================================================================
# Root & Info Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Synthetic Data Generation Gateway",
        "version": "1.0.0",
        "description": "API Gateway for orchestrating synthetic dataset generation",
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "generate_image": "/generate/image",
            "generate_batch": "/generate/batch",
            "augment_compose": "/augment/compose",
            "augment_batch": "/augment/compose-batch",
            "augment_validate": "/augment/validate",
            "augment_lighting": "/augment/lighting",
            "config_object_sizes": "/config/object-sizes",
            "depth_service": "/services/depth",
            "effects_service": "/services/effects",
            "augmentor_service": "/services/augmentor"
        },
        "documentation": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
