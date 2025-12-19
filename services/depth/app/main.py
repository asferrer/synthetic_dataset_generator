"""
Depth Service - FastAPI Application

REST API for depth estimation using Depth-Anything-3.
"""
import os
import time
import logging
from pathlib import Path
from typing import List
from contextlib import asynccontextmanager

import base64
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.depth_engine import get_depth_estimator, DepthEstimator
from app.models.schemas import (
    DepthEstimateRequest,
    DepthEstimateResponse,
    BatchEstimateRequest,
    BatchEstimateResponse,
    HealthResponse,
    InfoResponse,
    DepthZone
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - load model on startup."""
    logger.info("Starting Depth Service...")

    # Pre-load the model
    try:
        estimator = get_depth_estimator()
        logger.info(f"Model {estimator.model_name} loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Don't raise - allow service to start for health checks

    yield

    logger.info("Shutting down Depth Service...")


# Create FastAPI application
app = FastAPI(
    title="Depth Estimation Service",
    description="REST API for monocular depth estimation using Depth-Anything-3",
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


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns the service health status including model and GPU availability.
    """
    try:
        estimator = get_depth_estimator()
        gpu_info = estimator.get_gpu_info()

        status = "healthy" if estimator.is_loaded else "degraded"
        if not gpu_info["available"]:
            status = "degraded"

        return HealthResponse(
            status=status,
            model_loaded=estimator.is_loaded,
            gpu_available=gpu_info["available"],
            gpu_name=gpu_info.get("name"),
            gpu_memory_used=gpu_info.get("memory_used")
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            gpu_available=False
        )


@app.get("/info", response_model=InfoResponse)
async def service_info():
    """
    Get service information.

    Returns details about the loaded model and supported features.
    """
    try:
        estimator = get_depth_estimator()
        return InfoResponse(
            model=estimator.model_name,
            model_params=estimator.model_params,
            device=estimator.device
        )
    except Exception as e:
        logger.error(f"Failed to get service info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/estimate", response_model=DepthEstimateResponse)
async def estimate_depth(request: DepthEstimateRequest):
    """
    Estimate depth for a single image.

    The image should be accessible in the shared volume.
    Results are written to the shared volume.
    """
    logger.info(f"Depth estimation request for: {request.input_path}")

    try:
        estimator = get_depth_estimator()

        result = estimator.estimate_from_path(
            input_path=request.input_path,
            output_dir=request.output_dir,
            normalize=request.normalize,
            generate_preview=request.generate_preview,
            classify_zones=request.classify_zones,
            num_zones=request.num_zones
        )

        # Convert zones to schema format
        zones = None
        if result.get("zones"):
            zones = [
                DepthZone(
                    zone_id=z["zone_id"],
                    zone_name=z["zone_name"],
                    depth_range=z["depth_range"],
                    mask_path=z.get("mask_path")
                )
                for z in result["zones"]
            ]

        return DepthEstimateResponse(
            success=True,
            input_path=result["input_path"],
            depth_map_path=result["depth_map_path"],
            preview_path=result.get("preview_path"),
            shape=result["shape"],
            depth_range=result["depth_range"],
            zones=zones,
            processing_time_ms=result["processing_time_ms"]
        )

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))

    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Depth estimation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/estimate-batch", response_model=BatchEstimateResponse)
async def estimate_depth_batch(request: BatchEstimateRequest):
    """
    Estimate depth for a batch of images in a directory.

    All images matching the file pattern in the input directory will be processed.
    """
    logger.info(f"Batch depth estimation for: {request.input_dir}")
    start_time = time.time()

    try:
        estimator = get_depth_estimator()

        input_dir = Path(request.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Find all matching files
        patterns = request.file_pattern.split(",")
        image_files = []
        for pattern in patterns:
            image_files.extend(input_dir.glob(pattern.strip()))

        if not image_files:
            raise ValueError(f"No images found matching pattern: {request.file_pattern}")

        results = []
        processed = 0
        failed = 0

        for image_path in image_files:
            try:
                result = estimator.estimate_from_path(
                    input_path=str(image_path),
                    output_dir=request.output_dir,
                    normalize=request.normalize,
                    generate_preview=request.generate_preview,
                    classify_zones=False  # Skip zone classification for batch
                )

                results.append(DepthEstimateResponse(
                    success=True,
                    input_path=result["input_path"],
                    depth_map_path=result["depth_map_path"],
                    preview_path=result.get("preview_path"),
                    shape=result["shape"],
                    depth_range=result["depth_range"],
                    processing_time_ms=result["processing_time_ms"]
                ))
                processed += 1

            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results.append(DepthEstimateResponse(
                    success=False,
                    input_path=str(image_path),
                    depth_map_path="",
                    shape=(0, 0),
                    depth_range=(0.0, 0.0),
                    processing_time_ms=0,
                    error=str(e)
                ))
                failed += 1

        total_time = (time.time() - start_time) * 1000

        return BatchEstimateResponse(
            success=failed == 0,
            input_dir=str(input_dir),
            output_dir=request.output_dir,
            total_images=len(image_files),
            processed=processed,
            failed=failed,
            results=results,
            total_time_ms=total_time
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Batch estimation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/depth/estimate")
async def estimate_depth_upload(file: UploadFile = File(...)):
    """
    Estimate depth from uploaded image file.

    This endpoint accepts direct file uploads and returns the depth map
    as a base64-encoded numpy array.

    Returns:
        JSON with depth_map (base64), width, height
    """
    logger.info(f"Depth estimation request via upload: {file.filename}")
    start_time = time.time()

    try:
        # Read uploaded file
        contents = await file.read()

        # Decode image
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Could not decode image: {file.filename}")

        h, w = image.shape[:2]
        logger.info(f"Processing image {w}x{h}")

        # Get depth estimator and run inference
        estimator = get_depth_estimator()
        depth_map = estimator.estimate_depth(image, normalize=True)

        # Resize depth map to match original image dimensions if needed
        if depth_map.shape[0] != h or depth_map.shape[1] != w:
            logger.info(f"Resizing depth map from {depth_map.shape[1]}x{depth_map.shape[0]} to {w}x{h}")
            depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)

        # Encode depth map as base64
        depth_bytes = depth_map.astype(np.float32).tobytes()
        depth_b64 = base64.b64encode(depth_bytes).decode('utf-8')

        processing_time = (time.time() - start_time) * 1000
        logger.info(f"Depth estimation completed in {processing_time:.1f}ms")

        return {
            "success": True,
            "depth_map": depth_b64,
            "width": w,
            "height": h,
            "depth_range": [float(depth_map.min()), float(depth_map.max())],
            "processing_time_ms": processing_time
        }

    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Depth estimation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint - service information."""
    return {
        "service": "Depth Estimation Service",
        "version": "1.0.0",
        "model": "Depth-Anything-3",
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "estimate": "/estimate",
            "estimate_upload": "/depth/estimate",
            "batch": "/estimate-batch"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
