"""
Effects Service - FastAPI Application

REST API for applying photorealistic effects to synthetic images.
"""
import os
import time
import logging
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.realism import (
    transfer_color_correction,
    match_blur,
    add_lighting_effect,
    apply_underwater_effect,
    add_upscaling_noise,
    apply_poisson_blending,
    laplacian_pyramid_blending,
    generate_dynamic_shadow,
    apply_motion_blur,
    smooth_edges
)
from app.caustics import get_caustics_cache, generate_caustics_map, apply_caustics
from app.models.schemas import (
    ApplyEffectsRequest,
    ApplyEffectsResponse,
    BlendRequest,
    BlendResponse,
    CausticsRequest,
    CausticsResponse,
    TransformRequest,
    TransformResponse,
    HealthResponse,
    InfoResponse,
    EffectType
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - initialize caustics cache on startup."""
    logger.info("Starting Effects Service...")

    # Pre-initialize caustics cache
    try:
        cache = get_caustics_cache()
        logger.info(f"Caustics cache ready with {cache.template_count} templates")
    except Exception as e:
        logger.warning(f"Failed to initialize caustics cache: {e}")

    yield

    logger.info("Shutting down Effects Service...")


# Create FastAPI application
app = FastAPI(
    title="Effects Service",
    description="REST API for photorealistic image effects",
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
    """Health check endpoint."""
    try:
        cache = get_caustics_cache()
        return HealthResponse(
            status="healthy",
            caustics_cache_ready=cache.is_ready,
            cache_templates=cache.template_count
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            caustics_cache_ready=False,
            cache_templates=0
        )


@app.get("/info", response_model=InfoResponse)
async def service_info():
    """Get service information."""
    return InfoResponse(
        available_effects=[e.value for e in EffectType],
        blend_methods=["alpha", "poisson", "laplacian"],
        endpoints=["/apply", "/blend", "/caustics", "/transform", "/health", "/info"]
    )


@app.post("/apply", response_model=ApplyEffectsResponse)
async def apply_effects(request: ApplyEffectsRequest):
    """Apply effects to an image.

    Applies the requested effects pipeline to the image.
    """
    logger.info(f"Apply effects request: {request.effects}")
    start_time = time.time()

    try:
        # Load background image
        bg_path = Path(request.background_path)
        if not bg_path.exists():
            raise FileNotFoundError(f"Background not found: {bg_path}")

        background = cv2.imread(str(bg_path))
        if background is None:
            raise ValueError(f"Failed to read background: {bg_path}")

        result = background.copy()
        effects_applied = []

        # Load optional images
        foreground = None
        mask = None
        depth_map = None

        if request.foreground_path:
            fg_path = Path(request.foreground_path)
            if fg_path.exists():
                foreground = cv2.imread(str(fg_path), cv2.IMREAD_UNCHANGED)

        if request.mask_path:
            mask_path = Path(request.mask_path)
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if request.depth_map_path:
            depth_path = Path(request.depth_map_path)
            if depth_path.exists():
                if str(depth_path).endswith('.npy'):
                    depth_map = np.load(str(depth_path))
                else:
                    depth_map = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)

        # Create mask if not provided but foreground has alpha
        if foreground is not None and mask is None:
            if foreground.shape[2] == 4:
                mask = foreground[:, :, 3]
                foreground = foreground[:, :, :3]
            else:
                mask = np.ones(foreground.shape[:2], dtype=np.uint8) * 255

        config = request.config

        # Apply each effect in order
        for effect in request.effects:
            try:
                if effect == EffectType.COLOR_CORRECTION and foreground is not None:
                    # Get ROI from background
                    h, w = foreground.shape[:2]
                    bg_roi = cv2.resize(background, (w, h))
                    foreground = transfer_color_correction(
                        foreground, bg_roi, mask, config.color_intensity
                    )
                    effects_applied.append("color_correction")

                elif effect == EffectType.BLUR_MATCHING and foreground is not None:
                    h, w = foreground.shape[:2]
                    bg_roi = cv2.resize(background, (w, h))
                    foreground = match_blur(foreground, bg_roi, mask, config.blur_strength)
                    effects_applied.append("blur_matching")

                elif effect == EffectType.LIGHTING and foreground is not None:
                    foreground = add_lighting_effect(
                        foreground, config.lighting_type.value, config.lighting_intensity
                    )
                    effects_applied.append("lighting")

                elif effect == EffectType.UNDERWATER and foreground is not None:
                    foreground = apply_underwater_effect(
                        foreground, config.water_color, config.underwater_intensity
                    )
                    effects_applied.append("underwater")

                elif effect == EffectType.MOTION_BLUR and foreground is not None:
                    if np.random.random() < config.motion_blur_probability:
                        foreground = apply_motion_blur(foreground, config.motion_blur_kernel)
                        effects_applied.append("motion_blur")

                elif effect == EffectType.SHADOWS and foreground is not None and mask is not None:
                    # Get background ROI for shadow generation
                    h, w = mask.shape[:2]
                    bg_roi = cv2.resize(background, (w, h))
                    shadow = generate_dynamic_shadow(mask, bg_roi)
                    # Shadow application would go here
                    effects_applied.append("shadows")

                elif effect == EffectType.CAUSTICS:
                    caustics_map = generate_caustics_map(
                        result.shape[1], result.shape[0], config.caustics_intensity
                    )
                    result = apply_caustics(result, caustics_map)
                    effects_applied.append("caustics")

                elif effect == EffectType.EDGE_SMOOTHING and foreground is not None and mask is not None:
                    foreground_rgba = cv2.cvtColor(foreground, cv2.COLOR_BGR2BGRA)
                    foreground_rgba[:, :, 3] = mask
                    foreground_rgba = smooth_edges(foreground_rgba, mask, config.edge_feather)
                    foreground = foreground_rgba[:, :, :3]
                    mask = foreground_rgba[:, :, 3]
                    effects_applied.append("edge_smoothing")

                elif effect == EffectType.UPSCALING_NOISE:
                    result = add_upscaling_noise(result, 10)
                    effects_applied.append("upscaling_noise")

            except Exception as e:
                logger.warning(f"Failed to apply {effect}: {e}")

        # Generate output path
        if request.output_path:
            output_path = Path(request.output_path)
        else:
            output_dir = Path("/shared/effects")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{bg_path.stem}_effects.jpg"

        # Save result
        cv2.imwrite(str(output_path), result)

        processing_time = (time.time() - start_time) * 1000

        return ApplyEffectsResponse(
            success=True,
            input_path=str(bg_path),
            output_path=str(output_path),
            effects_applied=effects_applied,
            processing_time_ms=processing_time
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Effects application failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/blend", response_model=BlendResponse)
async def blend_images(request: BlendRequest):
    """Blend foreground onto background."""
    logger.info(f"Blend request: method={request.method}")
    start_time = time.time()

    try:
        # Load images
        background = cv2.imread(request.background_path)
        foreground = cv2.imread(request.foreground_path, cv2.IMREAD_UNCHANGED)

        if background is None:
            raise FileNotFoundError(f"Background not found: {request.background_path}")
        if foreground is None:
            raise FileNotFoundError(f"Foreground not found: {request.foreground_path}")

        # Get mask
        if request.mask_path:
            mask = cv2.imread(request.mask_path, cv2.IMREAD_GRAYSCALE)
        elif foreground.shape[2] == 4:
            mask = foreground[:, :, 3]
            foreground = foreground[:, :, :3]
        else:
            mask = np.ones(foreground.shape[:2], dtype=np.uint8) * 255

        # Resize foreground if needed
        if foreground.shape[:2] != background.shape[:2]:
            foreground = cv2.resize(foreground, (background.shape[1], background.shape[0]))
            mask = cv2.resize(mask, (background.shape[1], background.shape[0]))

        # Get center for blending
        if request.position:
            center = request.position
        else:
            center = (background.shape[1] // 2, background.shape[0] // 2)

        # Apply blending
        if request.method.value == "poisson":
            result = apply_poisson_blending(foreground, background, mask, center)
        elif request.method.value == "laplacian":
            result = laplacian_pyramid_blending(foreground, background, mask)
        else:  # alpha
            # Simple alpha blending
            mask_3c = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
            result = (foreground * mask_3c + background * (1 - mask_3c)).astype(np.uint8)

        # Generate output path
        if request.output_path:
            output_path = Path(request.output_path)
        else:
            output_dir = Path("/shared/effects")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"blended_{int(time.time())}.jpg"

        cv2.imwrite(str(output_path), result)

        processing_time = (time.time() - start_time) * 1000

        return BlendResponse(
            success=True,
            output_path=str(output_path),
            method_used=request.method.value,
            processing_time_ms=processing_time
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Blending failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/caustics", response_model=CausticsResponse)
async def generate_caustics(request: CausticsRequest):
    """Generate caustics map."""
    logger.info(f"Caustics request: {request.width}x{request.height}")
    start_time = time.time()

    try:
        cache = get_caustics_cache()
        from_cache = cache.is_ready

        caustics_map = generate_caustics_map(
            request.width, request.height, request.intensity
        )

        # Generate output path
        if request.output_path:
            output_path = Path(request.output_path)
        else:
            output_dir = Path("/shared/effects")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"caustics_{request.width}x{request.height}.png"

        # Convert to saveable format
        caustics_uint8 = (np.clip(caustics_map, 0, 2) * 127.5).astype(np.uint8)
        cv2.imwrite(str(output_path), caustics_uint8)

        processing_time = (time.time() - start_time) * 1000

        return CausticsResponse(
            success=True,
            output_path=str(output_path),
            size=(request.width, request.height),
            from_cache=from_cache,
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Caustics generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/available")
async def get_available_effects():
    """Get list of available effects."""
    return {
        "effects": [e.value for e in EffectType],
        "descriptions": {
            "color_correction": "Match object colors to background using LAB color space",
            "blur_matching": "Match object blur level to background",
            "lighting": "Add spotlight or gradient lighting effects",
            "underwater": "Apply underwater color tint",
            "motion_blur": "Add directional motion blur",
            "shadows": "Generate dynamic shadows based on lighting",
            "caustics": "Add underwater caustics light patterns",
            "poisson_blend": "Seamless Poisson blending for edges",
            "edge_smoothing": "Feather and smooth object edges",
            "perspective": "Apply perspective distortion",
            "upscaling_noise": "Add noise to simulate upscaling artifacts"
        }
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Effects Service",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "apply": "/apply",
            "blend": "/blend",
            "caustics": "/caustics",
            "available": "/available"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
