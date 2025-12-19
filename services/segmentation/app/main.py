"""
Segmentation Service
====================
FastAPI microservice for semantic scene analysis and SAM3 segmentation.

Uses SAM3 (Segment Anything Model 3) for Promptable Concept Segmentation (PCS).
SAM3 released 2025-11-19 - segments all instances matching a text concept.

Model: facebook/sam3 (848M parameters)
Docs: https://huggingface.co/facebook/sam3

Endpoints:
- POST /analyze - Analyze scene regions
- POST /check-compatibility - Check object-scene compatibility
- POST /suggest-placement - Get placement suggestions
- POST /segment-text - Text-driven segmentation (SAM3 PCS)
- GET /health - Health check

Debug Endpoints (Explainability):
- POST /debug/analyze - Analyze with full debug visualization and logs
- POST /debug/compatibility - Check compatibility with decision explanation
"""

import os
import time
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models.schemas import (
    AnalyzeSceneRequest,
    AnalyzeSceneResponse,
    CompatibilityCheckRequest,
    CompatibilityCheckResponse,
    SuggestPlacementRequest,
    SuggestPlacementResponse,
    SegmentTextRequest,
    SegmentTextResponse,
    HealthResponse,
    DebugAnalyzeRequest,
    DebugAnalyzeResponse,
    DebugCompatibilityRequest,
    DebugCompatibilityResponse,
)
from app.scene_analyzer import SemanticSceneAnalyzer

import base64


def encode_region_map_base64(region_map: np.ndarray) -> str:
    """Encode region map as base64 PNG string.

    The region_map is a uint8 array with values:
    0=unknown, 1=open_water, 2=seafloor, 3=surface, 4=vegetation, 5=rocky, 6=sandy, 7=murky
    """
    # Encode as PNG (lossless, efficient for indexed/mask images)
    success, encoded = cv2.imencode('.png', region_map)
    if success:
        return base64.b64encode(encoded.tobytes()).decode('utf-8')
    return None


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Global state
class ServiceState:
    """Global service state"""
    scene_analyzer = None
    debug_scene_analyzer = None  # Pre-initialized debug analyzer to avoid re-creation on each request
    sam3_model = None
    sam3_processor = None
    device: str = "cpu"
    sam3_available: bool = False
    gpu_available: bool = False


state = ServiceState()


def init_scene_analyzer():
    """Initialize scene analyzers with SAM3 model if available"""
    # Normal analyzer for regular requests
    state.scene_analyzer = SemanticSceneAnalyzer(
        use_sam3=state.sam3_available,
        device="cuda" if state.gpu_available else "cpu",
        sam3_model=state.sam3_model,
        sam3_processor=state.sam3_processor,
    )

    # Debug analyzer (pre-initialized to avoid creation on each debug request)
    debug_output_dir = "/shared/segmentation/debug"
    os.makedirs(debug_output_dir, exist_ok=True)
    state.debug_scene_analyzer = SemanticSceneAnalyzer(
        use_sam3=state.sam3_available,
        device="cuda" if state.gpu_available else "cpu",
        sam3_model=state.sam3_model,
        sam3_processor=state.sam3_processor,
        debug=True,
        debug_output_dir=debug_output_dir,
    )

    logger.info(f"Scene analyzers initialized (SAM3: {state.sam3_available})")


def init_sam3():
    """Initialize SAM3 model if available"""
    import torch

    # Check GPU availability
    state.gpu_available = torch.cuda.is_available()
    state.device = "cuda" if state.gpu_available else "cpu"

    if state.gpu_available:
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("No GPU available, using CPU")

    # Try to load SAM3 (Segment Anything Model 3)
    # Released 2025-11-19, uses Promptable Concept Segmentation (PCS)
    try:
        from transformers import Sam3Processor, Sam3Model

        # Official model: facebook/sam3 (848M parameters)
        model_id = os.environ.get("SAM3_MODEL_ID", "facebook/sam3")
        hf_token = os.environ.get("HF_TOKEN")

        logger.info(f"Loading SAM3 model: {model_id}")

        state.sam3_processor = Sam3Processor.from_pretrained(
            model_id,
            token=hf_token,
        )
        state.sam3_model = Sam3Model.from_pretrained(
            model_id,
            token=hf_token,
        ).to(state.device)
        state.sam3_model.eval()

        state.sam3_available = True
        logger.info("SAM3 model loaded successfully")

    except ImportError as e:
        logger.warning(f"SAM3 not available (transformers may need update): {e}")
        logger.info("Install transformers from main: pip install git+https://github.com/huggingface/transformers.git")
        logger.info("Falling back to heuristic-based analysis")
        state.sam3_available = False
    except Exception as e:
        logger.warning(f"SAM3 loading failed: {e}")
        logger.info("Falling back to heuristic-based analysis")
        state.sam3_available = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    logger.info("Starting Segmentation Service...")

    # Initialize models
    init_sam3()
    init_scene_analyzer()

    logger.info("Segmentation Service ready")
    yield

    # Cleanup
    logger.info("Shutting down Segmentation Service...")
    state.sam3_model = None
    state.scene_analyzer = None


# Create FastAPI app
app = FastAPI(
    title="Segmentation Service",
    description="Semantic scene analysis and SAM3 segmentation",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check service health"""
    return HealthResponse(
        status="healthy",
        sam3_available=state.sam3_available,
        gpu_available=state.gpu_available,
        model_loaded=state.scene_analyzer is not None,
        version="1.0.0",
    )


@app.post("/analyze", response_model=AnalyzeSceneResponse, tags=["Analysis"])
async def analyze_scene(request: AnalyzeSceneRequest):
    """Analyze scene regions in an image"""
    start_time = time.time()

    try:
        # Load image
        image_path = Path(request.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {request.image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {request.image_path}")

        # Analyze scene
        if hasattr(state.scene_analyzer, 'analyze_scene'):
            result = state.scene_analyzer.analyze_scene(image)

            # Handle both object and dict returns
            if hasattr(result, 'dominant_region'):
                # Object with attributes - encode region_map for transfer
                region_map_b64 = None
                if hasattr(result, 'region_map') and result.region_map is not None:
                    region_map_b64 = encode_region_map_base64(result.region_map)

                return AnalyzeSceneResponse(
                    success=True,
                    dominant_region=result.dominant_region.value if hasattr(result.dominant_region, 'value') else str(result.dominant_region),
                    region_scores=result.region_scores,
                    depth_zones={k: list(v) for k, v in result.depth_zones.items()},
                    scene_brightness=result.scene_brightness,
                    water_clarity=result.water_clarity,
                    color_temperature=result.color_temperature,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    region_map_base64=region_map_b64,
                )
            else:
                # Dict return
                return AnalyzeSceneResponse(
                    success=True,
                    dominant_region=result.get("dominant_region", "unknown"),
                    region_scores=result.get("region_scores", {}),
                    depth_zones=result.get("depth_zones", {}),
                    scene_brightness=result.get("scene_brightness", 0.5),
                    water_clarity=result.get("water_clarity", "moderate"),
                    color_temperature=result.get("color_temperature", "neutral"),
                    processing_time_ms=(time.time() - start_time) * 1000,
                )
        else:
            raise RuntimeError("Scene analyzer not initialized")

    except FileNotFoundError as e:
        return AnalyzeSceneResponse(
            success=False,
            dominant_region="unknown",
            region_scores={},
            depth_zones={},
            scene_brightness=0.0,
            water_clarity="unknown",
            color_temperature="unknown",
            processing_time_ms=(time.time() - start_time) * 1000,
            error=str(e),
        )
    except Exception as e:
        logger.error(f"Scene analysis failed: {e}")
        return AnalyzeSceneResponse(
            success=False,
            dominant_region="unknown",
            region_scores={},
            depth_zones={},
            scene_brightness=0.0,
            water_clarity="unknown",
            color_temperature="unknown",
            processing_time_ms=(time.time() - start_time) * 1000,
            error=str(e),
        )


@app.post("/check-compatibility", response_model=CompatibilityCheckResponse, tags=["Analysis"])
async def check_compatibility(request: CompatibilityCheckRequest):
    """Check if object placement is compatible with scene"""
    try:
        # Load image
        image_path = Path(request.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {request.image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {request.image_path}")

        h, w = image.shape[:2]

        # Analyze scene first
        if hasattr(state.scene_analyzer, 'analyze_scene'):
            scene_result = state.scene_analyzer.analyze_scene(image)
        else:
            raise RuntimeError("Scene analyzer not initialized")

        # Check compatibility
        position = (request.position_x, request.position_y)

        if hasattr(state.scene_analyzer, 'check_object_scene_compatibility'):
            score, reason = state.scene_analyzer.check_object_scene_compatibility(
                request.object_class,
                position,
                scene_result,
                (h, w),
            )
        else:
            score, reason = 0.6, "Default compatibility"

        # Get best region suggestion
        suggested_region = None
        if hasattr(state.scene_analyzer, 'get_best_placement_region'):
            best_region = state.scene_analyzer.get_best_placement_region(
                request.object_class,
                scene_result,
            )
            if best_region:
                suggested_region = best_region.value if hasattr(best_region, 'value') else str(best_region)

        return CompatibilityCheckResponse(
            success=True,
            is_compatible=score >= 0.4,
            score=score,
            reason=reason,
            suggested_region=suggested_region,
        )

    except Exception as e:
        logger.error(f"Compatibility check failed: {e}")
        return CompatibilityCheckResponse(
            success=False,
            is_compatible=False,
            score=0.0,
            reason="Error during check",
            error=str(e),
        )


@app.post("/suggest-placement", response_model=SuggestPlacementResponse, tags=["Analysis"])
async def suggest_placement(request: SuggestPlacementRequest):
    """Suggest best placement position for an object"""
    try:
        # Load image
        image_path = Path(request.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {request.image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {request.image_path}")

        h, w = image.shape[:2]

        # Analyze scene
        if hasattr(state.scene_analyzer, 'analyze_scene'):
            scene_result = state.scene_analyzer.analyze_scene(image)
        else:
            raise RuntimeError("Scene analyzer not initialized")

        # Convert existing positions
        existing = [(p[0], p[1]) for p in request.existing_positions]

        # Get suggestion
        if hasattr(state.scene_analyzer, 'suggest_placement_position'):
            position = state.scene_analyzer.suggest_placement_position(
                request.object_class,
                (request.object_width, request.object_height),
                scene_result,
                (h, w),
                existing,
                request.min_distance,
            )
        else:
            # Simple fallback
            import random
            margin = 50
            position = (
                random.randint(margin, w - request.object_width - margin),
                random.randint(margin, h - request.object_height - margin),
            )

        if position is None:
            return SuggestPlacementResponse(
                success=False,
                error="No valid placement position found",
            )

        # Get best region
        best_region = None
        if hasattr(state.scene_analyzer, 'get_best_placement_region'):
            region = state.scene_analyzer.get_best_placement_region(
                request.object_class,
                scene_result,
            )
            if region:
                best_region = region.value if hasattr(region, 'value') else str(region)

        # Get compatibility score
        score = 0.7
        if hasattr(state.scene_analyzer, 'check_object_scene_compatibility'):
            score, _ = state.scene_analyzer.check_object_scene_compatibility(
                request.object_class,
                position,
                scene_result,
                (h, w),
            )

        return SuggestPlacementResponse(
            success=True,
            position_x=position[0],
            position_y=position[1],
            best_region=best_region,
            compatibility_score=score,
        )

    except Exception as e:
        logger.error(f"Placement suggestion failed: {e}")
        return SuggestPlacementResponse(
            success=False,
            error=str(e),
        )


@app.post("/segment-text", response_model=SegmentTextResponse, tags=["Segmentation"])
async def segment_text(request: SegmentTextRequest):
    """
    Text-driven segmentation using SAM3 (Segment Anything Model 3).

    SAM3 uses Promptable Concept Segmentation (PCS) to segment all instances
    in an image that match a given text concept.
    """
    start_time = time.time()

    if not state.sam3_available:
        return SegmentTextResponse(
            success=False,
            processing_time_ms=(time.time() - start_time) * 1000,
            error="SAM3 not available. Install: pip install transformers>=4.45.0",
        )

    try:
        import torch
        from PIL import Image as PILImage

        # Load image
        image_path = Path(request.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {request.image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {request.image_path}")

        h, w = image.shape[:2]

        # Convert BGR to RGB PIL Image (required by SAM3)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(image_rgb)

        # Process with SAM3 text-prompted segmentation (Promptable Concept Segmentation)
        # Following official HuggingFace documentation: https://huggingface.co/facebook/sam3
        inputs = state.sam3_processor(
            images=pil_image,
            text=request.text_prompt,
            return_tensors="pt"
        ).to(state.device)

        with torch.no_grad():
            outputs = state.sam3_model(**inputs)

        # Post-process to get instance segmentation results
        # Use original_sizes from processor for accurate mask resizing
        target_sizes = inputs.get("original_sizes")
        if target_sizes is not None:
            target_sizes = target_sizes.tolist()
        else:
            target_sizes = [(h, w)]

        results = state.sam3_processor.post_process_instance_segmentation(
            outputs,
            threshold=request.threshold,
            mask_threshold=request.threshold,
            target_sizes=target_sizes
        )[0]

        # Combine all masks
        combined_mask = np.zeros((h, w), dtype=np.float32)
        max_confidence = 0.0

        if 'masks' in results and len(results['masks']) > 0:
            for mask, score in zip(results['masks'], results['scores']):
                mask_np = mask.cpu().numpy().astype(np.float32)
                score_val = score.cpu().item()

                # Resize if needed
                if mask_np.shape != (h, w):
                    mask_np = cv2.resize(mask_np, (w, h))

                combined_mask = np.maximum(combined_mask, mask_np)
                max_confidence = max(max_confidence, score_val)

        # Create binary mask
        mask_binary = (combined_mask > 0.5).astype(np.uint8) * 255

        # Save mask
        output_dir = Path("/shared/segmentation")
        output_dir.mkdir(parents=True, exist_ok=True)

        mask_filename = f"mask_{Path(request.image_path).stem}_{int(time.time())}.png"
        mask_path = output_dir / mask_filename
        cv2.imwrite(str(mask_path), mask_binary)

        # Calculate coverage
        coverage = float((mask_binary > 0).sum()) / (h * w)

        return SegmentTextResponse(
            success=True,
            mask_path=str(mask_path),
            mask_coverage=coverage,
            confidence=max_confidence,
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    except Exception as e:
        logger.error(f"Text segmentation failed: {e}")
        return SegmentTextResponse(
            success=False,
            processing_time_ms=(time.time() - start_time) * 1000,
            error=str(e),
        )


# =========================================================================
# DEBUG AND EXPLAINABILITY ENDPOINTS
# =========================================================================

@app.post("/debug/analyze", response_model=DebugAnalyzeResponse, tags=["Debug"])
async def debug_analyze_scene(request: DebugAnalyzeRequest):
    """
    Analyze scene with full debug information for explainability.

    Returns detailed information about how SAM3/heuristics made decisions,
    including region masks, confidence scores, and decision logs.
    """
    try:
        # Load image
        image_path = Path(request.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {request.image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {request.image_path}")

        # Use pre-initialized debug analyzer to avoid re-creation overhead
        debug_output_dir = "/shared/segmentation/debug"

        # Check if debug_scene_analyzer supports debug mode
        if state.debug_scene_analyzer is not None and hasattr(state.debug_scene_analyzer, 'analyze_scene_with_debug'):
            image_id = request.image_id or f"debug_{int(time.time())}"
            analysis, debug_info = state.debug_scene_analyzer.analyze_scene_with_debug(
                image,
                save_visualization=request.save_visualization,
                image_id=image_id,
            )

            # Get masks directory
            masks_dir = os.path.join(debug_output_dir, f"{image_id}_masks")

            # Encode region_map for transfer
            region_map_b64 = None
            if hasattr(analysis, 'region_map') and analysis.region_map is not None:
                region_map_b64 = encode_region_map_base64(analysis.region_map)

            return DebugAnalyzeResponse(
                success=True,
                dominant_region=analysis.dominant_region.value if hasattr(analysis.dominant_region, 'value') else str(analysis.dominant_region),
                region_scores=analysis.region_scores,
                scene_brightness=analysis.scene_brightness,
                water_clarity=analysis.water_clarity,
                color_temperature=analysis.color_temperature,
                analysis_method=debug_info.analysis_method,
                processing_time_ms=debug_info.processing_time_ms,
                sam3_prompts_used=debug_info.sam3_prompts_used,
                region_confidences=debug_info.region_confidences,
                decision_log=debug_info.decision_log,
                region_map_base64=region_map_b64,
                visualization_path=debug_info.visualization_path,
                masks_directory=masks_dir if os.path.exists(masks_dir) else None,
            )
        else:
            # Fallback: basic analysis without debug
            result = state.scene_analyzer.analyze_scene(image)
            return DebugAnalyzeResponse(
                success=True,
                dominant_region=result.get("dominant_region", "unknown") if isinstance(result, dict) else (result.dominant_region.value if hasattr(result.dominant_region, 'value') else str(result.dominant_region)),
                region_scores=result.get("region_scores", {}) if isinstance(result, dict) else result.region_scores,
                scene_brightness=result.get("scene_brightness", 0.5) if isinstance(result, dict) else result.scene_brightness,
                water_clarity=result.get("water_clarity", "unknown") if isinstance(result, dict) else result.water_clarity,
                color_temperature=result.get("color_temperature", "neutral") if isinstance(result, dict) else result.color_temperature,
                analysis_method="heuristic",
                processing_time_ms=0.0,
                sam3_prompts_used=[],
                region_confidences={},
                decision_log=["Debug mode not fully supported with current analyzer"],
            )

    except Exception as e:
        logger.error(f"Debug analysis failed: {e}")
        return DebugAnalyzeResponse(
            success=False,
            dominant_region="unknown",
            region_scores={},
            scene_brightness=0.0,
            water_clarity="unknown",
            color_temperature="unknown",
            analysis_method="error",
            processing_time_ms=0.0,
            sam3_prompts_used=[],
            region_confidences={},
            decision_log=[f"Error: {str(e)}"],
            error=str(e),
        )


@app.post("/debug/compatibility", response_model=DebugCompatibilityResponse, tags=["Debug"])
async def debug_check_compatibility(request: DebugCompatibilityRequest):
    """
    Check object-scene compatibility with detailed debug information.

    Returns the compatibility decision along with alternative positions
    and explanations for the decision.
    """
    try:
        # Load image
        image_path = Path(request.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {request.image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {request.image_path}")

        h, w = image.shape[:2]

        # Use pre-initialized debug analyzer to avoid re-creation overhead
        if state.debug_scene_analyzer is not None and hasattr(state.debug_scene_analyzer, 'check_object_scene_compatibility_with_debug'):
            # First analyze the scene
            analysis = state.debug_scene_analyzer.analyze_scene(image)

            # Then check compatibility with debug
            position = (request.position_x, request.position_y)
            score, reason, decision = state.debug_scene_analyzer.check_object_scene_compatibility_with_debug(
                request.object_class,
                position,
                analysis,
                (h, w),
            )

            # Convert alternatives to list format
            alternatives = [
                [float(x), float(y), float(s)]
                for x, y, s in decision.alternative_positions[:5]
            ] if decision.alternative_positions else []

            return DebugCompatibilityResponse(
                success=True,
                is_compatible=score >= 0.4,
                score=float(score),
                reason=reason,
                decision=decision.decision,
                region_at_position=decision.region_at_position,
                alternatives=alternatives,
            )
        else:
            # Fallback
            analysis = state.scene_analyzer.analyze_scene(image)
            position = (request.position_x, request.position_y)

            if hasattr(state.scene_analyzer, 'check_object_scene_compatibility'):
                score, reason = state.scene_analyzer.check_object_scene_compatibility(
                    request.object_class, position, analysis, (h, w)
                )
            else:
                score, reason = 0.6, "Default compatibility"

            return DebugCompatibilityResponse(
                success=True,
                is_compatible=score >= 0.4,
                score=float(score),
                reason=reason,
                decision="accepted" if score >= 0.4 else "rejected",
                region_at_position="unknown",
                alternatives=[],
            )

    except Exception as e:
        logger.error(f"Debug compatibility check failed: {e}")
        return DebugCompatibilityResponse(
            success=False,
            is_compatible=False,
            score=0.0,
            reason="Error during check",
            decision="error",
            region_at_position="unknown",
            alternatives=[],
            error=str(e),
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info",
    )
