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

Object Extraction Endpoints:
- POST /extract/analyze-dataset - Analyze dataset annotation types
- POST /extract/objects - Extract objects from dataset (async)
- GET /extract/jobs/{job_id} - Get extraction job status
- POST /extract/single-object - Extract single object (preview)

SAM3 Tool Endpoints:
- POST /sam3/segment-image - Segment single image with box/point prompt
- POST /sam3/convert-dataset - Convert bbox to segmentation (async)
- GET /sam3/jobs/{job_id} - Get conversion job status
"""

import os
import time
import json
import logging
import asyncio
import uuid
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, BackgroundTasks
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
from app.object_extractor import ObjectExtractor
from app.models.extraction_schemas import (
    AnalyzeDatasetRequest,
    AnalyzeDatasetResponse,
    CategoryInfo,
    ExtractObjectsRequest,
    ExtractObjectsResponse,
    ExtractionJobStatus,
    ExtractSingleObjectRequest,
    ExtractSingleObjectResponse,
    SAM3SegmentImageRequest,
    SAM3SegmentImageResponse,
    SAM3ConvertDatasetRequest,
    SAM3ConvertDatasetResponse,
    SAM3ConversionJobStatus,
    JobStatus,
    AnnotationType,
    ExtractionMethod,
)

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
    object_extractor: Optional[ObjectExtractor] = None


state = ServiceState()

# Job tracking for async operations
extraction_jobs: Dict[str, Dict[str, Any]] = {}
sam3_conversion_jobs: Dict[str, Dict[str, Any]] = {}

# Thread pool for CPU-bound operations
thread_pool = ThreadPoolExecutor(max_workers=2)


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


def init_object_extractor():
    """Initialize object extractor with shared SAM3 model"""
    state.object_extractor = ObjectExtractor(
        sam3_model=state.sam3_model,
        sam3_processor=state.sam3_processor,
        device=state.device
    )
    logger.info(f"ObjectExtractor initialized (SAM3: {state.sam3_available})")


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
    init_object_extractor()

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


# =========================================================================
# OBJECT EXTRACTION ENDPOINTS
# =========================================================================

@app.post("/extract/analyze-dataset", response_model=AnalyzeDatasetResponse, tags=["Object Extraction"])
async def analyze_dataset_for_extraction(request: AnalyzeDatasetRequest):
    """
    Analyze a COCO dataset to determine annotation types.

    Returns counts of annotations with segmentation vs bbox-only,
    and a recommendation for extraction method.
    """
    try:
        # Get COCO data
        coco_data = None
        if request.coco_data:
            coco_data = request.coco_data
        elif request.coco_json_path:
            json_path = Path(request.coco_json_path)
            if not json_path.exists():
                raise FileNotFoundError(f"COCO JSON not found: {request.coco_json_path}")
            with open(json_path, 'r') as f:
                coco_data = json.load(f)
        else:
            raise ValueError("Either coco_data or coco_json_path must be provided")

        # Analyze dataset
        analysis = state.object_extractor.analyze_dataset(coco_data)

        # Convert categories to CategoryInfo
        categories = [
            CategoryInfo(
                id=cat["id"],
                name=cat["name"],
                count=cat["count"],
                with_segmentation=cat["with_segmentation"],
                bbox_only=cat["bbox_only"]
            )
            for cat in analysis["categories"]
        ]

        return AnalyzeDatasetResponse(
            success=True,
            total_images=analysis["total_images"],
            total_annotations=analysis["total_annotations"],
            annotations_with_segmentation=analysis["annotations_with_segmentation"],
            annotations_bbox_only=analysis["annotations_bbox_only"],
            categories=categories,
            recommendation=analysis["recommendation"],
            sample_annotation=analysis["sample_annotation"]
        )

    except Exception as e:
        logger.error(f"Dataset analysis failed: {e}")
        return AnalyzeDatasetResponse(
            success=False,
            error=str(e)
        )


@app.post("/extract/objects", response_model=ExtractObjectsResponse, tags=["Object Extraction"])
async def extract_objects(request: ExtractObjectsRequest):
    """
    Extract objects from a COCO dataset as transparent PNG images.

    Runs asynchronously. Use GET /extract/jobs/{job_id} to track progress.
    """
    try:
        # Get COCO data
        coco_data = None
        source_path = ""
        if request.coco_data:
            coco_data = request.coco_data
            source_path = "uploaded_data"
            logger.info(f"Received COCO data with {len(coco_data.get('annotations', []))} annotations")
        elif request.coco_json_path:
            json_path = Path(request.coco_json_path)
            if not json_path.exists():
                raise FileNotFoundError(f"COCO JSON not found: {request.coco_json_path}")
            with open(json_path, 'r') as f:
                coco_data = json.load(f)
            source_path = request.coco_json_path
        else:
            raise ValueError("Either coco_data or coco_json_path must be provided")

        # Validate images directory
        images_dir = Path(request.images_dir)
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {request.images_dir}")

        # Create job
        job_id = str(uuid.uuid4())

        # Count total objects to extract
        categories = {c["id"]: c["name"] for c in coco_data.get("categories", [])}
        if request.categories_to_extract:
            valid_cat_ids = {cid for cid, name in categories.items() if name in request.categories_to_extract}
        else:
            valid_cat_ids = set(categories.keys())

        total_objects = sum(
            1 for ann in coco_data.get("annotations", [])
            if ann.get("category_id") in valid_cat_ids
            and ann.get("bbox", [0, 0, 0, 0])[2] * ann.get("bbox", [0, 0, 0, 0])[3] >= request.min_object_area
        )

        extraction_jobs[job_id] = {
            "job_id": job_id,
            "status": JobStatus.QUEUED,
            "total_objects": total_objects,
            "extracted_objects": 0,
            "failed_objects": 0,
            "current_category": "",
            "categories_progress": {},
            "output_dir": request.output_dir,
            "errors": [],
            "extracted_files": [],
            "processing_time_ms": 0.0,
            "started_at": None,
            "completed_at": None
        }

        # Define extraction task
        async def run_extraction():
            extraction_jobs[job_id]["status"] = JobStatus.PROCESSING
            extraction_jobs[job_id]["started_at"] = datetime.now().isoformat()

            def progress_callback(progress):
                # This callback runs from a thread pool, but dict updates are atomic in CPython
                extraction_jobs[job_id]["extracted_objects"] = progress["extracted"]
                extraction_jobs[job_id]["failed_objects"] = progress["failed"]
                extraction_jobs[job_id]["current_category"] = progress.get("current_category", "")
                extraction_jobs[job_id]["categories_progress"] = progress.get("by_category", {})

            try:
                result = await state.object_extractor.extract_from_dataset(
                    coco_data=coco_data,
                    images_dir=str(images_dir),
                    output_dir=request.output_dir,
                    categories_to_extract=request.categories_to_extract or None,
                    use_sam3_for_bbox=request.use_sam3_for_bbox,
                    padding=request.padding,
                    min_object_area=request.min_object_area,
                    save_individual_coco=request.save_individual_coco,
                    progress_callback=progress_callback
                )

                extraction_jobs[job_id]["status"] = JobStatus.COMPLETED
                extraction_jobs[job_id]["extracted_objects"] = result["extracted"]
                extraction_jobs[job_id]["failed_objects"] = result["failed"]
                extraction_jobs[job_id]["categories_progress"] = result["by_category"]
                extraction_jobs[job_id]["errors"] = result.get("errors", [])[:100]
                extraction_jobs[job_id]["extracted_files"] = result.get("extracted_files", [])[:1000]
                extraction_jobs[job_id]["processing_time_ms"] = result.get("processing_time_seconds", 0) * 1000

            except Exception as e:
                # Use logger.exception to get full traceback for debugging
                logger.exception(f"Extraction job {job_id} failed: {e}")
                extraction_jobs[job_id]["status"] = JobStatus.FAILED
                extraction_jobs[job_id]["errors"].append(str(e))

            finally:
                # Always set completed_at, even if job failed
                extraction_jobs[job_id]["completed_at"] = datetime.now().isoformat()

        # Run in background using asyncio.create_task for proper async execution
        asyncio.create_task(run_extraction())
        logger.info(f"Started extraction job {job_id} with {total_objects} objects")

        return ExtractObjectsResponse(
            success=True,
            job_id=job_id,
            status=JobStatus.QUEUED,
            message=f"Extraction job queued. {total_objects} objects to extract."
        )

    except Exception as e:
        logger.error(f"Failed to start extraction: {e}")
        return ExtractObjectsResponse(
            success=False,
            error=str(e)
        )


@app.get("/extract/jobs", tags=["Object Extraction"])
async def list_extraction_jobs():
    """List all extraction jobs."""
    jobs = []
    for job_id, job in extraction_jobs.items():
        # Convert JobStatus enum to string for JSON serialization
        status = job.get("status", "unknown")
        status_str = status.value if hasattr(status, 'value') else str(status)
        jobs.append({
            "job_id": job_id,
            "job_type": "extraction",
            "status": status_str,
            "total_objects": job.get("total_objects", 0),
            "extracted_objects": job.get("extracted_objects", 0),
            "failed_objects": job.get("failed_objects", 0),
            "current_category": job.get("current_category", ""),
            "output_dir": job.get("output_dir", ""),
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
            "processing_time_ms": job.get("processing_time_ms", 0),
        })
    return {"jobs": jobs, "total": len(jobs)}


@app.get("/extract/jobs/{job_id}", response_model=ExtractionJobStatus, tags=["Object Extraction"])
async def get_extraction_job_status(job_id: str):
    """Get the status of an extraction job."""
    if job_id not in extraction_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = extraction_jobs[job_id]
    return ExtractionJobStatus(**job)


@app.post("/extract/single-object", response_model=ExtractSingleObjectResponse, tags=["Object Extraction"])
async def extract_single_object(request: ExtractSingleObjectRequest):
    """
    Extract a single object for preview.

    Returns the extracted object as base64-encoded PNG with transparency.
    """
    start_time = time.time()

    try:
        # Load image
        image = None
        if request.image_base64:
            img_data = base64.b64decode(request.image_base64)
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        elif request.image_path:
            image_path = Path(request.image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {request.image_path}")
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if image is None:
            raise ValueError("Failed to load image")

        # Extract object
        result = await state.object_extractor.extract_single_object(
            image=image,
            annotation=request.annotation,
            category_name=request.category_name,
            use_sam3=request.use_sam3,
            padding=request.padding
        )

        processing_time = (time.time() - start_time) * 1000

        if not result["success"]:
            return ExtractSingleObjectResponse(
                success=False,
                processing_time_ms=processing_time,
                error=result.get("error", "Unknown error")
            )

        return ExtractSingleObjectResponse(
            success=True,
            cropped_image_base64=result["cropped_image_base64"],
            mask_base64=result.get("mask_base64"),
            annotation_type=AnnotationType(result["annotation_type"]),
            method_used=ExtractionMethod(result["method_used"]),
            original_bbox=result["original_bbox"],
            extracted_size=result["extracted_size"],
            mask_coverage=result["mask_coverage"],
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Single object extraction failed: {e}")
        return ExtractSingleObjectResponse(
            success=False,
            processing_time_ms=(time.time() - start_time) * 1000,
            error=str(e)
        )


# =========================================================================
# SAM3 TOOL ENDPOINTS
# =========================================================================

@app.post("/sam3/segment-image", response_model=SAM3SegmentImageResponse, tags=["SAM3 Tool"])
async def sam3_segment_image(request: SAM3SegmentImageRequest):
    """
    Segment an object in an image using SAM3 with box or point prompt.

    Returns the segmentation mask and polygon coordinates.
    """
    start_time = time.time()

    if not state.sam3_available:
        return SAM3SegmentImageResponse(
            success=False,
            processing_time_ms=(time.time() - start_time) * 1000,
            error="SAM3 not available"
        )

    try:
        import torch
        from PIL import Image as PILImage

        # Load image
        image = None
        if request.image_base64:
            img_data = base64.b64decode(request.image_base64)
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif request.image_path:
            image_path = Path(request.image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {request.image_path}")
            image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError("Failed to load image")

        h, w = image.shape[:2]

        # Convert to PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(image_rgb)

        # Prepare inputs based on prompt type
        inputs = None
        if request.bbox:
            x, y, bw, bh = request.bbox
            input_box = [[x, y, x + bw, y + bh]]
            inputs = state.sam3_processor(
                images=pil_image,
                input_boxes=[input_box],
                return_tensors="pt"
            ).to(state.device)
        elif request.point:
            input_point = [[request.point]]
            input_label = [[1]]  # 1 = foreground
            inputs = state.sam3_processor(
                images=pil_image,
                input_points=input_point,
                input_labels=input_label,
                return_tensors="pt"
            ).to(state.device)
        elif request.text_prompt:
            inputs = state.sam3_processor(
                images=pil_image,
                text=request.text_prompt,
                return_tensors="pt"
            ).to(state.device)
        else:
            raise ValueError("Must provide bbox, point, or text_prompt")

        # Run SAM3
        with torch.no_grad():
            outputs = state.sam3_model(**inputs)

        # Post-process masks
        masks = state.sam3_processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"]
        )

        if len(masks) == 0 or len(masks[0]) == 0:
            return SAM3SegmentImageResponse(
                success=False,
                processing_time_ms=(time.time() - start_time) * 1000,
                error="No mask generated"
            )

        # Get best mask
        if hasattr(outputs, 'iou_scores') and outputs.iou_scores is not None:
            best_idx = outputs.iou_scores[0].argmax().item()
            mask = masks[0][best_idx]
            confidence = float(outputs.iou_scores[0][best_idx].cpu().item())
        else:
            mask = masks[0][0]
            confidence = 1.0

        mask_np = mask.cpu().numpy().astype(np.uint8) * 255

        # Ensure correct size
        if mask_np.shape != (h, w):
            mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

        # Calculate bbox and area
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, bw, bh = cv2.boundingRect(np.concatenate(contours))
            bbox = [float(x), float(y), float(bw), float(bh)]
        else:
            bbox = [0.0, 0.0, float(w), float(h)]

        area = int(np.sum(mask_np > 0))

        # Convert to polygon if requested
        segmentation_polygon = None
        segmentation_coco = None
        if request.return_polygon:
            polygons = state.object_extractor.mask_to_polygon(
                mask_np,
                simplify=request.simplify_polygon,
                tolerance=request.simplify_tolerance
            )
            if polygons:
                segmentation_coco = polygons
                # Convert to [[x,y], [x,y], ...] format
                segmentation_polygon = [
                    [[polygons[0][i], polygons[0][i+1]] for i in range(0, len(polygons[0]), 2)]
                ]

        # Encode mask if requested
        mask_base64 = None
        if request.return_mask:
            success, encoded = cv2.imencode('.png', mask_np)
            if success:
                mask_base64 = base64.b64encode(encoded.tobytes()).decode('utf-8')

        return SAM3SegmentImageResponse(
            success=True,
            mask_base64=mask_base64,
            segmentation_polygon=segmentation_polygon[0] if segmentation_polygon else None,
            segmentation_coco=segmentation_coco,
            bbox=bbox,
            area=area,
            confidence=confidence,
            processing_time_ms=(time.time() - start_time) * 1000
        )

    except Exception as e:
        logger.error(f"SAM3 segmentation failed: {e}")
        return SAM3SegmentImageResponse(
            success=False,
            processing_time_ms=(time.time() - start_time) * 1000,
            error=str(e)
        )


@app.post("/sam3/convert-dataset", response_model=SAM3ConvertDatasetResponse, tags=["SAM3 Tool"])
async def sam3_convert_dataset(request: SAM3ConvertDatasetRequest):
    """
    Convert bbox-only annotations to segmentations using SAM3.

    Runs asynchronously. Use GET /sam3/jobs/{job_id} to track progress.
    """
    if not state.sam3_available:
        return SAM3ConvertDatasetResponse(
            success=False,
            error="SAM3 not available"
        )

    try:
        # Get COCO data
        coco_data = None
        if request.coco_data:
            coco_data = request.coco_data
        elif request.coco_json_path:
            json_path = Path(request.coco_json_path)
            if not json_path.exists():
                raise FileNotFoundError(f"COCO JSON not found: {request.coco_json_path}")
            with open(json_path, 'r') as f:
                coco_data = json.load(f)
        else:
            raise ValueError("Either coco_data or coco_json_path must be provided")

        # Validate images directory
        images_dir = Path(request.images_dir)
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {request.images_dir}")

        # Create job
        job_id = str(uuid.uuid4())

        # Count annotations to convert
        categories = {c["id"]: c["name"] for c in coco_data.get("categories", [])}
        if request.categories_to_convert:
            valid_cat_ids = {cid for cid, name in categories.items() if name in request.categories_to_convert}
        else:
            valid_cat_ids = set(categories.keys())

        total_annotations = 0
        for ann in coco_data.get("annotations", []):
            if ann.get("category_id") not in valid_cat_ids:
                continue
            ann_type = state.object_extractor.detect_annotation_type(ann)
            if ann_type == AnnotationType.BBOX_ONLY or request.overwrite_existing:
                total_annotations += 1

        sam3_conversion_jobs[job_id] = {
            "job_id": job_id,
            "status": JobStatus.QUEUED,
            "total_annotations": total_annotations,
            "converted_annotations": 0,
            "skipped_annotations": 0,
            "failed_annotations": 0,
            "current_image": "",
            "categories_progress": {},
            "output_path": request.output_path,
            "errors": [],
            "processing_time_ms": 0.0,
            "started_at": None,
            "completed_at": None
        }

        # Define conversion task
        async def run_conversion():
            sam3_conversion_jobs[job_id]["status"] = JobStatus.PROCESSING
            sam3_conversion_jobs[job_id]["started_at"] = datetime.now().isoformat()

            def progress_callback(progress):
                # This callback runs from a thread pool, but dict updates are atomic in CPython
                sam3_conversion_jobs[job_id]["converted_annotations"] = progress["converted"]
                sam3_conversion_jobs[job_id]["skipped_annotations"] = progress["skipped"]
                sam3_conversion_jobs[job_id]["failed_annotations"] = progress["failed"]
                sam3_conversion_jobs[job_id]["current_image"] = progress.get("current_image", "")
                sam3_conversion_jobs[job_id]["categories_progress"] = progress.get("by_category", {})

            try:
                result = await state.object_extractor.convert_bbox_to_segmentation(
                    coco_data=coco_data,
                    images_dir=str(images_dir),
                    output_path=request.output_path,
                    categories_to_convert=request.categories_to_convert or None,
                    overwrite_existing=request.overwrite_existing,
                    simplify_polygons=request.simplify_polygons,
                    simplify_tolerance=request.simplify_tolerance,
                    progress_callback=progress_callback
                )

                if result.get("success"):
                    sam3_conversion_jobs[job_id]["status"] = JobStatus.COMPLETED
                else:
                    sam3_conversion_jobs[job_id]["status"] = JobStatus.FAILED

                sam3_conversion_jobs[job_id]["converted_annotations"] = result.get("converted", 0)
                sam3_conversion_jobs[job_id]["skipped_annotations"] = result.get("skipped", 0)
                sam3_conversion_jobs[job_id]["failed_annotations"] = result.get("failed", 0)
                sam3_conversion_jobs[job_id]["categories_progress"] = result.get("by_category", {})
                sam3_conversion_jobs[job_id]["errors"] = result.get("errors", [])[:100]
                sam3_conversion_jobs[job_id]["processing_time_ms"] = result.get("processing_time_seconds", 0) * 1000

            except Exception as e:
                # Use logger.exception to get full traceback for debugging
                logger.exception(f"Conversion job {job_id} failed: {e}")
                sam3_conversion_jobs[job_id]["status"] = JobStatus.FAILED
                sam3_conversion_jobs[job_id]["errors"].append(str(e))

            finally:
                # Always set completed_at, even if job failed
                sam3_conversion_jobs[job_id]["completed_at"] = datetime.now().isoformat()

        # Run in background using asyncio.create_task for proper async execution
        asyncio.create_task(run_conversion())
        logger.info(f"Started SAM3 conversion job {job_id} with {total_annotations} annotations")

        return SAM3ConvertDatasetResponse(
            success=True,
            job_id=job_id,
            status=JobStatus.QUEUED,
            message=f"Conversion job queued. {total_annotations} annotations to convert."
        )

    except Exception as e:
        logger.error(f"Failed to start conversion: {e}")
        return SAM3ConvertDatasetResponse(
            success=False,
            error=str(e)
        )


@app.get("/sam3/jobs", tags=["SAM3 Tool"])
async def list_sam3_conversion_jobs():
    """List all SAM3 conversion jobs."""
    jobs = []
    for job_id, job in sam3_conversion_jobs.items():
        # Convert JobStatus enum to string for JSON serialization
        status = job.get("status", "unknown")
        status_str = status.value if hasattr(status, 'value') else str(status)
        jobs.append({
            "job_id": job_id,
            "job_type": "sam3_conversion",
            "status": status_str,
            "total_annotations": job.get("total_annotations", 0),
            "converted_annotations": job.get("converted_annotations", 0),
            "skipped_annotations": job.get("skipped_annotations", 0),
            "failed_annotations": job.get("failed_annotations", 0),
            "current_image": job.get("current_image", ""),
            "output_path": job.get("output_path", ""),
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
            "processing_time_ms": job.get("processing_time_ms", 0),
        })
    return {"jobs": jobs, "total": len(jobs)}


@app.get("/sam3/jobs/{job_id}", response_model=SAM3ConversionJobStatus, tags=["SAM3 Tool"])
async def get_sam3_conversion_job_status(job_id: str):
    """Get the status of a SAM3 conversion job."""
    if job_id not in sam3_conversion_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = sam3_conversion_jobs[job_id]
    return SAM3ConversionJobStatus(**job)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info",
    )
