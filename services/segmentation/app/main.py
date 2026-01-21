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
    ExtractCustomObjectsRequest,
    ExtractCustomObjectsResponse,
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
    # Labeling schemas
    StartLabelingRequest,
    StartRelabelingRequest,
    LabelingJobResponse,
    LabelingJobStatus,
    LabelingResultResponse,
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
labeling_jobs: Dict[str, Dict[str, Any]] = {}

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
            "completed_at": None,
            "duplicates_prevented": 0,
            "deduplication_enabled": request.deduplication.enabled if request.deduplication else True
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
                    force_bbox_only=request.force_bbox_only,
                    force_sam3_resegmentation=request.force_sam3_resegmentation,
                    force_sam3_text_prompt=request.force_sam3_text_prompt,
                    padding=request.padding,
                    min_object_area=request.min_object_area,
                    save_individual_coco=request.save_individual_coco,
                    progress_callback=progress_callback,
                    deduplication_config=request.deduplication
                )
                extraction_jobs[job_id]["extracted_objects"] = result["extracted"]
                extraction_jobs[job_id]["failed_objects"] = result["failed"]
                extraction_jobs[job_id]["categories_progress"] = result["by_category"]
                extraction_jobs[job_id]["errors"] = result.get("errors", [])[:100]
                extraction_jobs[job_id]["extracted_files"] = result.get("extracted_files", [])[:1000]
                extraction_jobs[job_id]["processing_time_ms"] = result.get("processing_time_seconds", 0) * 1000
                extraction_jobs[job_id]["duplicates_prevented"] = result.get("deduplication_stats", {}).get("duplicates_prevented", 0)

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

        dedup_status = "enabled (IoU=0.7)" if (request.deduplication and request.deduplication.enabled) or request.deduplication is None else "disabled"
        logger.info(f"Started extraction job {job_id} with {total_objects} objects (deduplication: {dedup_status})")

        return ExtractObjectsResponse(
            success=True,
            job_id=job_id,
            status=JobStatus.QUEUED,
            message=f"Extraction job queued. {total_objects} objects to extract. Deduplication: {dedup_status}"
        )

    except Exception as e:
        logger.error(f"Failed to start extraction: {e}")
        return ExtractObjectsResponse(
            success=False,
            error=str(e)
        )


@app.post("/extract/custom-objects", response_model=ExtractCustomObjectsResponse, tags=["Object Extraction"])
async def extract_custom_objects(request: ExtractCustomObjectsRequest):
    """
    Extract custom objects using text prompts (no COCO JSON required).

    This endpoint allows you to specify object names directly and segment them
    from images using SAM3 text prompt mode.

    Process:
    1. Scans all images in images_dir
    2. For each object name in the list, runs SAM3 text prompt segmentation
    3. Extracts all detected instances as transparent PNGs
    4. Organizes results by object type: output_dir/{object_name}/

    Runs asynchronously. Use GET /extract/jobs/{job_id} to track progress.
    """
    try:
        # Validate images directory
        images_dir = Path(request.images_dir)
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {request.images_dir}")

        # Validate object names
        if not request.object_names or len(request.object_names) == 0:
            raise ValueError("At least one object name must be provided")

        # Create job
        job_id = str(uuid.uuid4())

        # Count images for progress tracking
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        total_images = sum(
            1 for f in images_dir.iterdir()
            if f.suffix.lower() in image_extensions
        )

        if total_images == 0:
            raise ValueError(f"No images found in {request.images_dir}")

        extraction_jobs[job_id] = {
            "job_id": job_id,
            "status": JobStatus.QUEUED,
            "total_objects": 0,  # Unknown until extraction
            "extracted_objects": 0,
            "failed_objects": 0,
            "current_category": "",
            "categories_progress": {name: 0 for name in request.object_names},
            "output_dir": request.output_dir,
            "errors": [],
            "extracted_files": [],
            "processing_time_ms": 0.0,
            "started_at": None,
            "completed_at": None,
            "duplicates_prevented": 0,
            "deduplication_enabled": request.deduplication.enabled if request.deduplication else True,
            "total_images": total_images,
            "current_image": ""
        }

        # Define extraction task
        async def run_extraction():
            extraction_jobs[job_id]["status"] = JobStatus.PROCESSING
            extraction_jobs[job_id]["started_at"] = datetime.now().isoformat()

            def progress_callback(progress):
                # Update job status with progress info
                extraction_jobs[job_id]["extracted_objects"] = progress.get("extracted", 0)
                extraction_jobs[job_id]["failed_objects"] = progress.get("failed", 0)
                extraction_jobs[job_id]["duplicates_prevented"] = progress.get("duplicates_prevented", 0)
                extraction_jobs[job_id]["current_image"] = progress.get("current_image", "")

            try:
                result = await state.object_extractor.extract_custom_objects(
                    images_dir=str(images_dir),
                    output_dir=request.output_dir,
                    object_names=request.object_names,
                    padding=request.padding,
                    min_object_area=request.min_object_area,
                    save_individual_coco=request.save_individual_coco,
                    deduplication_config=request.deduplication.dict() if request.deduplication else None,
                    progress_callback=progress_callback
                )

                # Update final job status
                extraction_jobs[job_id]["total_objects"] = result.get("total_objects_extracted", 0)
                extraction_jobs[job_id]["extracted_objects"] = result.get("total_objects_extracted", 0)
                extraction_jobs[job_id]["failed_objects"] = result.get("failed_extractions", 0)
                extraction_jobs[job_id]["categories_progress"] = result.get("by_category", {})
                extraction_jobs[job_id]["errors"] = result.get("errors", [])[:100]
                extraction_jobs[job_id]["processing_time_ms"] = result.get("processing_time_seconds", 0) * 1000
                extraction_jobs[job_id]["duplicates_prevented"] = result.get("duplicates_prevented", 0)

                if result.get("success"):
                    extraction_jobs[job_id]["status"] = JobStatus.COMPLETED
                    logger.info(f"Custom extraction job {job_id} completed: {result['total_objects_extracted']} objects")
                else:
                    extraction_jobs[job_id]["status"] = JobStatus.FAILED
                    logger.error(f"Custom extraction job {job_id} failed")

            except Exception as e:
                logger.exception(f"Custom extraction job {job_id} failed: {e}")
                extraction_jobs[job_id]["status"] = JobStatus.FAILED
                extraction_jobs[job_id]["errors"].append(str(e))

            finally:
                extraction_jobs[job_id]["completed_at"] = datetime.now().isoformat()

        # Run in background
        asyncio.create_task(run_extraction())

        dedup_status = "enabled (IoU=0.7)" if (request.deduplication and request.deduplication.enabled) or request.deduplication is None else "disabled"
        logger.info(
            f"Started custom extraction job {job_id} for {len(request.object_names)} object types "
            f"across {total_images} images (deduplication: {dedup_status})"
        )

        return ExtractCustomObjectsResponse(
            success=True,
            job_id=job_id,
            status=JobStatus.QUEUED,
            message=f"Custom extraction job queued. Will search for {len(request.object_names)} object types in {total_images} images. Deduplication: {dedup_status}"
        )

    except Exception as e:
        logger.error(f"Failed to start custom extraction: {e}")
        return ExtractCustomObjectsResponse(
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
            padding=request.padding,
            force_bbox_only=request.force_bbox_only,
            force_sam3_resegmentation=request.force_sam3_resegmentation,
            force_sam3_text_prompt=request.force_sam3_text_prompt
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


@app.post("/extract/imagenet", tags=["Object Extraction"])
async def extract_from_imagenet(
    root_dir: str,
    output_dir: str,
    padding: int = 5,
    min_object_area: int = 100,
    max_objects_per_class: Optional[int] = None
):
    """
    Extract objects from ImageNet-style directory structure.

    Expected structure:
        root_dir/
        ├── class1/
        │   ├── img001.jpg
        │   └── img002.jpg
        ├── class2/
        │   └── ...

    Uses SAM3 with class name as text prompt for segmentation.
    Runs asynchronously. Use GET /extract/jobs/{job_id} to track progress.
    """
    if not state.sam3_available:
        return {
            "success": False,
            "error": "SAM3 is required for ImageNet extraction but not available"
        }

    try:
        # Validate root directory
        root_path = Path(root_dir)
        if not root_path.exists():
            raise FileNotFoundError(f"Root directory not found: {root_dir}")

        if not root_path.is_dir():
            raise ValueError(f"Path is not a directory: {root_dir}")

        # Create job
        job_id = str(uuid.uuid4())

        # Count classes
        try:
            classes = [d for d in os.listdir(root_dir)
                      if os.path.isdir(os.path.join(root_dir, d))]
            num_classes = len(classes)
        except Exception as e:
            raise ValueError(f"Failed to read root directory: {e}")

        extraction_jobs[job_id] = {
            "job_id": job_id,
            "status": JobStatus.QUEUED,
            "total_objects": 0,  # Will be updated as we discover images
            "extracted_objects": 0,
            "failed_objects": 0,
            "current_category": "",
            "categories_progress": {},
            "output_dir": output_dir,
            "errors": [],
            "extracted_files": [],
            "processing_time_ms": 0.0,
            "started_at": None,
            "completed_at": None,
            "extraction_type": "imagenet"
        }

        # Define extraction task
        async def run_imagenet_extraction():
            extraction_jobs[job_id]["status"] = JobStatus.PROCESSING
            extraction_jobs[job_id]["started_at"] = datetime.now().isoformat()

            def progress_callback(progress):
                # Update job progress
                extraction_jobs[job_id]["current_category"] = progress.get("current_class", "")
                extraction_jobs[job_id]["extracted_objects"] = progress.get("extracted", 0)
                extraction_jobs[job_id]["failed_objects"] = progress.get("failed", 0)

            try:
                start_time = time.time()
                result = await state.object_extractor.extract_from_imagenet_structure(
                    root_dir=root_dir,
                    output_dir=output_dir,
                    padding=padding,
                    min_object_area=min_object_area,
                    max_objects_per_class=max_objects_per_class,
                    progress_callback=progress_callback
                )

                if result.get("success"):
                    extraction_jobs[job_id]["status"] = JobStatus.COMPLETED
                else:
                    extraction_jobs[job_id]["status"] = JobStatus.FAILED

                extraction_jobs[job_id]["total_objects"] = result.get("total_extracted", 0) + result.get("total_failed", 0)
                extraction_jobs[job_id]["extracted_objects"] = result.get("total_extracted", 0)
                extraction_jobs[job_id]["failed_objects"] = result.get("total_failed", 0)
                extraction_jobs[job_id]["categories_progress"] = result.get("classes", {})
                extraction_jobs[job_id]["errors"] = result.get("errors", [])[:100]
                extraction_jobs[job_id]["processing_time_ms"] = (time.time() - start_time) * 1000

            except Exception as e:
                logger.exception(f"ImageNet extraction job {job_id} failed: {e}")
                extraction_jobs[job_id]["status"] = JobStatus.FAILED
                extraction_jobs[job_id]["errors"].append(str(e))

            finally:
                extraction_jobs[job_id]["completed_at"] = datetime.now().isoformat()

        # Run in background
        asyncio.create_task(run_imagenet_extraction())
        logger.info(f"Started ImageNet extraction job {job_id} from {root_dir} ({num_classes} classes)")

        return {
            "success": True,
            "job_id": job_id,
            "status": "queued",
            "message": f"ImageNet extraction job queued. Processing {num_classes} classes."
        }

    except Exception as e:
        logger.error(f"Failed to start ImageNet extraction: {e}")
        return {
            "success": False,
            "error": str(e)
        }


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
                    progress_callback=progress_callback,
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


# =========================================================================
# LABELING TOOL ENDPOINTS
# =========================================================================

@app.post("/labeling/start", response_model=LabelingJobResponse, tags=["Labeling Tool"])
async def start_labeling_job(request: StartLabelingRequest):
    """
    Start a new labeling job to label images from scratch.

    Uses SAM3 text prompts to detect and segment specified classes.
    Supports multiple image directories and output formats.
    """
    if not state.sam3_available:
        return LabelingJobResponse(
            success=False,
            error="SAM3 not available. This feature requires SAM3 for text-based segmentation."
        )

    try:
        import torch
        from PIL import Image as PILImage

        # Validate directories and count images
        total_images = 0
        all_image_paths = []
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        for dir_path in request.image_directories:
            dir_obj = Path(dir_path)
            if not dir_obj.exists():
                return LabelingJobResponse(
                    success=False,
                    error=f"Directory not found: {dir_path}"
                )

            for ext in image_extensions:
                for img_path in dir_obj.glob(f"*{ext}"):
                    all_image_paths.append(str(img_path))
                for img_path in dir_obj.glob(f"*{ext.upper()}"):
                    all_image_paths.append(str(img_path))

        total_images = len(all_image_paths)

        if total_images == 0:
            return LabelingJobResponse(
                success=False,
                error="No images found in specified directories"
            )

        # Create output directory
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create job
        job_id = str(uuid.uuid4())

        # Determine final classes for tracking (use mapped names if mapping exists)
        if request.class_mapping:
            final_classes = list(dict.fromkeys(
                request.class_mapping.get(cls, cls) for cls in request.classes
            ))
        else:
            final_classes = request.classes

        labeling_jobs[job_id] = {
            "job_id": job_id,
            "job_type": "labeling",
            "status": JobStatus.QUEUED,
            "total_images": total_images,
            "processed_images": 0,
            "total_objects_found": 0,
            "objects_by_class": {cls: 0 for cls in final_classes},
            "current_image": "",
            "output_dir": request.output_dir,
            "output_formats": request.output_formats,
            "errors": [],
            "processing_time_ms": 0.0,
            "started_at": None,
            "completed_at": None,
            # Store request params for processing
            "_image_paths": all_image_paths,
            "_classes": request.classes,
            "_task_type": request.task_type,
            "_min_confidence": request.min_confidence,
            "_min_area": request.min_area,
            "_max_instances": request.max_instances_per_image,
            "_simplify_polygons": request.simplify_polygons,
            "_simplify_tolerance": request.simplify_tolerance,
            "_save_visualizations": request.save_visualizations,
            "_class_mapping": request.class_mapping,  # Maps prompts to final class names
        }

        # Start background task
        async def run_labeling():
            labeling_jobs[job_id]["status"] = JobStatus.PROCESSING
            labeling_jobs[job_id]["started_at"] = datetime.now().isoformat()

            try:
                await _process_labeling_job(job_id)
                labeling_jobs[job_id]["status"] = JobStatus.COMPLETED
            except Exception as e:
                logger.exception(f"Labeling job {job_id} failed: {e}")
                labeling_jobs[job_id]["status"] = JobStatus.FAILED
                labeling_jobs[job_id]["errors"].append(str(e))
            finally:
                labeling_jobs[job_id]["completed_at"] = datetime.now().isoformat()

        asyncio.create_task(run_labeling())

        logger.info(f"Started labeling job {job_id} with {total_images} images, {len(request.classes)} classes")

        return LabelingJobResponse(
            success=True,
            job_id=job_id,
            status=JobStatus.QUEUED,
            message=f"Labeling job started. {total_images} images, {len(request.classes)} classes.",
            total_images=total_images
        )

    except Exception as e:
        logger.error(f"Failed to start labeling job: {e}")
        return LabelingJobResponse(
            success=False,
            error=str(e)
        )


@app.post("/labeling/relabel", response_model=LabelingJobResponse, tags=["Labeling Tool"])
async def start_relabeling_job(request: StartRelabelingRequest):
    """
    Start a relabeling job for an existing dataset.

    Modes:
    - add: Add new class annotations while keeping existing ones
    - replace: Replace all annotations with new labeling
    - improve_segmentation: Convert bbox-only annotations to segmentations
    """
    if not state.sam3_available:
        return LabelingJobResponse(
            success=False,
            error="SAM3 not available. This feature requires SAM3."
        )

    try:
        # Get image paths from directories and/or existing dataset
        all_image_paths = []
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        # Build image lookup from directories
        image_lookup = {}  # filename -> full path
        for dir_path in request.image_directories:
            dir_obj = Path(dir_path)
            if dir_obj.exists():
                for ext in image_extensions:
                    for img_path in dir_obj.glob(f"*{ext}"):
                        image_lookup[img_path.name] = str(img_path)
                    for img_path in dir_obj.glob(f"*{ext.upper()}"):
                        image_lookup[img_path.name] = str(img_path)

        # If we have COCO data, get images from there
        if request.coco_data:
            for img in request.coco_data.get("images", []):
                filename = img.get("file_name", "")
                if filename in image_lookup:
                    all_image_paths.append(image_lookup[filename])
                elif Path(filename).exists():
                    all_image_paths.append(filename)
        else:
            # Just use all images from directories
            all_image_paths = list(image_lookup.values())

        total_images = len(all_image_paths)

        if total_images == 0:
            return LabelingJobResponse(
                success=False,
                error="No images found in specified directories"
            )

        # Determine classes to label
        classes_to_label = request.new_classes if request.new_classes else []

        if request.relabel_mode == "improve_segmentation" and request.coco_data:
            # Get classes from existing dataset
            classes_to_label = [c["name"] for c in request.coco_data.get("categories", [])]

        # Create output directory
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create job
        job_id = str(uuid.uuid4())

        labeling_jobs[job_id] = {
            "job_id": job_id,
            "job_type": "relabeling",
            "status": JobStatus.QUEUED,
            "total_images": total_images,
            "processed_images": 0,
            "total_objects_found": 0,
            "objects_by_class": {cls: 0 for cls in classes_to_label},
            "current_image": "",
            "output_dir": request.output_dir,
            "output_formats": request.output_formats,
            "errors": [],
            "processing_time_ms": 0.0,
            "started_at": None,
            "completed_at": None,
            # Store request params
            "_image_paths": all_image_paths,
            "_image_lookup": image_lookup,
            "_classes": classes_to_label,
            "_relabel_mode": request.relabel_mode,
            "_coco_data": request.coco_data,
            "_min_confidence": request.min_confidence,
            "_simplify_polygons": request.simplify_polygons,
        }

        # Start background task
        async def run_relabeling():
            logger.info(f"[RELABEL] Background task started for job {job_id}")
            labeling_jobs[job_id]["status"] = JobStatus.PROCESSING
            labeling_jobs[job_id]["started_at"] = datetime.now().isoformat()

            try:
                logger.info(f"[RELABEL] Calling _process_relabeling_job for {job_id}")
                await _process_relabeling_job(job_id)
                labeling_jobs[job_id]["status"] = JobStatus.COMPLETED
                logger.info(f"[RELABEL] Job {job_id} completed successfully")
            except Exception as e:
                logger.exception(f"Relabeling job {job_id} failed: {e}")
                labeling_jobs[job_id]["status"] = JobStatus.FAILED
                labeling_jobs[job_id]["errors"].append(str(e))
            finally:
                labeling_jobs[job_id]["completed_at"] = datetime.now().isoformat()

        asyncio.create_task(run_relabeling())

        logger.info(f"Started relabeling job {job_id} - mode: {request.relabel_mode}, {total_images} images")

        return LabelingJobResponse(
            success=True,
            job_id=job_id,
            status=JobStatus.QUEUED,
            message=f"Relabeling job started. Mode: {request.relabel_mode}, {total_images} images.",
            total_images=total_images
        )

    except Exception as e:
        logger.error(f"Failed to start relabeling job: {e}")
        return LabelingJobResponse(
            success=False,
            error=str(e)
        )


@app.get("/labeling/jobs", tags=["Labeling Tool"])
async def list_labeling_jobs():
    """List all labeling jobs."""
    jobs = []
    for job_id, job in labeling_jobs.items():
        status = job.get("status", "unknown")
        status_str = status.value if hasattr(status, 'value') else str(status)

        # Check if job can be resumed (has checkpoint and is failed/cancelled)
        can_resume = False
        if status in [JobStatus.FAILED, JobStatus.CANCELLED]:
            output_dir = Path(job.get("output_dir", ""))
            checkpoint_path = output_dir / "checkpoint.json"
            can_resume = checkpoint_path.exists()

        jobs.append({
            "job_id": job_id,
            "job_type": job.get("job_type", "labeling"),
            "status": status_str,
            "total_images": job.get("total_images", 0),
            "processed_images": job.get("processed_images", 0),
            "total_objects_found": job.get("total_objects_found", 0),
            "objects_by_class": job.get("objects_by_class", {}),
            "output_dir": job.get("output_dir", ""),
            "current_image": job.get("current_image", ""),  # For progress display
            "errors": job.get("errors", [])[:5],  # Include first 5 errors
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
            "processing_time_ms": job.get("processing_time_ms", 0),
            "can_resume": can_resume,
        })
    return {"jobs": jobs, "total": len(jobs)}


@app.get("/labeling/jobs/{job_id}", response_model=LabelingJobStatus, tags=["Labeling Tool"])
async def get_labeling_job_status(job_id: str):
    """Get status of a labeling job."""
    if job_id not in labeling_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = labeling_jobs[job_id]

    # Check if job can be resumed (has checkpoint and is failed/cancelled)
    can_resume = False
    status = job.get("status")
    if status in [JobStatus.FAILED, JobStatus.CANCELLED]:
        output_dir = Path(job.get("output_dir", ""))
        checkpoint_path = output_dir / "checkpoint.json"
        can_resume = checkpoint_path.exists()

    return LabelingJobStatus(
        job_id=job_id,
        job_type=job.get("job_type", "labeling"),
        status=status or JobStatus.QUEUED,
        total_images=job.get("total_images", 0),
        processed_images=job.get("processed_images", 0),
        total_objects_found=job.get("total_objects_found", 0),
        objects_by_class=job.get("objects_by_class", {}),
        current_image=job.get("current_image", ""),
        output_dir=job.get("output_dir", ""),
        output_formats=job.get("output_formats", []),
        errors=job.get("errors", [])[:50],
        processing_time_ms=job.get("processing_time_ms", 0),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        can_resume=can_resume,
    )


@app.get("/labeling/jobs/{job_id}/result", response_model=LabelingResultResponse, tags=["Labeling Tool"])
async def get_labeling_result(job_id: str):
    """Get the result of a completed labeling job."""
    if job_id not in labeling_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = labeling_jobs[job_id]

    if job.get("status") != JobStatus.COMPLETED:
        return LabelingResultResponse(
            success=False,
            error=f"Job is not completed. Current status: {job.get('status')}"
        )

    # Load the COCO result
    output_dir = Path(job.get("output_dir", ""))
    coco_path = output_dir / "annotations.json"

    if not coco_path.exists():
        return LabelingResultResponse(
            success=False,
            error="Result file not found"
        )

    try:
        with open(coco_path, 'r') as f:
            coco_data = json.load(f)

        # Build output files map
        output_files = {"coco": str(coco_path)}

        yolo_dir = output_dir / "yolo"
        if yolo_dir.exists():
            output_files["yolo"] = str(yolo_dir)

        voc_dir = output_dir / "voc"
        if voc_dir.exists():
            output_files["voc"] = str(voc_dir)

        return LabelingResultResponse(
            success=True,
            data=coco_data,
            output_files=output_files,
            summary={
                "total_images": len(coco_data.get("images", [])),
                "total_annotations": len(coco_data.get("annotations", [])),
                "categories": [c.get("name") for c in coco_data.get("categories", [])],
            }
        )

    except Exception as e:
        return LabelingResultResponse(
            success=False,
            error=str(e)
        )


@app.post("/labeling/jobs/{job_id}/resume", response_model=LabelingJobResponse, tags=["Labeling Tool"])
async def resume_labeling_job(job_id: str):
    """
    Resume a failed or interrupted labeling job.

    Loads the checkpoint and continues from the last processed image.
    """
    if job_id not in labeling_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = labeling_jobs[job_id]

    # Only allow resuming failed jobs
    if job.get("status") not in [JobStatus.FAILED, JobStatus.CANCELLED]:
        return LabelingJobResponse(
            success=False,
            error=f"Job cannot be resumed. Current status: {job.get('status')}. "
                  f"Only failed or cancelled jobs can be resumed."
        )

    # Check if checkpoint exists
    output_dir = Path(job.get("output_dir", ""))
    checkpoint_path = output_dir / "checkpoint.json"

    if not checkpoint_path.exists():
        return LabelingJobResponse(
            success=False,
            error="No checkpoint found. Job must be restarted from the beginning."
        )

    # Load checkpoint to get resume point
    try:
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        resume_from = checkpoint.get("last_processed_idx", 0)
    except Exception as e:
        return LabelingJobResponse(
            success=False,
            error=f"Failed to load checkpoint: {str(e)}"
        )

    # Reset job status and start processing
    job["status"] = JobStatus.PROCESSING
    job["errors"] = []  # Clear previous errors
    job["started_at"] = datetime.now().isoformat()
    job["completed_at"] = None

    # Start background task to resume
    async def run_resume():
        try:
            await _process_labeling_job(job_id, resume_from=resume_from)
            labeling_jobs[job_id]["status"] = JobStatus.COMPLETED
        except Exception as e:
            logger.exception(f"Resumed labeling job {job_id} failed: {e}")
            labeling_jobs[job_id]["status"] = JobStatus.FAILED
            labeling_jobs[job_id]["errors"].append(str(e))
        finally:
            labeling_jobs[job_id]["completed_at"] = datetime.now().isoformat()

    asyncio.create_task(run_resume())

    logger.info(f"Resumed labeling job {job_id} from image {resume_from}")

    return LabelingJobResponse(
        success=True,
        job_id=job_id,
        status=JobStatus.PROCESSING,
        message=f"Job resumed from image {resume_from + 1} of {job.get('total_images', 0)}",
        total_images=job.get("total_images", 0)
    )


async def _process_labeling_job(job_id: str, resume_from: int = 0):
    """Process a labeling job - detect and segment objects in images.

    Args:
        job_id: The job identifier
        resume_from: Image index to resume from (0 = start fresh)
    """
    import gc
    import torch
    from PIL import Image as PILImage

    job = labeling_jobs[job_id]
    start_time = time.time()

    image_paths = job["_image_paths"]
    classes = job["_classes"]  # These are the search prompts
    class_mapping = job.get("_class_mapping")  # Maps prompts to final class names
    min_confidence = job["_min_confidence"]
    min_area = job["_min_area"]
    max_instances = job["_max_instances"]
    simplify_polygons = job["_simplify_polygons"]
    simplify_tolerance = job["_simplify_tolerance"]
    task_type = job["_task_type"]
    output_dir = Path(job["output_dir"])
    checkpoint_path = output_dir / "checkpoint.json"
    checkpoint_interval = 10  # Save checkpoint every N images
    gc_interval = 5  # Run garbage collection every N images
    yield_interval = 1  # Yield to event loop every N images

    # Determine final category names
    # If class_mapping exists, use unique mapped values as categories
    # Otherwise, use the original class names
    if class_mapping:
        # Get unique final class names (values from the mapping)
        # Preserve order by using the first appearance
        final_classes = []
        seen = set()
        for cls in classes:
            final_name = class_mapping.get(cls, cls)
            if final_name not in seen:
                final_classes.append(final_name)
                seen.add(final_name)
    else:
        final_classes = classes

    # Initialize or load COCO structure
    coco_result = None
    annotation_id = 1

    # Try to load checkpoint if resuming
    if resume_from > 0 and checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            coco_result = checkpoint.get("coco_data")
            annotation_id = checkpoint.get("next_annotation_id", 1)
            # Restore objects_by_class counts
            if "objects_by_class" in checkpoint:
                job["objects_by_class"] = checkpoint["objects_by_class"]
                job["total_objects_found"] = sum(checkpoint["objects_by_class"].values())
            logger.info(f"Resuming job {job_id} from image {resume_from}, annotation_id {annotation_id}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}, starting fresh")
            coco_result = None

    # Initialize fresh if no checkpoint loaded
    if coco_result is None:
        coco_result = {
            "info": {
                "description": "Auto-labeled dataset",
                "date_created": datetime.now().isoformat(),
                "version": "1.0"
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [{"id": i + 1, "name": cls, "supercategory": ""} for i, cls in enumerate(final_classes)]
        }
        resume_from = 0  # Reset if no valid checkpoint

    # Map final class names to category IDs
    category_map = {cls: i + 1 for i, cls in enumerate(final_classes)}

    # Process images starting from resume_from
    for img_idx, img_path in enumerate(image_paths):
        # Skip already processed images when resuming
        if img_idx < resume_from:
            continue

        job["current_image"] = Path(img_path).name
        job["processed_images"] = img_idx

        try:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                job["errors"].append(f"Failed to load: {img_path}")
                continue

            h, w = image.shape[:2]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(image_rgb)

            # Add image to COCO
            image_id = img_idx + 1
            coco_result["images"].append({
                "id": image_id,
                "file_name": Path(img_path).name,
                "width": w,
                "height": h
            })

            # Process each class (cls is the search prompt)
            for cls in classes:
                # Get final class name (may be different if mapping exists)
                final_class = class_mapping.get(cls, cls) if class_mapping else cls

                # Run SAM3 text prompt segmentation using original prompt
                inputs = state.sam3_processor(
                    images=pil_image,
                    text=cls,  # Use original prompt for detection
                    return_tensors="pt"
                ).to(state.device)

                with torch.no_grad():
                    outputs = state.sam3_model(**inputs)

                # Post-process results
                target_sizes = inputs.get("original_sizes")
                if target_sizes is not None:
                    target_sizes = target_sizes.tolist()
                else:
                    target_sizes = [(h, w)]

                results = state.sam3_processor.post_process_instance_segmentation(
                    outputs,
                    threshold=min_confidence,
                    mask_threshold=min_confidence,
                    target_sizes=target_sizes
                )[0]

                if 'masks' not in results or len(results['masks']) == 0:
                    continue

                # Process each detected instance
                instances_added = 0
                for mask, score in zip(results['masks'], results['scores']):
                    if instances_added >= max_instances:
                        break

                    score_val = score.cpu().item()
                    if score_val < min_confidence:
                        continue

                    mask_np = mask.cpu().numpy().astype(np.uint8)
                    if mask_np.shape != (h, w):
                        mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

                    # Calculate area
                    area = int(np.sum(mask_np > 0))
                    if area < min_area:
                        continue

                    # Get bounding box
                    contours, _ = cv2.findContours(
                        (mask_np * 255).astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )

                    if not contours:
                        continue

                    x, y, bw, bh = cv2.boundingRect(np.concatenate(contours))
                    bbox = [float(x), float(y), float(bw), float(bh)]

                    # Build annotation (use final_class for category lookup)
                    annotation = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_map[final_class],
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0,
                    }

                    # Add segmentation if task requires it
                    if task_type in ["segmentation", "both"]:
                        polygons = state.object_extractor.mask_to_polygon(
                            (mask_np * 255).astype(np.uint8),
                            simplify=simplify_polygons,
                            tolerance=simplify_tolerance
                        )
                        if polygons:
                            annotation["segmentation"] = polygons

                    coco_result["annotations"].append(annotation)
                    annotation_id += 1
                    instances_added += 1

                    # Track by final class name
                    job["objects_by_class"][final_class] = job["objects_by_class"].get(final_class, 0) + 1
                    job["total_objects_found"] += 1

            # Save checkpoint periodically
            if (img_idx + 1) % checkpoint_interval == 0:
                _save_labeling_checkpoint(
                    checkpoint_path, coco_result, annotation_id,
                    img_idx + 1, job["objects_by_class"]
                )
                logger.debug(f"Saved checkpoint at image {img_idx + 1}")

            # Garbage collection periodically to prevent memory buildup
            if (img_idx + 1) % gc_interval == 0:
                # Clear image references
                del image, image_rgb, pil_image
                # Clear CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            # Yield to event loop to allow health checks and other operations
            if (img_idx + 1) % yield_interval == 0:
                await asyncio.sleep(0)

        except Exception as e:
            job["errors"].append(f"Error processing {img_path}: {str(e)}")
            logger.error(f"Error processing {img_path}: {e}")
            # Save checkpoint on error so we can resume
            _save_labeling_checkpoint(
                checkpoint_path, coco_result, annotation_id,
                img_idx, job["objects_by_class"]
            )
            # Clean up on error
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save results
    job["processed_images"] = len(image_paths)
    job["processing_time_ms"] = (time.time() - start_time) * 1000

    # Save COCO JSON
    coco_path = output_dir / "annotations.json"
    with open(coco_path, 'w') as f:
        json.dump(coco_result, f, indent=2)

    # Save in other formats if requested
    if "yolo" in job["output_formats"]:
        _export_to_yolo(coco_result, output_dir, image_paths)

    if "voc" in job["output_formats"]:
        _export_to_voc(coco_result, output_dir, image_paths)

    # Remove checkpoint file on successful completion
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
            logger.debug(f"Removed checkpoint file after successful completion")
        except Exception as e:
            logger.warning(f"Failed to remove checkpoint file: {e}")

    logger.info(f"Labeling job {job_id} completed: {job['total_objects_found']} objects found")


def _save_labeling_checkpoint(checkpoint_path: Path, coco_data: Dict, next_annotation_id: int,
                               last_processed_idx: int, objects_by_class: Dict[str, int]):
    """Save checkpoint for labeling job recovery."""
    checkpoint = {
        "coco_data": coco_data,
        "next_annotation_id": next_annotation_id,
        "last_processed_idx": last_processed_idx,
        "objects_by_class": objects_by_class,
        "saved_at": datetime.now().isoformat()
    }
    try:
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


async def _process_relabeling_job(job_id: str):
    """Process a relabeling job."""
    import gc
    import torch
    from PIL import Image as PILImage

    logger.info(f"[RELABEL] _process_relabeling_job started for {job_id}")

    job = labeling_jobs[job_id]
    start_time = time.time()

    relabel_mode = job["_relabel_mode"]
    logger.info(f"[RELABEL] Mode: {relabel_mode}, Classes: {job.get('_classes', [])[:5]}...")
    coco_data = job.get("_coco_data")
    image_paths = job["_image_paths"]
    image_lookup = job["_image_lookup"]
    classes = job["_classes"]
    min_confidence = job["_min_confidence"]
    simplify_polygons = job["_simplify_polygons"]
    output_dir = Path(job["output_dir"])
    gc_interval = 5  # Run garbage collection every N images

    # Initialize result based on mode
    if relabel_mode == "add" and coco_data:
        # Start with existing data
        coco_result = coco_data.copy()
        # Add new categories
        existing_cats = {c["name"] for c in coco_result.get("categories", [])}
        max_cat_id = max([c["id"] for c in coco_result.get("categories", [])], default=0)
        for cls in classes:
            if cls not in existing_cats:
                max_cat_id += 1
                coco_result["categories"].append({"id": max_cat_id, "name": cls, "supercategory": ""})
        annotation_id = max([a["id"] for a in coco_result.get("annotations", [])], default=0) + 1
    else:
        # Start fresh
        coco_result = {
            "info": {
                "description": "Relabeled dataset",
                "date_created": datetime.now().isoformat(),
                "version": "1.0"
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [{"id": i + 1, "name": cls, "supercategory": ""} for i, cls in enumerate(classes)]
        }
        annotation_id = 1

    category_map = {c["name"]: c["id"] for c in coco_result.get("categories", [])}
    logger.info(f"[RELABEL] Category map: {category_map}")
    logger.info(f"[RELABEL] Starting to process {len(image_paths)} images")

    # Process images
    for img_idx, img_path in enumerate(image_paths):
        job["current_image"] = Path(img_path).name
        job["processed_images"] = img_idx

        if img_idx == 0 or img_idx % 100 == 0:
            logger.info(f"[RELABEL] Processing image {img_idx + 1}/{len(image_paths)}: {Path(img_path).name}")

        try:
            image = cv2.imread(img_path)
            if image is None:
                continue

            h, w = image.shape[:2]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(image_rgb)

            image_id = img_idx + 1

            # Add image if not in replace mode with existing data
            if relabel_mode != "add":
                coco_result["images"].append({
                    "id": image_id,
                    "file_name": Path(img_path).name,
                    "width": w,
                    "height": h
                })

            # Handle improve_segmentation mode differently
            if relabel_mode == "improve_segmentation" and coco_data:
                # Get existing annotations for this image
                img_anns = [a for a in coco_data.get("annotations", [])
                           if a.get("image_id") == image_id and not a.get("segmentation")]

                for ann in img_anns:
                    bbox = ann.get("bbox", [0, 0, 0, 0])
                    if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
                        # Use SAM3 to generate segmentation from bbox
                        x, y, bw, bh = bbox
                        input_box = [[x, y, x + bw, y + bh]]

                        inputs = state.sam3_processor(
                            images=pil_image,
                            input_boxes=[input_box],
                            return_tensors="pt"
                        ).to(state.device)

                        with torch.no_grad():
                            outputs = state.sam3_model(**inputs)

                        masks = state.sam3_processor.post_process_masks(
                            outputs.pred_masks,
                            inputs["original_sizes"],
                            inputs["reshaped_input_sizes"]
                        )

                        if len(masks) > 0 and len(masks[0]) > 0:
                            mask = masks[0][0]
                            mask_np = mask.cpu().numpy().astype(np.uint8) * 255

                            if mask_np.shape != (h, w):
                                mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

                            polygons = state.object_extractor.mask_to_polygon(
                                mask_np,
                                simplify=simplify_polygons,
                                tolerance=2.0
                            )

                            if polygons:
                                # Update annotation with segmentation
                                ann["segmentation"] = polygons
                                ann["area"] = int(np.sum(mask_np > 0))
                                job["total_objects_found"] += 1

            else:
                # Regular labeling with text prompts
                for cls in classes:
                    if cls not in category_map:
                        continue

                    inputs = state.sam3_processor(
                        images=pil_image,
                        text=cls,
                        return_tensors="pt"
                    ).to(state.device)

                    with torch.no_grad():
                        outputs = state.sam3_model(**inputs)

                    target_sizes = [(h, w)]
                    results = state.sam3_processor.post_process_instance_segmentation(
                        outputs,
                        threshold=min_confidence,
                        mask_threshold=min_confidence,
                        target_sizes=target_sizes
                    )[0]

                    if 'masks' not in results:
                        continue

                    for mask, score in zip(results['masks'], results['scores']):
                        score_val = score.cpu().item()
                        if score_val < min_confidence:
                            continue

                        mask_np = mask.cpu().numpy().astype(np.uint8)
                        if mask_np.shape != (h, w):
                            mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

                        area = int(np.sum(mask_np > 0))
                        if area < 100:
                            continue

                        contours, _ = cv2.findContours(
                            (mask_np * 255).astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE
                        )

                        if not contours:
                            continue

                        x, y, bw, bh = cv2.boundingRect(np.concatenate(contours))

                        polygons = state.object_extractor.mask_to_polygon(
                            (mask_np * 255).astype(np.uint8),
                            simplify=simplify_polygons,
                            tolerance=2.0
                        )

                        annotation = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_map[cls],
                            "bbox": [float(x), float(y), float(bw), float(bh)],
                            "area": area,
                            "iscrowd": 0,
                        }

                        if polygons:
                            annotation["segmentation"] = polygons

                        coco_result["annotations"].append(annotation)
                        annotation_id += 1

                        job["objects_by_class"][cls] = job["objects_by_class"].get(cls, 0) + 1
                        job["total_objects_found"] += 1

            # Garbage collection and yielding
            if (img_idx + 1) % gc_interval == 0:
                del image, image_rgb, pil_image
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            # Yield to event loop
            await asyncio.sleep(0)

        except Exception as e:
            job["errors"].append(f"Error processing {img_path}: {str(e)}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save results
    job["processed_images"] = len(image_paths)
    job["processing_time_ms"] = (time.time() - start_time) * 1000

    coco_path = output_dir / "annotations.json"
    with open(coco_path, 'w') as f:
        json.dump(coco_result, f, indent=2)

    if "yolo" in job["output_formats"]:
        _export_to_yolo(coco_result, output_dir, image_paths)

    if "voc" in job["output_formats"]:
        _export_to_voc(coco_result, output_dir, image_paths)

    logger.info(f"Relabeling job {job_id} completed")


def _export_to_yolo(coco_data: Dict, output_dir: Path, image_paths: List[str]):
    """Export COCO data to YOLO format."""
    yolo_dir = output_dir / "yolo"
    yolo_dir.mkdir(exist_ok=True)

    labels_dir = yolo_dir / "labels"
    labels_dir.mkdir(exist_ok=True)

    # Build image lookup
    image_dims = {img["id"]: (img["width"], img["height"]) for img in coco_data.get("images", [])}
    image_names = {img["id"]: img["file_name"] for img in coco_data.get("images", [])}

    # Category mapping (YOLO uses 0-indexed)
    categories = coco_data.get("categories", [])
    cat_map = {c["id"]: i for i, c in enumerate(categories)}

    # Group annotations by image
    anns_by_image = {}
    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)

    # Write labels
    for img_id, anns in anns_by_image.items():
        if img_id not in image_dims:
            continue

        w, h = image_dims[img_id]
        filename = Path(image_names[img_id]).stem + ".txt"

        lines = []
        for ann in anns:
            cat_idx = cat_map.get(ann["category_id"], 0)
            bbox = ann.get("bbox", [0, 0, 0, 0])

            # Convert to YOLO format (center x, center y, width, height) normalized
            x_center = (bbox[0] + bbox[2] / 2) / w
            y_center = (bbox[1] + bbox[3] / 2) / h
            bw = bbox[2] / w
            bh = bbox[3] / h

            lines.append(f"{cat_idx} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

        with open(labels_dir / filename, 'w') as f:
            f.write('\n'.join(lines))

    # Write classes.txt
    with open(yolo_dir / "classes.txt", 'w') as f:
        for cat in categories:
            f.write(f"{cat['name']}\n")


def _export_to_voc(coco_data: Dict, output_dir: Path, image_paths: List[str]):
    """Export COCO data to Pascal VOC format."""
    voc_dir = output_dir / "voc"
    voc_dir.mkdir(exist_ok=True)

    annotations_dir = voc_dir / "Annotations"
    annotations_dir.mkdir(exist_ok=True)

    # Build lookups
    image_info = {img["id"]: img for img in coco_data.get("images", [])}
    cat_names = {c["id"]: c["name"] for c in coco_data.get("categories", [])}

    # Group annotations by image
    anns_by_image = {}
    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)

    # Write XML files
    for img_id, anns in anns_by_image.items():
        if img_id not in image_info:
            continue

        img = image_info[img_id]
        filename = Path(img["file_name"]).stem + ".xml"

        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<annotation>',
            f'  <filename>{img["file_name"]}</filename>',
            '  <size>',
            f'    <width>{img["width"]}</width>',
            f'    <height>{img["height"]}</height>',
            '    <depth>3</depth>',
            '  </size>',
        ]

        for ann in anns:
            bbox = ann.get("bbox", [0, 0, 0, 0])
            cat_name = cat_names.get(ann["category_id"], "unknown")

            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[0] + bbox[2])
            ymax = int(bbox[1] + bbox[3])

            xml_lines.extend([
                '  <object>',
                f'    <name>{cat_name}</name>',
                '    <bndbox>',
                f'      <xmin>{xmin}</xmin>',
                f'      <ymin>{ymin}</ymin>',
                f'      <xmax>{xmax}</xmax>',
                f'      <ymax>{ymax}</ymax>',
                '    </bndbox>',
                '  </object>',
            ])

        xml_lines.append('</annotation>')

        with open(annotations_dir / filename, 'w') as f:
            f.write('\n'.join(xml_lines))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info",
    )
