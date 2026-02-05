"""
Segmentation Service Schemas
============================
Pydantic models for segmentation API.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class SceneRegionType(str, Enum):
    """Types of scene regions"""
    OPEN_WATER = "open_water"
    SEAFLOOR = "seafloor"
    SURFACE = "surface"
    VEGETATION = "vegetation"
    ROCKY = "rocky"
    SANDY = "sandy"
    MURKY = "murky"
    UNKNOWN = "unknown"


class AnalyzeSceneRequest(BaseModel):
    """Request to analyze a scene"""
    image_path: str = Field(..., description="Path to image to analyze")
    use_sam3: bool = Field(default=False, description="Use SAM3 for segmentation")


class AnalyzeSceneResponse(BaseModel):
    """Response from scene analysis"""
    success: bool
    dominant_region: str
    region_scores: Dict[str, float]
    depth_zones: Dict[str, List[float]]
    scene_brightness: float
    water_clarity: str
    color_temperature: str
    processing_time_ms: float
    region_map_base64: Optional[str] = Field(
        default=None,
        description="Region map as base64-encoded PNG (pixel values: 1=open_water, 2=seafloor, etc.)"
    )
    error: Optional[str] = None


class CompatibilityCheckRequest(BaseModel):
    """Request to check object-scene compatibility"""
    image_path: str = Field(..., description="Path to background image")
    object_class: str = Field(..., description="Object class name")
    position_x: int = Field(..., description="X position in image")
    position_y: int = Field(..., description="Y position in image")


class CompatibilityCheckResponse(BaseModel):
    """Response from compatibility check"""
    success: bool
    is_compatible: bool
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str
    suggested_region: Optional[str] = None
    error: Optional[str] = None


class SuggestPlacementRequest(BaseModel):
    """Request for placement suggestion"""
    image_path: str = Field(..., description="Path to background image")
    object_class: str = Field(..., description="Object class name")
    object_width: int = Field(..., gt=0, description="Object width in pixels")
    object_height: int = Field(..., gt=0, description="Object height in pixels")
    existing_positions: List[List[int]] = Field(
        default=[],
        description="List of [x, y] positions of existing objects"
    )
    min_distance: int = Field(default=50, description="Minimum distance from existing objects")


class SuggestPlacementResponse(BaseModel):
    """Response with suggested placement"""
    success: bool
    position_x: Optional[int] = None
    position_y: Optional[int] = None
    best_region: Optional[str] = None
    compatibility_score: float = 0.0
    error: Optional[str] = None


class SegmentTextRequest(BaseModel):
    """Request for text-driven segmentation"""
    image_path: str = Field(..., description="Path to image")
    text_prompt: str = Field(..., description="Text description of what to segment")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence threshold")


class SegmentTextResponse(BaseModel):
    """Response with segmentation mask"""
    success: bool
    mask_path: Optional[str] = None
    mask_coverage: float = 0.0  # Percentage of image covered
    confidence: float = 0.0
    processing_time_ms: float = 0.0
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    sam3_available: bool
    sam3_loading: bool = False  # True while SAM3 is being loaded in background
    sam3_load_progress: str = ""  # Progress message during loading
    sam3_load_error: Optional[str] = None  # Error message if loading failed
    gpu_available: bool
    model_loaded: bool
    version: str = "1.0.0"


# =========================================================================
# DEBUG AND EXPLAINABILITY SCHEMAS
# =========================================================================

class DebugAnalyzeRequest(BaseModel):
    """Request for debug analysis with full explainability"""
    image_path: str = Field(..., description="Path to image to analyze")
    image_id: str = Field(default=None, description="Unique identifier for this image")
    save_visualization: bool = Field(default=True, description="Save debug visualization")


class PlacementDecisionInfo(BaseModel):
    """Information about a placement decision"""
    object_class: str
    requested_position: List[int]
    region_at_position: str
    compatibility_score: float
    reason: str
    decision: str  # 'accepted', 'rejected', 'relocated'
    alternatives: List[List[float]]  # [(x, y, score), ...]


class DebugAnalyzeResponse(BaseModel):
    """Response with full debug information for explainability"""
    success: bool

    # Analysis results
    dominant_region: str
    region_scores: Dict[str, float]
    scene_brightness: float
    water_clarity: str
    color_temperature: str

    # Debug info
    analysis_method: str  # 'sam3' or 'heuristic'
    processing_time_ms: float
    sam3_prompts_used: List[str]
    region_confidences: Dict[str, float]
    decision_log: List[str]

    # Region map (SAM3 segmentation result)
    region_map_base64: Optional[str] = Field(
        default=None,
        description="Region map as base64-encoded PNG (pixel values: 1=open_water, 2=seafloor, etc.)"
    )

    # Output files
    visualization_path: Optional[str] = None
    masks_directory: Optional[str] = None
    report_path: Optional[str] = None

    error: Optional[str] = None


class DebugCompatibilityRequest(BaseModel):
    """Request for compatibility check with debug info"""
    image_path: str = Field(..., description="Path to background image")
    object_class: str = Field(..., description="Object class name")
    position_x: int = Field(..., description="X position in image")
    position_y: int = Field(..., description="Y position in image")


class DebugCompatibilityResponse(BaseModel):
    """Response with detailed compatibility debug info"""
    success: bool
    is_compatible: bool
    score: float
    reason: str

    # Placement decision details
    decision: str  # 'accepted', 'rejected', 'relocated'
    region_at_position: str
    alternatives: List[List[float]]  # Top alternative positions with scores

    error: Optional[str] = None
