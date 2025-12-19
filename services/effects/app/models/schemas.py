"""
Pydantic schemas for Effects Service API
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class EffectType(str, Enum):
    """Available effect types"""
    COLOR_CORRECTION = "color_correction"
    BLUR_MATCHING = "blur_matching"
    LIGHTING = "lighting"
    UNDERWATER = "underwater"
    MOTION_BLUR = "motion_blur"
    SHADOWS = "shadows"
    CAUSTICS = "caustics"
    POISSON_BLEND = "poisson_blend"
    EDGE_SMOOTHING = "edge_smoothing"
    PERSPECTIVE = "perspective"
    UPSCALING_NOISE = "upscaling_noise"


class BlendMethod(str, Enum):
    """Blending methods"""
    ALPHA = "alpha"
    POISSON = "poisson"
    LAPLACIAN = "laplacian"


class LightingType(str, Enum):
    """Lighting effect types"""
    SPOTLIGHT = "spotlight"
    GRADIENT = "gradient"
    AMBIENT = "ambient"


class EffectConfig(BaseModel):
    """Configuration for individual effects"""
    # Color correction
    color_intensity: float = Field(0.4, ge=0.0, le=1.0, description="Color correction intensity (lower preserves object colors better)")

    # Blur matching
    blur_strength: float = Field(1.0, ge=0.0, le=3.0, description="Blur strength multiplier")

    # Lighting
    lighting_type: LightingType = Field(LightingType.SPOTLIGHT, description="Type of lighting effect")
    lighting_intensity: float = Field(0.5, ge=0.0, le=1.0, description="Lighting intensity")

    # Underwater
    underwater_intensity: float = Field(0.25, ge=0.0, le=1.0, description="Underwater tint intensity")
    water_color: tuple[int, int, int] = Field((120, 80, 20), description="BGR water tint color")

    # Motion blur
    motion_blur_probability: float = Field(0.2, ge=0.0, le=1.0, description="Probability of motion blur")
    motion_blur_kernel: int = Field(15, ge=3, le=50, description="Motion blur kernel size")

    # Shadows
    shadow_opacity: float = Field(0.4, ge=0.0, le=1.0, description="Shadow opacity")
    shadow_blur: int = Field(21, ge=3, le=51, description="Shadow blur kernel size")

    # Caustics
    caustics_intensity: float = Field(0.15, ge=0.0, le=0.5, description="Caustics intensity")

    # Edge smoothing
    edge_feather: int = Field(5, ge=1, le=20, description="Edge feathering radius")


class ApplyEffectsRequest(BaseModel):
    """Request to apply effects to an image"""
    background_path: str = Field(..., description="Path to background image")
    foreground_path: Optional[str] = Field(None, description="Path to foreground/object image")
    mask_path: Optional[str] = Field(None, description="Path to mask image")
    depth_map_path: Optional[str] = Field(None, description="Path to depth map")
    output_path: Optional[str] = Field(None, description="Output path (auto-generated if not provided)")
    effects: List[EffectType] = Field(
        default=[EffectType.COLOR_CORRECTION, EffectType.BLUR_MATCHING],
        description="List of effects to apply"
    )
    config: EffectConfig = Field(default_factory=EffectConfig, description="Effect configuration")


class ApplyEffectsResponse(BaseModel):
    """Response from effects application"""
    success: bool
    input_path: str
    output_path: str
    effects_applied: List[str]
    processing_time_ms: float
    error: Optional[str] = None


class BlendRequest(BaseModel):
    """Request to blend foreground onto background"""
    background_path: str = Field(..., description="Path to background image")
    foreground_path: str = Field(..., description="Path to foreground image (RGBA)")
    mask_path: Optional[str] = Field(None, description="Optional mask path")
    output_path: Optional[str] = Field(None, description="Output path")
    method: BlendMethod = Field(BlendMethod.ALPHA, description="Blending method")
    position: Optional[tuple[int, int]] = Field(None, description="Position (x, y) to place foreground")


class BlendResponse(BaseModel):
    """Response from blending operation"""
    success: bool
    output_path: str
    method_used: str
    processing_time_ms: float
    error: Optional[str] = None


class CausticsRequest(BaseModel):
    """Request to generate caustics map"""
    width: int = Field(..., ge=64, le=4096, description="Output width")
    height: int = Field(..., ge=64, le=4096, description="Output height")
    intensity: float = Field(0.15, ge=0.0, le=0.5, description="Caustics intensity")
    output_path: Optional[str] = Field(None, description="Output path")
    use_cache: bool = Field(True, description="Use cached templates if available")


class CausticsResponse(BaseModel):
    """Response from caustics generation"""
    success: bool
    output_path: str
    size: tuple[int, int]
    from_cache: bool
    processing_time_ms: float


class TransformRequest(BaseModel):
    """Request for geometric transformations"""
    input_path: str = Field(..., description="Path to input image")
    output_path: Optional[str] = Field(None, description="Output path")
    rotation: Optional[float] = Field(None, ge=-180, le=180, description="Rotation angle in degrees")
    scale: Optional[float] = Field(None, ge=0.1, le=5.0, description="Scale factor")
    perspective_magnitude: float = Field(0.08, ge=0.0, le=0.2, description="Perspective distortion magnitude")
    apply_perspective: bool = Field(False, description="Apply random perspective transform")


class TransformResponse(BaseModel):
    """Response from transformation"""
    success: bool
    input_path: str
    output_path: str
    transformations_applied: List[str]
    processing_time_ms: float
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    caustics_cache_ready: bool
    cache_templates: int


class InfoResponse(BaseModel):
    """Service information"""
    service: str = "effects"
    version: str = "1.0.0"
    available_effects: List[str]
    blend_methods: List[str]
    endpoints: List[str]
