"""
Pydantic schemas for Gateway Service API
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class ServiceStatus(str, Enum):
    """Service health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


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


class GenerationConfig(BaseModel):
    """Configuration for synthetic image generation"""
    # Depth settings
    depth_aware: bool = Field(True, description="Enable depth-aware processing")

    # Effects to apply
    effects: List[EffectType] = Field(
        default=[EffectType.COLOR_CORRECTION, EffectType.BLUR_MATCHING, EffectType.CAUSTICS],
        description="Effects to apply to generated images"
    )

    # Object placement
    max_objects: int = Field(5, ge=1, le=20, description="Maximum objects per image")
    overlap_threshold: float = Field(0.1, ge=0.0, le=1.0, description="Maximum allowed overlap")

    # Size constraints
    min_area_ratio: float = Field(0.01, ge=0.001, le=0.5, description="Minimum object area ratio")
    max_area_ratio: float = Field(0.3, ge=0.05, le=1.0, description="Maximum object area ratio")

    # Effect intensities
    color_intensity: float = Field(0.4, ge=0.0, le=1.0, description="Color correction intensity (lower preserves object colors better)")
    underwater_intensity: float = Field(0.25, ge=0.0, le=1.0)
    caustics_intensity: float = Field(0.15, ge=0.0, le=0.5)


class ObjectInfo(BaseModel):
    """Information about an object to place"""
    image_path: str = Field(..., description="Path to object image in shared volume")
    class_name: str = Field(..., description="Object class name")
    scale: Optional[float] = Field(None, ge=0.1, le=5.0, description="Scale factor (auto if None)")
    position: Optional[tuple[int, int]] = Field(None, description="Position (x, y) or auto-place")


class GenerateImageRequest(BaseModel):
    """Request to generate a single synthetic image"""
    background_path: str = Field(..., description="Path to background image")
    objects: List[ObjectInfo] = Field(default=[], description="Objects to place")
    config: GenerationConfig = Field(default_factory=GenerationConfig)
    output_path: Optional[str] = Field(None, description="Output path (auto-generated if None)")


class AnnotationBox(BaseModel):
    """Bounding box annotation"""
    x: int
    y: int
    width: int
    height: int
    class_name: str
    confidence: float = 1.0


class GenerateImageResponse(BaseModel):
    """Response from image generation"""
    success: bool
    output_path: str
    depth_map_path: Optional[str] = None
    annotations: List[AnnotationBox] = []
    objects_placed: int = 0
    effects_applied: List[str] = []
    processing_time_ms: float
    error: Optional[str] = None


class BatchGenerateRequest(BaseModel):
    """Request for batch generation"""
    backgrounds_dir: str = Field(..., description="Directory with background images")
    objects_dir: str = Field(..., description="Directory with object images")
    output_dir: str = Field("/shared/output", description="Output directory")
    num_images: int = Field(10, ge=1, le=1000, description="Number of images to generate")
    config: GenerationConfig = Field(default_factory=GenerationConfig)


class BatchGenerateResponse(BaseModel):
    """Response from batch generation"""
    success: bool
    job_id: str
    output_dir: str
    total_requested: int
    generated: int
    failed: int
    processing_time_ms: float
    error: Optional[str] = None


class ServiceHealth(BaseModel):
    """Individual service health"""
    name: str
    status: ServiceStatus
    url: str
    latency_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Aggregated health response"""
    status: ServiceStatus
    services: List[ServiceHealth]
    all_healthy: bool


class InfoResponse(BaseModel):
    """Gateway service information"""
    service: str = "gateway"
    version: str = "1.0.0"
    description: str = "API Gateway for Synthetic Dataset Generation"
    available_services: List[str]
    endpoints: List[str]
