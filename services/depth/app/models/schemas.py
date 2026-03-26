"""
Pydantic schemas for Depth Service API
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class DepthModel(str, Enum):
    """Available Depth-Anything-3 models"""
    DA3_SMALL = "DA3-SMALL"
    DA3_BASE = "DA3-BASE"
    DA3_LARGE = "DA3-LARGE"
    DA3_GIANT = "DA3-GIANT"


class DepthZone(BaseModel):
    """Represents a depth zone classification"""
    zone_id: int = Field(..., description="Zone identifier (0=near, 1=mid, 2=far)")
    zone_name: str = Field(..., description="Human-readable zone name")
    depth_range: tuple[float, float] = Field(..., description="Min and max depth values for this zone")
    mask_path: Optional[str] = Field(None, description="Path to zone mask image")


class DepthEstimateRequest(BaseModel):
    """Request for single image depth estimation"""
    input_path: str = Field(..., description="Path to input image in shared volume")
    output_dir: Optional[str] = Field("/shared/depth", description="Output directory for depth map")
    normalize: bool = Field(True, description="Normalize depth values to [0, 1]")
    generate_preview: bool = Field(True, description="Generate PNG preview of depth map")
    classify_zones: bool = Field(True, description="Classify depth into near/mid/far zones")
    num_zones: int = Field(3, ge=2, le=5, description="Number of depth zones to classify")


class DepthEstimateResponse(BaseModel):
    """Response from depth estimation"""
    success: bool
    input_path: str
    depth_map_path: str = Field(..., description="Path to .npy depth map file")
    preview_path: Optional[str] = Field(None, description="Path to PNG preview")
    shape: tuple[int, int] = Field(..., description="Height x Width of depth map")
    depth_range: tuple[float, float] = Field(..., description="Min and max depth values")
    zones: Optional[List[DepthZone]] = Field(None, description="Classified depth zones")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    error: Optional[str] = None


class BatchEstimateRequest(BaseModel):
    """Request for batch depth estimation"""
    input_dir: str = Field(..., description="Directory containing input images")
    output_dir: Optional[str] = Field("/shared/depth", description="Output directory")
    normalize: bool = Field(True, description="Normalize depth values")
    generate_preview: bool = Field(True, description="Generate PNG previews")
    file_pattern: str = Field("*.jpg,*.jpeg,*.png", description="File patterns to process")
    max_images: int = Field(100, ge=1, le=500, description="Maximum number of images to process (prevents OOM)")
    skip_existing: bool = Field(True, description="Skip images that already have depth maps")


class BatchEstimateResponse(BaseModel):
    """Response from batch depth estimation"""
    success: bool
    input_dir: str
    output_dir: str
    total_images: int
    processed: int
    failed: int
    results: List[DepthEstimateResponse]
    total_time_ms: float
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status: healthy, degraded, unhealthy")
    model_loaded: bool = Field(..., description="Whether the depth model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    gpu_name: Optional[str] = Field(None, description="GPU device name")
    gpu_memory_used: Optional[str] = Field(None, description="GPU memory usage")


class InfoResponse(BaseModel):
    """Service information response"""
    service: str = "depth"
    version: str = "1.0.0"
    model: str = Field(..., description="Current model name")
    model_params: str = Field(..., description="Model parameters count")
    device: str = Field(..., description="Compute device (cuda:0, cpu)")
    supported_formats: List[str] = ["jpg", "jpeg", "png", "bmp", "tiff"]
    endpoints: List[str] = ["/estimate", "/estimate-batch", "/health", "/info"]
