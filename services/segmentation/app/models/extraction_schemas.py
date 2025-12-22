"""
Object Extraction Schemas
=========================
Pydantic models for object extraction and SAM3 tool endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime


class AnnotationType(str, Enum):
    """Types of annotation detected in COCO dataset"""
    POLYGON = "polygon"
    RLE = "rle"
    BBOX_ONLY = "bbox_only"


class ExtractionMethod(str, Enum):
    """Method used to extract object"""
    POLYGON_MASK = "polygon_mask"
    RLE_MASK = "rle_mask"
    SAM3_FROM_BBOX = "sam3_from_bbox"
    BBOX_CROP = "bbox_crop"


class JobStatus(str, Enum):
    """Status of async job"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# =============================================================================
# DATASET ANALYSIS
# =============================================================================

class CategoryInfo(BaseModel):
    """Information about a category in the dataset"""
    id: int
    name: str
    count: int
    with_segmentation: int
    bbox_only: int


class AnalyzeDatasetRequest(BaseModel):
    """Request to analyze a COCO dataset for extraction"""
    coco_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="COCO dataset as JSON object"
    )
    coco_json_path: Optional[str] = Field(
        default=None,
        description="Path to COCO JSON file"
    )
    images_dir: Optional[str] = Field(
        default=None,
        description="Directory containing images"
    )


class AnalyzeDatasetResponse(BaseModel):
    """Response from dataset analysis"""
    success: bool
    total_images: int = 0
    total_annotations: int = 0
    annotations_with_segmentation: int = 0
    annotations_bbox_only: int = 0
    categories: List[CategoryInfo] = []
    recommendation: str = Field(
        default="",
        description="'use_masks' if all have segmentation, 'use_sam3' if none, 'mixed' if both"
    )
    sample_annotation: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# =============================================================================
# OBJECT EXTRACTION
# =============================================================================

class ExtractObjectsRequest(BaseModel):
    """Request to extract objects from dataset"""
    coco_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="COCO dataset as JSON object"
    )
    coco_json_path: Optional[str] = Field(
        default=None,
        description="Path to COCO JSON file"
    )
    images_dir: str = Field(..., description="Directory containing source images")
    output_dir: str = Field(..., description="Output directory for extracted objects")
    categories_to_extract: List[str] = Field(
        default=[],
        description="Categories to extract (empty = all)"
    )
    use_sam3_for_bbox: bool = Field(
        default=True,
        description="Use SAM3 to segment objects that only have bounding boxes"
    )
    padding: int = Field(
        default=5,
        ge=0,
        le=50,
        description="Pixels of padding around extracted objects"
    )
    min_object_area: int = Field(
        default=100,
        description="Minimum object area in pixels to extract"
    )
    save_individual_coco: bool = Field(
        default=True,
        description="Save individual COCO JSON for each extracted object"
    )


class ExtractObjectsResponse(BaseModel):
    """Response from object extraction request"""
    success: bool
    job_id: str = ""
    status: JobStatus = JobStatus.QUEUED
    message: str = ""
    error: Optional[str] = None


class ExtractedObjectInfo(BaseModel):
    """Information about a single extracted object"""
    annotation_id: int
    category_name: str
    image_path: str
    json_path: Optional[str] = None
    method: ExtractionMethod
    original_bbox: List[float]
    extracted_size: List[int]  # [width, height]


class ExtractionJobStatus(BaseModel):
    """Status of an extraction job"""
    job_id: str
    status: JobStatus
    total_objects: int = 0
    extracted_objects: int = 0
    failed_objects: int = 0
    current_category: str = ""
    categories_progress: Dict[str, int] = {}
    output_dir: str = ""
    errors: List[str] = []
    extracted_files: List[ExtractedObjectInfo] = []
    processing_time_ms: float = 0.0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class ExtractSingleObjectRequest(BaseModel):
    """Request to extract a single object (for preview)"""
    image_path: Optional[str] = Field(
        default=None,
        description="Path to source image"
    )
    image_base64: Optional[str] = Field(
        default=None,
        description="Image as base64 string (alternative to path)"
    )
    annotation: Dict[str, Any] = Field(..., description="COCO annotation for the object")
    category_name: str = Field(..., description="Category name of the object")
    use_sam3: bool = Field(
        default=False,
        description="Use SAM3 to segment (if annotation has no mask)"
    )
    padding: int = Field(default=5, ge=0, le=50)


class ExtractSingleObjectResponse(BaseModel):
    """Response with extracted object preview"""
    success: bool
    cropped_image_base64: str = ""
    mask_base64: Optional[str] = None
    annotation_type: AnnotationType = AnnotationType.BBOX_ONLY
    method_used: ExtractionMethod = ExtractionMethod.BBOX_CROP
    original_bbox: List[float] = []
    extracted_size: List[int] = []  # [width, height]
    mask_coverage: float = 0.0  # Percentage of bbox covered by mask
    processing_time_ms: float = 0.0
    error: Optional[str] = None


# =============================================================================
# SAM3 TOOL - INDIVIDUAL SEGMENTATION
# =============================================================================

class SAM3SegmentImageRequest(BaseModel):
    """Request to segment a single image with SAM3"""
    image_path: Optional[str] = Field(
        default=None,
        description="Path to image file"
    )
    image_base64: Optional[str] = Field(
        default=None,
        description="Image as base64 string"
    )
    bbox: Optional[List[float]] = Field(
        default=None,
        description="Bounding box [x, y, width, height] to segment within"
    )
    point: Optional[List[int]] = Field(
        default=None,
        description="Point [x, y] to segment around"
    )
    text_prompt: Optional[str] = Field(
        default=None,
        description="Text prompt for text-driven segmentation"
    )
    return_polygon: bool = Field(
        default=True,
        description="Return segmentation as polygon coordinates"
    )
    return_mask: bool = Field(
        default=True,
        description="Return segmentation mask as base64 PNG"
    )
    simplify_polygon: bool = Field(
        default=True,
        description="Simplify polygon to reduce points"
    )
    simplify_tolerance: float = Field(
        default=2.0,
        description="Tolerance for polygon simplification"
    )


class SAM3SegmentImageResponse(BaseModel):
    """Response with SAM3 segmentation result"""
    success: bool
    mask_base64: Optional[str] = None
    segmentation_polygon: Optional[List[List[float]]] = Field(
        default=None,
        description="Polygon as list of [x, y] coordinates"
    )
    segmentation_coco: Optional[List[List[float]]] = Field(
        default=None,
        description="Segmentation in COCO format (flat list)"
    )
    bbox: List[float] = Field(
        default=[],
        description="Bounding box of segmented region [x, y, w, h]"
    )
    area: int = 0
    confidence: float = 0.0
    processing_time_ms: float = 0.0
    error: Optional[str] = None


# =============================================================================
# SAM3 TOOL - DATASET CONVERSION
# =============================================================================

class SAM3ConvertDatasetRequest(BaseModel):
    """Request to convert bbox annotations to segmentations using SAM3"""
    coco_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="COCO dataset as JSON object"
    )
    coco_json_path: Optional[str] = Field(
        default=None,
        description="Path to input COCO JSON file"
    )
    images_dir: str = Field(..., description="Directory containing images")
    output_path: str = Field(..., description="Path for output COCO JSON with segmentations")
    categories_to_convert: List[str] = Field(
        default=[],
        description="Categories to convert (empty = all)"
    )
    overwrite_existing: bool = Field(
        default=False,
        description="Overwrite existing segmentations"
    )
    simplify_polygons: bool = Field(
        default=True,
        description="Simplify generated polygons"
    )
    simplify_tolerance: float = Field(
        default=2.0,
        description="Tolerance for polygon simplification"
    )


class SAM3ConvertDatasetResponse(BaseModel):
    """Response from dataset conversion request"""
    success: bool
    job_id: str = ""
    status: JobStatus = JobStatus.QUEUED
    message: str = ""
    error: Optional[str] = None


class SAM3ConversionJobStatus(BaseModel):
    """Status of a SAM3 conversion job"""
    job_id: str
    status: JobStatus
    total_annotations: int = 0
    converted_annotations: int = 0
    skipped_annotations: int = 0
    failed_annotations: int = 0
    current_image: str = ""
    categories_progress: Dict[str, int] = {}
    output_path: str = ""
    errors: List[str] = []
    processing_time_ms: float = 0.0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


# =============================================================================
# EXTRACTION SUMMARY
# =============================================================================

class ExtractionSummary(BaseModel):
    """Summary of extraction job for saving to file"""
    extraction_date: str
    source_dataset: str
    images_dir: str
    output_dir: str
    total_objects_extracted: int
    categories: Dict[str, Dict[str, Any]]  # {name: {count, method}}
    failed_extractions: int
    errors: List[str]
    processing_time_seconds: float
