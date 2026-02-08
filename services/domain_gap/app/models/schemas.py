"""
Domain Gap Reduction Service - Pydantic Schemas
================================================
Request/response models for domain gap metrics, analysis,
reference management, and post-processing techniques.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================

class DGRTechnique(str, Enum):
    """Available domain gap reduction techniques."""
    DOMAIN_RANDOMIZATION = "domain_randomization"
    STYLE_TRANSFER = "neural_style_transfer"
    CYCLEGAN = "cyclegan_translation"


class GapLevel(str, Enum):
    """Domain gap severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IssueSeverity(str, Enum):
    """Severity of a detected gap issue."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class IssueCategory(str, Enum):
    """Categories of domain gap issues."""
    COLOR = "color"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    TEXTURE = "texture"
    EDGES = "edges"
    NOISE = "noise"
    FREQUENCY = "frequency"


class ImpactLevel(str, Enum):
    """Expected impact of a parameter suggestion."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# =============================================================================
# Reference Management
# =============================================================================

class ReferenceImageStats(BaseModel):
    """Pre-computed statistics for a reference image set."""
    channel_means_lab: List[float] = Field(description="Mean per LAB channel [L, A, B]")
    channel_stds_lab: List[float] = Field(description="Std per LAB channel [L, A, B]")
    channel_means_rgb: List[float] = Field(description="Mean per RGB channel [R, G, B]")
    channel_stds_rgb: List[float] = Field(description="Std per RGB channel [R, G, B]")
    avg_edge_variance: float = Field(description="Average Laplacian variance (edge sharpness)")
    avg_brightness: float = Field(description="Average brightness (L channel mean)")
    image_count: int = Field(description="Number of images in set")
    inception_features_path: Optional[str] = Field(
        None, description="Path to cached Inception features (.npy)"
    )


class ReferenceImageSet(BaseModel):
    """A set of real reference images for domain gap analysis."""
    set_id: str
    name: str
    description: str = ""
    domain_id: str = Field(description="Associated domain ID")
    image_count: int = 0
    image_dir: str = Field(description="Directory containing the images")
    stats: Optional[ReferenceImageStats] = None
    created_at: datetime = Field(default_factory=datetime.now)


class ReferenceUploadResponse(BaseModel):
    """Response after uploading reference images."""
    success: bool
    set_id: str
    name: str
    image_count: int
    stats: Optional[ReferenceImageStats] = None
    message: str = ""


class ReferenceListResponse(BaseModel):
    """Response listing all reference sets."""
    sets: List[ReferenceImageSet]
    total: int


class ReferenceCreateResponse(BaseModel):
    """Response for the create phase of chunked upload."""
    success: bool
    set_id: str
    name: str
    image_dir: str = ""
    message: str = ""


class ReferenceBatchResponse(BaseModel):
    """Response after adding a batch of images to an existing set."""
    success: bool
    set_id: str
    images_added: int
    total_images: int
    message: str = ""


class ReferenceFinalizeResponse(BaseModel):
    """Response after finalizing a reference set (stats computed)."""
    success: bool
    set_id: str
    name: str
    image_count: int
    stats: Optional[ReferenceImageStats] = None
    message: str = ""


class ReferenceFromDirectoryRequest(BaseModel):
    """Create a reference set from images in a server-side directory."""
    name: str
    description: str = ""
    domain_id: str = "default"
    directory_path: str = Field(description="Server-side path containing reference images")


# =============================================================================
# Metrics
# =============================================================================

class MetricsRequest(BaseModel):
    """Request to compute domain gap metrics between two image sets."""
    synthetic_dir: str = Field(description="Directory with synthetic images")
    reference_set_id: str = Field(description="ID of the real reference image set")
    max_images: int = Field(100, ge=5, le=5000, description="Max images to sample per set")
    compute_fid: bool = True
    compute_kid: bool = True
    compute_color_distribution: bool = True


class ChannelStats(BaseModel):
    """Per-channel statistics for an image set."""
    mean: float
    std: float
    min: float
    max: float


class ColorDistribution(BaseModel):
    """Color distribution comparison between synthetic and real."""
    emd_l: float = Field(description="Earth Mover's Distance for L channel")
    emd_a: float = Field(description="Earth Mover's Distance for A channel")
    emd_b: float = Field(description="Earth Mover's Distance for B channel")
    emd_total: float = Field(description="Weighted total EMD")
    synthetic_stats: Dict[str, ChannelStats] = Field(
        description="Per-channel stats for synthetic: {L, A, B}"
    )
    real_stats: Dict[str, ChannelStats] = Field(
        description="Per-channel stats for real: {L, A, B}"
    )


class MetricsResult(BaseModel):
    """Result of domain gap metric computation."""
    fid_score: Optional[float] = Field(None, description="Fréchet Inception Distance (lower=better)")
    kid_score: Optional[float] = Field(None, description="Kernel Inception Distance (lower=better)")
    kid_std: Optional[float] = Field(None, description="KID standard deviation")
    color_distribution: Optional[ColorDistribution] = None
    overall_gap_score: float = Field(
        description="Normalized gap score 0-100 (0=identical, 100=max gap)"
    )
    gap_level: GapLevel = Field(description="Qualitative gap level")
    synthetic_count: int = Field(description="Number of synthetic images analyzed")
    real_count: int = Field(description="Number of real images analyzed")
    processing_time_ms: float = 0
    timestamp: datetime = Field(default_factory=datetime.now)


class MetricsCompareRequest(BaseModel):
    """Request to compare metrics before and after processing."""
    original_synthetic_dir: str = Field(description="Original synthetic images")
    processed_synthetic_dir: str = Field(description="Processed synthetic images")
    reference_set_id: str
    max_images: int = Field(100, ge=5, le=5000)


class MetricsCompareResponse(BaseModel):
    """Comparison of metrics before and after domain gap reduction."""
    success: bool
    before: MetricsResult
    after: MetricsResult
    improvement_pct: Dict[str, float] = Field(
        description="% improvement per metric (positive=better)"
    )
    processing_time_ms: float = 0


class MetricsHistoryEntry(BaseModel):
    """A single entry in metrics history."""
    metrics: MetricsResult
    technique_applied: Optional[str] = None
    config_snapshot: Optional[Dict[str, Any]] = None


class MetricsHistoryResponse(BaseModel):
    """Historical metrics for tracking improvement over time."""
    entries: List[MetricsHistoryEntry]
    total: int


# =============================================================================
# Gap Analysis & Advisory
# =============================================================================

class GapIssue(BaseModel):
    """A detected domain gap issue."""
    category: IssueCategory
    severity: IssueSeverity
    description: str = Field(description="Human-readable description of the issue")
    metric_name: str = Field(description="Name of the metric that detected this")
    metric_value: float = Field(description="Actual metric value")
    reference_value: float = Field(description="Expected/ideal value")


class ParameterSuggestion(BaseModel):
    """A suggested parameter adjustment to reduce domain gap."""
    parameter_path: str = Field(
        description="Dot-notation path to parameter, e.g. 'effects.color_correction.intensity'"
    )
    current_value: Optional[float] = Field(None, description="Current parameter value")
    suggested_value: float = Field(description="Suggested new value")
    reason: str = Field(description="Why this change is suggested")
    expected_impact: ImpactLevel = Field(description="Expected impact on domain gap")


class AnalyzeRequest(BaseModel):
    """Request for full domain gap analysis with suggestions."""
    synthetic_dir: str = Field(description="Directory with synthetic images")
    reference_set_id: str = Field(description="ID of real reference set")
    max_images: int = Field(50, ge=5, le=500, description="Images to analyze (smaller=faster)")
    current_config: Optional[Dict[str, Any]] = Field(
        None, description="Current pipeline configuration for parameter suggestion context"
    )


class GapAnalysis(BaseModel):
    """Complete domain gap analysis with metrics, issues, and suggestions."""
    gap_score: float = Field(description="Overall gap score 0-100")
    gap_level: GapLevel
    metrics: MetricsResult
    issues: List[GapIssue] = Field(description="Detected issues ordered by severity")
    suggestions: List[ParameterSuggestion] = Field(
        description="Parameter suggestions ordered by expected impact"
    )
    sample_synthetic_paths: List[str] = Field(
        default_factory=list, description="Paths to sample synthetic images for visual comparison"
    )
    sample_real_paths: List[str] = Field(
        default_factory=list, description="Paths to sample real images for visual comparison"
    )
    processing_time_ms: float = 0


# =============================================================================
# Domain Randomization
# =============================================================================

class RandomizationConfig(BaseModel):
    """Configuration for domain randomization post-processing."""
    num_variants: int = Field(3, ge=1, le=10, description="Variants to generate per image")
    intensity: float = Field(0.5, ge=0, le=1, description="Overall randomization intensity")
    preserve_annotations: bool = Field(True, description="Copy annotations for each variant")
    # Generic domain-agnostic parameters
    color_jitter: float = Field(0.3, ge=0, le=1, description="Color variation strength")
    brightness_range: Tuple[float, float] = Field(
        (0.7, 1.3), description="Brightness multiplier range"
    )
    contrast_range: Tuple[float, float] = Field(
        (0.8, 1.2), description="Contrast multiplier range"
    )
    saturation_range: Tuple[float, float] = Field(
        (0.7, 1.3), description="Saturation multiplier range"
    )
    noise_intensity: float = Field(0.02, ge=0, le=0.2, description="Gaussian noise sigma")
    blur_range: Tuple[float, float] = Field(
        (0.0, 1.5), description="Gaussian blur sigma range"
    )
    # Reference-based histogram matching
    reference_set_id: Optional[str] = Field(
        None, description="Reference set for histogram matching"
    )
    histogram_match_strength: float = Field(
        0.5, ge=0, le=1, description="How strongly to match reference histogram"
    )


class RandomizationApplyRequest(BaseModel):
    """Request to apply domain randomization to a single image."""
    image_path: str
    config: RandomizationConfig = Field(default_factory=RandomizationConfig)
    output_dir: str
    annotations_path: Optional[str] = None


class RandomizationBatchRequest(BaseModel):
    """Request to apply domain randomization to a batch."""
    images_dir: str
    config: RandomizationConfig = Field(default_factory=RandomizationConfig)
    output_dir: str
    annotations_dir: Optional[str] = None


class RandomizationResponse(BaseModel):
    """Response after applying domain randomization."""
    success: bool
    variants_created: int
    output_dir: str
    processing_time_ms: float = 0
    error: Optional[str] = None


# =============================================================================
# Style Transfer
# =============================================================================

class StyleTransferConfig(BaseModel):
    """Configuration for neural style transfer."""
    reference_set_id: str = Field(description="Reference set defining target style")
    style_weight: float = Field(0.6, ge=0, le=1, description="Style transfer strength")
    content_weight: float = Field(1.0, ge=0, le=2, description="Content preservation strength")
    depth_guided: bool = Field(True, description="Use depth maps to guide transfer spatially")
    preserve_structure: float = Field(
        0.8, ge=0, le=1, description="How much to preserve structural content"
    )
    color_only: bool = Field(False, description="Only transfer color, not texture patterns")


class StyleTransferApplyRequest(BaseModel):
    """Request to apply style transfer to a single image."""
    image_path: str
    config: StyleTransferConfig
    output_path: str


class StyleTransferBatchRequest(BaseModel):
    """Request to apply style transfer to a batch."""
    images_dir: str
    config: StyleTransferConfig
    output_dir: str


class StyleTransferResponse(BaseModel):
    """Response after applying style transfer."""
    success: bool
    output_path: str
    processing_time_ms: float = 0
    error: Optional[str] = None


# =============================================================================
# CycleGAN
# =============================================================================

class CycleGANTrainConfig(BaseModel):
    """Configuration for CycleGAN training."""
    reference_set_id: str = Field(description="Real images reference set")
    synthetic_images_dir: str = Field(description="Synthetic images for training")
    epochs: int = Field(100, ge=10, le=500)
    batch_size: int = Field(1, ge=1, le=8)
    learning_rate: float = Field(2e-4, gt=0)
    lambda_cycle: float = Field(10.0, ge=0, description="Cycle consistency loss weight")
    lambda_identity: float = Field(0.5, ge=0, description="Identity loss weight")
    lambda_lpips: float = Field(1.0, ge=0, description="LPIPS perceptual loss weight")
    image_size: int = Field(512, ge=256, le=1024)
    save_interval: int = Field(10, ge=1, description="Save checkpoint every N epochs")
    use_attention: bool = Field(True, description="Use attention in generator")


class CycleGANTranslateConfig(BaseModel):
    """Configuration for CycleGAN inference."""
    model_id: str = Field(description="ID of trained CycleGAN model")
    preserve_annotations: bool = Field(True, description="Copy annotations alongside")


class CycleGANTranslateRequest(BaseModel):
    """Request to translate images with CycleGAN."""
    image_path: str
    config: CycleGANTranslateConfig
    output_path: str


class CycleGANTranslateBatchRequest(BaseModel):
    """Request to translate a batch with CycleGAN."""
    images_dir: str
    config: CycleGANTranslateConfig
    output_dir: str


class CycleGANModel(BaseModel):
    """Information about a trained CycleGAN model."""
    model_id: str
    reference_set_id: str
    epochs_trained: int
    image_size: int
    final_fid: Optional[float] = None
    model_dir: str
    created_at: datetime = Field(default_factory=datetime.now)


class CycleGANModelsResponse(BaseModel):
    """Response listing trained CycleGAN models."""
    models: List[CycleGANModel]
    total: int


# =============================================================================
# Job Management
# =============================================================================

class JobResponse(BaseModel):
    """Response for a submitted async job."""
    success: bool
    job_id: str
    message: str = ""


class JobStatusResponse(BaseModel):
    """Status of an async job."""
    job_id: str
    status: str
    progress: float = 0
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    progress_details: Optional[Dict[str, Any]] = None


class JobListResponse(BaseModel):
    """List of jobs."""
    jobs: List[JobStatusResponse]
    total: int


# =============================================================================
# Service Health & Info
# =============================================================================

class HealthResponse(BaseModel):
    """Service health status."""
    status: str = Field(description="healthy|degraded|unhealthy")
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory: Optional[str] = None
    engines_loaded: Dict[str, bool] = Field(default_factory=dict)
    reference_sets_count: int = 0
    cyclegan_models_count: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)


class InfoResponse(BaseModel):
    """Service capabilities information."""
    service: str = "domain_gap"
    version: str = "1.0.0"
    techniques: List[str] = Field(default_factory=lambda: [t.value for t in DGRTechnique])
    metrics_available: List[str] = Field(
        default_factory=lambda: ["fid", "kid", "color_distribution", "edge_analysis"]
    )
    gpu_available: bool = False
    max_image_size: int = 1024
