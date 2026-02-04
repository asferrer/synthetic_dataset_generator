// Service health types
export type ServiceStatusType = 'healthy' | 'unhealthy' | 'degraded'

export interface ServiceHealth {
  name: string
  status: ServiceStatusType
  url?: string
  latency_ms?: number
  details?: Record<string, unknown>
  error?: string
}

export interface HealthStatus {
  status: ServiceStatusType
  services: ServiceHealth[]
  all_healthy: boolean
}

// Job types
export type JobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'interrupted'
export type JobSource = 'augmentation' | 'labeling' | 'extraction' | 'sam3'

export interface Job {
  job_id: string
  type: string
  status: JobStatus
  progress: number
  created_at: string
  started_at?: string
  completed_at?: string
  error?: string
  result?: Record<string, unknown>
  logs?: string[]
  source?: JobSource
}

// Dataset types
export interface DatasetInfo {
  name: string
  path: string
  num_images: number
  num_annotations: number
  num_categories: number
  categories: CategoryInfo[]
  created_at?: string
  modified_at?: string
}

export interface CategoryInfo {
  id: number
  name: string
  count: number
  supercategory?: string
}

export interface DatasetAnalysis {
  total_images: number
  total_annotations: number
  categories: CategoryInfo[]
  images_per_category: Record<string, number>
  annotations_per_image: {
    min: number
    max: number
    mean: number
    median: number
  }
  image_sizes: {
    widths: number[]
    heights: number[]
  }
}

// ============================================
// EFFECTS CONFIGURATION - Complete Backend API
// ============================================

export type LightingType = 'ambient' | 'spotlight' | 'gradient'
export type WaterClarity = 'clear' | 'murky' | 'very_murky'
export type BlendMethod = 'alpha' | 'poisson' | 'laplacian'
export type DepthCategory = 'shallow' | 'mid' | 'deep'

// Basic Augmentation Effects (traditional data augmentation)
export interface BasicAugmentationConfig {
  blur: {
    enabled: boolean
    min_radius: number  // 0-10
    max_radius: number  // 0-10
  }
  noise: {
    enabled: boolean
    min_intensity: number  // 0-0.2
    max_intensity: number  // 0-0.2
  }
  brightness: {
    enabled: boolean
    min_factor: number  // 0.5-1.5
    max_factor: number  // 0.5-1.5
  }
  contrast: {
    enabled: boolean
    min_factor: number  // 0.5-1.5
    max_factor: number  // 0.5-1.5
  }
  rotation: {
    enabled: boolean
    min_angle: number  // -180 to 180
    max_angle: number  // -180 to 180
  }
  scale: {
    enabled: boolean
    min_factor: number  // 0.5-2.0
    max_factor: number  // 0.5-2.0
  }
  flip_horizontal: {
    enabled: boolean
    probability: number  // 0-1
  }
  flip_vertical: {
    enabled: boolean
    probability: number  // 0-1
  }
}

// Color Correction Effect - Match object colors to background
export interface ColorCorrectionConfig {
  enabled: boolean
  color_intensity: number  // 0.0-1.0, default 0.12
}

// Blur Matching Effect - Match object blur to background
export interface BlurMatchingConfig {
  enabled: boolean
  blur_strength: number  // 0.0-2.0, default 0.5
}

// Lighting Effect - Add lighting effects
export interface LightingEffectConfig {
  enabled: boolean
  lighting_type: LightingType  // ambient, spotlight, gradient
  lighting_intensity: number  // 0.0-1.0, default 0.5
}

// Underwater Effect - Apply underwater color tint
export interface UnderwaterConfig {
  enabled: boolean
  underwater_intensity: number  // 0.0-1.0, default 0.15
  water_color: [number, number, number]  // BGR, default [120, 80, 20]
  water_clarity: WaterClarity  // clear, murky, very_murky
}

// Motion Blur Effect - Add directional blur
export interface MotionBlurConfig {
  enabled: boolean
  motion_blur_probability: number  // 0.0-1.0, default 0.2
  motion_blur_kernel: number  // 3-50, default 11 (must be odd)
}

// Shadows Effect - Generate dynamic shadows
export interface ShadowsConfig {
  enabled: boolean
  shadow_opacity: number  // 0.0-0.5, default 0.10
  shadow_blur: number  // 3-61, default 25 (must be odd)
}

// Caustics Effect - Add underwater light patterns
export interface CausticsConfig {
  enabled: boolean
  caustics_intensity: number  // 0.0-0.5, default 0.10
  caustics_deterministic: boolean  // default true
}

// Edge Smoothing Effect
export interface EdgeSmoothingConfig {
  enabled: boolean
  edge_feather: number  // 1-15, default 4
}

// Perspective Effect
export interface PerspectiveConfig {
  enabled: boolean
  perspective_magnitude: number  // 0.0-0.2, default 0.08
}

// Poisson Blending Configuration
export interface BlendingConfig {
  blend_method: BlendMethod  // alpha, poisson, laplacian
  use_binary_alpha: boolean  // default true
  alpha_feather_radius: number  // 0-3, default 1
}

// Complete Effects Configuration
export interface EffectsConfig {
  // Basic augmentation
  basic: BasicAugmentationConfig

  // Advanced realism effects
  color_correction: ColorCorrectionConfig
  blur_matching: BlurMatchingConfig
  lighting: LightingEffectConfig
  underwater: UnderwaterConfig
  motion_blur: MotionBlurConfig
  shadows: ShadowsConfig
  caustics: CausticsConfig
  edge_smoothing: EdgeSmoothingConfig
  perspective: PerspectiveConfig

  // Blending configuration
  blending: BlendingConfig

  // Blur budget for small objects
  max_blur_budget: number  // 3.0-20.0, default 10.0
}

// ============================================
// OBJECT PLACEMENT CONFIGURATION
// ============================================

export interface ObjectPlacementConfig {
  min_object_size_ratio: number  // 0.01-0.1, default 0.025
  absolute_min_size: number  // 10-30, default 15
  min_area_ratio: number  // 0.001-0.5, default 0.01
  max_area_ratio: number  // 0.05-1.0, default 0.3
  overlap_threshold: number  // 0.0-1.0, default 0.1
  max_objects_per_image: number  // 1-20, default 5
  min_objects_per_image: number  // 1-10, default 1
}

// ============================================
// VALIDATION CONFIGURATION
// ============================================

export interface ValidationConfig {
  // Identity validation
  validate_identity: boolean  // default false
  max_color_shift: number  // 10-100, default 50
  min_sharpness_ratio: number  // 0.05-1.0, default 0.15
  min_contrast_ratio: number  // 0.05-1.0, default 0.2

  // Quality validation
  validate_quality: boolean  // default false
  min_perceptual_quality: number  // 0.0-1.0, default 0.7
  min_anomaly_score: number  // 0.0-1.0, default 0.6

  // Physics validation
  validate_physics: boolean  // default false
  reject_invalid: boolean  // default true
}

// ============================================
// LIGHTING ESTIMATION CONFIGURATION
// ============================================

export interface LightingEstimationConfig {
  max_light_sources: number  // 1-5, default 3
  intensity_threshold: number  // 0.3-0.9, default 0.6
  estimate_hdr: boolean  // default false
  apply_water_attenuation: boolean  // default false
  depth_category: DepthCategory  // near, mid, far
}

// ============================================
// BATCH PROCESSING CONFIGURATION
// ============================================

export interface BatchConfig {
  parallel: boolean  // default true
  concurrent_limit: number  // 1-8, default 4
  vram_threshold: number  // 0.5-0.9, default 0.7
  save_pipeline_debug: boolean  // default false
}

// ============================================
// DATASET METADATA
// ============================================

export interface DatasetMetadata {
  name: string
  description: string
  version: string
  year: number
  contributor: string
  url: string
  license_name: string
  license_url: string
}

// ============================================
// GENERATION REQUEST - Complete
// ============================================

export interface GenerationRequest {
  source_dataset: string
  output_dir: string
  target_counts: Record<string, number>

  // Effects configuration
  effects_config: EffectsConfig

  // Object placement
  placement_config: ObjectPlacementConfig

  // Validation
  validation_config: ValidationConfig

  // Lighting
  lighting_config: LightingEstimationConfig

  // Batch processing
  batch_config: BatchConfig

  // Dataset metadata
  metadata: DatasetMetadata

  // Paths
  backgrounds_dir?: string

  // Feature flags
  use_depth: boolean
  use_segmentation: boolean
  depth_aware_placement: boolean
}

export interface GenerationResult {
  job_id: string
  output_path: string
  generated_images: number
  generated_annotations: number
  duration_seconds: number
}

// ============================================
// OBJECT SIZE CONFIGURATION
// ============================================

export interface ObjectSizeConfig {
  class_name: string
  real_world_size: number  // meters
  reference_distance: number  // meters, default 2.0
}

// ============================================
// LABELING TYPES
// ============================================

export type LabelingTaskType = 'detection' | 'segmentation' | 'both'
export type RelabelMode = 'add' | 'replace' | 'improve_segmentation'

export interface LabelingRequest {
  image_directories: string[]
  classes: string[]
  output_dir: string
  min_confidence?: number
  task_type?: LabelingTaskType
  use_sam2?: boolean
  box_threshold?: number
  text_threshold?: number
}

export interface RelabelingRequest {
  image_directories: string[]
  output_dir: string
  relabel_mode?: RelabelMode
  new_classes?: string[]
  min_confidence?: number
  existing_annotations?: string
  task_type?: LabelingTaskType
  use_sam2?: boolean
}

export interface LabelingResult {
  job_id: string
  output_path: string
  labeled_images: number
  total_annotations: number
  annotations_per_class: Record<string, number>
}

export interface LabelingJob {
  job_id: string
  status: JobStatus
  progress: number
  current_image?: string
  processed_images: number
  total_images: number
  annotations_created: number
  created_at: string
  started_at?: string
  completed_at?: string
  error?: string
  config: {
    classes: string[]
    task_type: LabelingTaskType
    min_confidence: number
  }
}

// ============================================
// OBJECT EXTRACTION TYPES
// ============================================

export interface ExtractionRequest {
  coco_json_path: string
  images_dir: string
  output_dir: string
  categories?: string[]
  min_size?: number
  include_masks?: boolean
  use_sam3?: boolean
  padding?: number
  deduplicate?: boolean
}

export interface ExtractionResult {
  job_id: string
  status: JobStatus
  output_path?: string
  objects_extracted: number
  categories_processed: string[]
  duration_seconds?: number
}

// ============================================
// EXPORT TYPES
// ============================================

export type ExportFormat = 'coco' | 'yolo' | 'voc'

export interface ExportRequest {
  source_path: string
  images_dir: string
  output_dir: string
  format: ExportFormat
  include_images?: boolean
  split_name?: string
}

export interface ExportResult {
  success: boolean
  format: string
  output_dir: string
  images_exported: number
  annotations_exported: number
  classes_file?: string
  yaml_file?: string
  imageset_file?: string
}

// ============================================
// COMBINE/SPLIT TYPES
// ============================================

export interface CombineRequest {
  dataset_paths: string[]
  output_dir: string
  merge_categories?: boolean
  deduplicate?: boolean
}

export interface SplitRequest {
  dataset_path: string
  output_dir: string
  train_ratio: number
  val_ratio: number
  test_ratio: number
  stratified?: boolean
  random_seed?: number
}

// ============================================
// SEGMENTATION TYPES
// ============================================

export interface SegmentationRequest {
  image_path: string
  points?: { x: number; y: number; label: number }[]
  boxes?: { x1: number; y1: number; x2: number; y2: number }[]
  text_prompt?: string
}

export interface SegmentationResult {
  masks: string[]
  scores: number[]
  bboxes: number[][]
}

// ============================================
// DOMAIN CONFIGURATION TYPES
// ============================================

export interface DomainRegion {
  id: string
  name: string
  display_name: string
  color_rgb: [number, number, number]
  sam3_prompt: string | null
  detection_heuristics?: Record<string, unknown>
}

export interface DomainObject {
  class_name: string
  display_name: string
  real_world_size_meters: number
  keywords: string[]
  physics_properties: {
    density_relative: number
    behavior?: string
  }
}

export interface DomainEffectsConfig {
  domain_specific: Array<{
    effect_id: string
    enabled_by_default: boolean
    parameters: Record<string, unknown>
  }>
  disabled: string[]
  universal_overrides: Record<string, Record<string, unknown>>
}

export interface DomainPhysicsConfig {
  physics_type: 'buoyancy' | 'aerial' | 'gravity' | 'neutral'
  medium_density: number
  float_threshold: number | null
  sink_threshold: number | null
  surface_zone: number | null
  bottom_zone: number | null
  gravity_direction: 'down' | 'up'
}

export interface DomainPreset {
  id: string
  name: string
  description: string
  icon: string
  config: Record<string, unknown>
}

export interface DomainLabelingTemplate {
  id: string
  name: string
  description: string
  icon: string
  classes: string[]
}

export interface Domain {
  domain_id: string
  name: string
  description: string
  version: string
  icon: string
  is_builtin: boolean
  regions: DomainRegion[]
  objects: DomainObject[]
  compatibility_matrix: Record<string, Record<string, number>>
  effects: DomainEffectsConfig
  physics: DomainPhysicsConfig
  presets: DomainPreset[]
  labeling_templates: DomainLabelingTemplate[]
}

export interface DomainSummary {
  domain_id: string
  name: string
  description: string
  icon: string
  version: string
  is_builtin: boolean
  region_count: number
  object_count: number
}

export interface DomainCreateRequest {
  domain_id: string
  name: string
  description?: string
  version?: string
  icon?: string
  regions: Omit<DomainRegion, 'value'>[]
  objects?: DomainObject[]
  compatibility_matrix?: Record<string, Record<string, number>>
  effects?: Partial<DomainEffectsConfig>
  physics?: Partial<DomainPhysicsConfig>
  presets?: DomainPreset[]
  labeling_templates?: DomainLabelingTemplate[]
}

export interface DomainUpdateRequest {
  name?: string
  description?: string
  version?: string
  icon?: string
  regions?: Omit<DomainRegion, 'value'>[]
  objects?: DomainObject[]
  compatibility_matrix?: Record<string, Record<string, number>>
  effects?: Partial<DomainEffectsConfig>
  physics?: Partial<DomainPhysicsConfig>
  presets?: DomainPreset[]
  labeling_templates?: DomainLabelingTemplate[]
}

export interface CompatibilityCheckRequest {
  object_class: string
  region_id: string
  domain_id?: string
}

export interface CompatibilityCheckResponse {
  object_class: string
  region_id: string
  domain_id: string
  score: number
}

// ============================================
// API RESPONSE WRAPPER
// ============================================

export interface ApiResponse<T> {
  success: boolean
  data?: T
  error?: string
  message?: string
}
