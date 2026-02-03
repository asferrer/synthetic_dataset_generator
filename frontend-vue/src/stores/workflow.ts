import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type {
  EffectsConfig,
  DatasetAnalysis,
  ObjectPlacementConfig,
  ValidationConfig,
  LightingEstimationConfig,
  BatchConfig,
  DatasetMetadata,
} from '@/types/api'
import type { COCODataset } from '@/types/coco'

// ============================================
// DEFAULT CONFIGURATIONS
// ============================================

const defaultEffectsConfig: EffectsConfig = {
  // Basic augmentation effects
  basic: {
    blur: { enabled: false, min_radius: 1, max_radius: 3 },
    noise: { enabled: false, min_intensity: 0.01, max_intensity: 0.05 },
    brightness: { enabled: true, min_factor: 0.8, max_factor: 1.2 },
    contrast: { enabled: true, min_factor: 0.9, max_factor: 1.1 },
    rotation: { enabled: true, min_angle: -15, max_angle: 15 },
    scale: { enabled: true, min_factor: 0.8, max_factor: 1.2 },
    flip_horizontal: { enabled: true, probability: 0.5 },
    flip_vertical: { enabled: false, probability: 0.3 },
  },

  // Advanced realism effects
  color_correction: { enabled: true, color_intensity: 0.12 },
  blur_matching: { enabled: true, blur_strength: 0.5 },
  lighting: { enabled: true, lighting_type: 'ambient', lighting_intensity: 0.5 },
  underwater: {
    enabled: true,
    underwater_intensity: 0.15,
    water_color: [120, 80, 20],
    water_clarity: 'clear',
  },
  motion_blur: { enabled: false, motion_blur_probability: 0.2, motion_blur_kernel: 11 },
  shadows: { enabled: true, shadow_opacity: 0.10, shadow_blur: 25 },
  caustics: { enabled: true, caustics_intensity: 0.10, caustics_deterministic: true },
  edge_smoothing: { enabled: true, edge_feather: 4 },
  perspective: { enabled: false, perspective_magnitude: 0.08 },

  // Blending configuration
  blending: {
    blend_method: 'alpha',
    use_binary_alpha: true,
    alpha_feather_radius: 1,
  },

  // Blur budget
  max_blur_budget: 10.0,
}

const defaultPlacementConfig: ObjectPlacementConfig = {
  min_object_size_ratio: 0.025,
  absolute_min_size: 15,
  min_area_ratio: 0.01,
  max_area_ratio: 0.3,
  overlap_threshold: 0.1,
  max_objects_per_image: 5,
  min_objects_per_image: 1,
}

const defaultValidationConfig: ValidationConfig = {
  validate_identity: false,
  max_color_shift: 50,
  min_sharpness_ratio: 0.15,
  min_contrast_ratio: 0.2,
  validate_quality: false,
  min_perceptual_quality: 0.7,
  min_anomaly_score: 0.6,
  validate_physics: false,
  reject_invalid: true,
}

const defaultLightingConfig: LightingEstimationConfig = {
  max_light_sources: 3,
  intensity_threshold: 0.6,
  estimate_hdr: false,
  apply_water_attenuation: false,
  depth_category: 'mid',
}

const defaultBatchConfig: BatchConfig = {
  parallel: true,
  concurrent_limit: 4,
  vram_threshold: 0.7,
  save_pipeline_debug: false,
}

const defaultMetadata: DatasetMetadata = {
  name: 'Synthetic Dataset',
  description: '',
  version: '1.0',
  year: new Date().getFullYear(),
  contributor: '',
  url: '',
  license_name: '',
  license_url: '',
}

// ============================================
// STORE DEFINITION
// ============================================

export const useWorkflowStore = defineStore('workflow', () => {
  // State
  const currentStep = ref(1)
  const completedSteps = ref<number[]>([])
  const sourceDataset = ref<COCODataset | null>(null)
  const sourceDatasetPath = ref<string | null>(null)
  const datasetAnalysis = ref<DatasetAnalysis | null>(null)

  // Configuration state
  const effectsConfig = ref<EffectsConfig>(JSON.parse(JSON.stringify(defaultEffectsConfig)))
  const placementConfig = ref<ObjectPlacementConfig>(JSON.parse(JSON.stringify(defaultPlacementConfig)))
  const validationConfig = ref<ValidationConfig>(JSON.parse(JSON.stringify(defaultValidationConfig)))
  const lightingConfig = ref<LightingEstimationConfig>(JSON.parse(JSON.stringify(defaultLightingConfig)))
  const batchConfig = ref<BatchConfig>(JSON.parse(JSON.stringify(defaultBatchConfig)))
  const metadata = ref<DatasetMetadata>(JSON.parse(JSON.stringify(defaultMetadata)))

  // Generation targets
  const balancingTargets = ref<Record<string, number>>({})

  // Paths
  const backgroundsDir = ref<string | null>(null)
  const outputDir = ref<string | null>(null)

  // Feature flags
  const useDepth = ref(true)
  const useSegmentation = ref(true)
  const depthAwarePlacement = ref(true)

  // Active job
  const activeJobId = ref<string | null>(null)

  // ============================================
  // GETTERS
  // ============================================

  const isStepCompleted = computed(() => (step: number) => completedSteps.value.includes(step))

  const canProceedToStep = computed(() => (step: number) => {
    if (step === 1) return true
    return completedSteps.value.includes(step - 1)
  })

  const hasSourceDataset = computed(() => sourceDatasetPath.value !== null || datasetAnalysis.value !== null)

  // Use categories from datasetAnalysis (set by analyzeDataset API call)
  const categories = computed(() => datasetAnalysis.value?.categories || [])

  const totalTargetImages = computed(() =>
    Object.values(balancingTargets.value).reduce((sum, val) => sum + val, 0)
  )

  // ============================================
  // ACTIONS
  // ============================================

  function setCurrentStep(step: number) {
    currentStep.value = step
  }

  function markStepCompleted(step: number) {
    if (!completedSteps.value.includes(step)) {
      completedSteps.value.push(step)
      completedSteps.value.sort((a, b) => a - b)
    }
  }

  function setSourceDataset(dataset: COCODataset, path: string) {
    sourceDataset.value = dataset
    sourceDatasetPath.value = path

    // Initialize balancing targets based on categories
    const targets: Record<string, number> = {}
    for (const cat of dataset.categories) {
      targets[cat.name] = 100
    }
    balancingTargets.value = targets
  }

  function setDatasetAnalysis(analysis: DatasetAnalysis) {
    datasetAnalysis.value = analysis

    // Initialize balancing targets from categories if they're empty or different
    if (analysis.categories && analysis.categories.length > 0) {
      const currentCats = Object.keys(balancingTargets.value)
      const newCats = analysis.categories.map(c => c.name)

      // Check if categories changed
      const catsChanged = currentCats.length !== newCats.length ||
        !newCats.every(cat => currentCats.includes(cat))

      if (catsChanged || currentCats.length === 0) {
        const targets: Record<string, number> = {}
        for (const cat of analysis.categories) {
          // Preserve existing target if category exists, otherwise default to 100
          targets[cat.name] = balancingTargets.value[cat.name] ?? 100
        }
        balancingTargets.value = targets
      }
    }
  }

  // Effects config updates
  function updateEffectsConfig(config: Partial<EffectsConfig>) {
    effectsConfig.value = { ...effectsConfig.value, ...config }
  }

  function updateBasicEffect(effectKey: string, config: any) {
    (effectsConfig.value.basic as any)[effectKey] = {
      ...(effectsConfig.value.basic as any)[effectKey],
      ...config,
    }
  }

  function updateAdvancedEffect(effectKey: string, config: any) {
    (effectsConfig.value as any)[effectKey] = {
      ...(effectsConfig.value as any)[effectKey],
      ...config,
    }
  }

  // Placement config updates
  function updatePlacementConfig(config: Partial<ObjectPlacementConfig>) {
    placementConfig.value = { ...placementConfig.value, ...config }
  }

  // Validation config updates
  function updateValidationConfig(config: Partial<ValidationConfig>) {
    validationConfig.value = { ...validationConfig.value, ...config }
  }

  // Lighting config updates
  function updateLightingConfig(config: Partial<LightingEstimationConfig>) {
    lightingConfig.value = { ...lightingConfig.value, ...config }
  }

  // Batch config updates
  function updateBatchConfig(config: Partial<BatchConfig>) {
    batchConfig.value = { ...batchConfig.value, ...config }
  }

  // Metadata updates
  function updateMetadata(data: Partial<DatasetMetadata>) {
    metadata.value = { ...metadata.value, ...data }
  }

  // Balancing targets
  function updateBalancingTarget(category: string, target: number) {
    balancingTargets.value[category] = Math.max(0, target)
  }

  function setAllTargets(value: number) {
    for (const key of Object.keys(balancingTargets.value)) {
      balancingTargets.value[key] = Math.max(0, value)
    }
  }

  // Paths
  function setBackgroundsDir(dir: string | null) {
    backgroundsDir.value = dir
  }

  function setOutputDir(dir: string) {
    outputDir.value = dir
  }

  // Feature flags
  function setUseDepth(value: boolean) {
    useDepth.value = value
  }

  function setUseSegmentation(value: boolean) {
    useSegmentation.value = value
  }

  function setDepthAwarePlacement(value: boolean) {
    depthAwarePlacement.value = value
  }

  // Active job
  function setActiveJobId(jobId: string | null) {
    activeJobId.value = jobId
  }

  // Reset functions
  function resetWorkflow() {
    currentStep.value = 1
    completedSteps.value = []
    sourceDataset.value = null
    sourceDatasetPath.value = null
    datasetAnalysis.value = null
    effectsConfig.value = JSON.parse(JSON.stringify(defaultEffectsConfig))
    placementConfig.value = JSON.parse(JSON.stringify(defaultPlacementConfig))
    validationConfig.value = JSON.parse(JSON.stringify(defaultValidationConfig))
    lightingConfig.value = JSON.parse(JSON.stringify(defaultLightingConfig))
    batchConfig.value = JSON.parse(JSON.stringify(defaultBatchConfig))
    metadata.value = JSON.parse(JSON.stringify(defaultMetadata))
    balancingTargets.value = {}
    backgroundsDir.value = null
    outputDir.value = null
    useDepth.value = true
    useSegmentation.value = true
    depthAwarePlacement.value = true
    activeJobId.value = null
  }

  function resetEffectsConfig() {
    effectsConfig.value = JSON.parse(JSON.stringify(defaultEffectsConfig))
  }

  function resetPlacementConfig() {
    placementConfig.value = JSON.parse(JSON.stringify(defaultPlacementConfig))
  }

  function resetValidationConfig() {
    validationConfig.value = JSON.parse(JSON.stringify(defaultValidationConfig))
  }

  function resetLightingConfig() {
    lightingConfig.value = JSON.parse(JSON.stringify(defaultLightingConfig))
  }

  function resetBatchConfig() {
    batchConfig.value = JSON.parse(JSON.stringify(defaultBatchConfig))
  }

  return {
    // State
    currentStep,
    completedSteps,
    sourceDataset,
    sourceDatasetPath,
    datasetAnalysis,
    effectsConfig,
    placementConfig,
    validationConfig,
    lightingConfig,
    batchConfig,
    metadata,
    balancingTargets,
    backgroundsDir,
    outputDir,
    useDepth,
    useSegmentation,
    depthAwarePlacement,
    activeJobId,

    // Getters
    isStepCompleted,
    canProceedToStep,
    hasSourceDataset,
    categories,
    totalTargetImages,

    // Actions
    setCurrentStep,
    markStepCompleted,
    setSourceDataset,
    setDatasetAnalysis,
    updateEffectsConfig,
    updateBasicEffect,
    updateAdvancedEffect,
    updatePlacementConfig,
    updateValidationConfig,
    updateLightingConfig,
    updateBatchConfig,
    updateMetadata,
    updateBalancingTarget,
    setAllTargets,
    setBackgroundsDir,
    setOutputDir,
    setUseDepth,
    setUseSegmentation,
    setDepthAwarePlacement,
    setActiveJobId,
    resetWorkflow,
    resetEffectsConfig,
    resetPlacementConfig,
    resetValidationConfig,
    resetLightingConfig,
    resetBatchConfig,
  }
}, {
  persist: true,
})
