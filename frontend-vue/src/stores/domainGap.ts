import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import {
  uploadReferences,
  createReferenceSet,
  addReferenceBatch,
  finalizeReferenceSet,
  createReferenceFromDirectory,
  listReferences,
  getReference,
  deleteReference,
  computeMetrics,
  compareMetrics,
  analyzeGap,
  randomizeSingle,
  randomizeBatch,
  styleTransferSingle,
  styleTransferBatch,
  optimizeGap,
  getDomainGapInfo,
  listDomainGapJobs,
  getDomainGapJob,
  cancelDomainGapJob,
  diffusionRefine,
  diffusionRefineBatch,
  diffusionTrainLora,
  listLoraModels,
  deleteLoraModel,
  validateAnnotations,
} from '@/lib/api'

/** Extract error detail from Axios error responses */
function getErrorMessage(e: any, fallback: string): string {
  return e?.response?.data?.detail || e?.message || fallback
}

// ============================================
// Types
// ============================================

export interface ReferenceImageStats {
  channel_means_lab: number[]
  channel_stds_lab: number[]
  channel_means_rgb: number[]
  channel_stds_rgb: number[]
  avg_edge_variance: number
  avg_brightness: number
  image_count: number
}

export interface ReferenceSet {
  set_id: string
  name: string
  description: string
  domain_id: string
  image_count: number
  image_dir: string
  stats: ReferenceImageStats | null
  created_at: string
}

export interface GapIssue {
  category: string
  severity: string
  description: string
  metric_name: string
  metric_value: number
  reference_value: number
}

export interface ParameterSuggestion {
  parameter_path: string
  current_value: number | null
  suggested_value: number
  reason: string
  expected_impact: string
}

export interface ColorDistribution {
  channel_stats: Record<string, { synthetic_mean: number; synthetic_std: number; real_mean: number; real_std: number }>
  emd_scores: Record<string, number>
}

export interface MetricsResult {
  radio_mmd_score: number | null
  fd_radio_score: number | null
  fid_score: number | null
  kid_score: number | null
  kid_std: number | null
  cmmd_score: number | null
  precision: number | null
  recall: number | null
  density: number | null
  coverage: number | null
  overall_gap_score: number
  gap_level: string
  color_distribution: ColorDistribution | null
  synthetic_count: number
  real_count: number
  // v2.0 diagnostics
  sample_size_warning: string | null
  metrics_version: string
  pca_applied: boolean
  pca_dims: number | null
  sharpness_ratio: number | null
  synthetic_sharpness: number | null
  real_sharpness: number | null
}

export interface GapAnalysis {
  metrics: MetricsResult
  issues: GapIssue[]
  suggestions: ParameterSuggestion[]
  sample_synthetic: string[]
  sample_real: string[]
}

export interface UploadProgress {
  phase: 'creating' | 'uploading' | 'finalizing'
  currentBatch: number
  totalBatches: number
  filesUploaded: number
  totalFiles: number
  batchProgress: number // 0-100 for current batch network transfer
}

export interface DomainGapJob {
  job_id: string
  job_type: string
  status: string
  progress: number
  result: any
  error?: string | null
  created_at: string
  updated_at: string
  progress_details?: {
    phase: string
    phase_progress: number
    global_progress: number
  }
}

export interface AnalysisProgress {
  jobId: string
  phase: string
  phaseProgress: number
  globalProgress: number
}

export interface DiffusionRefinementConfig {
  method: 'controlnet' | 'ip_adapter' | 'lora'
  reference_set_id: string
  strength: number
  controlnet_conditioning_scale: number
  use_depth_conditioning: boolean
  ip_adapter_scale: number
  lora_model_id: string | null
  lora_weight: number
  guidance_scale: number
  num_inference_steps: number
  use_lcm: boolean
  seed: number | null
  validate_annotations: boolean
  annotation_threshold: number
  prompt: string
  negative_prompt: string
}

export interface AnnotationPreservationMetrics {
  edge_iou: number
  ssim: number
  mean_keypoint_displacement: number
  annotations_valid: boolean
}

export interface LoRAModel {
  model_id: string
  reference_set_id: string
  training_steps: number
  lora_rank: number
  resolution: number
  model_size_mb: number
  model_dir: string
  created_at: string
}

export interface DiffusionRefinementResult {
  success: boolean
  output_path: string
  method: string
  preservation_metrics: AnnotationPreservationMetrics | null
  processing_time_ms: number
  error: string | null
}

// ============================================
// Store
// ============================================

export const useDomainGapStore = defineStore('domainGap', () => {
  // ============================================
  // STATE
  // ============================================

  const referenceSets = ref<ReferenceSet[]>([])
  const selectedReferenceSetId = ref<string | null>(null)
  const latestAnalysis = ref<GapAnalysis | null>(null)
  const jobs = ref<DomainGapJob[]>([])

  // Loading states
  const isLoading = ref(false)
  const isUploading = ref(false)
  const isAnalyzing = ref(false)
  const error = ref<string | null>(null)

  // Upload progress tracking
  const uploadProgress = ref<UploadProgress | null>(null)

  // Analysis progress tracking
  const analysisProgress = ref<AnalysisProgress | null>(null)
  let _analysisPollTimer: ReturnType<typeof setInterval> | null = null

  // Diffusion refinement state
  const loraModels = ref<LoRAModel[]>([])
  const isDiffusionProcessing = ref(false)
  const isLoraTraining = ref(false)
  const diffusionPreviewBefore = ref<string | null>(null)
  const diffusionPreviewAfter = ref<string | null>(null)
  const lastPreservationMetrics = ref<AnnotationPreservationMetrics | null>(null)
  const lastDiffusionError = ref<string | null>(null)

  // ============================================
  // GETTERS
  // ============================================

  const selectedReferenceSet = computed(() =>
    referenceSets.value.find(s => s.set_id === selectedReferenceSetId.value) ?? null
  )

  const hasReferenceSets = computed(() => referenceSets.value.length > 0)

  const gapScore = computed(() => latestAnalysis.value?.metrics.overall_gap_score ?? null)

  const gapLevel = computed(() => latestAnalysis.value?.metrics.gap_level ?? null)

  const gapColor = computed(() => {
    const score = gapScore.value
    if (score === null) return 'gray'
    if (score < 30) return 'green'
    if (score < 60) return 'yellow'
    return 'red'
  })

  const activeJobs = computed(() =>
    jobs.value.filter(j => j.status === 'running' || j.status === 'pending')
  )

  const hasLoraModels = computed(() => loraModels.value.length > 0)

  // ============================================
  // ACTIONS
  // ============================================

  async function fetchReferenceSets(domainId?: string) {
    isLoading.value = true
    error.value = null
    try {
      const response = await listReferences(domainId)
      referenceSets.value = response.sets || response
    } catch (e: any) {
      error.value = getErrorMessage(e, 'Failed to fetch reference sets')
      console.error('Error fetching reference sets:', e)
    } finally {
      isLoading.value = false
    }
  }

  async function fetchReferenceSet(setId: string): Promise<ReferenceSet | null> {
    isLoading.value = true
    error.value = null
    try {
      return await getReference(setId)
    } catch (e: any) {
      error.value = getErrorMessage(e, 'Failed to fetch reference set')
      return null
    } finally {
      isLoading.value = false
    }
  }

  const BATCH_SIZE = 50

  async function uploadReferenceSet(
    name: string,
    description: string,
    domainId: string,
    files: File[],
  ): Promise<ReferenceSet | null> {
    isUploading.value = true
    error.value = null
    uploadProgress.value = null

    try {
      // Small uploads: use single-shot for backward compat
      if (files.length <= BATCH_SIZE) {
        const result = await uploadReferences(files, name, description, domainId)
        await fetchReferenceSets()
        return result
      }

      // Large uploads: use 3-phase chunked protocol
      const totalBatches = Math.ceil(files.length / BATCH_SIZE)

      // Phase 1: Create empty set
      uploadProgress.value = {
        phase: 'creating',
        currentBatch: 0,
        totalBatches,
        filesUploaded: 0,
        totalFiles: files.length,
        batchProgress: 0,
      }
      const createResult = await createReferenceSet(name, description, domainId)
      const setId = createResult.set_id

      // Phase 2: Upload batches
      uploadProgress.value.phase = 'uploading'
      for (let i = 0; i < totalBatches; i++) {
        const batch = files.slice(i * BATCH_SIZE, (i + 1) * BATCH_SIZE)
        uploadProgress.value.currentBatch = i + 1
        uploadProgress.value.batchProgress = 0

        await addReferenceBatch(setId, batch, (progressEvent: any) => {
          if (progressEvent.total && uploadProgress.value) {
            uploadProgress.value.batchProgress = Math.round(
              (progressEvent.loaded / progressEvent.total) * 100,
            )
          }
        })

        uploadProgress.value.filesUploaded = Math.min((i + 1) * BATCH_SIZE, files.length)
      }

      // Phase 3: Finalize (compute stats)
      uploadProgress.value.phase = 'finalizing'
      uploadProgress.value.batchProgress = 0
      const finalResult = await finalizeReferenceSet(setId)

      await fetchReferenceSets()
      uploadProgress.value = null
      return finalResult as any
    } catch (e: any) {
      error.value = getErrorMessage(e, 'Failed to upload reference set')
      uploadProgress.value = null
      return null
    } finally {
      isUploading.value = false
    }
  }

  async function createReferenceFromDir(
    name: string,
    description: string,
    domainId: string,
    directoryPath: string,
  ): Promise<ReferenceSet | null> {
    isUploading.value = true
    error.value = null
    try {
      const result = await createReferenceFromDirectory(name, description, domainId, directoryPath)
      await fetchReferenceSets()
      return result
    } catch (e: any) {
      error.value = getErrorMessage(e, 'Failed to create reference set from directory')
      return null
    } finally {
      isUploading.value = false
    }
  }

  async function removeReferenceSet(setId: string): Promise<boolean> {
    isLoading.value = true
    error.value = null
    try {
      await deleteReference(setId)
      referenceSets.value = referenceSets.value.filter(s => s.set_id !== setId)
      if (selectedReferenceSetId.value === setId) {
        selectedReferenceSetId.value = null
      }
      return true
    } catch (e: any) {
      error.value = getErrorMessage(e, 'Failed to delete reference set')
      return false
    } finally {
      isLoading.value = false
    }
  }

  function _stopAnalysisPoll() {
    if (_analysisPollTimer) {
      clearInterval(_analysisPollTimer)
      _analysisPollTimer = null
    }
  }

  async function runAnalysis(
    syntheticDir: string,
    referenceSetId: string,
    maxImages = 50,
    currentConfig?: Record<string, any>,
  ): Promise<GapAnalysis | null> {
    isAnalyzing.value = true
    error.value = null
    analysisProgress.value = null
    _stopAnalysisPoll()

    try {
      // 1. Start the async job
      const { job_id } = await analyzeGap({
        synthetic_dir: syntheticDir,
        reference_set_id: referenceSetId,
        max_images: maxImages,
        current_config: currentConfig,
      })

      analysisProgress.value = {
        jobId: job_id,
        phase: 'pending',
        phaseProgress: 0,
        globalProgress: 0,
      }

      // 2. Poll for progress every 2s
      return await new Promise<GapAnalysis | null>((resolve, reject) => {
        _analysisPollTimer = setInterval(async () => {
          try {
            const job = await getDomainGapJob(job_id)

            // Update progress from job details
            if (job.progress_details && analysisProgress.value) {
              analysisProgress.value.phase = job.progress_details.phase
              analysisProgress.value.phaseProgress = job.progress_details.phase_progress
              analysisProgress.value.globalProgress = job.progress_details.global_progress
            } else if (analysisProgress.value) {
              analysisProgress.value.globalProgress = job.progress ?? 0
            }

            if (job.status === 'completed') {
              _stopAnalysisPoll()
              analysisProgress.value = null
              isAnalyzing.value = false
              const result = job.result as GapAnalysis
              latestAnalysis.value = result
              resolve(result)
            } else if (job.status === 'failed' || job.status === 'cancelled') {
              _stopAnalysisPoll()
              analysisProgress.value = null
              isAnalyzing.value = false
              error.value = job.error || `Analysis ${job.status}`
              resolve(null)
            }
          } catch (pollErr: any) {
            console.error('Error polling analysis job:', pollErr)
          }
        }, 2000)
      })
    } catch (e: any) {
      error.value = getErrorMessage(e, 'Failed to start analysis')
      analysisProgress.value = null
      isAnalyzing.value = false
      return null
    }
  }

  async function runMetrics(
    syntheticDir: string,
    referenceSetId: string,
    maxImages = 100,
    options?: { computeFid?: boolean; computeKid?: boolean; computeColorDistribution?: boolean },
  ): Promise<MetricsResult | null> {
    isAnalyzing.value = true
    error.value = null
    try {
      return await computeMetrics({
        synthetic_dir: syntheticDir,
        reference_set_id: referenceSetId,
        max_images: maxImages,
        compute_fid: options?.computeFid,
        compute_kid: options?.computeKid,
        compute_color_distribution: options?.computeColorDistribution,
      })
    } catch (e: any) {
      error.value = getErrorMessage(e, 'Failed to compute metrics')
      return null
    } finally {
      isAnalyzing.value = false
    }
  }

  async function runCompare(
    syntheticDirBefore: string,
    syntheticDirAfter: string,
    referenceSetId: string,
    maxImages = 100,
  ) {
    isAnalyzing.value = true
    error.value = null
    try {
      return await compareMetrics({
        original_synthetic_dir: syntheticDirBefore,
        processed_synthetic_dir: syntheticDirAfter,
        reference_set_id: referenceSetId,
        max_images: maxImages,
      })
    } catch (e: any) {
      error.value = getErrorMessage(e, 'Failed to compare metrics')
      return null
    } finally {
      isAnalyzing.value = false
    }
  }

  async function applyRandomization(
    imagesDir: string,
    outputDir: string,
    config: Record<string, any>,
  ) {
    isLoading.value = true
    error.value = null
    try {
      return await randomizeSingle({
        image_path: imagesDir,
        config,
        output_dir: outputDir,
      })
    } catch (e: any) {
      error.value = getErrorMessage(e, 'Failed to apply randomization')
      return null
    } finally {
      isLoading.value = false
    }
  }

  async function applyRandomizationBatch(
    imagesDir: string,
    outputDir: string,
    config: Record<string, any>,
    annotationsDir?: string,
  ) {
    isLoading.value = true
    error.value = null
    try {
      const result = await randomizeBatch({
        images_dir: imagesDir,
        config,
        output_dir: outputDir,
        annotations_dir: annotationsDir,
      })
      await fetchJobs()
      return result
    } catch (e: any) {
      error.value = getErrorMessage(e, 'Failed to start batch randomization')
      return null
    } finally {
      isLoading.value = false
    }
  }

  async function applyStyleTransfer(
    imagePath: string,
    outputPath: string,
    config: {
      reference_set_id: string
      style_weight?: number
      content_weight?: number
      depth_guided?: boolean
      preserve_structure?: number
      color_only?: boolean
    },
  ) {
    isLoading.value = true
    error.value = null
    try {
      return await styleTransferSingle({
        image_path: imagePath,
        config,
        output_path: outputPath,
      })
    } catch (e: any) {
      error.value = getErrorMessage(e, 'Failed to apply style transfer')
      return null
    } finally {
      isLoading.value = false
    }
  }

  async function applyStyleTransferBatch(
    imagesDir: string,
    outputDir: string,
    config: {
      reference_set_id: string
      style_weight?: number
      content_weight?: number
      depth_guided?: boolean
      preserve_structure?: number
      color_only?: boolean
    },
  ) {
    isLoading.value = true
    error.value = null
    try {
      const result = await styleTransferBatch({
        images_dir: imagesDir,
        config,
        output_dir: outputDir,
      })
      await fetchJobs()
      return result
    } catch (e: any) {
      error.value = getErrorMessage(e, 'Failed to start style transfer batch')
      return null
    } finally {
      isLoading.value = false
    }
  }

  async function runOptimization(
    syntheticDir: string,
    referenceSetId: string,
    outputDir: string,
    options?: {
      targetGapScore?: number
      maxIterations?: number
      maxImages?: number
      techniques?: string[]
      currentConfig?: Record<string, any>
    },
  ) {
    isLoading.value = true
    error.value = null
    try {
      const result = await optimizeGap({
        synthetic_dir: syntheticDir,
        reference_set_id: referenceSetId,
        output_dir: outputDir,
        target_gap_score: options?.targetGapScore,
        max_iterations: options?.maxIterations,
        max_images: options?.maxImages,
        techniques: options?.techniques,
        current_config: options?.currentConfig,
      })
      await fetchJobs()
      return result
    } catch (e: any) {
      error.value = getErrorMessage(e, 'Failed to start optimization')
      return null
    } finally {
      isLoading.value = false
    }
  }

  async function fetchJobs() {
    try {
      const response = await listDomainGapJobs()
      jobs.value = response.jobs ?? []
    } catch (e: any) {
      console.error('Error fetching domain gap jobs:', e)
    }
  }

  async function fetchJob(jobId: string): Promise<DomainGapJob | null> {
    try {
      return await getDomainGapJob(jobId)
    } catch (e: any) {
      console.error('Error fetching job:', e)
      return null
    }
  }

  async function cancelJob(jobId: string): Promise<boolean> {
    try {
      await cancelDomainGapJob(jobId)
      await fetchJobs()
      return true
    } catch (e: any) {
      error.value = getErrorMessage(e, 'Failed to cancel job')
      return false
    }
  }

  async function fetchServiceInfo() {
    try {
      return await getDomainGapInfo()
    } catch (e: any) {
      console.error('Error fetching domain gap service info:', e)
      return null
    }
  }

  async function applyDiffusionRefine(
    imagePath: string,
    outputPath: string,
    config: DiffusionRefinementConfig,
    depthMapPath?: string,
    annotationsPath?: string,
  ): Promise<DiffusionRefinementResult | null> {
    isDiffusionProcessing.value = true
    lastDiffusionError.value = null
    lastPreservationMetrics.value = null
    try {
      const result = await diffusionRefine({
        image_path: imagePath,
        config,
        output_path: outputPath,
        depth_map_path: depthMapPath || null,
        annotations_path: annotationsPath || null,
      })
      if (result.preservation_metrics) {
        lastPreservationMetrics.value = result.preservation_metrics
      }
      return result
    } catch (e: any) {
      lastDiffusionError.value = getErrorMessage(e, 'Failed to apply diffusion refinement')
      return null
    } finally {
      isDiffusionProcessing.value = false
    }
  }

  async function applyDiffusionRefineBatch(
    imagesDir: string,
    outputDir: string,
    config: DiffusionRefinementConfig,
    annotationsDir?: string,
  ): Promise<string | null> {
    isDiffusionProcessing.value = true
    lastDiffusionError.value = null
    try {
      const result = await diffusionRefineBatch({
        images_dir: imagesDir,
        config,
        output_dir: outputDir,
        annotations_dir: annotationsDir || null,
      })
      return result.job_id || null
    } catch (e: any) {
      lastDiffusionError.value = getErrorMessage(e, 'Failed to start batch diffusion refinement')
      return null
    } finally {
      isDiffusionProcessing.value = false
    }
  }

  async function trainLora(
    referenceSetId: string,
    modelId: string,
    config?: Partial<{
      training_steps: number
      learning_rate: number
      lora_rank: number
      resolution: number
      batch_size: number
      prompt_template: string
    }>,
  ): Promise<string | null> {
    isLoraTraining.value = true
    lastDiffusionError.value = null
    try {
      const result = await diffusionTrainLora({
        reference_set_id: referenceSetId,
        model_id: modelId,
        ...config,
      })
      return result.job_id || null
    } catch (e: any) {
      lastDiffusionError.value = getErrorMessage(e, 'Failed to start LoRA training')
      return null
    } finally {
      isLoraTraining.value = false
    }
  }

  async function fetchLoraModels(): Promise<void> {
    try {
      const result = await listLoraModels()
      loraModels.value = result.models || []
    } catch (e: any) {
      console.error('Failed to fetch LoRA models:', e)
    }
  }

  async function removeLoraModel(modelId: string): Promise<boolean> {
    try {
      await deleteLoraModel(modelId)
      loraModels.value = loraModels.value.filter(m => m.model_id !== modelId)
      return true
    } catch (e: any) {
      lastDiffusionError.value = getErrorMessage(e, 'Failed to delete LoRA model')
      return false
    }
  }

  async function validateAnnotationPreservation(
    originalPath: string,
    refinedPath: string,
    threshold?: number,
  ): Promise<AnnotationPreservationMetrics | null> {
    try {
      const result = await validateAnnotations({
        original_path: originalPath,
        refined_path: refinedPath,
        threshold,
      })
      if (result.metrics) {
        lastPreservationMetrics.value = result.metrics
      }
      return result.metrics || null
    } catch (e: any) {
      lastDiffusionError.value = getErrorMessage(e, 'Failed to validate annotation preservation')
      return null
    }
  }

  function clearAnalysis() {
    latestAnalysis.value = null
  }

  function reset() {
    _stopAnalysisPoll()
    referenceSets.value = []
    selectedReferenceSetId.value = null
    latestAnalysis.value = null
    jobs.value = []
    isLoading.value = false
    isUploading.value = false
    isAnalyzing.value = false
    error.value = null
    uploadProgress.value = null
    analysisProgress.value = null
    loraModels.value = []
    isDiffusionProcessing.value = false
    isLoraTraining.value = false
    diffusionPreviewBefore.value = null
    diffusionPreviewAfter.value = null
    lastPreservationMetrics.value = null
    lastDiffusionError.value = null
  }

  // ============================================
  // RETURN
  // ============================================

  return {
    // State
    referenceSets,
    selectedReferenceSetId,
    latestAnalysis,
    jobs,
    isLoading,
    isUploading,
    isAnalyzing,
    error,
    uploadProgress,
    analysisProgress,

    // Getters
    selectedReferenceSet,
    hasReferenceSets,
    gapScore,
    gapLevel,
    gapColor,
    activeJobs,

    // Actions
    fetchReferenceSets,
    fetchReferenceSet,
    uploadReferenceSet,
    createReferenceFromDir,
    removeReferenceSet,
    runAnalysis,
    runMetrics,
    runCompare,
    applyRandomization,
    applyRandomizationBatch,
    applyStyleTransfer,
    applyStyleTransferBatch,
    runOptimization,
    fetchJobs,
    fetchJob,
    cancelJob,
    fetchServiceInfo,
    clearAnalysis,
    reset,

    // Diffusion state
    loraModels,
    isDiffusionProcessing,
    isLoraTraining,
    diffusionPreviewBefore,
    diffusionPreviewAfter,
    lastPreservationMetrics,
    lastDiffusionError,
    hasLoraModels,
    // Diffusion actions
    applyDiffusionRefine,
    applyDiffusionRefineBatch,
    trainLora,
    fetchLoraModels,
    removeLoraModel,
    validateAnnotationPreservation,
  }
})
