import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import {
  uploadReferences,
  listReferences,
  getReference,
  deleteReference,
  computeMetrics,
  compareMetrics,
  analyzeGap,
  randomizeSingle,
  randomizeBatch,
  getDomainGapInfo,
  listDomainGapJobs,
  getDomainGapJob,
  cancelDomainGapJob,
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
  fid_score: number | null
  kid_score: number | null
  kid_std: number | null
  overall_gap_score: number
  gap_level: string
  color_distribution: ColorDistribution | null
  synthetic_count: number
  real_count: number
}

export interface GapAnalysis {
  metrics: MetricsResult
  issues: GapIssue[]
  suggestions: ParameterSuggestion[]
  sample_synthetic: string[]
  sample_real: string[]
}

export interface DomainGapJob {
  job_id: string
  job_type: string
  status: string
  progress: number
  result: any
  created_at: string
  updated_at: string
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

  async function uploadReferenceSet(
    name: string,
    description: string,
    domainId: string,
    files: File[],
  ): Promise<ReferenceSet | null> {
    isUploading.value = true
    error.value = null
    try {
      const result = await uploadReferences(files, name, description, domainId)
      await fetchReferenceSets()
      return result
    } catch (e: any) {
      error.value = getErrorMessage(e, 'Failed to upload reference set')
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

  async function runAnalysis(
    syntheticDir: string,
    referenceSetId: string,
    maxImages = 50,
    currentConfig?: Record<string, any>,
  ): Promise<GapAnalysis | null> {
    isAnalyzing.value = true
    error.value = null
    try {
      const result = await analyzeGap({
        synthetic_dir: syntheticDir,
        reference_set_id: referenceSetId,
        max_images: maxImages,
        current_config: currentConfig,
      })
      latestAnalysis.value = result
      return result
    } catch (e: any) {
      error.value = getErrorMessage(e, 'Failed to analyze gap')
      return null
    } finally {
      isAnalyzing.value = false
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

  async function fetchJobs() {
    try {
      jobs.value = await listDomainGapJobs()
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

  function clearAnalysis() {
    latestAnalysis.value = null
  }

  function reset() {
    referenceSets.value = []
    selectedReferenceSetId.value = null
    latestAnalysis.value = null
    jobs.value = []
    isLoading.value = false
    isUploading.value = false
    isAnalyzing.value = false
    error.value = null
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
    removeReferenceSet,
    runAnalysis,
    runMetrics,
    runCompare,
    applyRandomization,
    applyRandomizationBatch,
    fetchJobs,
    fetchJob,
    cancelJob,
    fetchServiceInfo,
    clearAnalysis,
    reset,
  }
})
