/**
 * API Client for Synthetic Dataset Generator
 *
 * This module provides functions to interact with the backend services.
 * Includes automatic retry with exponential backoff for failed requests.
 */

import axios, { type AxiosInstance, type AxiosError, type InternalAxiosRequestConfig } from 'axios'
import type {
  HealthStatus,
  Job,
  JobStatus,
  DatasetInfo,
  DatasetAnalysis,
  GenerationRequest,
  GenerationResult,
  LabelingRequest,
  RelabelingRequest,
  LabelingJob,
  LabelingResult,
  ExtractionRequest,
  ExtractionResult,
  ExportRequest,
  ExportResult,
  SegmentationRequest,
  SegmentationResult,
  ObjectSizeConfig,
  Domain,
  DomainSummary,
  DomainCreateRequest,
  DomainUpdateRequest,
  CompatibilityCheckRequest,
  CompatibilityCheckResponse,
} from '@/types/api'

// API Base URL - uses Gateway service
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const SEGMENTATION_URL = import.meta.env.VITE_SEGMENTATION_URL || 'http://localhost:8002'

// Retry configuration
const MAX_RETRIES = 3
const INITIAL_RETRY_DELAY = 1000 // 1 second
const MAX_RETRY_DELAY = 10000 // 10 seconds

// Extend AxiosRequestConfig to include retry metadata
interface RetryConfig extends InternalAxiosRequestConfig {
  _retryCount?: number
  _isRetry?: boolean
}

/**
 * Calculate delay with exponential backoff and jitter
 */
function calculateRetryDelay(retryCount: number): number {
  const delay = Math.min(INITIAL_RETRY_DELAY * Math.pow(2, retryCount), MAX_RETRY_DELAY)
  // Add jitter (±20%)
  const jitter = delay * 0.2 * (Math.random() * 2 - 1)
  return Math.round(delay + jitter)
}

/**
 * Check if error is retryable
 */
function isRetryableError(error: AxiosError): boolean {
  // Network errors
  if (!error.response) return true

  // Server errors (5xx) except 501 Not Implemented
  const status = error.response.status
  if (status >= 500 && status !== 501) return true

  // Too Many Requests
  if (status === 429) return true

  // Request Timeout
  if (status === 408) return true

  return false
}

/**
 * Add retry interceptor to axios instance
 */
function addRetryInterceptor(instance: AxiosInstance): void {
  instance.interceptors.response.use(
    (response) => response,
    async (error: AxiosError) => {
      const config = error.config as RetryConfig

      if (!config) {
        return Promise.reject(error)
      }

      // Initialize retry count
      config._retryCount = config._retryCount ?? 0

      // Check if we should retry
      if (config._retryCount >= MAX_RETRIES || !isRetryableError(error)) {
        return Promise.reject(error)
      }

      // Increment retry count
      config._retryCount += 1
      config._isRetry = true

      // Calculate delay
      const delay = calculateRetryDelay(config._retryCount - 1)

      console.warn(
        `Request failed (${error.response?.status || 'network error'}), ` +
        `retrying in ${delay}ms (attempt ${config._retryCount}/${MAX_RETRIES})...`
      )

      // Wait before retrying
      await new Promise((resolve) => setTimeout(resolve, delay))

      // Retry the request
      return instance.request(config)
    }
  )
}

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json',
  },
})

const segmentationApi = axios.create({
  baseURL: SEGMENTATION_URL,
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Add retry interceptors
addRetryInterceptor(api)
addRetryInterceptor(segmentationApi)

// ===========================================
// HEALTH & STATUS
// ===========================================

export async function getHealthStatus(): Promise<HealthStatus> {
  const response = await api.get('/health')
  return response.data
}

// ===========================================
// JOBS
// ===========================================

export async function getJobs(status?: string, limit = 50): Promise<Job[]> {
  const params: Record<string, any> = { limit }
  if (status) params.status = status
  const response = await api.get('/augment/jobs', { params })
  return response.data.jobs || []
}

export async function getAllJobs(status?: string, limit = 100): Promise<Job[]> {
  // Fetch from all sources and combine
  const [augmentJobs, extractionJobs, sam3Jobs, labelingJobs] = await Promise.all([
    api.get('/augment/jobs', { params: { status, limit } }).then(r => r.data.jobs || []).catch(() => []),
    segmentationApi.get('/extract/jobs').then(r => r.data.jobs || []).catch(() => []),
    segmentationApi.get('/sam3/jobs').then(r => r.data.jobs || []).catch(() => []),
    segmentationApi.get('/labeling/jobs').then(r => r.data.jobs || []).catch(() => []),
  ])

  // Add source identifier to each job
  const jobs: Job[] = [
    ...augmentJobs.map((j: any) => ({ ...j, source: 'augmentation' })),
    ...extractionJobs.map((j: any) => ({ ...j, source: 'extraction' })),
    ...sam3Jobs.map((j: any) => ({ ...j, source: 'sam3' })),
    ...labelingJobs.map((j: any) => ({ ...j, source: 'labeling' })),
  ]

  // Sort by created_at descending
  jobs.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())

  return jobs.slice(0, limit)
}

export async function getJob(jobId: string): Promise<Job> {
  const response = await api.get(`/augment/jobs/${jobId}`)
  return response.data
}

export async function getJobLogs(jobId: string, level?: string, limit = 100): Promise<string[]> {
  const params: Record<string, any> = { limit }
  if (level) params.level = level
  const response = await api.get(`/augment/jobs/${jobId}/logs`, { params })
  return response.data.logs || []
}

export async function cancelJobBySource(jobId: string, source: string): Promise<{ success: boolean; message?: string }> {
  switch (source) {
    case 'augmentation':
      return cancelJob(jobId)
    case 'labeling':
      return cancelLabelingJob(jobId)
    case 'extraction':
      return cancelExtractionJob(jobId)
    case 'sam3':
      return cancelSam3Job(jobId)
    default:
      return cancelJob(jobId)
  }
}

export async function cancelJob(jobId: string): Promise<{ success: boolean; message?: string }> {
  const response = await api.delete(`/augment/jobs/${jobId}`)
  return response.data
}

export async function resumeJob(jobId: string): Promise<{ success: boolean; message?: string; job_id?: string }> {
  const response = await api.post(`/augment/jobs/${jobId}/resume`)
  return response.data
}

export async function retryJob(jobId: string): Promise<{ success: boolean; message?: string; new_job_id?: string }> {
  const response = await api.post(`/augment/jobs/${jobId}/retry`)
  return response.data
}

export async function deleteJob(jobId: string): Promise<{ success: boolean; message?: string }> {
  const response = await api.delete(`/augment/jobs/${jobId}`)
  return response.data
}

export async function regenerateDataset(jobId: string, force = false): Promise<{ success: boolean; message?: string; coco_path?: string }> {
  const response = await api.post(`/augment/jobs/${jobId}/regenerate-dataset`, {}, { params: { force } })
  return response.data
}

// ===========================================
// DATASETS
// ===========================================

export async function listDatasets(type?: string, limit = 50): Promise<DatasetInfo[]> {
  const params: Record<string, any> = { limit }
  if (type) params.dataset_type = type
  const response = await api.get('/augment/datasets', { params })
  return response.data.datasets || []
}

export async function analyzeDataset(path: string): Promise<DatasetAnalysis> {
  const response = await api.post('/datasets/analyze', { path })
  return response.data
}

export async function uploadDataset(file: File, name?: string): Promise<{ success: boolean; path: string }> {
  const formData = new FormData()
  formData.append('file', file)
  if (name) formData.append('name', name)
  const response = await api.post('/datasets/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 300000, // 5 minutes for large files
  })
  return response.data
}

// ===========================================
// GENERATION
// ===========================================

export async function startGeneration(request: GenerationRequest): Promise<GenerationResult> {
  const response = await api.post('/augment/compose-batch', request)
  return response.data
}

// ===========================================
// EXPORT
// ===========================================

export async function exportDataset(request: ExportRequest): Promise<ExportResult> {
  const response = await api.post('/datasets/export', request)
  return response.data
}

// ===========================================
// COMBINE & SPLIT
// ===========================================

export interface CombineResult {
  success: boolean
  output_path: string
  total_images: number
  total_annotations: number
  categories: string[]
}

export async function combineDatasets(
  datasetPaths: string[],
  outputDir: string,
  mergeCategories = true,
  deduplicate = false
): Promise<CombineResult> {
  const response = await api.post('/datasets/combine', {
    dataset_paths: datasetPaths,
    output_dir: outputDir,
    merge_categories: mergeCategories,
    deduplicate,
  })
  return response.data
}

export interface SplitResult {
  success: boolean
  output_dir: string
  train_count: number
  val_count: number
  test_count: number
}

export async function splitDataset(
  datasetPath: string,
  outputDir: string,
  trainRatio: number,
  valRatio: number,
  testRatio: number,
  stratified = true,
  randomSeed?: number
): Promise<SplitResult> {
  const response = await api.post('/datasets/split', {
    dataset_path: datasetPath,
    output_dir: outputDir,
    train_ratio: trainRatio,
    val_ratio: valRatio,
    test_ratio: testRatio,
    stratified,
    random_seed: randomSeed,
  })
  return response.data
}

export interface KFoldResult {
  success: boolean
  output_dir: string
  num_folds: number
  fold_counts: number[]
}

export async function kFoldSplit(
  datasetPath: string,
  outputDir: string,
  numFolds: number,
  stratified = true,
  randomSeed?: number
): Promise<KFoldResult> {
  const response = await api.post('/datasets/kfold', {
    dataset_path: datasetPath,
    output_dir: outputDir,
    num_folds: numFolds,
    stratified,
    random_seed: randomSeed,
  })
  return response.data
}

// ===========================================
// LABEL MANAGEMENT
// ===========================================

export async function renameCategory(
  datasetPath: string,
  oldName: string,
  newName: string
): Promise<{ success: boolean; renamed_count: number }> {
  const response = await api.put('/datasets/categories/rename', {
    dataset_path: datasetPath,
    old_name: oldName,
    new_name: newName,
  })
  return response.data
}

export async function deleteCategory(
  datasetPath: string,
  categoryName: string
): Promise<{ success: boolean; deleted_count: number }> {
  const response = await api.delete('/datasets/categories/delete', {
    data: {
      dataset_path: datasetPath,
      category_name: categoryName,
    },
  })
  return response.data
}

// ===========================================
// FILE SYSTEM
// ===========================================

export interface MountPoint {
  id: string
  name: string
  path: string
  description: string
  purpose: 'input' | 'output' | 'both'
  icon: string
  is_writable: boolean
  exists: boolean
}

export interface MountPointsResponse {
  mount_points: MountPoint[]
  default_input: string
  default_output: string
}

export async function getMountPoints(): Promise<MountPointsResponse> {
  const response = await api.get('/filesystem/mount-points')
  return response.data
}

export async function listDirectories(path?: string): Promise<string[]> {
  const params: Record<string, any> = {}
  if (path) params.path = path
  const response = await api.get('/filesystem/directories', { params })
  return response.data.directories || []
}

export async function listFiles(path: string, extensions?: string | string[]): Promise<string[]> {
  const params: Record<string, any> = { path }
  if (extensions) {
    // Accept both string and array formats
    params.extensions = Array.isArray(extensions) ? extensions.join(',') : extensions
  }
  const response = await api.get('/filesystem/files', { params })
  return response.data.files || []
}

export async function checkPathExists(path: string): Promise<{ exists: boolean }> {
  try {
    const response = await api.get('/filesystem/exists', { params: { path } })
    return { exists: response.data.exists ?? false }
  } catch {
    return { exists: false }
  }
}

export function getImageUrl(imagePath: string): string {
  return `${API_BASE_URL}/filesystem/images?path=${encodeURIComponent(imagePath)}`
}

// ===========================================
// LABELING (Segmentation Service)
// ===========================================

export async function startLabeling(request: LabelingRequest): Promise<LabelingResult> {
  const response = await segmentationApi.post('/labeling/start', request)
  return response.data
}

export async function startRelabeling(request: RelabelingRequest): Promise<LabelingResult> {
  const response = await segmentationApi.post('/labeling/relabel', request)
  return response.data
}

export async function getLabelingStatus(jobId: string): Promise<LabelingJob> {
  const response = await segmentationApi.get(`/labeling/jobs/${jobId}`)
  return response.data
}

export async function getActiveLabelingJobs(): Promise<LabelingJob[]> {
  const response = await segmentationApi.get('/labeling/jobs')
  const jobs = response.data.jobs || []
  return jobs.filter((j: LabelingJob) => j.status === 'running' || j.status === 'pending')
}

export async function cancelLabelingJob(jobId: string): Promise<{ success: boolean; message?: string }> {
  const response = await segmentationApi.delete(`/labeling/jobs/${jobId}`)
  return response.data
}

export async function deleteLabelingJob(jobId: string, deleteFiles = false): Promise<{ success: boolean; message?: string }> {
  const response = await segmentationApi.post(`/labeling/jobs/${jobId}/delete`, {}, { params: { delete_files: deleteFiles } })
  return response.data
}

export interface LabelingPreview {
  filename: string
  image_data: string
  timestamp: string
}

export interface LabelingPreviewsResponse {
  job_id: string
  previews: LabelingPreview[]
  total: number
  message?: string
}

export async function getLabelingPreviews(jobId: string, limit = 10): Promise<LabelingPreviewsResponse> {
  const response = await segmentationApi.get(`/labeling/jobs/${jobId}/previews`, { params: { limit } })
  return response.data
}

// ===========================================
// EXTRACTION (Segmentation Service)
// ===========================================

export async function startExtraction(request: ExtractionRequest): Promise<ExtractionResult> {
  const response = await segmentationApi.post('/extract/objects', request)
  return response.data
}

export async function getExtractionStatus(jobId: string): Promise<ExtractionResult> {
  const response = await segmentationApi.get(`/extract/jobs/${jobId}`)
  return response.data
}

export async function cancelExtractionJob(jobId: string): Promise<{ success: boolean; message?: string }> {
  const response = await segmentationApi.delete(`/extract/jobs/${jobId}`)
  return response.data
}

// ===========================================
// SAM3 SEGMENTATION (Segmentation Service)
// ===========================================

export async function startSamSegmentation(request: SegmentationRequest): Promise<{ job_id: string }> {
  const response = await segmentationApi.post('/sam3/segment-image', request)
  return response.data
}

export async function getSamSegmentationStatus(jobId: string): Promise<SegmentationResult & { status: JobStatus }> {
  const response = await segmentationApi.get(`/sam3/jobs/${jobId}`)
  return response.data
}

export async function cancelSam3Job(jobId: string): Promise<{ success: boolean; message?: string }> {
  const response = await segmentationApi.delete(`/sam3/jobs/${jobId}`)
  return response.data
}

// ===========================================
// OBJECT SIZES
// ===========================================

export async function getObjectSizes(): Promise<ObjectSizeConfig[]> {
  const response = await api.get('/config/object-sizes')
  return response.data.sizes || []
}

export async function updateObjectSize(className: string, size: number): Promise<{ success: boolean }> {
  const response = await api.put(`/config/object-sizes/${encodeURIComponent(className)}`, { size })
  return response.data
}

export async function deleteObjectSize(className: string): Promise<{ success: boolean }> {
  const response = await api.delete(`/config/object-sizes/${encodeURIComponent(className)}`)
  return response.data
}

export async function updateMultipleObjectSizes(sizes: Record<string, number>): Promise<{ success: boolean; updated_count: number }> {
  const response = await api.post('/config/object-sizes/batch', { sizes })
  return response.data
}

// ===========================================
// SAM3 TEXT SEGMENTATION & CONVERSION
// ===========================================

export async function segmentWithText(imagePath: string, textPrompt: string): Promise<SegmentationResult> {
  const response = await segmentationApi.post('/segment-text', {
    image_path: imagePath,
    text_prompt: textPrompt,
  })
  return response.data
}

export async function sam3ConvertDataset(
  cocoJsonPath: string,
  imagesDir: string,
  outputDir: string
): Promise<{ job_id: string }> {
  const response = await segmentationApi.post('/sam3/convert-dataset', {
    coco_json_path: cocoJsonPath,
    images_dir: imagesDir,
    output_dir: outputDir,
  })
  return response.data
}

export async function getSam3JobStatus(jobId: string): Promise<Job> {
  const response = await segmentationApi.get(`/sam3/jobs/${jobId}`)
  return response.data
}

// ===========================================
// DOMAINS
// ===========================================

export async function listDomains(): Promise<DomainSummary[]> {
  const response = await api.get('/domains')
  return response.data
}

export async function getDomain(domainId: string): Promise<Domain> {
  const response = await api.get(`/domains/${domainId}`)
  return response.data
}

export async function getActiveDomain(): Promise<{ active_domain_id: string; domain: Domain }> {
  const response = await api.get('/domains/active')
  return response.data
}

export async function activateDomain(domainId: string): Promise<{ success: boolean; active_domain_id: string; message: string }> {
  const response = await api.post(`/domains/${domainId}/activate`)
  return response.data
}

export async function createDomain(request: DomainCreateRequest): Promise<Domain> {
  const response = await api.post('/domains', request)
  return response.data
}

export async function updateDomain(domainId: string, request: DomainUpdateRequest): Promise<Domain> {
  const response = await api.put(`/domains/${domainId}`, request)
  return response.data
}

export async function deleteDomain(domainId: string): Promise<{ success: boolean; message: string }> {
  const response = await api.delete(`/domains/${domainId}`)
  return response.data
}

export async function exportDomain(domainId: string): Promise<Domain> {
  const response = await api.get(`/domains/${domainId}/export`)
  return response.data
}

export async function importDomain(domainData: Domain, overwrite = false): Promise<Domain> {
  const response = await api.post('/domains/import', domainData, { params: { overwrite } })
  return response.data
}

export async function checkCompatibility(request: CompatibilityCheckRequest): Promise<CompatibilityCheckResponse> {
  const response = await api.post('/domains/compatibility', request)
  return response.data
}

export async function getDomainRegions(domainId: string): Promise<{ domain_id: string; regions: Domain['regions'] }> {
  const response = await api.get(`/domains/${domainId}/regions`)
  return response.data
}

export async function getDomainObjects(domainId: string): Promise<{ domain_id: string; objects: Domain['objects'] }> {
  const response = await api.get(`/domains/${domainId}/objects`)
  return response.data
}

export async function getDomainEffects(domainId: string): Promise<{ domain_id: string } & Domain['effects']> {
  const response = await api.get(`/domains/${domainId}/effects`)
  return response.data
}

export async function getDomainPresets(domainId: string): Promise<{ domain_id: string; presets: Domain['presets'] }> {
  const response = await api.get(`/domains/${domainId}/presets`)
  return response.data
}

export async function getDomainSam3Prompts(domainId: string): Promise<{ domain_id: string; prompts: Array<{ text: string; region_id: string }> }> {
  const response = await api.get(`/domains/${domainId}/sam3-prompts`)
  return response.data
}

export async function getDomainLabelingTemplates(domainId: string): Promise<{ domain_id: string; labeling_templates: Domain['labeling_templates'] }> {
  const response = await api.get(`/domains/${domainId}/labeling-templates`)
  return response.data
}

// ===========================================
// DOMAIN OVERRIDES (Built-in Domain Customization)
// ===========================================

export interface BuiltinOverrideRequest {
  regions?: Domain['regions']
  objects?: Domain['objects']
  compatibility_matrix?: Domain['compatibility_matrix']
  effects?: Domain['effects']
  physics?: Domain['physics']
  presets?: Domain['presets']
  labeling_templates?: Domain['labeling_templates']
  name?: string
  description?: string
  icon?: string
}

export interface BuiltinOverrideResponse {
  success: boolean
  domain_id: string
  message: string
  has_override: boolean
  domain?: Domain
}

export interface OverrideStatusResponse {
  domain_id: string
  has_override: boolean
  is_builtin: boolean
  current_version: string
  original?: Domain
}

/**
 * Create or update an override for a built-in domain.
 * This allows modifying built-in domains (like 'underwater', 'aerial_birds')
 * by creating a user-space copy that takes precedence over the original.
 */
export async function createBuiltinOverride(
  domainId: string,
  updates: BuiltinOverrideRequest
): Promise<BuiltinOverrideResponse> {
  const response = await api.post(`/domains/${domainId}/override`, updates)
  return response.data
}

/**
 * Reset a built-in domain override to its original configuration.
 * This removes any user customizations and restores the original settings.
 */
export async function resetBuiltinOverride(
  domainId: string
): Promise<BuiltinOverrideResponse> {
  const response = await api.delete(`/domains/${domainId}/override`)
  return response.data
}

/**
 * Check if a domain has a user override.
 */
export async function getOverrideStatus(domainId: string): Promise<OverrideStatusResponse> {
  const response = await api.get(`/domains/${domainId}/override-status`)
  return response.data
}

/**
 * Update SAM3 prompts for a domain's regions.
 * Convenience function that creates an override with only region updates.
 */
export async function updateDomainSam3Prompts(
  domainId: string,
  regionPrompts: Array<{ id: string; sam3_prompt: string | null }>
): Promise<BuiltinOverrideResponse> {
  // Convert to the full region update format
  const regions = regionPrompts.map(rp => ({
    id: rp.id,
    name: rp.id,
    display_name: rp.id,
    sam3_prompt: rp.sam3_prompt,
  }))

  return createBuiltinOverride(domainId, { regions })
}


// ============================================================================
// Domain Gap Reduction
// ============================================================================

/**
 * Upload reference images to create a new reference set.
 */
export async function uploadReferences(
  files: File[],
  name: string,
  description: string = '',
  domainId: string = 'default',
): Promise<any> {
  const formData = new FormData()
  files.forEach(f => formData.append('files', f))
  formData.append('name', name)
  formData.append('description', description)
  formData.append('domain_id', domainId)

  const response = await api.post('/domain-gap/references/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000,
  })
  return response.data
}

/**
 * Create an empty reference set (phase 1 of chunked upload).
 */
export async function createReferenceSet(
  name: string,
  description: string = '',
  domainId: string = 'default',
): Promise<{ success: boolean; set_id: string; name: string; image_dir: string; message: string }> {
  const formData = new FormData()
  formData.append('name', name)
  formData.append('description', description)
  formData.append('domain_id', domainId)

  const response = await api.post('/domain-gap/references/create', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 30000,
  })
  return response.data
}

/**
 * Add a batch of images to an existing reference set (phase 2 of chunked upload).
 */
export async function addReferenceBatch(
  setId: string,
  files: File[],
  onUploadProgress?: (progressEvent: any) => void,
): Promise<{ success: boolean; set_id: string; images_added: number; total_images: number; message: string }> {
  const formData = new FormData()
  files.forEach(f => formData.append('files', f))

  const response = await api.post(`/domain-gap/references/${setId}/add-batch`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 300000, // 5 min per batch
    onUploadProgress,
  })
  return response.data
}

/**
 * Finalize a reference set: compute stats (phase 3 of chunked upload).
 */
export async function finalizeReferenceSet(
  setId: string,
): Promise<{ success: boolean; set_id: string; name: string; image_count: number; stats: any; message: string }> {
  const response = await api.post(`/domain-gap/references/${setId}/finalize`, null, {
    timeout: 600000, // 10 min for stats
  })
  return response.data
}

/**
 * Create a reference set from images in a server-side directory (zero network transfer).
 */
export async function createReferenceFromDirectory(
  name: string,
  description: string = '',
  domainId: string = 'default',
  directoryPath: string,
): Promise<any> {
  const response = await api.post('/domain-gap/references/from-directory', {
    name,
    description,
    domain_id: domainId,
    directory_path: directoryPath,
  }, {
    timeout: 600000, // 10 min
  })
  return response.data
}

/**
 * List all reference sets.
 */
export async function listReferences(domainId?: string): Promise<any> {
  const params = domainId ? { domain_id: domainId } : {}
  const response = await api.get('/domain-gap/references', { params })
  return response.data
}

/**
 * Get reference set details.
 */
export async function getReference(setId: string): Promise<any> {
  const response = await api.get(`/domain-gap/references/${setId}`)
  return response.data
}

/**
 * Delete a reference set.
 */
export async function deleteReference(setId: string): Promise<any> {
  const response = await api.delete(`/domain-gap/references/${setId}`)
  return response.data
}

/**
 * Compute domain gap metrics between synthetic images and a reference set.
 */
export async function computeMetrics(request: {
  synthetic_dir: string
  reference_set_id: string
  max_images?: number
  compute_fid?: boolean
  compute_kid?: boolean
  compute_color_distribution?: boolean
}): Promise<any> {
  const response = await api.post('/domain-gap/metrics/compute', request, {
    timeout: 600000, // 10 min for FID/KID on large sets
  })
  return response.data
}

/**
 * Compare domain gap metrics before and after processing.
 */
export async function compareMetrics(request: {
  original_synthetic_dir: string
  processed_synthetic_dir: string
  reference_set_id: string
  max_images?: number
}): Promise<any> {
  const response = await api.post('/domain-gap/metrics/compare', request, {
    timeout: 600000, // 10 min — runs metrics twice (before + after)
  })
  return response.data
}

/**
 * Start domain gap analysis (async job). Returns job_id for polling.
 */
export async function analyzeGap(request: {
  synthetic_dir: string
  reference_set_id: string
  max_images?: number
  current_config?: Record<string, any>
}): Promise<{ success: boolean; job_id: string; status: string; message: string }> {
  const response = await api.post('/domain-gap/analyze', request)
  return response.data
}

/**
 * Apply domain randomization to a single image.
 */
export async function randomizeSingle(request: {
  image_path: string
  config: Record<string, any>
  output_dir: string
  annotations_path?: string
}): Promise<any> {
  const response = await api.post('/domain-gap/randomize/apply', request)
  return response.data
}

/**
 * Apply domain randomization to a batch (async job).
 */
export async function randomizeBatch(request: {
  images_dir: string
  config: Record<string, any>
  output_dir: string
  annotations_dir?: string
}): Promise<any> {
  const response = await api.post('/domain-gap/randomize/apply-batch', request)
  return response.data
}

/**
 * Get domain gap service info.
 */
export async function getDomainGapInfo(): Promise<any> {
  const response = await api.get('/domain-gap/info')
  return response.data
}

/**
 * List domain gap jobs.
 */
export async function listDomainGapJobs(): Promise<any> {
  const response = await api.get('/domain-gap/jobs')
  return response.data
}

/**
 * Get domain gap job status.
 */
export async function getDomainGapJob(jobId: string): Promise<any> {
  const response = await api.get(`/domain-gap/jobs/${jobId}`)
  return response.data
}

/**
 * Cancel a domain gap job.
 */
export async function cancelDomainGapJob(jobId: string): Promise<any> {
  const response = await api.delete(`/domain-gap/jobs/${jobId}`)
  return response.data
}
