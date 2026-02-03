/**
 * API Client for Synthetic Dataset Generator
 *
 * This module provides functions to interact with the backend services.
 */

import axios from 'axios'
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
} from '@/types/api'

// API Base URL - uses Gateway service
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const SEGMENTATION_URL = import.meta.env.VITE_SEGMENTATION_URL || 'http://localhost:8002'

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
    segmentationApi.get('/extraction/jobs').then(r => r.data.jobs || []).catch(() => []),
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
  const response = await api.post(`/augment/jobs/${jobId}/cancel`)
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
  const response = await api.post('/augment/analyze', { path })
  return response.data
}

export async function uploadDataset(file: File, name?: string): Promise<{ success: boolean; path: string }> {
  const formData = new FormData()
  formData.append('file', file)
  if (name) formData.append('name', name)
  const response = await api.post('/augment/datasets/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 300000, // 5 minutes for large files
  })
  return response.data
}

// ===========================================
// GENERATION
// ===========================================

export async function startGeneration(request: GenerationRequest): Promise<GenerationResult> {
  const response = await api.post('/augment/compose/batch', request)
  return response.data
}

// ===========================================
// EXPORT
// ===========================================

export async function exportDataset(request: ExportRequest): Promise<ExportResult> {
  const response = await api.post('/augment/export', request)
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
  const response = await api.post('/augment/combine', {
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
  const response = await api.post('/augment/split', {
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
  const response = await api.post('/augment/kfold', {
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
  const response = await api.post('/augment/categories/rename', {
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
  const response = await api.post('/augment/categories/delete', {
    dataset_path: datasetPath,
    category_name: categoryName,
  })
  return response.data
}

// ===========================================
// FILE SYSTEM
// ===========================================

export async function listDirectories(path = '/data'): Promise<string[]> {
  const response = await api.get('/filesystem/directories', { params: { path } })
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

export async function getLabelingPreviews(jobId: string, limit = 10): Promise<{ previews: any[] }> {
  const response = await segmentationApi.get(`/labeling/jobs/${jobId}/previews`, { params: { limit } })
  return response.data
}

// ===========================================
// EXTRACTION (Segmentation Service)
// ===========================================

export async function startExtraction(request: ExtractionRequest): Promise<ExtractionResult> {
  const response = await segmentationApi.post('/extraction/start', request)
  return response.data
}

export async function getExtractionStatus(jobId: string): Promise<ExtractionResult> {
  const response = await segmentationApi.get(`/extraction/jobs/${jobId}`)
  return response.data
}

export async function cancelExtractionJob(jobId: string): Promise<{ success: boolean; message?: string }> {
  const response = await segmentationApi.delete(`/extraction/jobs/${jobId}`)
  return response.data
}

// ===========================================
// SAM3 SEGMENTATION (Segmentation Service)
// ===========================================

export async function startSamSegmentation(request: SegmentationRequest): Promise<{ job_id: string }> {
  const response = await segmentationApi.post('/sam3/segment', request)
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
  const response = await api.get('/augment/object-sizes')
  return response.data.sizes || []
}

export async function updateObjectSize(config: ObjectSizeConfig): Promise<{ success: boolean }> {
  const response = await api.post('/augment/object-sizes', config)
  return response.data
}

export async function deleteObjectSize(className: string): Promise<{ success: boolean }> {
  const response = await api.delete(`/augment/object-sizes/${encodeURIComponent(className)}`)
  return response.data
}

export async function updateMultipleObjectSizes(configs: ObjectSizeConfig[]): Promise<{ success: boolean; updated_count: number }> {
  const response = await api.post('/augment/object-sizes/batch', { sizes: configs })
  return response.data
}

// ===========================================
// SAM3 TEXT SEGMENTATION & CONVERSION
// ===========================================

export async function segmentWithText(imagePath: string, textPrompt: string): Promise<SegmentationResult> {
  const response = await segmentationApi.post('/sam3/segment-text', {
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
  const response = await segmentationApi.post('/sam3/convert', {
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
