<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed } from 'vue'
import {
  getAllJobs,
  cancelJobBySource,
  getJobLogs,
  resumeJob,
  retryJob,
  deleteJob,
  regenerateDataset,
  cancelLabelingJob,
  cancelExtractionJob,
  cancelSam3Job,
} from '@/lib/api'
import { useUiStore } from '@/stores/ui'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseSelect from '@/components/ui/BaseSelect.vue'
import AlertBox from '@/components/common/AlertBox.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import {
  Activity,
  CheckCircle,
  XCircle,
  Clock,
  RefreshCw,
  X,
  Terminal,
  Filter,
  Play,
  RotateCcw,
  Trash2,
  FileJson,
  Sparkles,
  Tags,
  Scissors,
  Wand2,
} from 'lucide-vue-next'
import type { Job, JobStatus, JobSource } from '@/types/api'

const uiStore = useUiStore()

const loading = ref(true)
const jobs = ref<Job[]>([])
const selectedJob = ref<Job | null>(null)
const jobLogs = ref<string[]>([])
const logsLoading = ref(false)
const statusFilter = ref<string>('all')
const sourceFilter = ref<string>('all')
const error = ref<string | null>(null)
let pollingInterval: ReturnType<typeof setInterval> | null = null

const statusOptions = [
  { value: 'all', label: 'All Status' },
  { value: 'pending', label: 'Pending' },
  { value: 'running', label: 'Running' },
  { value: 'completed', label: 'Completed' },
  { value: 'failed', label: 'Failed' },
]

const sourceOptions = [
  { value: 'all', label: 'All Types' },
  { value: 'augmentation', label: 'Generation' },
  { value: 'labeling', label: 'Labeling' },
  { value: 'extraction', label: 'Extraction' },
  { value: 'sam3', label: 'SAM3' },
]

async function loadJobs() {
  try {
    const status = statusFilter.value === 'all' ? undefined : statusFilter.value
    jobs.value = await getAllJobs(status, 100)
    error.value = null
  } catch (e: any) {
    error.value = e.message || 'Failed to load jobs'
  } finally {
    loading.value = false
  }
}

async function loadJobLogs(jobId: string) {
  logsLoading.value = true
  try {
    jobLogs.value = await getJobLogs(jobId)
  } catch (e) {
    jobLogs.value = ['Failed to load logs']
  } finally {
    logsLoading.value = false
  }
}

async function handleCancelJob(job: Job) {
  try {
    await cancelJobBySource(job.job_id, job.source || 'augmentation')
    uiStore.showSuccess('Job Cancelled', `Job ${job.job_id.slice(0, 8)} has been cancelled`)
    await loadJobs()
  } catch (e: any) {
    uiStore.showError('Cancel Failed', e.message)
  }
}

async function handleResumeJob(jobId: string) {
  try {
    const result = await resumeJob(jobId)
    uiStore.showSuccess('Job Resumed', result.message || `Job ${jobId.slice(0, 8)} has been resumed`)
    await loadJobs()
  } catch (e: any) {
    uiStore.showError('Resume Failed', e.message)
  }
}

async function handleRetryJob(jobId: string) {
  try {
    const result = await retryJob(jobId)
    uiStore.showSuccess('Job Retried', `New job created: ${result.new_job_id?.slice(0, 8) || 'unknown'}`)
    await loadJobs()
  } catch (e: any) {
    uiStore.showError('Retry Failed', e.message)
  }
}

async function handleDeleteJob(jobId: string) {
  if (!confirm('Are you sure you want to delete this job? This action cannot be undone.')) {
    return
  }
  try {
    await deleteJob(jobId)
    uiStore.showSuccess('Job Deleted', `Job ${jobId.slice(0, 8)} has been deleted`)
    if (selectedJob.value?.job_id === jobId) {
      selectedJob.value = null
      jobLogs.value = []
    }
    await loadJobs()
  } catch (e: any) {
    uiStore.showError('Delete Failed', e.message)
  }
}

async function handleRegenerateDataset(jobId: string) {
  try {
    const result = await regenerateDataset(jobId, false)
    uiStore.showSuccess('Dataset Regenerated', result.message || 'Dataset JSON has been regenerated')
  } catch (e: any) {
    uiStore.showError('Regeneration Failed', e.message)
  }
}

function selectJob(job: Job) {
  selectedJob.value = job
  loadJobLogs(job.job_id)
}

function getStatusIcon(status: JobStatus) {
  switch (status) {
    case 'completed': return CheckCircle
    case 'running': return Activity
    case 'failed': return XCircle
    default: return Clock
  }
}

function getStatusColor(status: JobStatus) {
  switch (status) {
    case 'completed': return 'text-green-400'
    case 'running': return 'text-blue-400'
    case 'failed': return 'text-red-400'
    default: return 'text-gray-400'
  }
}

function getSourceIcon(source?: string) {
  switch (source) {
    case 'augmentation': return Sparkles
    case 'labeling': return Tags
    case 'extraction': return Scissors
    case 'sam3': return Wand2
    default: return Sparkles
  }
}

function getSourceLabel(source?: string) {
  switch (source) {
    case 'augmentation': return 'Generation'
    case 'labeling': return 'Labeling'
    case 'extraction': return 'Extraction'
    case 'sam3': return 'SAM3'
    default: return 'Job'
  }
}

function getSourceColor(source?: string) {
  switch (source) {
    case 'augmentation': return 'text-purple-400'
    case 'labeling': return 'text-blue-400'
    case 'extraction': return 'text-green-400'
    case 'sam3': return 'text-yellow-400'
    default: return 'text-gray-400'
  }
}

function formatDate(dateString: string) {
  return new Date(dateString).toLocaleString()
}

const filteredJobs = computed(() => {
  let filtered = jobs.value

  // Filter by status
  if (statusFilter.value !== 'all') {
    filtered = filtered.filter(j => j.status === statusFilter.value)
  }

  // Filter by source
  if (sourceFilter.value !== 'all') {
    filtered = filtered.filter(j => j.source === sourceFilter.value)
  }

  return filtered
})

const runningJobsCount = computed(() =>
  jobs.value.filter(j => j.status === 'running').length
)

onMounted(() => {
  loadJobs()
  pollingInterval = setInterval(loadJobs, 5000)
})

onUnmounted(() => {
  if (pollingInterval) clearInterval(pollingInterval)
})
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h2 class="text-2xl font-bold text-white">Job Monitor</h2>
        <p class="mt-1 text-gray-400">
          Monitor and manage background jobs.
          <span v-if="runningJobsCount > 0" class="text-blue-400">
            {{ runningJobsCount }} running
          </span>
        </p>
      </div>
      <div class="flex items-center gap-4">
        <BaseSelect
          v-model="sourceFilter"
          :options="sourceOptions"
          class="w-36"
        />
        <BaseSelect
          v-model="statusFilter"
          :options="statusOptions"
          class="w-36"
        />
        <BaseButton variant="outline" @click="loadJobs" :disabled="loading">
          <RefreshCw :class="['h-5 w-5', loading ? 'animate-spin' : '']" />
        </BaseButton>
      </div>
    </div>

    <!-- Error -->
    <AlertBox v-if="error" type="error" :title="error" dismissible @dismiss="error = null" />

    <!-- Loading -->
    <div v-if="loading" class="flex justify-center py-12">
      <LoadingSpinner size="lg" message="Loading jobs..." />
    </div>

    <!-- Empty State -->
    <EmptyState
      v-else-if="filteredJobs.length === 0"
      :icon="Activity"
      title="No jobs found"
      description="Jobs will appear here when you start generation, export, or other operations."
    />

    <!-- Jobs Grid -->
    <div v-else class="grid gap-6 lg:grid-cols-3">
      <!-- Jobs List -->
      <div class="lg:col-span-2 space-y-3">
        <div
          v-for="job in filteredJobs"
          :key="job.job_id"
          @click="selectJob(job)"
          :class="[
            'card p-4 cursor-pointer transition-colors',
            selectedJob?.job_id === job.job_id ? 'border-primary' : 'hover:bg-gray-700/30',
          ]"
        >
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-3">
              <component
                :is="getStatusIcon(job.status)"
                :class="['h-6 w-6', getStatusColor(job.status)]"
              />
              <div>
                <div class="flex items-center gap-2">
                  <p class="font-medium text-white">{{ job.type }}</p>
                  <span :class="['flex items-center gap-1 text-xs', getSourceColor(job.source)]">
                    <component :is="getSourceIcon(job.source)" class="h-3 w-3" />
                    {{ getSourceLabel(job.source) }}
                  </span>
                </div>
                <p class="text-sm text-gray-400">{{ job.job_id.slice(0, 8) }}...</p>
              </div>
            </div>
            <div class="flex items-center gap-2">
              <span
                :class="[
                  'badge',
                  job.status === 'completed' ? 'badge-success' :
                  job.status === 'running' ? 'badge-info' :
                  job.status === 'failed' ? 'badge-error' : 'badge-gray'
                ]"
              >
                {{ job.status }}
              </span>
              <!-- Cancel button for running jobs -->
              <BaseButton
                v-if="job.status === 'running'"
                variant="danger"
                size="sm"
                title="Cancel job"
                @click.stop="handleCancelJob(job)"
              >
                <X class="h-4 w-4" />
              </BaseButton>
              <!-- Resume button for interrupted jobs (only for augmentation) -->
              <BaseButton
                v-if="(job.status === 'interrupted' || job.status === 'cancelled') && job.source === 'augmentation'"
                variant="outline"
                size="sm"
                title="Resume job"
                @click.stop="handleResumeJob(job.job_id)"
              >
                <Play class="h-4 w-4" />
              </BaseButton>
              <!-- Retry button for failed jobs (only for augmentation) -->
              <BaseButton
                v-if="job.status === 'failed' && job.source === 'augmentation'"
                variant="outline"
                size="sm"
                title="Retry job"
                @click.stop="handleRetryJob(job.job_id)"
              >
                <RotateCcw class="h-4 w-4" />
              </BaseButton>
            </div>
          </div>

          <!-- Progress bar for running jobs -->
          <div v-if="job.status === 'running'" class="mt-3">
            <div class="flex justify-between text-sm text-gray-400 mb-1">
              <span>Progress</span>
              <span>{{ job.progress }}%</span>
            </div>
            <div class="h-2 bg-background-tertiary rounded-full overflow-hidden">
              <div
                class="h-full bg-primary transition-all"
                :style="{ width: `${job.progress}%` }"
              />
            </div>
          </div>

          <p class="mt-2 text-xs text-gray-500">
            Created: {{ formatDate(job.created_at) }}
          </p>
        </div>
      </div>

      <!-- Job Details Panel -->
      <div class="card p-6 h-fit sticky top-6">
        <h3 class="text-lg font-semibold text-white mb-4">Job Details</h3>

        <div v-if="!selectedJob" class="text-center py-8 text-gray-400">
          Select a job to view details
        </div>

        <div v-else class="space-y-4">
          <div>
            <p class="text-sm text-gray-400">Job ID</p>
            <p class="text-white font-mono text-sm">{{ selectedJob.job_id }}</p>
          </div>
          <div class="grid grid-cols-2 gap-4">
            <div>
              <p class="text-sm text-gray-400">Type</p>
              <p class="text-white">{{ selectedJob.type }}</p>
            </div>
            <div>
              <p class="text-sm text-gray-400">Source</p>
              <div class="flex items-center gap-2">
                <component
                  :is="getSourceIcon(selectedJob.source)"
                  :class="['h-4 w-4', getSourceColor(selectedJob.source)]"
                />
                <span :class="getSourceColor(selectedJob.source)">{{ getSourceLabel(selectedJob.source) }}</span>
              </div>
            </div>
          </div>
          <div>
            <p class="text-sm text-gray-400">Status</p>
            <div class="flex items-center gap-2">
              <component
                :is="getStatusIcon(selectedJob.status)"
                :class="['h-4 w-4', getStatusColor(selectedJob.status)]"
              />
              <span class="text-white capitalize">{{ selectedJob.status }}</span>
            </div>
          </div>
          <div v-if="selectedJob.error">
            <p class="text-sm text-gray-400">Error</p>
            <p class="text-red-400 text-sm">{{ selectedJob.error }}</p>
          </div>

          <!-- Logs -->
          <div>
            <div class="flex items-center gap-2 mb-2">
              <Terminal class="h-4 w-4 text-gray-400" />
              <p class="text-sm text-gray-400">Logs</p>
            </div>
            <div v-if="logsLoading" class="flex justify-center py-4">
              <LoadingSpinner size="sm" />
            </div>
            <div
              v-else
              class="bg-background-primary rounded-lg p-3 max-h-60 overflow-y-auto font-mono text-xs"
            >
              <p v-for="(log, i) in jobLogs" :key="i" class="text-gray-300">
                {{ log }}
              </p>
              <p v-if="jobLogs.length === 0" class="text-gray-500">No logs available</p>
            </div>
          </div>

          <!-- Actions -->
          <div class="pt-4 border-t border-gray-700 space-y-2">
            <p class="text-sm text-gray-400 mb-2">Actions</p>
            <div class="flex flex-wrap gap-2">
              <!-- Regenerate Dataset for completed augmentation jobs -->
              <BaseButton
                v-if="selectedJob.status === 'completed' && selectedJob.source === 'augmentation'"
                variant="outline"
                size="sm"
                @click="handleRegenerateDataset(selectedJob.job_id)"
              >
                <FileJson class="h-4 w-4" />
                Regenerate Dataset
              </BaseButton>
              <!-- Resume for interrupted/cancelled augmentation jobs -->
              <BaseButton
                v-if="(selectedJob.status === 'interrupted' || selectedJob.status === 'cancelled') && selectedJob.source === 'augmentation'"
                variant="outline"
                size="sm"
                @click="handleResumeJob(selectedJob.job_id)"
              >
                <Play class="h-4 w-4" />
                Resume
              </BaseButton>
              <!-- Retry for failed augmentation jobs -->
              <BaseButton
                v-if="selectedJob.status === 'failed' && selectedJob.source === 'augmentation'"
                variant="outline"
                size="sm"
                @click="handleRetryJob(selectedJob.job_id)"
              >
                <RotateCcw class="h-4 w-4" />
                Retry
              </BaseButton>
              <!-- Delete for all non-running augmentation jobs -->
              <BaseButton
                v-if="selectedJob.status !== 'running' && selectedJob.source === 'augmentation'"
                variant="danger"
                size="sm"
                @click="handleDeleteJob(selectedJob.job_id)"
              >
                <Trash2 class="h-4 w-4" />
                Delete
              </BaseButton>
              <!-- No actions message for non-augmentation jobs -->
              <p
                v-if="selectedJob.source !== 'augmentation' && selectedJob.status !== 'running'"
                class="text-sm text-gray-500"
              >
                Actions limited for {{ getSourceLabel(selectedJob.source) }} jobs
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
