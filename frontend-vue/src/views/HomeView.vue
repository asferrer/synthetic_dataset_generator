<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { getHealthStatus, getJobs, listDatasets } from '@/lib/api'
import MetricCard from '@/components/common/MetricCard.vue'
import AlertBox from '@/components/common/AlertBox.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import {
  Database,
  Image,
  Activity,
  Server,
  Sparkles,
  ArrowRight,
  CheckCircle,
  Clock,
  AlertCircle,
} from 'lucide-vue-next'
import type { HealthStatus, Job, DatasetInfo } from '@/types/api'

const router = useRouter()

const loading = ref(true)
const health = ref<HealthStatus | null>(null)
const recentJobs = ref<Job[]>([])
const datasets = ref<DatasetInfo[]>([])
const error = ref<string | null>(null)

async function fetchData() {
  loading.value = true
  error.value = null

  try {
    const [healthData, jobsData, datasetsData] = await Promise.all([
      getHealthStatus().catch(() => null),
      getJobs(undefined, 5).catch(() => []),
      listDatasets().catch(() => []),
    ])

    health.value = healthData
    recentJobs.value = jobsData
    datasets.value = datasetsData
  } catch (e) {
    error.value = 'Failed to load dashboard data'
  } finally {
    loading.value = false
  }
}

function getHealthyServices(): number {
  if (!health.value) return 0
  return Object.values(health.value).filter(s => s.status === 'healthy').length
}

function getTotalServices(): number {
  if (!health.value) return 5
  return Object.keys(health.value).length
}

function getJobStatusIcon(status: string) {
  switch (status) {
    case 'completed':
      return CheckCircle
    case 'running':
      return Activity
    case 'failed':
      return AlertCircle
    default:
      return Clock
  }
}

function getJobStatusColor(status: string) {
  switch (status) {
    case 'completed':
      return 'text-green-400'
    case 'running':
      return 'text-blue-400'
    case 'failed':
      return 'text-red-400'
    default:
      return 'text-gray-400'
  }
}

const workflowSteps = [
  { name: 'Analysis', description: 'Analyze your dataset structure', path: '/analysis' },
  { name: 'Configure', description: 'Set up augmentation effects', path: '/configure' },
  { name: 'Source', description: 'Select source images', path: '/source-selection' },
  { name: 'Generate', description: 'Create synthetic data', path: '/generation' },
  { name: 'Export', description: 'Export in various formats', path: '/export' },
]

onMounted(fetchData)
</script>

<template>
  <div class="space-y-8">
    <!-- Welcome Header -->
    <div class="flex items-start justify-between">
      <div>
        <h1 class="text-3xl font-bold text-white">Welcome Back</h1>
        <p class="mt-2 text-gray-400">
          Manage your synthetic datasets and monitor generation jobs
        </p>
      </div>
      <button
        @click="router.push('/analysis')"
        class="btn-primary flex items-center gap-2"
      >
        <Sparkles class="h-5 w-5" />
        Start New Generation
      </button>
    </div>

    <!-- Loading State -->
    <div v-if="loading" class="flex justify-center py-12">
      <LoadingSpinner size="lg" message="Loading dashboard..." />
    </div>

    <!-- Error State -->
    <AlertBox v-else-if="error" type="error" :title="error" dismissible @dismiss="error = null">
      Please check your connection to the backend services.
    </AlertBox>

    <!-- Dashboard Content -->
    <template v-else>
      <!-- Metrics Grid -->
      <div class="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Total Datasets"
          :value="datasets.length"
          :icon="Database"
          variant="info"
        />
        <MetricCard
          title="Total Images"
          :value="datasets.reduce((sum, d) => sum + d.num_images, 0)"
          :icon="Image"
        />
        <MetricCard
          title="Active Jobs"
          :value="recentJobs.filter(j => j.status === 'running').length"
          :icon="Activity"
          variant="warning"
        />
        <MetricCard
          title="Services Online"
          :value="`${getHealthyServices()}/${getTotalServices()}`"
          :icon="Server"
          :variant="getHealthyServices() === getTotalServices() ? 'success' : 'warning'"
        />
      </div>

      <!-- Quick Start Workflow -->
      <div class="card p-6">
        <h2 class="text-lg font-semibold text-white mb-4">Quick Start Workflow</h2>
        <div class="flex items-center gap-4 overflow-x-auto pb-2">
          <template v-for="(step, index) in workflowSteps" :key="step.path">
            <button
              @click="router.push(step.path)"
              class="flex min-w-[180px] flex-col items-center rounded-xl bg-background-tertiary p-4 transition-colors hover:bg-gray-600"
            >
              <div class="flex h-10 w-10 items-center justify-center rounded-full bg-primary/20 text-primary font-semibold">
                {{ index + 1 }}
              </div>
              <span class="mt-3 font-medium text-white">{{ step.name }}</span>
              <span class="mt-1 text-xs text-gray-400 text-center">{{ step.description }}</span>
            </button>
            <ArrowRight
              v-if="index < workflowSteps.length - 1"
              class="h-5 w-5 flex-shrink-0 text-gray-500"
            />
          </template>
        </div>
      </div>

      <!-- Recent Jobs -->
      <div class="card p-6">
        <div class="flex items-center justify-between mb-4">
          <h2 class="text-lg font-semibold text-white">Recent Jobs</h2>
          <button
            @click="router.push('/tools/job-monitor')"
            class="text-sm text-primary hover:text-primary-hover"
          >
            View All
          </button>
        </div>

        <div v-if="recentJobs.length === 0" class="text-center py-8 text-gray-400">
          No jobs yet. Start a generation to see jobs here.
        </div>

        <div v-else class="space-y-3">
          <div
            v-for="job in recentJobs"
            :key="job.job_id"
            class="flex items-center justify-between rounded-lg bg-background-tertiary p-4"
          >
            <div class="flex items-center gap-3">
              <component
                :is="getJobStatusIcon(job.status)"
                :class="['h-5 w-5', getJobStatusColor(job.status)]"
              />
              <div>
                <p class="font-medium text-white">{{ job.type }}</p>
                <p class="text-sm text-gray-400">{{ job.job_id.slice(0, 8) }}...</p>
              </div>
            </div>
            <div class="text-right">
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
              <p v-if="job.status === 'running'" class="mt-1 text-sm text-gray-400">
                {{ job.progress }}%
              </p>
            </div>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>
