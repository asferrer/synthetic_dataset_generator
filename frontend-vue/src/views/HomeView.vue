<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { getHealthStatus, getJobs, listDatasets } from '@/lib/api'
import { useDomainStore } from '@/stores/domain'
import MetricCard from '@/components/common/MetricCard.vue'
import AlertBox from '@/components/common/AlertBox.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import DomainSelector from '@/components/domain/DomainSelector.vue'
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
  Globe,
} from 'lucide-vue-next'
import type { HealthStatus, Job, DatasetInfo } from '@/types/api'

const router = useRouter()
const domainStore = useDomainStore()
const { t } = useI18n()

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
    error.value = t('common.errors.loadFailed', { item: 'dashboard' })
  } finally {
    loading.value = false
  }
}

function getHealthyServices(): number {
  if (!health.value || !health.value.services) return 0
  return health.value.services.filter(s => s.status === 'healthy').length
}

function getTotalServices(): number {
  if (!health.value || !health.value.services) return 5
  return health.value.services.length
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
  { nameKey: 'workflow.steps.analysis.name', descKey: 'workflow.steps.analysis.description', path: '/analysis' },
  { nameKey: 'workflow.steps.configure.name', descKey: 'workflow.steps.configure.description', path: '/configure' },
  { nameKey: 'workflow.steps.source.name', descKey: 'workflow.steps.source.description', path: '/source-selection' },
  { nameKey: 'workflow.steps.generate.name', descKey: 'workflow.steps.generate.description', path: '/generation' },
  { nameKey: 'workflow.steps.export.name', descKey: 'workflow.steps.export.description', path: '/export' },
]

onMounted(() => {
  fetchData()
  domainStore.fetchDomains()
})
</script>

<template>
  <div class="space-y-8">
    <!-- Welcome Header -->
    <div class="flex items-start justify-between">
      <div>
        <h1 class="text-3xl font-bold text-white">{{ t('workflow.home.welcome') }}</h1>
        <p class="mt-2 text-gray-400">
          {{ t('workflow.home.subtitle') }}
        </p>
      </div>
      <button
        @click="router.push('/analysis')"
        class="btn-primary flex items-center gap-2"
      >
        <Sparkles class="h-5 w-5" />
        {{ t('workflow.home.startGeneration') }}
      </button>
    </div>

    <!-- Loading State -->
    <div v-if="loading" class="flex justify-center py-12">
      <LoadingSpinner size="lg" :message="t('common.status.loading')" />
    </div>

    <!-- Error State -->
    <AlertBox v-else-if="error" type="error" :title="error" dismissible @dismiss="error = null">
      {{ t('common.errors.networkError') }}
    </AlertBox>

    <!-- Dashboard Content -->
    <template v-else>
      <!-- Metrics Grid -->
      <div class="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          :title="t('workflow.metrics.totalDatasets')"
          :value="datasets.length"
          :icon="Database"
          variant="info"
        />
        <MetricCard
          :title="t('workflow.metrics.totalImages')"
          :value="datasets.reduce((sum, d) => sum + d.num_images, 0)"
          :icon="Image"
        />
        <MetricCard
          :title="t('workflow.metrics.activeJobs')"
          :value="recentJobs.filter(j => j.status === 'running').length"
          :icon="Activity"
          variant="warning"
        />
        <MetricCard
          :title="t('workflow.metrics.servicesOnline')"
          :value="`${getHealthyServices()}/${getTotalServices()}`"
          :icon="Server"
          :variant="getHealthyServices() === getTotalServices() ? 'success' : 'warning'"
        />
      </div>

      <!-- Active Domain -->
      <div class="card p-6">
        <div class="flex items-center justify-between mb-4">
          <h2 class="text-lg font-semibold text-white">{{ t('workflow.home.activeDomain') }}</h2>
          <button
            @click="router.push('/domains')"
            class="text-sm text-primary hover:text-primary-hover"
          >
            {{ t('common.actions.manageDomains') }}
          </button>
        </div>
        <DomainSelector :show-manage-link="false" />
        <p class="mt-3 text-sm text-gray-500">
          {{ t('workflow.home.domainDescription') }}
        </p>
      </div>

      <!-- Quick Start Workflow -->
      <div class="card p-6">
        <h2 class="text-lg font-semibold text-white mb-4">{{ t('workflow.home.quickStart') }}</h2>
        <div class="flex items-center gap-4 overflow-x-auto pb-2">
          <template v-for="(step, index) in workflowSteps" :key="step.path">
            <button
              @click="router.push(step.path)"
              class="flex min-w-[180px] flex-col items-center rounded-xl bg-background-tertiary p-4 transition-colors hover:bg-gray-600"
            >
              <div class="flex h-10 w-10 items-center justify-center rounded-full bg-primary/20 text-primary font-semibold">
                {{ index + 1 }}
              </div>
              <span class="mt-3 font-medium text-white">{{ t(step.nameKey) }}</span>
              <span class="mt-1 text-xs text-gray-400 text-center">{{ t(step.descKey) }}</span>
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
          <h2 class="text-lg font-semibold text-white">{{ t('workflow.home.recentJobs') }}</h2>
          <button
            @click="router.push('/tools/job-monitor')"
            class="text-sm text-primary hover:text-primary-hover"
          >
            {{ t('common.actions.viewAll') }}
          </button>
        </div>

        <div v-if="recentJobs.length === 0" class="text-center py-8 text-gray-400">
          {{ t('workflow.home.noJobs') }}
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
