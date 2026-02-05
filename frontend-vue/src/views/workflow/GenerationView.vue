<script setup lang="ts">
import { ref, computed, onUnmounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { useWorkflowStore } from '@/stores/workflow'
import { useUiStore } from '@/stores/ui'
import { startGeneration, getJob, getActiveLabelingJobs } from '@/lib/api'
import MetricCard from '@/components/common/MetricCard.vue'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseInput from '@/components/ui/BaseInput.vue'
import BaseSelect from '@/components/ui/BaseSelect.vue'
import AlertBox from '@/components/common/AlertBox.vue'
import {
  Switch,
  TabGroup,
  TabList,
  Tab,
  TabPanels,
  TabPanel,
  Disclosure,
  DisclosureButton,
  DisclosurePanel,
} from '@headlessui/vue'
import {
  ArrowRight,
  ArrowLeft,
  Play,
  CheckCircle,
  XCircle,
  Sparkles,
  Image,
  Layers,
  Clock,
  Settings,
  Shield,
  Cpu,
  Sun,
  FileText,
  ChevronDown,
  Target,
} from 'lucide-vue-next'
import type { Job, LabelingJob, ValidationConfig, BatchConfig, LightingEstimationConfig, DatasetMetadata } from '@/types/api'

const router = useRouter()
const workflowStore = useWorkflowStore()
const uiStore = useUiStore()
const { t } = useI18n()

const generating = ref(false)
const currentJob = ref<Job | null>(null)
const activeLabelingJobs = ref<LabelingJob[]>([])
const error = ref<string | null>(null)
const pollingInterval = ref<ReturnType<typeof setInterval> | null>(null)

// Balancing targets local state
const targets = ref<Record<string, number>>({ ...workflowStore.balancingTargets })
const useDepth = ref(workflowStore.useDepth)
const useSegmentation = ref(workflowStore.useSegmentation)
const depthAwarePlacement = ref(workflowStore.depthAwarePlacement)

// Advanced config local states
const validationConfig = ref<ValidationConfig>(JSON.parse(JSON.stringify(workflowStore.validationConfig)))
const batchConfig = ref<BatchConfig>(JSON.parse(JSON.stringify(workflowStore.batchConfig)))
const lightingConfig = ref<LightingEstimationConfig>(JSON.parse(JSON.stringify(workflowStore.lightingConfig)))
const metadata = ref<DatasetMetadata>(JSON.parse(JSON.stringify(workflowStore.metadata)))

// Tab options
const tabs = computed(() => [
  { nameKey: 'workflow.generation.tabs.targets', icon: Target },
  { nameKey: 'workflow.generation.tabs.validation', icon: Shield },
  { nameKey: 'workflow.generation.tabs.processing', icon: Cpu },
  { nameKey: 'workflow.generation.tabs.lighting', icon: Sun },
  { nameKey: 'workflow.generation.tabs.metadata', icon: FileText },
])

// Options for selects
const depthCategoryOptions = computed(() => [
  { value: 'shallow', label: t('workflow.generation.lighting.depth.shallow') },
  { value: 'mid', label: t('workflow.generation.lighting.depth.mid') },
  { value: 'deep', label: t('workflow.generation.lighting.depth.deep') },
])

// Computed
const canGenerate = computed(() => {
  return workflowStore.sourceDatasetPath && workflowStore.outputDir
})

const totalTargetImages = computed(() =>
  Object.values(targets.value).reduce((sum, val) => sum + val, 0)
)

// Watch for store changes
watch(() => workflowStore.validationConfig, (newVal) => {
  validationConfig.value = JSON.parse(JSON.stringify(newVal))
}, { deep: true })

watch(() => workflowStore.batchConfig, (newVal) => {
  batchConfig.value = JSON.parse(JSON.stringify(newVal))
}, { deep: true })

watch(() => workflowStore.lightingConfig, (newVal) => {
  lightingConfig.value = JSON.parse(JSON.stringify(newVal))
}, { deep: true })

watch(() => workflowStore.metadata, (newVal) => {
  metadata.value = JSON.parse(JSON.stringify(newVal))
}, { deep: true })

async function startGenerationJob() {
  if (!canGenerate.value) {
    uiStore.showError(t('workflow.generation.configRequired'), t('workflow.generation.configRequiredMsg'))
    return
  }

  // Save all configs to store
  workflowStore.updateValidationConfig(validationConfig.value)
  workflowStore.updateBatchConfig(batchConfig.value)
  workflowStore.updateLightingConfig(lightingConfig.value)
  workflowStore.updateMetadata(metadata.value)
  workflowStore.setUseDepth(useDepth.value)
  workflowStore.setUseSegmentation(useSegmentation.value)
  workflowStore.setDepthAwarePlacement(depthAwarePlacement.value)

  generating.value = true
  error.value = null

  try {
    const response = await startGeneration({
      source_dataset: workflowStore.sourceDatasetPath!,
      output_dir: workflowStore.outputDir!,
      target_counts: targets.value,
      effects_config: workflowStore.effectsConfig,
      placement_config: workflowStore.placementConfig,
      validation_config: validationConfig.value,
      batch_config: batchConfig.value,
      lighting_config: lightingConfig.value,
      metadata: metadata.value,
      backgrounds_dir: workflowStore.backgroundsDir || undefined,
      use_depth: useDepth.value,
      use_segmentation: useSegmentation.value,
      depth_aware_placement: depthAwarePlacement.value,
    })

    workflowStore.setActiveJobId(response.job_id)
    uiStore.showSuccess(t('workflow.generation.notifications.started'), t('workflow.generation.notifications.startedMsg', { id: response.job_id.slice(0, 8) }))
    startPolling(response.job_id)
  } catch (e: any) {
    error.value = e.message || t('workflow.generation.notifications.failed')
    uiStore.showError(t('workflow.generation.notifications.failed'), error.value)
    generating.value = false
  }
}

async function pollJobStatus(jobId: string) {
  try {
    const job = await getJob(jobId)
    currentJob.value = job

    if (job.status === 'completed') {
      stopPolling()
      generating.value = false
      workflowStore.markStepCompleted(4)
      uiStore.showSuccess(t('workflow.generation.notifications.completed'), t('workflow.generation.notifications.completedMsg'))
    } else if (job.status === 'failed') {
      stopPolling()
      generating.value = false
      error.value = job.error || t('workflow.generation.notifications.failed')
      uiStore.showError(t('workflow.generation.notifications.failed'), error.value)
    }
  } catch (e) {
    // Ignore polling errors
  }
}

function startPolling(jobId: string) {
  pollingInterval.value = setInterval(() => pollJobStatus(jobId), 2000)
}

function stopPolling() {
  if (pollingInterval.value) {
    clearInterval(pollingInterval.value)
    pollingInterval.value = null
  }
}

async function loadActiveLabelingJobs() {
  try {
    activeLabelingJobs.value = await getActiveLabelingJobs()
  } catch (e) {
    // Ignore
  }
}

function updateTarget(category: string, value: number) {
  targets.value[category] = Math.max(0, value)
}

function setAllTargets(value: number) {
  for (const key of Object.keys(targets.value)) {
    targets.value[key] = Math.max(0, value)
  }
}

function goBack() {
  router.push('/source-selection')
}

function continueToExport() {
  router.push('/export')
}

// Initialize targets from categories
if (workflowStore.categories.length > 0 && Object.keys(targets.value).length === 0) {
  workflowStore.categories.forEach(cat => {
    targets.value[cat.name] = 100
  })
}

// Load active labeling jobs
loadActiveLabelingJobs()

// Cleanup on unmount
onUnmounted(() => {
  stopPolling()
})
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div>
      <h2 class="text-2xl font-bold text-white">{{ t('workflow.generation.title') }}</h2>
      <p class="mt-2 text-gray-400">
        {{ t('workflow.generation.subtitle') }}
      </p>
    </div>

    <!-- Error Alert -->
    <AlertBox v-if="error" type="error" :title="error" dismissible @dismiss="error = null" />

    <!-- Warning if missing configuration -->
    <AlertBox v-if="!canGenerate" type="warning" :title="t('workflow.generation.configRequired')">
      {{ t('workflow.generation.configRequiredMsg') }}
    </AlertBox>

    <!-- Active Labeling Jobs -->
    <AlertBox v-if="activeLabelingJobs.length > 0" type="info" :title="t('workflow.generation.activeLabelingJobs')">
      {{ t('workflow.generation.labelingJobsMsg', { count: activeLabelingJobs.length }) }}
      <router-link to="/tools/labeling" class="text-primary hover:underline ml-1">
        {{ t('common.actions.viewJobs') }}
      </router-link>
    </AlertBox>

    <!-- Tabs -->
    <TabGroup>
      <TabList class="flex space-x-1 rounded-xl bg-gray-800/50 p-1 overflow-x-auto">
        <Tab
          v-for="tab in tabs"
          :key="tab.nameKey"
          v-slot="{ selected }"
          class="rounded-lg py-2 text-sm font-medium leading-5 transition-all focus:outline-none flex-shrink-0"
        >
          <div
            :class="[
              selected
                ? 'bg-primary text-white shadow'
                : 'text-gray-400 hover:bg-gray-700/50 hover:text-white',
              'flex items-center gap-2 rounded-lg px-4 py-2',
            ]"
          >
            <component :is="tab.icon" class="h-4 w-4" />
            {{ t(tab.nameKey) }}
          </div>
        </Tab>
      </TabList>

      <TabPanels class="mt-6">
        <!-- Targets Tab -->
        <TabPanel class="space-y-6">
          <div class="grid gap-6 lg:grid-cols-2">
            <!-- Target Counts -->
            <div class="card p-6">
              <h3 class="text-lg font-semibold text-white mb-4">{{ t('workflow.generation.targets.title') }}</h3>
              <p class="text-sm text-gray-400 mb-4">
                {{ t('workflow.generation.targets.description') }}
              </p>

              <!-- Bulk actions -->
              <div class="flex gap-2 mb-4">
                <button @click="setAllTargets(50)" class="btn-outline text-xs px-2 py-1">{{ t('workflow.generation.targets.setAll') }}: 50</button>
                <button @click="setAllTargets(100)" class="btn-outline text-xs px-2 py-1">{{ t('workflow.generation.targets.setAll') }}: 100</button>
                <button @click="setAllTargets(200)" class="btn-outline text-xs px-2 py-1">{{ t('workflow.generation.targets.setAll') }}: 200</button>
                <button @click="setAllTargets(500)" class="btn-outline text-xs px-2 py-1">{{ t('workflow.generation.targets.setAll') }}: 500</button>
              </div>

              <div v-if="Object.keys(targets).length === 0" class="text-center py-8 text-gray-400">
                {{ t('workflow.generation.targets.noCategories') }}
              </div>

              <div v-else class="space-y-3 max-h-80 overflow-y-auto pr-2">
                <div
                  v-for="(count, category) in targets"
                  :key="category"
                  class="flex items-center gap-3"
                >
                  <span class="w-28 text-sm text-gray-300 truncate" :title="category as string">{{ category }}</span>
                  <input
                    type="number"
                    :value="count"
                    @input="updateTarget(category as string, parseInt(($event.target as HTMLInputElement).value) || 0)"
                    class="input w-20 text-sm"
                    min="0"
                    step="10"
                  />
                  <input
                    type="range"
                    :value="count"
                    @input="updateTarget(category as string, parseInt(($event.target as HTMLInputElement).value))"
                    min="0"
                    max="500"
                    class="flex-1 accent-primary"
                  />
                </div>
              </div>

              <div class="mt-4 pt-4 border-t border-gray-700">
                <p class="text-sm text-gray-400">
                  {{ t('workflow.generation.targets.totalImages', { count: totalTargetImages }) }}
                </p>
              </div>
            </div>

            <!-- Generation Options -->
            <div class="card p-6">
              <h3 class="text-lg font-semibold text-white mb-4">{{ t('workflow.generation.options.title') }}</h3>

              <div class="space-y-4">
                <div class="space-y-3">
                  <label class="flex items-center gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      v-model="useDepth"
                      class="h-5 w-5 rounded border-gray-600 bg-background-tertiary text-primary focus:ring-primary"
                    />
                    <div>
                      <span class="text-gray-300">{{ t('workflow.generation.options.useDepth') }}</span>
                      <p class="text-xs text-gray-500">{{ t('workflow.generation.options.useDepthDesc') }}</p>
                    </div>
                  </label>

                  <label class="flex items-center gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      v-model="useSegmentation"
                      class="h-5 w-5 rounded border-gray-600 bg-background-tertiary text-primary focus:ring-primary"
                    />
                    <div>
                      <span class="text-gray-300">{{ t('workflow.generation.options.useSegmentation') }}</span>
                      <p class="text-xs text-gray-500">{{ t('workflow.generation.options.useSegmentationDesc') }}</p>
                    </div>
                  </label>

                  <label class="flex items-center gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      v-model="depthAwarePlacement"
                      class="h-5 w-5 rounded border-gray-600 bg-background-tertiary text-primary focus:ring-primary"
                    />
                    <div>
                      <span class="text-gray-300">{{ t('workflow.generation.options.depthAwarePlacement') }}</span>
                      <p class="text-xs text-gray-500">{{ t('workflow.generation.options.depthAwarePlacementDesc') }}</p>
                    </div>
                  </label>
                </div>
              </div>
            </div>
          </div>
        </TabPanel>

        <!-- Validation Tab -->
        <TabPanel class="space-y-6">
          <div class="card p-6">
            <h3 class="text-lg font-semibold text-white mb-4">{{ t('workflow.generation.validation.title') }}</h3>
            <p class="text-sm text-gray-400 mb-6">
              {{ t('workflow.generation.validation.description') }}
            </p>

            <div class="grid gap-6 md:grid-cols-2">
              <!-- Identity Validation -->
              <div class="space-y-4 p-4 bg-background-tertiary rounded-lg">
                <div class="flex items-center justify-between">
                  <div>
                    <h4 class="font-medium text-white">{{ t('workflow.generation.validation.identity.title') }}</h4>
                    <p class="text-xs text-gray-400">{{ t('workflow.generation.validation.identity.description') }}</p>
                  </div>
                  <Switch
                    v-model="validationConfig.validate_identity"
                    :class="[validationConfig.validate_identity ? 'bg-primary' : 'bg-gray-600', 'relative inline-flex h-6 w-11 items-center rounded-full transition-colors']"
                  >
                    <span :class="[validationConfig.validate_identity ? 'translate-x-6' : 'translate-x-1', 'inline-block h-4 w-4 transform rounded-full bg-white transition-transform']" />
                  </Switch>
                </div>
                <div v-if="validationConfig.validate_identity" class="space-y-3">
                  <div>
                    <label class="text-sm text-gray-400 flex justify-between mb-1">
                      <span>{{ t('workflow.generation.validation.identity.maxColorShift') }}</span>
                      <span class="text-white font-mono">{{ validationConfig.max_color_shift }}</span>
                    </label>
                    <input type="range" v-model.number="validationConfig.max_color_shift" min="10" max="100" step="5" class="w-full accent-primary" />
                  </div>
                  <div>
                    <label class="text-sm text-gray-400 flex justify-between mb-1">
                      <span>{{ t('workflow.generation.validation.identity.minSharpnessRatio') }}</span>
                      <span class="text-white font-mono">{{ validationConfig.min_sharpness_ratio.toFixed(2) }}</span>
                    </label>
                    <input type="range" v-model.number="validationConfig.min_sharpness_ratio" min="0" max="1" step="0.05" class="w-full accent-primary" />
                  </div>
                  <div>
                    <label class="text-sm text-gray-400 flex justify-between mb-1">
                      <span>{{ t('workflow.generation.validation.identity.minContrastRatio') }}</span>
                      <span class="text-white font-mono">{{ validationConfig.min_contrast_ratio.toFixed(2) }}</span>
                    </label>
                    <input type="range" v-model.number="validationConfig.min_contrast_ratio" min="0" max="1" step="0.05" class="w-full accent-primary" />
                  </div>
                </div>
              </div>

              <!-- Quality Validation -->
              <div class="space-y-4 p-4 bg-background-tertiary rounded-lg">
                <div class="flex items-center justify-between">
                  <div>
                    <h4 class="font-medium text-white">{{ t('workflow.generation.validation.quality.title') }}</h4>
                    <p class="text-xs text-gray-400">{{ t('workflow.generation.validation.quality.description') }}</p>
                  </div>
                  <Switch
                    v-model="validationConfig.validate_quality"
                    :class="[validationConfig.validate_quality ? 'bg-primary' : 'bg-gray-600', 'relative inline-flex h-6 w-11 items-center rounded-full transition-colors']"
                  >
                    <span :class="[validationConfig.validate_quality ? 'translate-x-6' : 'translate-x-1', 'inline-block h-4 w-4 transform rounded-full bg-white transition-transform']" />
                  </Switch>
                </div>
                <div v-if="validationConfig.validate_quality" class="space-y-3">
                  <div>
                    <label class="text-sm text-gray-400 flex justify-between mb-1">
                      <span>{{ t('workflow.generation.validation.quality.minPerceptualQuality') }}</span>
                      <span class="text-white font-mono">{{ validationConfig.min_perceptual_quality.toFixed(2) }}</span>
                    </label>
                    <input type="range" v-model.number="validationConfig.min_perceptual_quality" min="0" max="1" step="0.05" class="w-full accent-primary" />
                  </div>
                  <div>
                    <label class="text-sm text-gray-400 flex justify-between mb-1">
                      <span>{{ t('workflow.generation.validation.quality.minAnomalyScore') }}</span>
                      <span class="text-white font-mono">{{ validationConfig.min_anomaly_score.toFixed(2) }}</span>
                    </label>
                    <input type="range" v-model.number="validationConfig.min_anomaly_score" min="0" max="1" step="0.05" class="w-full accent-primary" />
                  </div>
                </div>
              </div>

              <!-- Physics Validation -->
              <div class="space-y-4 p-4 bg-background-tertiary rounded-lg">
                <div class="flex items-center justify-between">
                  <div>
                    <h4 class="font-medium text-white">{{ t('workflow.generation.validation.physics.title') }}</h4>
                    <p class="text-xs text-gray-400">{{ t('workflow.generation.validation.physics.description') }}</p>
                  </div>
                  <Switch
                    v-model="validationConfig.validate_physics"
                    :class="[validationConfig.validate_physics ? 'bg-primary' : 'bg-gray-600', 'relative inline-flex h-6 w-11 items-center rounded-full transition-colors']"
                  >
                    <span :class="[validationConfig.validate_physics ? 'translate-x-6' : 'translate-x-1', 'inline-block h-4 w-4 transform rounded-full bg-white transition-transform']" />
                  </Switch>
                </div>
              </div>

              <!-- Reject Invalid -->
              <div class="space-y-4 p-4 bg-background-tertiary rounded-lg">
                <div class="flex items-center justify-between">
                  <div>
                    <h4 class="font-medium text-white">{{ t('workflow.generation.validation.rejectInvalid.title') }}</h4>
                    <p class="text-xs text-gray-400">{{ t('workflow.generation.validation.rejectInvalid.description') }}</p>
                  </div>
                  <Switch
                    v-model="validationConfig.reject_invalid"
                    :class="[validationConfig.reject_invalid ? 'bg-primary' : 'bg-gray-600', 'relative inline-flex h-6 w-11 items-center rounded-full transition-colors']"
                  >
                    <span :class="[validationConfig.reject_invalid ? 'translate-x-6' : 'translate-x-1', 'inline-block h-4 w-4 transform rounded-full bg-white transition-transform']" />
                  </Switch>
                </div>
              </div>
            </div>
          </div>
        </TabPanel>

        <!-- Processing Tab -->
        <TabPanel class="space-y-6">
          <div class="card p-6">
            <h3 class="text-lg font-semibold text-white mb-4">{{ t('workflow.generation.processing.title') }}</h3>
            <p class="text-sm text-gray-400 mb-6">
              {{ t('workflow.generation.processing.description') }}
            </p>

            <div class="grid gap-6 md:grid-cols-2">
              <div class="space-y-4 p-4 bg-background-tertiary rounded-lg">
                <div class="flex items-center justify-between">
                  <div>
                    <h4 class="font-medium text-white">{{ t('workflow.generation.processing.parallel.title') }}</h4>
                    <p class="text-xs text-gray-400">{{ t('workflow.generation.processing.parallel.description') }}</p>
                  </div>
                  <Switch
                    v-model="batchConfig.parallel"
                    :class="[batchConfig.parallel ? 'bg-primary' : 'bg-gray-600', 'relative inline-flex h-6 w-11 items-center rounded-full transition-colors']"
                  >
                    <span :class="[batchConfig.parallel ? 'translate-x-6' : 'translate-x-1', 'inline-block h-4 w-4 transform rounded-full bg-white transition-transform']" />
                  </Switch>
                </div>

                <div v-if="batchConfig.parallel">
                  <label class="text-sm text-gray-400 flex justify-between mb-1">
                    <span>{{ t('workflow.generation.processing.parallel.concurrentLimit') }}</span>
                    <span class="text-white font-mono">{{ batchConfig.concurrent_limit }}</span>
                  </label>
                  <input type="range" v-model.number="batchConfig.concurrent_limit" min="1" max="16" step="1" class="w-full accent-primary" />
                  <p class="text-xs text-gray-500 mt-1">{{ t('workflow.generation.processing.parallel.concurrentLimitDesc') }}</p>
                </div>
              </div>

              <div class="space-y-4 p-4 bg-background-tertiary rounded-lg">
                <h4 class="font-medium text-white">{{ t('workflow.generation.processing.vram.title') }}</h4>
                <div>
                  <label class="text-sm text-gray-400 flex justify-between mb-1">
                    <span>{{ t('workflow.generation.processing.vram.threshold') }}</span>
                    <span class="text-white font-mono">{{ (batchConfig.vram_threshold * 100).toFixed(0) }}%</span>
                  </label>
                  <input type="range" v-model.number="batchConfig.vram_threshold" min="0.3" max="0.95" step="0.05" class="w-full accent-primary" />
                  <p class="text-xs text-gray-500 mt-1">{{ t('workflow.generation.processing.vram.thresholdDesc') }}</p>
                </div>
              </div>

              <div class="space-y-4 p-4 bg-background-tertiary rounded-lg">
                <div class="flex items-center justify-between">
                  <div>
                    <h4 class="font-medium text-white">{{ t('workflow.generation.processing.debug.title') }}</h4>
                    <p class="text-xs text-gray-400">{{ t('workflow.generation.processing.debug.description') }}</p>
                  </div>
                  <Switch
                    v-model="batchConfig.save_pipeline_debug"
                    :class="[batchConfig.save_pipeline_debug ? 'bg-primary' : 'bg-gray-600', 'relative inline-flex h-6 w-11 items-center rounded-full transition-colors']"
                  >
                    <span :class="[batchConfig.save_pipeline_debug ? 'translate-x-6' : 'translate-x-1', 'inline-block h-4 w-4 transform rounded-full bg-white transition-transform']" />
                  </Switch>
                </div>
              </div>
            </div>
          </div>
        </TabPanel>

        <!-- Lighting Tab -->
        <TabPanel class="space-y-6">
          <div class="card p-6">
            <h3 class="text-lg font-semibold text-white mb-4">{{ t('workflow.generation.lighting.title') }}</h3>
            <p class="text-sm text-gray-400 mb-6">
              {{ t('workflow.generation.lighting.description') }}
            </p>

            <div class="grid gap-6 md:grid-cols-2">
              <div class="space-y-4 p-4 bg-background-tertiary rounded-lg">
                <h4 class="font-medium text-white">{{ t('workflow.generation.lighting.detection.title') }}</h4>
                <div>
                  <label class="text-sm text-gray-400 flex justify-between mb-1">
                    <span>{{ t('workflow.generation.lighting.detection.maxSources') }}</span>
                    <span class="text-white font-mono">{{ lightingConfig.max_light_sources }}</span>
                  </label>
                  <input type="range" v-model.number="lightingConfig.max_light_sources" min="1" max="10" step="1" class="w-full accent-primary" />
                </div>
                <div>
                  <label class="text-sm text-gray-400 flex justify-between mb-1">
                    <span>{{ t('workflow.generation.lighting.detection.intensityThreshold') }}</span>
                    <span class="text-white font-mono">{{ lightingConfig.intensity_threshold.toFixed(2) }}</span>
                  </label>
                  <input type="range" v-model.number="lightingConfig.intensity_threshold" min="0.1" max="1" step="0.05" class="w-full accent-primary" />
                </div>
              </div>

              <div class="space-y-4 p-4 bg-background-tertiary rounded-lg">
                <h4 class="font-medium text-white">{{ t('workflow.generation.lighting.advanced.title') }}</h4>
                <div class="flex items-center justify-between">
                  <span class="text-sm text-gray-400">{{ t('workflow.generation.lighting.advanced.estimateHdr') }}</span>
                  <Switch
                    v-model="lightingConfig.estimate_hdr"
                    :class="[lightingConfig.estimate_hdr ? 'bg-primary' : 'bg-gray-600', 'relative inline-flex h-5 w-9 items-center rounded-full transition-colors']"
                  >
                    <span :class="[lightingConfig.estimate_hdr ? 'translate-x-5' : 'translate-x-1', 'inline-block h-3 w-3 transform rounded-full bg-white transition-transform']" />
                  </Switch>
                </div>
                <div class="flex items-center justify-between">
                  <span class="text-sm text-gray-400">{{ t('workflow.generation.lighting.advanced.waterAttenuation') }}</span>
                  <Switch
                    v-model="lightingConfig.apply_water_attenuation"
                    :class="[lightingConfig.apply_water_attenuation ? 'bg-primary' : 'bg-gray-600', 'relative inline-flex h-5 w-9 items-center rounded-full transition-colors']"
                  >
                    <span :class="[lightingConfig.apply_water_attenuation ? 'translate-x-5' : 'translate-x-1', 'inline-block h-3 w-3 transform rounded-full bg-white transition-transform']" />
                  </Switch>
                </div>
              </div>

              <div class="space-y-4 p-4 bg-background-tertiary rounded-lg">
                <h4 class="font-medium text-white">{{ t('workflow.generation.lighting.depth.title') }}</h4>
                <BaseSelect
                  v-model="lightingConfig.depth_category"
                  :label="t('workflow.generation.lighting.depth.label')"
                  :options="depthCategoryOptions"
                />
                <p class="text-xs text-gray-500">{{ t('workflow.generation.lighting.depth.description') }}</p>
              </div>
            </div>
          </div>
        </TabPanel>

        <!-- Metadata Tab -->
        <TabPanel class="space-y-6">
          <div class="card p-6">
            <h3 class="text-lg font-semibold text-white mb-4">{{ t('workflow.generation.metadata.title') }}</h3>
            <p class="text-sm text-gray-400 mb-6">
              {{ t('workflow.generation.metadata.description') }}
            </p>

            <div class="grid gap-4 md:grid-cols-2">
              <BaseInput
                v-model="metadata.name"
                :label="t('workflow.generation.metadata.name')"
                placeholder="My Synthetic Dataset"
              />
              <BaseInput
                v-model="metadata.version"
                :label="t('workflow.generation.metadata.version')"
                placeholder="1.0"
              />
              <div class="md:col-span-2">
                <label class="block text-sm font-medium text-gray-300 mb-1">{{ t('workflow.generation.metadata.descriptionField') }}</label>
                <textarea
                  v-model="metadata.description"
                  class="input w-full h-24 resize-none"
                  :placeholder="t('workflow.generation.metadata.descriptionPlaceholder')"
                />
              </div>
              <BaseInput
                v-model="metadata.contributor"
                :label="t('workflow.generation.metadata.contributor')"
                placeholder="Your name or organization"
              />
              <BaseInput
                v-model="metadata.url"
                :label="t('workflow.generation.metadata.url')"
                placeholder="https://example.com"
              />
              <BaseInput
                v-model="metadata.license_name"
                :label="t('workflow.generation.metadata.license')"
                placeholder="CC BY 4.0"
              />
              <BaseInput
                v-model="metadata.license_url"
                :label="t('workflow.generation.metadata.licenseUrl')"
                placeholder="https://creativecommons.org/licenses/by/4.0/"
              />
            </div>
          </div>
        </TabPanel>
      </TabPanels>
    </TabGroup>

    <!-- Job Progress -->
    <div v-if="currentJob" class="card p-6">
      <h3 class="text-lg font-semibold text-white mb-4">{{ t('workflow.generation.progress.title') }}</h3>

      <div class="flex items-center gap-4 mb-4">
        <component
          :is="currentJob.status === 'completed' ? CheckCircle : currentJob.status === 'failed' ? XCircle : Sparkles"
          :class="[
            'h-8 w-8',
            currentJob.status === 'completed' ? 'text-green-400' :
            currentJob.status === 'failed' ? 'text-red-400' :
            'text-primary animate-pulse'
          ]"
        />
        <div class="flex-1">
          <p class="font-medium text-white">
            {{ currentJob.status === 'running' ? t('workflow.generation.progress.generating') : currentJob.status }}
          </p>
          <p class="text-sm text-gray-400">{{ t('workflow.generation.progress.jobId', { id: currentJob.job_id.slice(0, 8) }) }}...</p>
        </div>
        <span class="text-2xl font-bold text-primary">{{ currentJob.progress }}%</span>
      </div>

      <div class="h-3 bg-background-tertiary rounded-full overflow-hidden">
        <div
          class="h-full bg-gradient-to-r from-primary to-blue-400 transition-all duration-500"
          :style="{ width: `${currentJob.progress}%` }"
        />
      </div>
    </div>

    <!-- Metrics (after completion) -->
    <div v-if="currentJob?.status === 'completed'" class="grid gap-6 sm:grid-cols-3">
      <MetricCard
        :title="t('workflow.generation.results.imagesGenerated')"
        :value="(currentJob.result as any)?.generated_images || 0"
        :icon="Image"
        variant="success"
      />
      <MetricCard
        :title="t('workflow.generation.results.annotations')"
        :value="(currentJob.result as any)?.generated_annotations || 0"
        :icon="Layers"
      />
      <MetricCard
        :title="t('workflow.generation.results.duration')"
        :value="`${((currentJob.result as any)?.duration_seconds || 0).toFixed(1)}s`"
        :icon="Clock"
      />
    </div>

    <!-- Action Buttons -->
    <div class="flex justify-between pt-4">
      <BaseButton variant="outline" @click="goBack">
        <ArrowLeft class="h-5 w-5" />
        {{ t('common.actions.back') }}
      </BaseButton>

      <div class="flex gap-4">
        <BaseButton
          v-if="!generating && !currentJob?.status"
          :disabled="!canGenerate"
          @click="startGenerationJob"
        >
          <Play class="h-5 w-5" />
          {{ t('workflow.generation.actions.startGeneration') }}
        </BaseButton>

        <BaseButton
          v-if="currentJob?.status === 'completed'"
          @click="continueToExport"
        >
          {{ t('workflow.generation.actions.continueToExport') }}
          <ArrowRight class="h-5 w-5" />
        </BaseButton>
      </div>
    </div>
  </div>
</template>
