<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useUiStore } from '@/stores/ui'
import {
  startLabeling,
  startRelabeling,
  getLabelingStatus,
  getActiveLabelingJobs,
  cancelLabelingJob,
  listDirectories,
  getLabelingPreviews,
  type LabelingPreview,
} from '@/lib/api'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseInput from '@/components/ui/BaseInput.vue'
import BaseSelect from '@/components/ui/BaseSelect.vue'
import DirectoryBrowser from '@/components/common/DirectoryBrowser.vue'
import AlertBox from '@/components/common/AlertBox.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import {
  Tags,
  Play,
  FolderOpen,
  Plus,
  X,
  CheckCircle,
  RefreshCw,
  Settings,
  Sparkles,
  StopCircle,
  Fish,
  Car,
  Users,
  Building,
  TreePine,
  Utensils,
  Shirt,
} from 'lucide-vue-next'
import type { LabelingJob, LabelingTaskType, RelabelMode } from '@/types/api'

const uiStore = useUiStore()
const { t } = useI18n()

// Predefined labeling templates
interface LabelingTemplate {
  id: string
  nameKey: string
  descKey: string
  icon: any
  classes: string[]
  taskType: LabelingTaskType
  confidence: number
}

const labelingTemplates: LabelingTemplate[] = [
  {
    id: 'marine_life',
    nameKey: 'tools.labeling.templates.marineLife',
    descKey: 'tools.labeling.templates.marineLifeDesc',
    icon: Fish,
    classes: ['fish', 'coral', 'shark', 'turtle', 'jellyfish', 'octopus', 'starfish', 'crab', 'dolphin', 'whale'],
    taskType: 'both',
    confidence: 0.35,
  },
  {
    id: 'vehicles',
    nameKey: 'tools.labeling.templates.vehicles',
    descKey: 'tools.labeling.templates.vehiclesDesc',
    icon: Car,
    classes: ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'boat', 'airplane', 'train'],
    taskType: 'detection',
    confidence: 0.4,
  },
  {
    id: 'people',
    nameKey: 'tools.labeling.templates.people',
    descKey: 'tools.labeling.templates.peopleDesc',
    icon: Users,
    classes: ['person', 'face', 'hand', 'head'],
    taskType: 'both',
    confidence: 0.35,
  },
  {
    id: 'urban',
    nameKey: 'tools.labeling.templates.urban',
    descKey: 'tools.labeling.templates.urbanDesc',
    icon: Building,
    classes: ['building', 'traffic light', 'stop sign', 'street sign', 'bench', 'parking meter', 'fire hydrant'],
    taskType: 'detection',
    confidence: 0.4,
  },
  {
    id: 'nature',
    nameKey: 'tools.labeling.templates.nature',
    descKey: 'tools.labeling.templates.natureDesc',
    icon: TreePine,
    classes: ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'tree', 'flower'],
    taskType: 'both',
    confidence: 0.35,
  },
  {
    id: 'food',
    nameKey: 'tools.labeling.templates.food',
    descKey: 'tools.labeling.templates.foodDesc',
    icon: Utensils,
    classes: ['apple', 'banana', 'orange', 'pizza', 'sandwich', 'cake', 'cup', 'bowl', 'bottle', 'knife', 'fork', 'spoon'],
    taskType: 'detection',
    confidence: 0.4,
  },
  {
    id: 'fashion',
    nameKey: 'tools.labeling.templates.fashion',
    descKey: 'tools.labeling.templates.fashionDesc',
    icon: Shirt,
    classes: ['shirt', 'pants', 'dress', 'shoe', 'hat', 'bag', 'tie', 'watch', 'glasses', 'backpack'],
    taskType: 'detection',
    confidence: 0.35,
  },
]

const selectedTemplate = ref<string | null>(null)

function applyTemplate(template: LabelingTemplate) {
  selectedTemplate.value = template.id
  classes.value = [...template.classes]
  taskType.value = template.taskType
  minConfidence.value = template.confidence
  uiStore.showSuccess(
    t('tools.labeling.notifications.templateApplied'),
    t('tools.labeling.notifications.templateAppliedMsg', { name: t(template.nameKey), count: template.classes.length })
  )
}

function clearTemplate() {
  selectedTemplate.value = null
  classes.value = ['']
}

// Form state
const imageDirectories = ref<string[]>([''])
const classes = ref<string[]>([''])
const outputDir = ref('/app/output/labeled')
const minConfidence = ref(0.3)
const taskType = ref<LabelingTaskType>('detection')
const outputFormats = ref<string[]>(['coco'])

// Preview mode state
const previewMode = ref(false)
const previewCount = ref(20)

// Deduplication strategy
const deduplicationStrategy = ref<'confidence' | 'area'>('confidence')

// Relabeling state
const isRelabeling = ref(false)
const relabelMode = ref<RelabelMode>('add')
const existingAnnotations = ref('')

// UI state
const loading = ref(false)
let pollingErrorCount = 0
const directories = ref<string[]>([])
const activeJobs = ref<LabelingJob[]>([])
const currentJob = ref<LabelingJob | null>(null)
const error = ref<string | null>(null)
let pollingInterval: ReturnType<typeof setInterval> | null = null

// Preview state
const recentPreviews = ref<LabelingPreview[]>([])
const selectedPreview = ref<LabelingPreview | null>(null)
let previewPollingInterval: ReturnType<typeof setInterval> | null = null

const taskTypeOptions = computed(() => [
  { value: 'detection', label: t('tools.labeling.output.taskTypes.detection') },
  { value: 'segmentation', label: t('tools.labeling.output.taskTypes.segmentation') },
  { value: 'both', label: t('tools.labeling.output.taskTypes.both') },
])

const relabelModeOptions = computed(() => [
  { value: 'add', label: t('tools.labeling.relabeling.modes.add') },
  { value: 'replace', label: t('tools.labeling.relabeling.modes.replace') },
  { value: 'improve_segmentation', label: t('tools.labeling.relabeling.modes.improve') },
])

async function loadDirectories() {
  try {
    directories.value = await listDirectories()
  } catch (e) {
    // Ignore
  }
}

async function loadActiveJobs() {
  try {
    activeJobs.value = await getActiveLabelingJobs()
  } catch (e) {
    // Ignore
  }
}

function addDirectory() {
  imageDirectories.value.push('')
}

function removeDirectory(index: number) {
  imageDirectories.value.splice(index, 1)
}

function addClass() {
  classes.value.push('')
}

function removeClass(index: number) {
  classes.value.splice(index, 1)
}

async function startJob() {
  const validDirs = imageDirectories.value.filter(d => d.trim())
  const validClasses = classes.value.filter(c => c.trim())

  if (validDirs.length === 0) {
    uiStore.showError(t('tools.labeling.notifications.missingInput'), t('tools.labeling.notifications.missingDirectories'))
    return
  }

  if (!isRelabeling.value && validClasses.length === 0) {
    uiStore.showError(t('tools.labeling.notifications.missingInput'), t('tools.labeling.notifications.missingClasses'))
    return
  }

  loading.value = true
  error.value = null

  try {
    let response

    if (isRelabeling.value) {
      response = await startRelabeling({
        image_directories: validDirs,
        output_dir: outputDir.value,
        relabel_mode: relabelMode.value,
        new_classes: validClasses.length > 0 ? validClasses : undefined,
        min_confidence: minConfidence.value,
        coco_json_path: existingAnnotations.value || undefined,
        output_formats: outputFormats.value,
        preview_mode: previewMode.value,
        preview_count: previewCount.value,
        deduplication_strategy: deduplicationStrategy.value,
      })
    } else {
      response = await startLabeling({
        image_directories: validDirs,
        classes: validClasses,
        output_dir: outputDir.value,
        min_confidence: minConfidence.value,
        task_type: taskType.value,
        output_formats: outputFormats.value,
        preview_mode: previewMode.value,
        preview_count: previewCount.value,
        deduplication_strategy: deduplicationStrategy.value,
      })
    }

    uiStore.showSuccess(t('tools.labeling.notifications.started'), t('tools.labeling.notifications.startedMsg', { id: response.job_id.slice(0, 8) }))
    startPolling(response.job_id)
  } catch (e: any) {
    error.value = e.message || t('tools.labeling.notifications.failed')
    uiStore.showError(t('tools.labeling.notifications.failed'), error.value)
  } finally {
    loading.value = false
  }
}

async function pollJobStatus(jobId: string) {
  try {
    const job = await getLabelingStatus(jobId)
    currentJob.value = job
    pollingErrorCount = 0 // Reset on successful poll

    if (job.status === 'completed' || job.status === 'failed') {
      stopPolling()
      if (job.status === 'completed') {
        uiStore.showSuccess(t('tools.labeling.notifications.completed'), t('tools.labeling.notifications.completedMsg', { count: job.processed_images }))
      } else {
        uiStore.showError(t('tools.labeling.notifications.failed'), job.error || t('common.errors.generic'))
      }
    }
  } catch (e: any) {
    pollingErrorCount++
    if (pollingErrorCount >= 3) {
      // After 3 consecutive errors, show warning to user
      uiStore.showWarning(
        t('tools.labeling.notifications.pollingError'),
        t('tools.labeling.notifications.pollingErrorMsg')
      )
      // Stop polling after 10 consecutive failures
      if (pollingErrorCount >= 10) {
        stopPolling()
        uiStore.showError(t('tools.labeling.notifications.connectionLost'))
      }
    }
  }
}

async function pollPreviews(jobId: string) {
  try {
    const response = await getLabelingPreviews(jobId, 5)
    recentPreviews.value = response.previews || []
  } catch (e) {
    // Ignore preview polling errors
  }
}

function startPolling(jobId: string) {
  pollingInterval = setInterval(() => pollJobStatus(jobId), 2000)
  // Poll previews less frequently (every 5 seconds)
  previewPollingInterval = setInterval(() => pollPreviews(jobId), 5000)
  // Initial preview load
  pollPreviews(jobId)
}

async function cancelCurrentJob() {
  if (!currentJob.value) return
  try {
    await cancelLabelingJob(currentJob.value.job_id)
    stopPolling()
    currentJob.value.status = 'cancelled'
    uiStore.showInfo(t('tools.labeling.notifications.cancelled'))
  } catch (e: any) {
    uiStore.showError(t('tools.labeling.notifications.cancelFailed'), e.message)
  }
}

function stopPolling() {
  if (pollingInterval) {
    clearInterval(pollingInterval)
    pollingInterval = null
  }
  if (previewPollingInterval) {
    clearInterval(previewPollingInterval)
    previewPollingInterval = null
  }
  // Clear previews on job completion
  recentPreviews.value = []
  selectedPreview.value = null
}

onMounted(() => {
  loadDirectories()
  loadActiveJobs()
})

onUnmounted(() => {
  stopPolling()
})
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div>
      <h2 class="text-2xl font-bold text-white">{{ t('tools.labeling.title') }}</h2>
      <p class="mt-1 text-gray-400">
        {{ t('tools.labeling.subtitle') }}
      </p>
    </div>

    <!-- Active Jobs Alert -->
    <AlertBox v-if="activeJobs.length > 0" type="info" :title="t('tools.labeling.activeJobs')">
      {{ t('tools.labeling.activeJobsMsg', { count: activeJobs.length }) }}
    </AlertBox>

    <!-- Error -->
    <AlertBox v-if="error" type="error" :title="error" dismissible @dismiss="error = null" />

    <!-- Mode Toggle -->
    <div class="flex gap-4">
      <BaseButton
        :variant="!isRelabeling ? 'primary' : 'outline'"
        @click="isRelabeling = false"
      >
        <Tags class="h-5 w-5" />
        {{ t('tools.labeling.mode.new') }}
      </BaseButton>
      <BaseButton
        :variant="isRelabeling ? 'primary' : 'outline'"
        @click="isRelabeling = true"
      >
        <RefreshCw class="h-5 w-5" />
        {{ t('tools.labeling.mode.relabel') }}
      </BaseButton>
    </div>

    <!-- Templates Section (only for new labeling) -->
    <div v-if="!isRelabeling" class="card p-6">
      <div class="flex items-center justify-between mb-4">
        <div>
          <h3 class="text-lg font-semibold text-white flex items-center gap-2">
            <Sparkles class="h-5 w-5 text-yellow-400" />
            {{ t('tools.labeling.templates.title') }}
          </h3>
          <p class="text-sm text-gray-400">{{ t('tools.labeling.templates.description') }}</p>
        </div>
        <BaseButton
          v-if="selectedTemplate"
          variant="ghost"
          size="sm"
          @click="clearTemplate"
        >
          <X class="h-4 w-4" />
          {{ t('common.actions.clear') }}
        </BaseButton>
      </div>

      <div class="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <button
          v-for="template in labelingTemplates"
          :key="template.id"
          @click="applyTemplate(template)"
          :class="[
            'flex flex-col items-center gap-2 p-4 rounded-lg transition-all text-center',
            selectedTemplate === template.id
              ? 'bg-primary/20 border-2 border-primary'
              : 'bg-background-tertiary hover:bg-gray-600 border-2 border-transparent'
          ]"
        >
          <component :is="template.icon" class="h-8 w-8 text-primary" />
          <div>
            <p class="text-sm font-medium text-white">{{ t(template.nameKey) }}</p>
            <p class="text-xs text-gray-400">{{ t(template.descKey) }}</p>
          </div>
        </button>
      </div>
    </div>

    <div class="grid gap-6 lg:grid-cols-2">
      <!-- Input Configuration -->
      <div class="card p-6">
        <h3 class="text-lg font-semibold text-white mb-4">{{ t('tools.labeling.input.imageDirectories') }}</h3>

        <div class="space-y-3">
          <div
            v-for="(dir, index) in imageDirectories"
            :key="index"
            class="flex gap-2 items-start"
          >
            <DirectoryBrowser
              v-model="imageDirectories[index]"
              :label="index === 0 ? '' : undefined"
              placeholder="/app/datasets/images"
              path-mode="input"
              class="flex-1"
            />
            <BaseButton
              v-if="imageDirectories.length > 1"
              variant="ghost"
              size="sm"
              class="mt-8"
              @click="removeDirectory(index)"
            >
              <X class="h-4 w-4" />
            </BaseButton>
          </div>
          <BaseButton variant="outline" size="sm" @click="addDirectory">
            <Plus class="h-4 w-4" />
            {{ t('tools.labeling.input.addDirectory') }}
          </BaseButton>
        </div>

        <!-- Classes (for new labeling) -->
        <div v-if="!isRelabeling" class="mt-6">
          <h4 class="text-md font-medium text-white mb-3">{{ t('tools.labeling.input.classesToDetect') }}</h4>
          <div class="space-y-3">
            <div
              v-for="(cls, index) in classes"
              :key="index"
              class="flex gap-2"
            >
              <BaseInput
                v-model="classes[index]"
                :placeholder="t('tools.labeling.input.classPlaceholder')"
                class="flex-1"
              />
              <BaseButton
                v-if="classes.length > 1"
                variant="ghost"
                size="sm"
                @click="removeClass(index)"
              >
                <X class="h-4 w-4" />
              </BaseButton>
            </div>
            <BaseButton variant="outline" size="sm" @click="addClass">
              <Plus class="h-4 w-4" />
              {{ t('tools.labeling.input.addClass') }}
            </BaseButton>
          </div>
        </div>

        <!-- Relabeling Options -->
        <div v-else class="mt-6 space-y-4">
          <BaseSelect
            v-model="relabelMode"
            :options="relabelModeOptions"
            :label="t('tools.labeling.relabeling.mode')"
          />
          <DirectoryBrowser
            v-model="existingAnnotations"
            :label="t('tools.labeling.relabeling.existingAnnotations')"
            placeholder="/app/datasets/annotations.json"
            :show-files="true"
            file-pattern="*.json"
            path-mode="input"
          />
        </div>
      </div>

      <!-- Output & Model Settings -->
      <div class="card p-6">
        <h3 class="text-lg font-semibold text-white mb-4">{{ t('tools.labeling.output.title') }}</h3>

        <div class="space-y-4">
          <DirectoryBrowser
            v-model="outputDir"
            :label="t('tools.labeling.output.outputDirectory')"
            placeholder="/app/output/labeled"
            path-mode="output"
          />

          <BaseSelect
            v-model="taskType"
            :options="taskTypeOptions"
            :label="t('tools.labeling.output.taskType')"
          />

          <div>
            <label class="text-sm text-gray-400 flex justify-between mb-2">
              <span>{{ t('tools.labeling.output.minConfidence') }}</span>
              <span class="text-white">{{ (minConfidence * 100).toFixed(0) }}%</span>
            </label>
            <input
              type="range"
              v-model.number="minConfidence"
              min="0.1"
              max="0.9"
              step="0.05"
              class="w-full accent-primary"
            />
          </div>

          <div>
            <label class="text-sm text-gray-400 mb-2 block">{{ t('tools.labeling.output.outputFormats') }}</label>
            <div class="flex gap-4">
              <label
                v-for="fmt in ['coco', 'yolo', 'voc']"
                :key="fmt"
                class="flex items-center gap-2 cursor-pointer"
              >
                <input
                  type="checkbox"
                  :value="fmt"
                  v-model="outputFormats"
                  class="h-4 w-4 rounded border-gray-600 bg-background-tertiary text-primary focus:ring-primary"
                />
                <span class="text-gray-300 text-sm uppercase">{{ fmt }}</span>
              </label>
            </div>
          </div>

          <!-- Deduplication Strategy -->
          <div>
            <label class="text-sm text-gray-400 mb-2 block">{{ t('tools.labeling.output.deduplicationStrategy') }}</label>
            <div class="flex gap-4">
              <label class="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  value="confidence"
                  v-model="deduplicationStrategy"
                  class="h-4 w-4 border-gray-600 bg-background-tertiary text-primary focus:ring-primary"
                />
                <div class="flex flex-col">
                  <span class="text-gray-300 text-sm">{{ t('tools.labeling.output.confidenceStrategy') }}</span>
                  <span class="text-xs text-gray-500">{{ t('tools.labeling.output.confidenceStrategyDesc') }}</span>
                </div>
              </label>
              <label class="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  value="area"
                  v-model="deduplicationStrategy"
                  class="h-4 w-4 border-gray-600 bg-background-tertiary text-primary focus:ring-primary"
                />
                <div class="flex flex-col">
                  <span class="text-gray-300 text-sm">{{ t('tools.labeling.output.areaStrategy') }}</span>
                  <span class="text-xs text-gray-500">{{ t('tools.labeling.output.areaStrategyDesc') }}</span>
                </div>
              </label>
            </div>
          </div>

          <!-- Preview Mode -->
          <div class="border-t border-gray-700 pt-4">
            <div class="flex items-center gap-3 mb-3">
              <input
                type="checkbox"
                id="previewMode"
                v-model="previewMode"
                class="h-4 w-4 rounded border-gray-600 bg-background-tertiary text-primary focus:ring-primary"
              />
              <label for="previewMode" class="text-sm text-gray-300 cursor-pointer flex items-center gap-2">
                <Sparkles class="h-4 w-4 text-yellow-400" />
                {{ t('tools.labeling.output.previewMode') }}
              </label>
            </div>
            <p class="text-xs text-gray-500 mb-3">{{ t('tools.labeling.output.previewModeDesc') }}</p>
            <div v-if="previewMode">
              <label class="text-sm text-gray-400 flex justify-between mb-2">
                <span>{{ t('tools.labeling.output.previewCount') }}</span>
                <span class="text-white">{{ previewCount }} {{ t('tools.labeling.output.images') }}</span>
              </label>
              <input
                type="range"
                v-model.number="previewCount"
                min="5"
                max="50"
                step="5"
                class="w-full accent-primary"
              />
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Current Job Progress -->
    <div v-if="currentJob" class="card p-6">
      <h3 class="text-lg font-semibold text-white mb-4">{{ t('tools.labeling.progress.title') }}</h3>

      <div class="flex items-center gap-4 mb-4">
        <component
          :is="currentJob.status === 'completed' ? CheckCircle : Tags"
          :class="[
            'h-8 w-8',
            currentJob.status === 'completed' ? 'text-green-400' :
            currentJob.status === 'failed' ? 'text-red-400' :
            'text-primary animate-pulse'
          ]"
        />
        <div class="flex-1">
          <p class="font-medium text-white">
            {{ currentJob.status === 'running' ? t('tools.labeling.progress.processing') : currentJob.status }}
          </p>
          <p class="text-sm text-gray-400">
            {{ t('tools.labeling.progress.imagesProcessed', { processed: currentJob.processed_images, total: currentJob.total_images }) }} |
            {{ t('tools.labeling.progress.annotationsCreated', { count: currentJob.annotations_created }) }}
          </p>
        </div>
        <span class="text-2xl font-bold text-primary">{{ currentJob.progress }}%</span>
        <BaseButton
          v-if="currentJob.status === 'running'"
          variant="ghost"
          size="sm"
          @click="cancelCurrentJob"
        >
          <StopCircle class="h-4 w-4" />
          {{ t('tools.labeling.actions.cancel') }}
        </BaseButton>
      </div>

      <div class="h-3 bg-background-tertiary rounded-full overflow-hidden">
        <div
          class="h-full bg-gradient-to-r from-primary to-green-400 transition-all duration-500"
          :style="{ width: `${currentJob.progress}%` }"
        />
      </div>

      <!-- Detections by Class -->
      <div v-if="currentJob.objects_by_class && Object.keys(currentJob.objects_by_class).length > 0" class="mt-4">
        <h4 class="text-sm font-medium text-gray-400 mb-2">{{ t('tools.labeling.progress.byClass') }}</h4>
        <div class="flex flex-wrap gap-2">
          <div
            v-for="(count, className) in currentJob.objects_by_class"
            :key="className"
            class="bg-background-tertiary rounded-lg px-3 py-1.5 flex items-center gap-2"
          >
            <span class="text-sm text-gray-300">{{ className }}</span>
            <span class="text-sm font-semibold text-primary">{{ count }}</span>
          </div>
        </div>
      </div>

      <!-- Quality Metrics -->
      <div v-if="currentJob.quality_metrics" class="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
        <div class="bg-background-tertiary rounded-lg p-3">
          <p class="text-xs text-gray-500 uppercase">{{ t('tools.labeling.metrics.avgConfidence') }}</p>
          <p class="text-lg font-semibold text-white">
            {{ (currentJob.quality_metrics.avg_confidence * 100).toFixed(1) }}%
          </p>
        </div>
        <div class="bg-background-tertiary rounded-lg p-3">
          <p class="text-xs text-gray-500 uppercase">{{ t('tools.labeling.metrics.withDetections') }}</p>
          <p class="text-lg font-semibold text-green-400">
            {{ currentJob.quality_metrics.images_with_detections }}
          </p>
        </div>
        <div class="bg-background-tertiary rounded-lg p-3">
          <p class="text-xs text-gray-500 uppercase">{{ t('tools.labeling.metrics.noDetections') }}</p>
          <p class="text-lg font-semibold text-yellow-400">
            {{ currentJob.quality_metrics.images_without_detections }}
          </p>
        </div>
        <div class="bg-background-tertiary rounded-lg p-3">
          <p class="text-xs text-gray-500 uppercase">{{ t('tools.labeling.metrics.lowConfidence') }}</p>
          <p class="text-lg font-semibold text-orange-400">
            {{ currentJob.quality_metrics.low_confidence_count }}
          </p>
        </div>
      </div>

      <!-- Warnings -->
      <AlertBox
        v-if="currentJob.warnings && currentJob.warnings.length > 0"
        type="warning"
        :title="t('tools.labeling.warnings.title')"
        class="mt-4"
      >
        <ul class="list-disc list-inside text-sm">
          <li v-for="(warning, idx) in currentJob.warnings" :key="idx">{{ warning }}</li>
        </ul>
      </AlertBox>

      <!-- Recent Previews -->
      <div v-if="recentPreviews.length > 0" class="mt-4">
        <h4 class="text-sm font-medium text-gray-400 mb-2">{{ t('tools.labeling.previews.title') }}</h4>
        <div class="grid grid-cols-5 gap-2">
          <button
            v-for="preview in recentPreviews"
            :key="preview.filename"
            @click="selectedPreview = preview"
            class="relative aspect-square rounded overflow-hidden hover:ring-2 ring-primary transition-all"
          >
            <img :src="`data:image/jpeg;base64,${preview.image_data}`" class="w-full h-full object-cover" />
          </button>
        </div>
      </div>
    </div>

    <!-- Preview Modal -->
    <div
      v-if="selectedPreview"
      class="fixed inset-0 bg-black/80 flex items-center justify-center z-50"
      @click="selectedPreview = null"
    >
      <div class="max-w-4xl max-h-[90vh] p-4" @click.stop>
        <img
          :src="`data:image/jpeg;base64,${selectedPreview.image_data}`"
          class="max-w-full max-h-[85vh] rounded-lg shadow-2xl"
        />
        <p class="text-center text-gray-400 mt-2">{{ selectedPreview.filename }}</p>
      </div>
    </div>

    <!-- Start Button -->
    <div class="flex justify-end">
      <BaseButton
        :loading="loading"
        :disabled="loading || (currentJob?.status === 'running')"
        @click="startJob"
        size="lg"
      >
        <Play class="h-5 w-5" />
        {{ isRelabeling ? t('tools.labeling.actions.startRelabeling') : t('tools.labeling.actions.startLabeling') }}
      </BaseButton>
    </div>
  </div>
</template>
