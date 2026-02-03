<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useUiStore } from '@/stores/ui'
import {
  startLabeling,
  startRelabeling,
  getLabelingStatus,
  getActiveLabelingJobs,
  listDirectories,
} from '@/lib/api'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseInput from '@/components/ui/BaseInput.vue'
import BaseSelect from '@/components/ui/BaseSelect.vue'
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

// Predefined labeling templates
interface LabelingTemplate {
  id: string
  name: string
  icon: any
  description: string
  classes: string[]
  taskType: LabelingTaskType
  confidence: number
}

const labelingTemplates: LabelingTemplate[] = [
  {
    id: 'marine_life',
    name: 'Marine Life',
    icon: Fish,
    description: 'Fish, coral, sea creatures',
    classes: ['fish', 'coral', 'shark', 'turtle', 'jellyfish', 'octopus', 'starfish', 'crab', 'dolphin', 'whale'],
    taskType: 'both',
    confidence: 0.35,
  },
  {
    id: 'vehicles',
    name: 'Vehicles',
    icon: Car,
    description: 'Cars, trucks, motorcycles',
    classes: ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'boat', 'airplane', 'train'],
    taskType: 'detection',
    confidence: 0.4,
  },
  {
    id: 'people',
    name: 'People & Poses',
    icon: Users,
    description: 'Person detection and poses',
    classes: ['person', 'face', 'hand', 'head'],
    taskType: 'both',
    confidence: 0.35,
  },
  {
    id: 'urban',
    name: 'Urban Scene',
    icon: Building,
    description: 'Buildings, streets, signs',
    classes: ['building', 'traffic light', 'stop sign', 'street sign', 'bench', 'parking meter', 'fire hydrant'],
    taskType: 'detection',
    confidence: 0.4,
  },
  {
    id: 'nature',
    name: 'Nature & Wildlife',
    icon: TreePine,
    description: 'Animals, plants, landscapes',
    classes: ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'tree', 'flower'],
    taskType: 'both',
    confidence: 0.35,
  },
  {
    id: 'food',
    name: 'Food & Kitchen',
    icon: Utensils,
    description: 'Food items and kitchenware',
    classes: ['apple', 'banana', 'orange', 'pizza', 'sandwich', 'cake', 'cup', 'bowl', 'bottle', 'knife', 'fork', 'spoon'],
    taskType: 'detection',
    confidence: 0.4,
  },
  {
    id: 'fashion',
    name: 'Fashion & Clothing',
    icon: Shirt,
    description: 'Clothes and accessories',
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
  uiStore.showSuccess('Template Applied', `Loaded ${template.name} template with ${template.classes.length} classes`)
}

function clearTemplate() {
  selectedTemplate.value = null
  classes.value = ['']
}

// Form state
const imageDirectories = ref<string[]>([''])
const classes = ref<string[]>([''])
const outputDir = ref('/data/labeled')
const minConfidence = ref(0.3)
const taskType = ref<LabelingTaskType>('detection')
const useSam2 = ref(true)
const boxThreshold = ref(0.25)
const textThreshold = ref(0.25)

// Relabeling state
const isRelabeling = ref(false)
const relabelMode = ref<RelabelMode>('add')
const existingAnnotations = ref('')

// UI state
const loading = ref(false)
const directories = ref<string[]>([])
const activeJobs = ref<LabelingJob[]>([])
const currentJob = ref<LabelingJob | null>(null)
const error = ref<string | null>(null)
let pollingInterval: ReturnType<typeof setInterval> | null = null

const taskTypeOptions = [
  { value: 'detection', label: 'Detection (Bounding Boxes)' },
  { value: 'segmentation', label: 'Segmentation (Masks)' },
  { value: 'both', label: 'Both Detection & Segmentation' },
]

const relabelModeOptions = [
  { value: 'add', label: 'Add - Keep existing, add new classes' },
  { value: 'replace', label: 'Replace - Remove all, relabel everything' },
  { value: 'improve_segmentation', label: 'Improve - Enhance segmentation quality' },
]

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
    uiStore.showError('Missing Input', 'Please specify at least one image directory')
    return
  }

  if (!isRelabeling.value && validClasses.length === 0) {
    uiStore.showError('Missing Input', 'Please specify at least one class to detect')
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
        existing_annotations: existingAnnotations.value || undefined,
        task_type: taskType.value,
        use_sam2: useSam2.value,
      })
    } else {
      response = await startLabeling({
        image_directories: validDirs,
        classes: validClasses,
        output_dir: outputDir.value,
        min_confidence: minConfidence.value,
        task_type: taskType.value,
        use_sam2: useSam2.value,
        box_threshold: boxThreshold.value,
        text_threshold: textThreshold.value,
      })
    }

    uiStore.showSuccess('Job Started', `Labeling job ${response.job_id.slice(0, 8)} started`)
    startPolling(response.job_id)
  } catch (e: any) {
    error.value = e.message || 'Failed to start labeling job'
    uiStore.showError('Job Failed', error.value)
  } finally {
    loading.value = false
  }
}

async function pollJobStatus(jobId: string) {
  try {
    const job = await getLabelingStatus(jobId)
    currentJob.value = job

    if (job.status === 'completed' || job.status === 'failed') {
      stopPolling()
      if (job.status === 'completed') {
        uiStore.showSuccess('Labeling Complete', `Processed ${job.processed_images} images`)
      } else {
        uiStore.showError('Labeling Failed', job.error || 'Unknown error')
      }
    }
  } catch (e) {
    // Ignore polling errors
  }
}

function startPolling(jobId: string) {
  pollingInterval = setInterval(() => pollJobStatus(jobId), 2000)
}

function stopPolling() {
  if (pollingInterval) {
    clearInterval(pollingInterval)
    pollingInterval = null
  }
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
      <h2 class="text-2xl font-bold text-white">Auto Labeling</h2>
      <p class="mt-1 text-gray-400">
        Automatically label images using AI models (GroundingDINO + SAM2)
      </p>
    </div>

    <!-- Active Jobs Alert -->
    <AlertBox v-if="activeJobs.length > 0" type="info" title="Active Jobs">
      {{ activeJobs.length }} labeling job(s) are currently running.
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
        New Labeling
      </BaseButton>
      <BaseButton
        :variant="isRelabeling ? 'primary' : 'outline'"
        @click="isRelabeling = true"
      >
        <RefreshCw class="h-5 w-5" />
        Relabeling
      </BaseButton>
    </div>

    <!-- Templates Section (only for new labeling) -->
    <div v-if="!isRelabeling" class="card p-6">
      <div class="flex items-center justify-between mb-4">
        <div>
          <h3 class="text-lg font-semibold text-white flex items-center gap-2">
            <Sparkles class="h-5 w-5 text-yellow-400" />
            Quick Templates
          </h3>
          <p class="text-sm text-gray-400">Select a predefined template or configure manually</p>
        </div>
        <BaseButton
          v-if="selectedTemplate"
          variant="ghost"
          size="sm"
          @click="clearTemplate"
        >
          <X class="h-4 w-4" />
          Clear
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
            <p class="text-sm font-medium text-white">{{ template.name }}</p>
            <p class="text-xs text-gray-400">{{ template.description }}</p>
          </div>
        </button>
      </div>
    </div>

    <div class="grid gap-6 lg:grid-cols-2">
      <!-- Input Configuration -->
      <div class="card p-6">
        <h3 class="text-lg font-semibold text-white mb-4">Image Directories</h3>

        <div class="space-y-3">
          <div
            v-for="(dir, index) in imageDirectories"
            :key="index"
            class="flex gap-2"
          >
            <BaseInput
              v-model="imageDirectories[index]"
              placeholder="/data/images"
              class="flex-1"
            />
            <BaseButton
              v-if="imageDirectories.length > 1"
              variant="ghost"
              size="sm"
              @click="removeDirectory(index)"
            >
              <X class="h-4 w-4" />
            </BaseButton>
          </div>
          <BaseButton variant="outline" size="sm" @click="addDirectory">
            <Plus class="h-4 w-4" />
            Add Directory
          </BaseButton>
        </div>

        <!-- Classes (for new labeling) -->
        <div v-if="!isRelabeling" class="mt-6">
          <h4 class="text-md font-medium text-white mb-3">Classes to Detect</h4>
          <div class="space-y-3">
            <div
              v-for="(cls, index) in classes"
              :key="index"
              class="flex gap-2"
            >
              <BaseInput
                v-model="classes[index]"
                placeholder="e.g., person, car, dog"
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
              Add Class
            </BaseButton>
          </div>
        </div>

        <!-- Relabeling Options -->
        <div v-else class="mt-6 space-y-4">
          <BaseSelect
            v-model="relabelMode"
            :options="relabelModeOptions"
            label="Relabel Mode"
          />
          <BaseInput
            v-model="existingAnnotations"
            label="Existing Annotations Path (optional)"
            placeholder="/data/annotations.json"
          />
        </div>
      </div>

      <!-- Output & Model Settings -->
      <div class="card p-6">
        <h3 class="text-lg font-semibold text-white mb-4">Output Settings</h3>

        <div class="space-y-4">
          <BaseInput
            v-model="outputDir"
            label="Output Directory"
            placeholder="/data/labeled"
          />

          <BaseSelect
            v-model="taskType"
            :options="taskTypeOptions"
            label="Task Type"
          />

          <div>
            <label class="text-sm text-gray-400 flex justify-between mb-2">
              <span>Min Confidence</span>
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

          <label class="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              v-model="useSam2"
              class="h-5 w-5 rounded border-gray-600 bg-background-tertiary text-primary focus:ring-primary"
            />
            <div>
              <span class="text-gray-300">Use SAM2 for Segmentation</span>
              <p class="text-sm text-gray-500">Generate precise segmentation masks</p>
            </div>
          </label>
        </div>
      </div>
    </div>

    <!-- Current Job Progress -->
    <div v-if="currentJob" class="card p-6">
      <h3 class="text-lg font-semibold text-white mb-4">Current Job Progress</h3>

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
            {{ currentJob.status === 'running' ? 'Processing...' : currentJob.status }}
          </p>
          <p class="text-sm text-gray-400">
            {{ currentJob.processed_images }}/{{ currentJob.total_images }} images |
            {{ currentJob.annotations_created }} annotations
          </p>
        </div>
        <span class="text-2xl font-bold text-primary">{{ currentJob.progress }}%</span>
      </div>

      <div class="h-3 bg-background-tertiary rounded-full overflow-hidden">
        <div
          class="h-full bg-gradient-to-r from-primary to-green-400 transition-all duration-500"
          :style="{ width: `${currentJob.progress}%` }"
        />
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
        {{ isRelabeling ? 'Start Relabeling' : 'Start Labeling' }}
      </BaseButton>
    </div>
  </div>
</template>
