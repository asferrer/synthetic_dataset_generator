<script setup lang="ts">
import { ref, onUnmounted } from 'vue'
import { useUiStore } from '@/stores/ui'
import { startExtraction, getExtractionStatus, listDatasets } from '@/lib/api'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseInput from '@/components/ui/BaseInput.vue'
import AlertBox from '@/components/common/AlertBox.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import DirectoryBrowser from '@/components/common/DirectoryBrowser.vue'
import {
  Scissors,
  Play,
  CheckCircle,
  Settings,
  Wand2,
  Copy,
  Box,
} from 'lucide-vue-next'
import type { DatasetInfo, Job } from '@/types/api'

const uiStore = useUiStore()

const loading = ref(false)
const extracting = ref(false)
const datasets = ref<DatasetInfo[]>([])
const cocoJsonPath = ref('')
const imagesDir = ref('')
const outputDir = ref('/data/extracted')
const minSize = ref(32)
const includeMasks = ref(true)
const useSam3 = ref(false)
const padding = ref(10)
const deduplicate = ref(false)
const showAdvanced = ref(false)
const currentJob = ref<Job | null>(null)
const error = ref<string | null>(null)
let pollingInterval: ReturnType<typeof setInterval> | null = null

async function loadDatasets() {
  loading.value = true
  try {
    datasets.value = await listDatasets()
  } catch (e) {
    // Ignore
  } finally {
    loading.value = false
  }
}

async function startExtractionJob() {
  if (!cocoJsonPath.value || !imagesDir.value) {
    uiStore.showError('Missing Input', 'Please specify COCO JSON path and images directory')
    return
  }

  extracting.value = true
  error.value = null
  currentJob.value = null

  try {
    const response = await startExtraction({
      coco_json_path: cocoJsonPath.value,
      images_dir: imagesDir.value,
      output_dir: outputDir.value,
      min_size: minSize.value,
      include_masks: includeMasks.value,
      use_sam3: useSam3.value,
      padding: padding.value,
      deduplicate: deduplicate.value,
    })

    uiStore.showSuccess('Extraction Started', `Job ${response.job_id.slice(0, 8)} started`)
    startPolling(response.job_id)
  } catch (e: any) {
    error.value = e.message || 'Failed to start extraction'
    uiStore.showError('Extraction Failed', error.value)
    extracting.value = false
  }
}

async function pollJobStatus(jobId: string) {
  try {
    const job = await getExtractionStatus(jobId)
    currentJob.value = job

    if (job.status === 'completed') {
      stopPolling()
      extracting.value = false
      uiStore.showSuccess('Extraction Complete', 'Objects extracted successfully')
    } else if (job.status === 'failed') {
      stopPolling()
      extracting.value = false
      error.value = job.error || 'Extraction failed'
      uiStore.showError('Extraction Failed', error.value)
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

onUnmounted(() => {
  stopPolling()
})

// Load datasets on mount
loadDatasets()
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div>
      <h2 class="text-2xl font-bold text-white">Object Extraction</h2>
      <p class="mt-1 text-gray-400">
        Extract individual objects from a dataset for use in synthetic generation.
      </p>
    </div>

    <!-- Error -->
    <AlertBox v-if="error" type="error" :title="error" dismissible @dismiss="error = null" />

    <div class="grid gap-6 lg:grid-cols-2">
      <!-- Input -->
      <div class="card p-6">
        <h3 class="text-lg font-semibold text-white mb-4">Source Dataset</h3>

        <div class="space-y-4">
          <DirectoryBrowser
            v-model="cocoJsonPath"
            label="COCO JSON Path"
            placeholder="/data/dataset/annotations.json"
            :show-files="true"
            file-pattern="*.json"
          />
          <DirectoryBrowser
            v-model="imagesDir"
            label="Images Directory"
            placeholder="/data/dataset/images"
          />
        </div>
      </div>

      <!-- Output -->
      <div class="card p-6">
        <h3 class="text-lg font-semibold text-white mb-4">Output Settings</h3>

        <div class="space-y-4">
          <DirectoryBrowser
            v-model="outputDir"
            label="Output Directory"
            placeholder="/data/extracted"
          />

          <div>
            <label class="text-sm text-gray-400 flex justify-between mb-2">
              <span>Minimum Object Size</span>
              <span class="text-white">{{ minSize }}px</span>
            </label>
            <input
              type="range"
              v-model.number="minSize"
              min="8"
              max="256"
              step="8"
              class="w-full accent-primary"
            />
          </div>

          <label class="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              v-model="includeMasks"
              class="h-5 w-5 rounded border-gray-600 bg-background-tertiary text-primary focus:ring-primary"
            />
            <div>
              <span class="text-gray-300">Include segmentation masks</span>
              <p class="text-xs text-gray-500">Extract masks along with object crops</p>
            </div>
          </label>
        </div>
      </div>
    </div>

    <!-- Advanced Options -->
    <div class="card p-6">
      <button
        class="flex items-center justify-between w-full"
        @click="showAdvanced = !showAdvanced"
      >
        <h3 class="text-lg font-semibold text-white flex items-center gap-2">
          <Settings class="h-5 w-5" />
          Advanced Options
        </h3>
        <span class="text-gray-400">{{ showAdvanced ? '▲' : '▼' }}</span>
      </button>

      <div v-if="showAdvanced" class="mt-6 space-y-6">
        <div class="grid gap-6 md:grid-cols-2">
          <!-- SAM3 Option -->
          <div class="p-4 bg-background-tertiary rounded-lg">
            <label class="flex items-center gap-3 cursor-pointer">
              <input
                type="checkbox"
                v-model="useSam3"
                class="h-5 w-5 rounded border-gray-600 bg-background-primary text-primary focus:ring-primary"
              />
              <div class="flex-1">
                <div class="flex items-center gap-2">
                  <Wand2 class="h-4 w-4 text-yellow-400" />
                  <span class="text-gray-300 font-medium">Use SAM3 for masks</span>
                </div>
                <p class="text-xs text-gray-500 mt-1">
                  Generate precise segmentation masks using Segment Anything Model 3.
                  More accurate but slower.
                </p>
              </div>
            </label>
          </div>

          <!-- Deduplicate Option -->
          <div class="p-4 bg-background-tertiary rounded-lg">
            <label class="flex items-center gap-3 cursor-pointer">
              <input
                type="checkbox"
                v-model="deduplicate"
                class="h-5 w-5 rounded border-gray-600 bg-background-primary text-primary focus:ring-primary"
              />
              <div class="flex-1">
                <div class="flex items-center gap-2">
                  <Copy class="h-4 w-4 text-blue-400" />
                  <span class="text-gray-300 font-medium">Remove duplicates</span>
                </div>
                <p class="text-xs text-gray-500 mt-1">
                  Skip objects that are visually similar to already extracted ones.
                </p>
              </div>
            </label>
          </div>
        </div>

        <!-- Padding Slider -->
        <div class="p-4 bg-background-tertiary rounded-lg">
          <div class="flex items-center gap-2 mb-3">
            <Box class="h-4 w-4 text-green-400" />
            <label class="text-sm text-gray-300 font-medium">Padding around objects</label>
          </div>
          <div class="flex items-center gap-4">
            <input
              type="range"
              v-model.number="padding"
              min="0"
              max="50"
              step="5"
              class="flex-1 accent-primary"
            />
            <span class="text-white font-mono w-16 text-right">{{ padding }}px</span>
          </div>
          <p class="text-xs text-gray-500 mt-2">
            Extra pixels to include around each object bounding box. Helps with edge artifacts.
          </p>
        </div>
      </div>
    </div>

    <!-- Progress -->
    <div v-if="currentJob" class="card p-6">
      <h3 class="text-lg font-semibold text-white mb-4">Extraction Progress</h3>

      <div class="flex items-center gap-4 mb-4">
        <component
          :is="currentJob.status === 'completed' ? CheckCircle : Scissors"
          :class="[
            'h-8 w-8',
            currentJob.status === 'completed' ? 'text-green-400' : 'text-primary animate-pulse'
          ]"
        />
        <div class="flex-1">
          <p class="font-medium text-white">
            {{ currentJob.status === 'running' ? 'Extracting...' : currentJob.status }}
          </p>
          <p class="text-sm text-gray-400">Job ID: {{ currentJob.job_id.slice(0, 8) }}...</p>
        </div>
        <span class="text-2xl font-bold text-primary">{{ currentJob.progress }}%</span>
      </div>

      <div class="h-3 bg-background-tertiary rounded-full overflow-hidden">
        <div
          class="h-full bg-primary transition-all"
          :style="{ width: `${currentJob.progress}%` }"
        />
      </div>
    </div>

    <!-- Start Button -->
    <div class="flex justify-end">
      <BaseButton
        :loading="extracting"
        :disabled="extracting || !cocoJsonPath || !imagesDir"
        @click="startExtractionJob"
        size="lg"
      >
        <Scissors class="h-5 w-5" />
        Start Extraction
      </BaseButton>
    </div>
  </div>
</template>
