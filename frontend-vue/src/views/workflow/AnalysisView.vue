<script setup lang="ts">
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useWorkflowStore } from '@/stores/workflow'
import { useUiStore } from '@/stores/ui'
import { analyzeDataset, listDatasets, uploadDataset } from '@/lib/api'
import MetricCard from '@/components/common/MetricCard.vue'
import AlertBox from '@/components/common/AlertBox.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseSelect from '@/components/ui/BaseSelect.vue'
import {
  Upload,
  Database,
  Image,
  Tag,
  Layers,
  ArrowRight,
  FileJson,
  BarChart3,
} from 'lucide-vue-next'
import type { DatasetInfo, DatasetAnalysis } from '@/types/api'

const router = useRouter()
const workflowStore = useWorkflowStore()
const uiStore = useUiStore()

const loading = ref(false)
const analyzing = ref(false)
const datasets = ref<DatasetInfo[]>([])
const selectedDatasetPath = ref<string | null>(null)
const analysis = ref<DatasetAnalysis | null>(null)
const uploadedFile = ref<File | null>(null)
const error = ref<string | null>(null)

// Load available datasets
async function loadDatasets() {
  loading.value = true
  try {
    datasets.value = await listDatasets()
  } catch (e) {
    error.value = 'Failed to load datasets'
  } finally {
    loading.value = false
  }
}

// Analyze selected dataset
async function analyzeSelected() {
  if (!selectedDatasetPath.value) return

  analyzing.value = true
  error.value = null

  try {
    analysis.value = await analyzeDataset(selectedDatasetPath.value)
    workflowStore.setDatasetAnalysis(analysis.value)
    workflowStore.sourceDatasetPath = selectedDatasetPath.value
    workflowStore.markStepCompleted(1)
    uiStore.showSuccess('Analysis Complete', 'Dataset analyzed successfully')
  } catch (e: any) {
    error.value = e.message || 'Failed to analyze dataset'
    uiStore.showError('Analysis Failed', error.value)
  } finally {
    analyzing.value = false
  }
}

// Handle file upload
async function handleFileUpload(event: Event) {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (!file) return

  if (!file.name.endsWith('.json')) {
    uiStore.showError('Invalid File', 'Please upload a COCO JSON file')
    return
  }

  uploadedFile.value = file
  loading.value = true

  try {
    const result = await uploadDataset(file)
    selectedDatasetPath.value = result.path
    await loadDatasets()
    uiStore.showSuccess('Upload Complete', 'Dataset uploaded successfully. Analyzing...')

    // Automatically analyze the uploaded dataset
    await analyzeSelected()
  } catch (e: any) {
    error.value = e.message || 'Failed to upload dataset'
    uiStore.showError('Upload Failed', error.value)
  } finally {
    loading.value = false
  }
}

// Continue to next step
function continueToNext() {
  router.push('/configure')
}

// Dataset options for select
const datasetOptions = computed(() =>
  datasets.value.map(d => ({
    value: d.path,
    label: `${d.name} (${d.num_images} images)`,
  }))
)

// Load datasets on mount
loadDatasets()
</script>

<template>
  <div class="space-y-8">
    <!-- Header -->
    <div>
      <h2 class="text-2xl font-bold text-white">Dataset Analysis</h2>
      <p class="mt-2 text-gray-400">
        Upload or select a COCO dataset to analyze its structure and prepare for generation.
      </p>
    </div>

    <!-- Error Alert -->
    <AlertBox v-if="error" type="error" :title="error" dismissible @dismiss="error = null" />

    <!-- Upload Section -->
    <div class="card p-6">
      <h3 class="text-lg font-semibold text-white mb-4">Upload Dataset</h3>
      <div class="flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-gray-600 p-8 hover:border-primary transition-colors">
        <Upload class="h-12 w-12 text-gray-400 mb-4" />
        <p class="text-gray-400 mb-4">Drag and drop a COCO JSON file or click to browse</p>
        <label class="btn-primary cursor-pointer">
          <FileJson class="h-5 w-5" />
          Select File
          <input
            type="file"
            accept=".json"
            class="hidden"
            @change="handleFileUpload"
          />
        </label>
        <p v-if="uploadedFile" class="mt-4 text-sm text-green-400">
          Uploaded: {{ uploadedFile.name }}
        </p>
      </div>
    </div>

    <!-- Select Existing Dataset -->
    <div class="card p-6">
      <h3 class="text-lg font-semibold text-white mb-4">Or Select Existing Dataset</h3>

      <div v-if="loading" class="flex justify-center py-8">
        <LoadingSpinner message="Loading datasets..." />
      </div>

      <div v-else-if="datasets.length === 0" class="text-center py-8 text-gray-400">
        No datasets found. Upload a dataset to get started.
      </div>

      <div v-else class="space-y-4">
        <BaseSelect
          v-model="selectedDatasetPath"
          :options="datasetOptions"
          label="Select Dataset"
          placeholder="Choose a dataset..."
        />

        <BaseButton
          :disabled="!selectedDatasetPath"
          :loading="analyzing"
          @click="analyzeSelected"
        >
          <BarChart3 class="h-5 w-5" />
          Analyze Dataset
        </BaseButton>
      </div>
    </div>

    <!-- Analysis Results -->
    <div v-if="analysis" class="space-y-6">
      <h3 class="text-lg font-semibold text-white">Analysis Results</h3>

      <!-- Metrics -->
      <div class="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Total Images"
          :value="analysis.total_images"
          :icon="Image"
        />
        <MetricCard
          title="Total Annotations"
          :value="analysis.total_annotations"
          :icon="Layers"
        />
        <MetricCard
          title="Categories"
          :value="analysis.categories.length"
          :icon="Tag"
        />
        <MetricCard
          title="Avg. Annotations/Image"
          :value="analysis.annotations_per_image.mean.toFixed(1)"
          :icon="Database"
        />
      </div>

      <!-- Category Distribution -->
      <div class="card p-6">
        <h4 class="text-md font-semibold text-white mb-4">Category Distribution</h4>
        <div class="space-y-3">
          <div
            v-for="category in analysis.categories"
            :key="category.id"
            class="flex items-center gap-4"
          >
            <span class="w-32 text-sm text-gray-300 truncate">{{ category.name }}</span>
            <div class="flex-1 h-4 bg-background-tertiary rounded-full overflow-hidden">
              <div
                class="h-full bg-primary rounded-full transition-all"
                :style="{
                  width: `${(category.count / analysis.total_annotations) * 100}%`
                }"
              />
            </div>
            <span class="w-20 text-sm text-gray-400 text-right">{{ category.count }}</span>
          </div>
        </div>
      </div>

      <!-- Continue Button -->
      <div class="flex justify-end">
        <BaseButton @click="continueToNext">
          Continue to Configuration
          <ArrowRight class="h-5 w-5" />
        </BaseButton>
      </div>
    </div>
  </div>
</template>
