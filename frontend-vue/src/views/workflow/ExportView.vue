<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useWorkflowStore } from '@/stores/workflow'
import { useUiStore } from '@/stores/ui'
import { exportDataset, listDatasets, analyzeDataset } from '@/lib/api'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseSelect from '@/components/ui/BaseSelect.vue'
import BaseInput from '@/components/ui/BaseInput.vue'
import DirectoryBrowser from '@/components/common/DirectoryBrowser.vue'
import AlertBox from '@/components/common/AlertBox.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import {
  ArrowRight,
  ArrowLeft,
  Download,
  FileJson,
  FileText,
  Database,
  CheckCircle,
  FolderOpen,
} from 'lucide-vue-next'
import type { ExportFormat, DatasetInfo, DatasetAnalysis, ExportResult } from '@/types/api'

const router = useRouter()
const workflowStore = useWorkflowStore()
const uiStore = useUiStore()

const loading = ref(false)
const exporting = ref(false)
const datasets = ref<DatasetInfo[]>([])
const selectedDataset = ref<string | null>(workflowStore.outputDir)
const selectedImagesDir = ref('')
const datasetAnalysis = ref<DatasetAnalysis | null>(null)
const exportFormat = ref<ExportFormat>('yolo')
const outputDir = ref('/app/output/exports')
const includeImages = ref(true)
const exportResult = ref<ExportResult | null>(null)
const error = ref<string | null>(null)

const formatOptions = [
  { value: 'yolo', label: 'YOLO (Ultralytics)' },
  { value: 'voc', label: 'Pascal VOC (XML)' },
  { value: 'coco', label: 'COCO JSON' },
]

const formatIcons: Record<string, any> = {
  coco: FileJson,
  yolo: FileText,
  voc: Database,
}

const formatDescriptions: Record<string, string> = {
  yolo: 'Normalized bbox coordinates in TXT files. Creates data.yaml for Ultralytics training.',
  voc: 'XML annotation files per image. Compatible with Pascal VOC format tools.',
  coco: 'Standard COCO format JSON. Copy with optional images.',
}

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

async function loadDatasetAnalysis() {
  if (!selectedDataset.value) return

  loading.value = true
  try {
    datasetAnalysis.value = await analyzeDataset(selectedDataset.value)

    // Try to infer images directory from dataset path
    const datasetPath = selectedDataset.value
    const dirPath = datasetPath.substring(0, datasetPath.lastIndexOf('/'))
    selectedImagesDir.value = `${dirPath}/images`
  } catch (e) {
    // Ignore
  } finally {
    loading.value = false
  }
}

watch(selectedDataset, () => {
  loadDatasetAnalysis()
})

const datasetOptions = computed(() =>
  datasets.value.map(d => ({
    value: d.path,
    label: `${d.name} (${d.num_images} images)`,
  }))
)

async function startExport() {
  if (!selectedDataset.value) {
    uiStore.showError('Missing Dataset', 'Please select a dataset to export')
    return
  }

  if (!selectedImagesDir.value) {
    uiStore.showError('Missing Images Directory', 'Please specify the images directory')
    return
  }

  exporting.value = true
  error.value = null
  exportResult.value = null

  try {
    const result = await exportDataset({
      source_path: selectedDataset.value,
      images_dir: selectedImagesDir.value,
      output_dir: outputDir.value,
      format: exportFormat.value,
      include_images: includeImages.value,
    })

    exportResult.value = result
    workflowStore.markStepCompleted(5)
    uiStore.showSuccess('Export Complete', `Dataset exported to ${result.output_dir}`)
  } catch (e: any) {
    error.value = e.message || 'Failed to export dataset'
    uiStore.showError('Export Failed', error.value)
  } finally {
    exporting.value = false
  }
}

function goBack() {
  router.push('/generation')
}

function continueToCombine() {
  router.push('/combine')
}

// Load datasets on mount
loadDatasets()
</script>

<template>
  <div class="space-y-8">
    <!-- Header -->
    <div>
      <h2 class="text-2xl font-bold text-white">Export Dataset</h2>
      <p class="mt-2 text-gray-400">
        Export your dataset to various formats for use in different ML frameworks.
      </p>
    </div>

    <!-- Error Alert -->
    <AlertBox v-if="error" type="error" :title="error" dismissible @dismiss="error = null" />

    <!-- Export Configuration -->
    <div class="grid gap-6 lg:grid-cols-2">
      <!-- Dataset Selection -->
      <div class="card p-6">
        <h3 class="text-lg font-semibold text-white mb-4">Select Dataset</h3>

        <div v-if="loading && datasets.length === 0" class="flex justify-center py-8">
          <LoadingSpinner message="Loading datasets..." />
        </div>

        <div v-else class="space-y-4">
          <BaseSelect
            v-model="selectedDataset"
            :options="datasetOptions"
            label="Dataset (COCO JSON)"
            placeholder="Choose a dataset to export..."
          />

          <DirectoryBrowser
            v-model="selectedImagesDir"
            label="Images Directory"
            placeholder="/app/datasets/images"
            path-mode="input"
          />

          <!-- Dataset Info -->
          <div v-if="datasetAnalysis" class="mt-4 p-4 bg-background-tertiary rounded-lg">
            <p class="text-sm text-gray-400">
              <span class="text-white font-medium">{{ datasetAnalysis.total_images }}</span> images,
              <span class="text-white font-medium">{{ datasetAnalysis.total_annotations }}</span> annotations,
              <span class="text-white font-medium">{{ datasetAnalysis.categories.length }}</span> categories
            </p>
          </div>
        </div>
      </div>

      <!-- Export Options -->
      <div class="card p-6">
        <h3 class="text-lg font-semibold text-white mb-4">Export Options</h3>

        <div class="space-y-4">
          <BaseSelect
            v-model="exportFormat"
            :options="formatOptions"
            label="Export Format"
          />

          <DirectoryBrowser
            v-model="outputDir"
            label="Output Directory"
            placeholder="/app/output/exports"
            path-mode="output"
          />

          <label class="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              v-model="includeImages"
              class="h-5 w-5 rounded border-gray-600 bg-background-tertiary text-primary focus:ring-primary"
            />
            <span class="text-gray-300">Include images in export</span>
          </label>
        </div>
      </div>
    </div>

    <!-- Format Information -->
    <div class="card p-6">
      <h3 class="text-lg font-semibold text-white mb-4">Format Details</h3>
      <div class="grid gap-4 sm:grid-cols-3">
        <div
          v-for="format in formatOptions"
          :key="format.value"
          :class="[
            'rounded-lg p-4 transition-colors cursor-pointer',
            exportFormat === format.value
              ? 'bg-primary/20 border-2 border-primary'
              : 'bg-background-tertiary border-2 border-transparent hover:bg-gray-600',
          ]"
          @click="exportFormat = format.value as ExportFormat"
        >
          <div class="flex items-center gap-3 mb-2">
            <component :is="formatIcons[format.value]" class="h-5 w-5 text-primary" />
            <span class="font-medium text-white">{{ format.label }}</span>
          </div>
          <p class="text-sm text-gray-400">
            {{ formatDescriptions[format.value] }}
          </p>
        </div>
      </div>
    </div>

    <!-- Export Result -->
    <div v-if="exportResult" class="card p-6">
      <div class="flex items-center gap-4 mb-4">
        <CheckCircle class="h-8 w-8 text-green-400" />
        <div>
          <h3 class="text-lg font-semibold text-white">Export Complete</h3>
          <p class="text-gray-400">Dataset successfully exported to {{ exportResult.format.toUpperCase() }} format</p>
        </div>
      </div>

      <div class="grid gap-4 sm:grid-cols-3 mt-4">
        <div class="p-4 bg-background-tertiary rounded-lg text-center">
          <p class="text-2xl font-bold text-primary">{{ exportResult.images_exported }}</p>
          <p class="text-sm text-gray-400">Images Exported</p>
        </div>
        <div class="p-4 bg-background-tertiary rounded-lg text-center">
          <p class="text-2xl font-bold text-primary">{{ exportResult.annotations_exported }}</p>
          <p class="text-sm text-gray-400">Annotations</p>
        </div>
        <div class="p-4 bg-background-tertiary rounded-lg text-center">
          <p class="text-lg font-medium text-white truncate" :title="exportResult.output_dir">
            {{ exportResult.output_dir.split('/').pop() }}
          </p>
          <p class="text-sm text-gray-400">Output Directory</p>
        </div>
      </div>

      <!-- YOLO-specific info -->
      <div v-if="exportResult.yaml_file" class="mt-4 p-4 bg-green-900/20 border border-green-700/50 rounded-lg">
        <p class="text-sm text-gray-300">
          <span class="text-green-400 font-medium">YOLO Config:</span>
          {{ exportResult.yaml_file }}
        </p>
        <p class="text-sm text-gray-400 mt-1">
          Use this data.yaml file to train with Ultralytics YOLO.
        </p>
      </div>
    </div>

    <!-- Action Buttons -->
    <div class="flex justify-between">
      <BaseButton variant="outline" @click="goBack">
        <ArrowLeft class="h-5 w-5" />
        Back
      </BaseButton>

      <div class="flex gap-4">
        <BaseButton
          v-if="!exportResult"
          :disabled="!selectedDataset || !selectedImagesDir"
          :loading="exporting"
          @click="startExport"
        >
          <Download class="h-5 w-5" />
          Export Dataset
        </BaseButton>

        <BaseButton
          v-if="exportResult"
          @click="continueToCombine"
        >
          Continue to Combine
          <ArrowRight class="h-5 w-5" />
        </BaseButton>
      </div>
    </div>
  </div>
</template>
