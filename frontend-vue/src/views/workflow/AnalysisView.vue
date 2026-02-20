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
  PackagePlus,
} from 'lucide-vue-next'
import { useI18n } from 'vue-i18n'
import type { DatasetInfo, DatasetAnalysis } from '@/types/api'

const router = useRouter()
const workflowStore = useWorkflowStore()
const uiStore = useUiStore()
const { t } = useI18n()

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
    workflowStore.disableCustomObjectsMode()
    workflowStore.setDatasetAnalysis(analysis.value)
    workflowStore.sourceDatasetPath = selectedDatasetPath.value
    workflowStore.markStepCompleted(1)
    uiStore.showSuccess(t('workflow.analysis.notifications.analysisComplete'), t('workflow.analysis.notifications.analysisCompleteMsg'))
  } catch (e: any) {
    error.value = e.message || t('workflow.analysis.notifications.analysisFailed')
    uiStore.showError(t('workflow.analysis.notifications.analysisFailed'), error.value)
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
    uiStore.showError(t('workflow.analysis.notifications.invalidFile'), t('workflow.analysis.notifications.invalidFileMsg'))
    return
  }

  uploadedFile.value = file
  loading.value = true

  try {
    const result = await uploadDataset(file)
    selectedDatasetPath.value = result.path
    await loadDatasets()
    uiStore.showSuccess(t('workflow.analysis.notifications.uploadComplete'), t('workflow.analysis.notifications.uploadCompleteMsg'))

    // Automatically analyze the uploaded dataset
    await analyzeSelected()
  } catch (e: any) {
    error.value = e.message || t('workflow.analysis.notifications.uploadFailed')
    uiStore.showError(t('workflow.analysis.notifications.uploadFailed'), error.value)
  } finally {
    loading.value = false
  }
}

// Skip COCO dataset and use custom objects mode
function skipToCustomObjects() {
  workflowStore.enableCustomObjectsMode()
  workflowStore.markStepCompleted(1)
  uiStore.showSuccess(t('workflow.analysis.notifications.customModeEnabled'), t('workflow.analysis.notifications.customModeEnabledMsg'))
  router.push('/configure')
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
      <h2 class="text-2xl font-bold text-white">{{ t('workflow.analysis.title') }}</h2>
      <p class="mt-2 text-gray-400">
        {{ t('workflow.analysis.subtitle') }}
      </p>
    </div>

    <!-- Error Alert -->
    <AlertBox v-if="error" type="error" :title="error" dismissible @dismiss="error = null" />

    <!-- Custom Objects Mode - Skip COCO -->
    <div class="card p-6 border border-dashed border-primary/40 bg-primary/5">
      <div class="flex items-start gap-4">
        <div class="p-3 rounded-lg bg-primary/10">
          <PackagePlus class="h-8 w-8 text-primary" />
        </div>
        <div class="flex-1">
          <h3 class="text-lg font-semibold text-white mb-1">{{ t('workflow.analysis.customMode.title') }}</h3>
          <p class="text-sm text-gray-400 mb-4">
            {{ t('workflow.analysis.customMode.description') }}
          </p>
          <BaseButton variant="outline" @click="skipToCustomObjects">
            <PackagePlus class="h-5 w-5" />
            {{ t('workflow.analysis.customMode.button') }}
          </BaseButton>
        </div>
      </div>
    </div>

    <!-- Separator -->
    <div class="flex items-center gap-4">
      <div class="flex-1 h-px bg-gray-700" />
      <span class="text-sm text-gray-500 uppercase tracking-wider">{{ t('workflow.analysis.orSeparator') }}</span>
      <div class="flex-1 h-px bg-gray-700" />
    </div>

    <!-- Upload Section -->
    <div class="card p-6">
      <h3 class="text-lg font-semibold text-white mb-4">{{ t('workflow.analysis.uploadDataset') }}</h3>
      <div class="flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-gray-600 p-8 hover:border-primary transition-colors">
        <Upload class="h-12 w-12 text-gray-400 mb-4" />
        <p class="text-gray-400 mb-4">{{ t('workflow.analysis.dropzone') }}</p>
        <label class="btn-primary cursor-pointer">
          <FileJson class="h-5 w-5" />
          {{ t('workflow.analysis.selectDataset') }}
          <input
            type="file"
            accept=".json"
            class="hidden"
            @change="handleFileUpload"
          />
        </label>
        <p v-if="uploadedFile" class="mt-4 text-sm text-green-400">
          {{ t('workflow.analysis.notifications.uploadComplete') }}: {{ uploadedFile.name }}
        </p>
      </div>
    </div>

    <!-- Select Existing Dataset -->
    <div class="card p-6">
      <h3 class="text-lg font-semibold text-white mb-4">{{ t('workflow.analysis.selectExisting') }}</h3>

      <div v-if="loading" class="flex justify-center py-8">
        <LoadingSpinner :message="t('workflow.analysis.analyzing')" />
      </div>

      <div v-else-if="datasets.length === 0" class="text-center py-8 text-gray-400">
        {{ t('workflow.analysis.noDatasets') }}
      </div>

      <div v-else class="space-y-4">
        <BaseSelect
          v-model="selectedDatasetPath"
          :options="datasetOptions"
          :label="t('workflow.analysis.selectDataset')"
          :placeholder="t('workflow.analysis.selectDatasetPlaceholder')"
        />

        <BaseButton
          :disabled="!selectedDatasetPath"
          :loading="analyzing"
          @click="analyzeSelected"
        >
          <BarChart3 class="h-5 w-5" />
          {{ t('workflow.analysis.analyzeButton') }}
        </BaseButton>
      </div>
    </div>

    <!-- Analysis Results -->
    <div v-if="analysis" class="space-y-6">
      <h3 class="text-lg font-semibold text-white">{{ t('workflow.analysis.results.title') }}</h3>

      <!-- Metrics -->
      <div class="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          :title="t('workflow.analysis.results.totalImages')"
          :value="analysis.total_images"
          :icon="Image"
        />
        <MetricCard
          :title="t('workflow.analysis.results.totalAnnotations')"
          :value="analysis.total_annotations"
          :icon="Layers"
        />
        <MetricCard
          :title="t('workflow.analysis.results.categories')"
          :value="analysis.categories.length"
          :icon="Tag"
        />
        <MetricCard
          :title="t('workflow.analysis.results.avgAnnotationsPerImage')"
          :value="analysis.annotations_per_image.mean.toFixed(1)"
          :icon="Database"
        />
      </div>

      <!-- Category Distribution -->
      <div class="card p-6">
        <h4 class="text-md font-semibold text-white mb-4">{{ t('workflow.analysis.categoryDistribution') }}</h4>
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
          {{ t('workflow.analysis.continueButton') }}
          <ArrowRight class="h-5 w-5" />
        </BaseButton>
      </div>
    </div>
  </div>
</template>
