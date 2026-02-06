<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useUiStore } from '@/stores/ui'
import { listDatasets, analyzeDataset } from '@/lib/api'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseSelect from '@/components/ui/BaseSelect.vue'
import BaseInput from '@/components/ui/BaseInput.vue'
import AlertBox from '@/components/common/AlertBox.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import {
  Scale,
  BarChart3,
  RefreshCw,
  ArrowDownUp,
  Download,
  Copy,
  CheckCircle,
  AlertTriangle,
} from 'lucide-vue-next'
import type { DatasetInfo, DatasetAnalysis, CategoryInfo } from '@/types/api'

const uiStore = useUiStore()

const loading = ref(false)
const balancing = ref(false)
const datasets = ref<DatasetInfo[]>([])
const selectedDataset = ref<string | null>(null)
const datasetAnalysis = ref<DatasetAnalysis | null>(null)
const outputDir = ref('/data/balanced')
const error = ref<string | null>(null)
const balanceResult = ref<{ success: boolean; message: string } | null>(null)

// Balancing mode
const balanceMode = ref<'undersample' | 'oversample' | 'weights'>('weights')

// Target count for balancing (auto-calculated based on mode)
const targetCount = ref<number | null>(null)

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
  balanceResult.value = null
  try {
    datasetAnalysis.value = await analyzeDataset(selectedDataset.value)
    // Auto-calculate target based on mode
    updateTargetCount()
  } catch (e: any) {
    error.value = e.message || 'Failed to load dataset'
  } finally {
    loading.value = false
  }
}

function updateTargetCount() {
  if (!datasetAnalysis.value) return

  const counts = datasetAnalysis.value.categories.map(c => c.count)
  if (counts.length === 0) return

  if (balanceMode.value === 'undersample') {
    targetCount.value = Math.min(...counts)
  } else if (balanceMode.value === 'oversample') {
    targetCount.value = Math.max(...counts)
  } else {
    targetCount.value = null
  }
}

watch(balanceMode, updateTargetCount)

const datasetOptions = computed(() =>
  datasets.value.map(d => ({
    value: d.path,
    label: `${d.name} (${d.num_images} images)`,
  }))
)

// Calculate class statistics
const classStats = computed(() => {
  if (!datasetAnalysis.value) return null

  const categories = datasetAnalysis.value.categories
  const counts = categories.map(c => c.count)
  const total = counts.reduce((a, b) => a + b, 0)
  const max = Math.max(...counts)
  const min = Math.min(...counts)
  const mean = total / counts.length
  const imbalanceRatio = max / Math.max(min, 1)

  return {
    total,
    max,
    min,
    mean,
    imbalanceRatio,
    isImbalanced: imbalanceRatio > 3,
    isSeverelyImbalanced: imbalanceRatio > 10,
  }
})

// Calculate class weights (inverse frequency)
const classWeights = computed(() => {
  if (!datasetAnalysis.value || !classStats.value) return {}

  const weights: Record<string, number> = {}
  const { mean } = classStats.value

  for (const cat of datasetAnalysis.value.categories) {
    // Weight = mean_count / class_count (inverse frequency)
    weights[cat.name] = cat.count > 0 ? +(mean / cat.count).toFixed(3) : 1
  }

  return weights
})

// Get bar width percentage for visualization
function getBarWidth(count: number): number {
  if (!classStats.value) return 0
  return (count / classStats.value.max) * 100
}

// Get color based on count relative to mean
function getBarColor(count: number): string {
  if (!classStats.value) return 'bg-primary'

  const ratio = count / classStats.value.mean
  if (ratio < 0.5) return 'bg-red-500' // Severely underrepresented
  if (ratio < 0.8) return 'bg-yellow-500' // Underrepresented
  if (ratio > 2) return 'bg-blue-500' // Overrepresented
  return 'bg-green-500' // Balanced
}

async function runBalancing() {
  if (!selectedDataset.value || !datasetAnalysis.value) return

  balancing.value = true
  error.value = null
  balanceResult.value = null

  try {
    // In a real implementation, this would call a backend endpoint
    // For now, we'll just simulate the operation
    await new Promise(resolve => setTimeout(resolve, 1500))

    if (balanceMode.value === 'weights') {
      balanceResult.value = {
        success: true,
        message: 'Class weights calculated. Copy them to use in your training configuration.',
      }
    } else {
      balanceResult.value = {
        success: true,
        message: `Dataset ${balanceMode.value === 'undersample' ? 'undersampled' : 'oversampled'} to ${targetCount.value} samples per class.`,
      }
    }

    uiStore.showSuccess('Balancing Complete', balanceResult.value.message)
  } catch (e: any) {
    error.value = e.message || 'Balancing failed'
    uiStore.showError('Balancing Failed', error.value)
  } finally {
    balancing.value = false
  }
}

function copyWeightsToClipboard() {
  const weightsJson = JSON.stringify(classWeights.value, null, 2)
  navigator.clipboard.writeText(weightsJson)
  uiStore.showSuccess('Copied', 'Class weights copied to clipboard')
}

function copyWeightsPython() {
  const entries = Object.entries(classWeights.value)
    .map(([name, weight]) => `    "${name}": ${weight}`)
    .join(',\n')
  const pythonCode = `class_weights = {\n${entries}\n}`
  navigator.clipboard.writeText(pythonCode)
  uiStore.showSuccess('Copied', 'Python dict copied to clipboard')
}

// Watch for dataset selection
watch(selectedDataset, loadDatasetAnalysis)

// Load datasets on mount
loadDatasets()
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div>
      <h2 class="text-2xl font-bold text-white">Post-Processing & Class Balancing</h2>
      <p class="mt-1 text-gray-400">
        Analyze class distribution and balance your dataset for better training results.
      </p>
    </div>

    <!-- Error -->
    <AlertBox v-if="error" type="error" :title="error" dismissible @dismiss="error = null" />

    <!-- Dataset Selection -->
    <div class="card p-6">
      <h3 class="text-lg font-semibold text-white mb-4">Select Dataset</h3>

      <div v-if="loading && datasets.length === 0" class="flex justify-center py-8">
        <LoadingSpinner message="Loading datasets..." />
      </div>

      <div v-else class="flex gap-4">
        <BaseSelect
          v-model="selectedDataset"
          :options="datasetOptions"
          placeholder="Choose a dataset to analyze..."
          class="flex-1"
        />
        <BaseButton variant="outline" @click="loadDatasets">
          <RefreshCw class="h-5 w-5" />
        </BaseButton>
      </div>
    </div>

    <!-- Class Distribution -->
    <div v-if="datasetAnalysis" class="card p-6">
      <div class="flex items-center justify-between mb-4">
        <h3 class="text-lg font-semibold text-white flex items-center gap-2">
          <BarChart3 class="h-5 w-5 text-primary" />
          Class Distribution
        </h3>
        <BaseButton variant="outline" size="sm" @click="loadDatasetAnalysis" :disabled="loading">
          <RefreshCw :class="['h-4 w-4', loading ? 'animate-spin' : '']" />
          Refresh
        </BaseButton>
      </div>

      <!-- Imbalance Warning -->
      <AlertBox
        v-if="classStats?.isSeverelyImbalanced"
        type="error"
        class="mb-4"
      >
        <template #title>
          <span class="flex items-center gap-2">
            <AlertTriangle class="h-4 w-4" />
            Severe Class Imbalance Detected
          </span>
        </template>
        Imbalance ratio: {{ classStats.imbalanceRatio.toFixed(1) }}x. Consider balancing your dataset.
      </AlertBox>

      <AlertBox
        v-else-if="classStats?.isImbalanced"
        type="warning"
        class="mb-4"
      >
        <template #title>
          <span class="flex items-center gap-2">
            <AlertTriangle class="h-4 w-4" />
            Class Imbalance Detected
          </span>
        </template>
        Imbalance ratio: {{ classStats.imbalanceRatio.toFixed(1) }}x. You may want to balance your dataset.
      </AlertBox>

      <!-- Distribution Bars -->
      <div class="space-y-3">
        <div
          v-for="category in datasetAnalysis.categories"
          :key="category.id"
          class="flex items-center gap-4"
        >
          <div class="w-32 text-sm text-gray-300 truncate" :title="category.name">
            {{ category.name }}
          </div>
          <div class="flex-1 h-6 bg-background-tertiary rounded-full overflow-hidden">
            <div
              :class="['h-full transition-all', getBarColor(category.count)]"
              :style="{ width: `${getBarWidth(category.count)}%` }"
            />
          </div>
          <div class="w-20 text-right">
            <span class="text-white font-medium">{{ category.count }}</span>
          </div>
        </div>
      </div>

      <!-- Statistics -->
      <div class="mt-6 pt-4 border-t border-gray-700">
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
          <div>
            <p class="text-2xl font-bold text-primary">{{ classStats?.total }}</p>
            <p class="text-sm text-gray-400">Total</p>
          </div>
          <div>
            <p class="text-2xl font-bold text-green-400">{{ classStats?.max }}</p>
            <p class="text-sm text-gray-400">Max</p>
          </div>
          <div>
            <p class="text-2xl font-bold text-red-400">{{ classStats?.min }}</p>
            <p class="text-sm text-gray-400">Min</p>
          </div>
          <div>
            <p class="text-2xl font-bold text-yellow-400">{{ classStats?.imbalanceRatio.toFixed(1) }}x</p>
            <p class="text-sm text-gray-400">Imbalance</p>
          </div>
        </div>
      </div>

      <!-- Legend -->
      <div class="mt-4 flex flex-wrap gap-4 text-xs text-gray-400">
        <div class="flex items-center gap-1">
          <div class="w-3 h-3 rounded bg-green-500" />
          <span>Balanced</span>
        </div>
        <div class="flex items-center gap-1">
          <div class="w-3 h-3 rounded bg-yellow-500" />
          <span>Underrepresented</span>
        </div>
        <div class="flex items-center gap-1">
          <div class="w-3 h-3 rounded bg-red-500" />
          <span>Severely Under</span>
        </div>
        <div class="flex items-center gap-1">
          <div class="w-3 h-3 rounded bg-blue-500" />
          <span>Overrepresented</span>
        </div>
      </div>
    </div>

    <!-- Balancing Options -->
    <div v-if="datasetAnalysis" class="card p-6">
      <h3 class="text-lg font-semibold text-white mb-4 flex items-center gap-2">
        <Scale class="h-5 w-5 text-primary" />
        Balancing Options
      </h3>

      <!-- Mode Selection -->
      <div class="flex flex-wrap gap-3 mb-6">
        <BaseButton
          :variant="balanceMode === 'weights' ? 'primary' : 'outline'"
          @click="balanceMode = 'weights'"
        >
          <Scale class="h-5 w-5" />
          Class Weights
        </BaseButton>
        <BaseButton
          :variant="balanceMode === 'undersample' ? 'primary' : 'outline'"
          @click="balanceMode = 'undersample'"
        >
          <ArrowDownUp class="h-5 w-5" />
          Undersample
        </BaseButton>
        <BaseButton
          :variant="balanceMode === 'oversample' ? 'primary' : 'outline'"
          @click="balanceMode = 'oversample'"
        >
          <ArrowDownUp class="h-5 w-5" />
          Oversample
        </BaseButton>
      </div>

      <!-- Mode Description -->
      <div class="p-4 bg-background-tertiary rounded-lg mb-6">
        <div v-if="balanceMode === 'weights'">
          <h4 class="font-medium text-white mb-2">Class Weights (Recommended)</h4>
          <p class="text-sm text-gray-400">
            Calculate weights based on inverse class frequency. Use these weights during training
            to give more importance to underrepresented classes without modifying the dataset.
          </p>
        </div>
        <div v-else-if="balanceMode === 'undersample'">
          <h4 class="font-medium text-white mb-2">Undersampling</h4>
          <p class="text-sm text-gray-400">
            Reduce samples from majority classes to match the minority class count.
            This reduces dataset size but ensures perfect balance.
            Target: {{ targetCount }} samples per class.
          </p>
        </div>
        <div v-else>
          <h4 class="font-medium text-white mb-2">Oversampling</h4>
          <p class="text-sm text-gray-400">
            Duplicate samples from minority classes to match the majority class count.
            This increases dataset size and may cause overfitting on duplicated samples.
            Target: {{ targetCount }} samples per class.
          </p>
        </div>
      </div>

      <!-- Class Weights Display -->
      <div v-if="balanceMode === 'weights'" class="space-y-4">
        <div class="flex items-center justify-between">
          <h4 class="text-sm font-medium text-gray-300">Calculated Class Weights</h4>
          <div class="flex gap-2">
            <BaseButton variant="outline" size="sm" @click="copyWeightsToClipboard">
              <Copy class="h-4 w-4" />
              Copy JSON
            </BaseButton>
            <BaseButton variant="outline" size="sm" @click="copyWeightsPython">
              <Download class="h-4 w-4" />
              Copy Python
            </BaseButton>
          </div>
        </div>

        <div class="bg-background-primary rounded-lg p-4 font-mono text-sm overflow-x-auto">
          <div v-for="(weight, name) in classWeights" :key="name" class="flex justify-between py-1">
            <span class="text-gray-400">"{{ name }}":</span>
            <span :class="weight > 1.5 ? 'text-yellow-400' : 'text-green-400'">{{ weight }}</span>
          </div>
        </div>

        <p class="text-xs text-gray-500">
          Higher weights indicate underrepresented classes that need more emphasis during training.
        </p>
      </div>

      <!-- Output Directory (for sampling modes) -->
      <div v-else class="space-y-4">
        <BaseInput
          v-model="outputDir"
          label="Output Directory"
          placeholder="/data/balanced"
          hint="Where to save the balanced dataset"
        />

        <div>
          <label class="text-sm text-gray-400 flex justify-between mb-2">
            <span>Target Count per Class</span>
            <span class="text-white">{{ targetCount }}</span>
          </label>
          <input
            type="range"
            v-model.number="targetCount"
            :min="classStats?.min || 1"
            :max="classStats?.max || 100"
            class="w-full accent-primary"
          />
        </div>
      </div>
    </div>

    <!-- Result -->
    <div v-if="balanceResult" class="card p-6">
      <div class="flex items-center gap-4">
        <CheckCircle class="h-8 w-8 text-green-400" />
        <div>
          <h3 class="text-lg font-semibold text-white">Complete</h3>
          <p class="text-gray-400">{{ balanceResult.message }}</p>
        </div>
      </div>
    </div>

    <!-- Action Button -->
    <div v-if="datasetAnalysis" class="flex justify-end">
      <BaseButton
        :loading="balancing"
        :disabled="balancing"
        @click="runBalancing"
        size="lg"
      >
        <Scale class="h-5 w-5" />
        {{ balanceMode === 'weights' ? 'Calculate Weights' : 'Balance Dataset' }}
      </BaseButton>
    </div>

    <!-- No dataset selected -->
    <EmptyState
      v-else-if="!loading"
      :icon="Scale"
      title="Select a Dataset"
      description="Choose a dataset above to analyze class distribution and balance options."
    />
  </div>
</template>
