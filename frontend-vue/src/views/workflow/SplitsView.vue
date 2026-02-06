<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useUiStore } from '@/stores/ui'
import { splitDataset, kFoldSplit, listDatasets, type SplitResult, type KFoldResult } from '@/lib/api'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseSelect from '@/components/ui/BaseSelect.vue'
import BaseInput from '@/components/ui/BaseInput.vue'
import DirectoryBrowser from '@/components/common/DirectoryBrowser.vue'
import AlertBox from '@/components/common/AlertBox.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import {
  Split,
  CheckCircle,
  ArrowLeft,
  Layers,
  LayoutGrid,
} from 'lucide-vue-next'
import type { DatasetInfo } from '@/types/api'

const router = useRouter()
const uiStore = useUiStore()

// Split mode: 'ratio' for train/val/test, 'kfold' for K-Fold
const splitMode = ref<'ratio' | 'kfold'>('ratio')

const loading = ref(false)
const splitting = ref(false)
const datasets = ref<DatasetInfo[]>([])
const selectedDataset = ref<string | null>(null)
const outputDir = ref('/data/splits')

// Ratio mode options
const trainRatio = ref(70)
const valRatio = ref(20)
const testRatio = ref(10)

// K-Fold mode options
const numFolds = ref(5)
const valFold = ref(0) // Which fold to use as validation

const stratified = ref(true)
const randomSeed = ref(42)
const splitResult = ref<SplitResult | null>(null)
const kFoldResult = ref<KFoldResult | null>(null)
const error = ref<string | null>(null)

// Reset results when mode changes
watch(splitMode, () => {
  splitResult.value = null
  kFoldResult.value = null
  error.value = null
})

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

const datasetOptions = computed(() =>
  datasets.value.map(d => ({
    value: d.path,
    label: `${d.name} (${d.num_images} images)`,
  }))
)

const totalRatio = computed(() => trainRatio.value + valRatio.value + testRatio.value)
const ratioValid = computed(() => totalRatio.value === 100)

async function startSplit() {
  if (!selectedDataset.value) {
    uiStore.showError('Missing Dataset', 'Please select a dataset to split')
    return
  }

  if (splitMode.value === 'ratio' && !ratioValid.value) {
    uiStore.showError('Invalid Ratios', 'Train + Val + Test must equal 100%')
    return
  }

  splitting.value = true
  error.value = null
  splitResult.value = null
  kFoldResult.value = null

  try {
    if (splitMode.value === 'ratio') {
      const result = await splitDataset({
        dataset_path: selectedDataset.value,
        output_dir: outputDir.value,
        train_ratio: trainRatio.value / 100,
        val_ratio: valRatio.value / 100,
        test_ratio: testRatio.value / 100,
        stratified: stratified.value,
        random_seed: randomSeed.value,
      })
      splitResult.value = result
      uiStore.showSuccess('Split Complete', 'Dataset split successfully')
    } else {
      const result = await kFoldSplit({
        dataset_path: selectedDataset.value,
        output_dir: outputDir.value,
        num_folds: numFolds.value,
        val_fold: valFold.value,
        stratified: stratified.value,
        random_seed: randomSeed.value,
      })
      kFoldResult.value = result
      uiStore.showSuccess('K-Fold Split Complete', `Created ${numFolds.value} folds`)
    }
  } catch (e: any) {
    error.value = e.message || 'Failed to split dataset'
    uiStore.showError('Split Failed', error.value)
  } finally {
    splitting.value = false
  }
}

function goBack() {
  router.push('/combine')
}

function goToDashboard() {
  router.push('/')
}

// Load datasets on mount
loadDatasets()
</script>

<template>
  <div class="space-y-8">
    <!-- Header -->
    <div>
      <h2 class="text-2xl font-bold text-white">Dataset Splits</h2>
      <p class="mt-2 text-gray-400">
        Split your dataset into train, validation, and test sets.
      </p>
    </div>

    <!-- Mode Toggle -->
    <div class="flex gap-4">
      <BaseButton
        :variant="splitMode === 'ratio' ? 'primary' : 'outline'"
        @click="splitMode = 'ratio'"
      >
        <Split class="h-5 w-5" />
        Train/Val/Test Split
      </BaseButton>
      <BaseButton
        :variant="splitMode === 'kfold' ? 'primary' : 'outline'"
        @click="splitMode = 'kfold'"
      >
        <Layers class="h-5 w-5" />
        K-Fold Cross Validation
      </BaseButton>
    </div>

    <!-- Error Alert -->
    <AlertBox v-if="error" type="error" :title="error" dismissible @dismiss="error = null" />

    <!-- Ratio Warning -->
    <AlertBox v-if="splitMode === 'ratio' && !ratioValid" type="warning" title="Invalid Ratios">
      Train + Val + Test must equal 100%. Current total: {{ totalRatio }}%
    </AlertBox>

    <div class="grid gap-6 lg:grid-cols-2">
      <!-- Dataset Selection -->
      <div class="card p-6">
        <h3 class="text-lg font-semibold text-white mb-4">Select Dataset</h3>

        <div v-if="loading" class="flex justify-center py-8">
          <LoadingSpinner message="Loading datasets..." />
        </div>

        <div v-else>
          <BaseSelect
            v-model="selectedDataset"
            :options="datasetOptions"
            label="Dataset"
            placeholder="Choose a dataset to split..."
          />

          <DirectoryBrowser
            v-model="outputDir"
            label="Output Directory"
            placeholder="/data/splits"
            path-mode="output"
            class="mt-4"
          />
        </div>
      </div>

      <!-- Split Ratios (Ratio Mode) -->
      <div v-if="splitMode === 'ratio'" class="card p-6">
        <h3 class="text-lg font-semibold text-white mb-4">Split Ratios</h3>

        <div class="space-y-6">
          <!-- Train -->
          <div>
            <label class="text-sm text-gray-400 flex justify-between mb-2">
              <span>Train</span>
              <span class="text-white font-semibold">{{ trainRatio }}%</span>
            </label>
            <input
              type="range"
              v-model.number="trainRatio"
              min="0"
              max="100"
              step="5"
              class="w-full accent-green-500"
            />
          </div>

          <!-- Validation -->
          <div>
            <label class="text-sm text-gray-400 flex justify-between mb-2">
              <span>Validation</span>
              <span class="text-white font-semibold">{{ valRatio }}%</span>
            </label>
            <input
              type="range"
              v-model.number="valRatio"
              min="0"
              max="100"
              step="5"
              class="w-full accent-yellow-500"
            />
          </div>

          <!-- Test -->
          <div>
            <label class="text-sm text-gray-400 flex justify-between mb-2">
              <span>Test</span>
              <span class="text-white font-semibold">{{ testRatio }}%</span>
            </label>
            <input
              type="range"
              v-model.number="testRatio"
              min="0"
              max="100"
              step="5"
              class="w-full accent-red-500"
            />
          </div>

          <!-- Visual representation -->
          <div class="flex h-4 rounded-full overflow-hidden">
            <div
              class="bg-green-500 transition-all"
              :style="{ width: `${trainRatio}%` }"
            />
            <div
              class="bg-yellow-500 transition-all"
              :style="{ width: `${valRatio}%` }"
            />
            <div
              class="bg-red-500 transition-all"
              :style="{ width: `${testRatio}%` }"
            />
          </div>

          <div class="flex justify-between text-sm">
            <span class="text-green-400">Train: {{ trainRatio }}%</span>
            <span class="text-yellow-400">Val: {{ valRatio }}%</span>
            <span class="text-red-400">Test: {{ testRatio }}%</span>
          </div>
        </div>
      </div>

      <!-- K-Fold Configuration (K-Fold Mode) -->
      <div v-else class="card p-6">
        <h3 class="text-lg font-semibold text-white mb-4">K-Fold Configuration</h3>

        <div class="space-y-6">
          <!-- Number of folds -->
          <div>
            <label class="text-sm text-gray-400 flex justify-between mb-2">
              <span>Number of Folds (K)</span>
              <span class="text-white font-semibold">{{ numFolds }}</span>
            </label>
            <input
              type="range"
              v-model.number="numFolds"
              min="2"
              max="10"
              step="1"
              class="w-full accent-primary"
            />
            <p class="text-xs text-gray-500 mt-1">
              Each fold will contain ~{{ Math.round(100 / numFolds) }}% of the data
            </p>
          </div>

          <!-- Validation fold selector -->
          <div>
            <label class="text-sm text-gray-400 mb-2 block">Validation Fold</label>
            <div class="flex flex-wrap gap-2">
              <button
                v-for="fold in numFolds"
                :key="fold - 1"
                @click="valFold = fold - 1"
                :class="[
                  'px-4 py-2 rounded-lg text-sm font-medium transition-colors',
                  valFold === fold - 1
                    ? 'bg-primary text-white'
                    : 'bg-background-tertiary text-gray-300 hover:bg-gray-600'
                ]"
              >
                Fold {{ fold }}
              </button>
            </div>
            <p class="text-xs text-gray-500 mt-2">
              Fold {{ valFold + 1 }} will be used for validation, others for training
            </p>
          </div>

          <!-- Visual representation -->
          <div class="space-y-2">
            <p class="text-sm text-gray-400">Fold Distribution:</p>
            <div class="flex gap-1">
              <div
                v-for="fold in numFolds"
                :key="fold"
                :class="[
                  'h-8 rounded flex items-center justify-center text-xs font-medium flex-1',
                  fold - 1 === valFold
                    ? 'bg-yellow-500 text-black'
                    : 'bg-green-600 text-white'
                ]"
              >
                {{ fold - 1 === valFold ? 'Val' : 'Train' }}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Options -->
    <div class="card p-6">
      <h3 class="text-lg font-semibold text-white mb-4">Split Options</h3>

      <div class="grid gap-4 sm:grid-cols-2">
        <label class="flex items-center gap-3 cursor-pointer">
          <input
            type="checkbox"
            v-model="stratified"
            class="h-5 w-5 rounded border-gray-600 bg-background-tertiary text-primary focus:ring-primary"
          />
          <div>
            <span class="text-gray-300">Stratified split</span>
            <p class="text-sm text-gray-500">Maintain class distribution across splits</p>
          </div>
        </label>

        <BaseInput
          v-model="randomSeed"
          type="number"
          label="Random Seed"
          hint="Use same seed for reproducible splits"
        />
      </div>
    </div>

    <!-- Split Result (Ratio Mode) -->
    <div v-if="splitResult" class="card p-6">
      <div class="flex items-center gap-4 mb-4">
        <CheckCircle class="h-8 w-8 text-green-400" />
        <div>
          <h3 class="text-lg font-semibold text-white">Split Complete</h3>
          <p class="text-gray-400">Dataset split successfully</p>
        </div>
      </div>

      <div class="grid gap-4 sm:grid-cols-3 mt-4">
        <div v-if="splitResult.splits.train" class="p-4 bg-green-900/20 border border-green-700/50 rounded-lg">
          <p class="text-lg font-bold text-green-400">Train</p>
          <p class="text-2xl font-bold text-white">{{ splitResult.splits.train.images }}</p>
          <p class="text-sm text-gray-400">images ({{ splitResult.splits.train.annotations }} annotations)</p>
        </div>
        <div v-if="splitResult.splits.val" class="p-4 bg-yellow-900/20 border border-yellow-700/50 rounded-lg">
          <p class="text-lg font-bold text-yellow-400">Validation</p>
          <p class="text-2xl font-bold text-white">{{ splitResult.splits.val.images }}</p>
          <p class="text-sm text-gray-400">images ({{ splitResult.splits.val.annotations }} annotations)</p>
        </div>
        <div v-if="splitResult.splits.test" class="p-4 bg-red-900/20 border border-red-700/50 rounded-lg">
          <p class="text-lg font-bold text-red-400">Test</p>
          <p class="text-2xl font-bold text-white">{{ splitResult.splits.test.images }}</p>
          <p class="text-sm text-gray-400">images ({{ splitResult.splits.test.annotations }} annotations)</p>
        </div>
      </div>
    </div>

    <!-- K-Fold Result -->
    <div v-if="kFoldResult" class="card p-6">
      <div class="flex items-center gap-4 mb-4">
        <Layers class="h-8 w-8 text-green-400" />
        <div>
          <h3 class="text-lg font-semibold text-white">K-Fold Split Complete</h3>
          <p class="text-gray-400">Created {{ kFoldResult.num_folds }} folds successfully</p>
        </div>
      </div>

      <div class="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 mt-4">
        <div
          v-for="(fold, index) in kFoldResult.folds"
          :key="index"
          :class="[
            'p-4 rounded-lg border',
            index === kFoldResult.val_fold
              ? 'bg-yellow-900/20 border-yellow-700/50'
              : 'bg-green-900/20 border-green-700/50'
          ]"
        >
          <div class="flex items-center justify-between mb-2">
            <p :class="['text-lg font-bold', index === kFoldResult.val_fold ? 'text-yellow-400' : 'text-green-400']">
              Fold {{ index + 1 }}
            </p>
            <span :class="['text-xs px-2 py-1 rounded', index === kFoldResult.val_fold ? 'bg-yellow-500/20 text-yellow-300' : 'bg-green-500/20 text-green-300']">
              {{ index === kFoldResult.val_fold ? 'Validation' : 'Training' }}
            </span>
          </div>
          <p class="text-2xl font-bold text-white">{{ fold.images }}</p>
          <p class="text-sm text-gray-400">images ({{ fold.annotations }} annotations)</p>
        </div>
      </div>

      <div class="mt-4 p-3 bg-background-tertiary rounded-lg">
        <p class="text-sm text-gray-300">
          <strong>Output:</strong> {{ kFoldResult.output_dir }}
        </p>
        <p class="text-xs text-gray-500 mt-1">
          Each fold is saved as fold_1.json, fold_2.json, etc.
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
          v-if="!splitResult && !kFoldResult"
          :disabled="!selectedDataset || (splitMode === 'ratio' && !ratioValid)"
          :loading="splitting"
          @click="startSplit"
        >
          <component :is="splitMode === 'ratio' ? Split : Layers" class="h-5 w-5" />
          {{ splitMode === 'ratio' ? 'Split Dataset' : 'Create K-Fold Splits' }}
        </BaseButton>

        <BaseButton
          v-if="splitResult || kFoldResult"
          variant="success"
          @click="goToDashboard"
        >
          <CheckCircle class="h-5 w-5" />
          Finish Workflow
        </BaseButton>
      </div>
    </div>
  </div>
</template>
