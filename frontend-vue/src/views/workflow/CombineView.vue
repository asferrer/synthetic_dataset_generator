<script setup lang="ts">
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useUiStore } from '@/stores/ui'
import { combineDatasets, listDatasets, type CombineResult } from '@/lib/api'
import BaseButton from '@/components/ui/BaseButton.vue'
import DirectoryBrowser from '@/components/common/DirectoryBrowser.vue'
import AlertBox from '@/components/common/AlertBox.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import {
  ArrowRight,
  ArrowLeft,
  Combine,
  Plus,
  CheckCircle,
  Database,
} from 'lucide-vue-next'
import type { DatasetInfo } from '@/types/api'

const router = useRouter()
const uiStore = useUiStore()

const loading = ref(false)
const combining = ref(false)
const datasets = ref<DatasetInfo[]>([])
const selectedDatasets = ref<string[]>([])
const outputDir = ref('/app/output/combined')
const mergeCategories = ref(true)
const deduplicate = ref(true)
const combineResult = ref<CombineResult | null>(null)
const error = ref<string | null>(null)

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

function toggleDataset(path: string) {
  const index = selectedDatasets.value.indexOf(path)
  if (index > -1) {
    selectedDatasets.value.splice(index, 1)
  } else {
    selectedDatasets.value.push(path)
  }
}

function isSelected(path: string) {
  return selectedDatasets.value.includes(path)
}

const canCombine = computed(() => selectedDatasets.value.length >= 2)

async function startCombine() {
  if (!canCombine.value) {
    uiStore.showError('Not Enough Datasets', 'Please select at least 2 datasets to combine')
    return
  }

  combining.value = true
  error.value = null
  combineResult.value = null

  try {
    const result = await combineDatasets({
      dataset_paths: selectedDatasets.value,
      output_dir: outputDir.value,
      merge_categories: mergeCategories.value,
      deduplicate: deduplicate.value,
    })

    combineResult.value = result
    uiStore.showSuccess('Combine Complete', `Combined ${result.total_images} images into ${result.output_path}`)
  } catch (e: any) {
    error.value = e.message || 'Failed to combine datasets'
    uiStore.showError('Combine Failed', error.value)
  } finally {
    combining.value = false
  }
}

function goBack() {
  router.push('/export')
}

function continueToSplits() {
  router.push('/splits')
}

// Load datasets on mount
loadDatasets()
</script>

<template>
  <div class="space-y-8">
    <!-- Header -->
    <div>
      <h2 class="text-2xl font-bold text-white">Combine Datasets</h2>
      <p class="mt-2 text-gray-400">
        Merge multiple datasets into a single unified dataset.
      </p>
    </div>

    <!-- Error Alert -->
    <AlertBox v-if="error" type="error" :title="error" dismissible @dismiss="error = null" />

    <!-- Dataset Selection -->
    <div class="card p-6">
      <h3 class="text-lg font-semibold text-white mb-4">
        Select Datasets to Combine
        <span v-if="selectedDatasets.length > 0" class="text-sm text-gray-400 font-normal ml-2">
          ({{ selectedDatasets.length }} selected)
        </span>
      </h3>

      <div v-if="loading" class="flex justify-center py-8">
        <LoadingSpinner message="Loading datasets..." />
      </div>

      <div v-else-if="datasets.length === 0" class="text-center py-8 text-gray-400">
        <Database class="h-12 w-12 mx-auto mb-4 opacity-50" />
        <p>No datasets found</p>
      </div>

      <div v-else class="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        <button
          v-for="dataset in datasets"
          :key="dataset.path"
          @click="toggleDataset(dataset.path)"
          :class="[
            'flex items-center gap-3 rounded-lg p-4 text-left transition-colors',
            isSelected(dataset.path)
              ? 'bg-primary/20 border-2 border-primary'
              : 'bg-background-tertiary hover:bg-gray-600 border-2 border-transparent',
          ]"
        >
          <div
            :class="[
              'flex h-6 w-6 items-center justify-center rounded-md',
              isSelected(dataset.path) ? 'bg-primary text-white' : 'bg-gray-600 text-gray-400',
            ]"
          >
            <CheckCircle v-if="isSelected(dataset.path)" class="h-4 w-4" />
            <Plus v-else class="h-4 w-4" />
          </div>
          <div class="flex-1 min-w-0">
            <p class="font-medium text-white truncate">{{ dataset.name }}</p>
            <p class="text-sm text-gray-400">{{ dataset.num_images }} images</p>
          </div>
        </button>
      </div>
    </div>

    <!-- Combine Options -->
    <div class="card p-6">
      <h3 class="text-lg font-semibold text-white mb-4">Combine Options</h3>

      <div class="space-y-4">
        <DirectoryBrowser
          v-model="outputDir"
          label="Output Directory"
          placeholder="/app/output/combined"
          path-mode="output"
        />

        <label class="flex items-center gap-3 cursor-pointer">
          <input
            type="checkbox"
            v-model="mergeCategories"
            class="h-5 w-5 rounded border-gray-600 bg-background-tertiary text-primary focus:ring-primary"
          />
          <div>
            <span class="text-gray-300">Merge categories</span>
            <p class="text-sm text-gray-500">Combine categories with the same name</p>
          </div>
        </label>

        <label class="flex items-center gap-3 cursor-pointer">
          <input
            type="checkbox"
            v-model="deduplicate"
            class="h-5 w-5 rounded border-gray-600 bg-background-tertiary text-primary focus:ring-primary"
          />
          <div>
            <span class="text-gray-300">Deduplicate images</span>
            <p class="text-sm text-gray-500">Remove duplicate images based on filename</p>
          </div>
        </label>
      </div>
    </div>

    <!-- Combine Result -->
    <div v-if="combineResult" class="card p-6">
      <div class="flex items-center gap-4 mb-4">
        <CheckCircle class="h-8 w-8 text-green-400" />
        <div>
          <h3 class="text-lg font-semibold text-white">Combine Complete</h3>
          <p class="text-gray-400">Datasets merged successfully</p>
        </div>
      </div>

      <div class="grid gap-4 sm:grid-cols-3 mt-4">
        <div class="p-4 bg-background-tertiary rounded-lg text-center">
          <p class="text-2xl font-bold text-primary">{{ combineResult.total_images }}</p>
          <p class="text-sm text-gray-400">Total Images</p>
        </div>
        <div class="p-4 bg-background-tertiary rounded-lg text-center">
          <p class="text-2xl font-bold text-primary">{{ combineResult.total_annotations }}</p>
          <p class="text-sm text-gray-400">Total Annotations</p>
        </div>
        <div class="p-4 bg-background-tertiary rounded-lg text-center">
          <p class="text-2xl font-bold text-primary">{{ combineResult.total_categories }}</p>
          <p class="text-sm text-gray-400">Categories</p>
        </div>
      </div>

      <div class="mt-4 p-4 bg-green-900/20 border border-green-700/50 rounded-lg">
        <p class="text-sm text-gray-300">
          <span class="text-green-400 font-medium">Output:</span>
          {{ combineResult.output_path }}
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
          v-if="!combineResult"
          :disabled="!canCombine"
          :loading="combining"
          @click="startCombine"
        >
          <Combine class="h-5 w-5" />
          Combine Datasets
        </BaseButton>

        <BaseButton @click="continueToSplits">
          Continue to Splits
          <ArrowRight class="h-5 w-5" />
        </BaseButton>
      </div>
    </div>
  </div>
</template>
