<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useUiStore } from '@/stores/ui'
import {
  listDatasets,
  analyzeDataset,
  getObjectSizes,
  updateObjectSize,
  updateMultipleObjectSizes,
  deleteObjectSize,
} from '@/lib/api'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseSelect from '@/components/ui/BaseSelect.vue'
import BaseInput from '@/components/ui/BaseInput.vue'
import AlertBox from '@/components/common/AlertBox.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import MetricCard from '@/components/common/MetricCard.vue'
import {
  Ruler,
  RefreshCw,
  BarChart3,
  Maximize,
  Minimize,
  Plus,
  Trash2,
  Save,
  Fish,
  Car,
  Package,
  Settings,
} from 'lucide-vue-next'
import type { DatasetInfo, DatasetAnalysis } from '@/types/api'

const uiStore = useUiStore()

// Dataset analysis state
const loading = ref(false)
const analyzing = ref(false)
const datasets = ref<DatasetInfo[]>([])
const selectedDataset = ref<string | null>(null)
const analysis = ref<DatasetAnalysis | null>(null)
const error = ref<string | null>(null)

// Object size configuration state
const objectSizes = ref<Record<string, number>>({})
const loadingSizes = ref(false)
const savingSizes = ref(false)
const newClassName = ref('')
const newClassSize = ref(0.3)

// Default presets
const presets = {
  marine_life: {
    name: 'Marine Life',
    icon: Fish,
    sizes: {
      fish: 0.25,
      shark: 0.6,
      turtle: 0.5,
      jellyfish: 0.3,
      crab: 0.15,
      octopus: 0.4,
      seahorse: 0.1,
      starfish: 0.12,
      coral: 0.35,
      urchin: 0.08,
      ray: 0.55,
      dolphin: 0.7,
      whale: 1.0,
      lobster: 0.2,
      squid: 0.35,
    }
  },
  vehicles: {
    name: 'Vehicles',
    icon: Car,
    sizes: {
      car: 0.4,
      truck: 0.6,
      motorcycle: 0.25,
      bicycle: 0.2,
      bus: 0.7,
      boat: 0.5,
      airplane: 0.8,
      helicopter: 0.5,
      train: 0.9,
      scooter: 0.15,
    }
  },
  objects: {
    name: 'Common Objects',
    icon: Package,
    sizes: {
      bottle: 0.15,
      cup: 0.1,
      chair: 0.35,
      table: 0.5,
      laptop: 0.25,
      phone: 0.08,
      book: 0.12,
      bag: 0.2,
      umbrella: 0.3,
      ball: 0.15,
    }
  }
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

async function loadObjectSizes() {
  loadingSizes.value = true
  try {
    objectSizes.value = await getObjectSizes()
  } catch (e) {
    // API might not have sizes yet, start with empty
    objectSizes.value = {}
  } finally {
    loadingSizes.value = false
  }
}

async function analyzeSelected() {
  if (!selectedDataset.value) return

  analyzing.value = true
  error.value = null

  try {
    analysis.value = await analyzeDataset(selectedDataset.value)
    uiStore.showSuccess('Analysis Complete', 'Dataset analyzed successfully')
  } catch (e: any) {
    error.value = e.message || 'Failed to analyze dataset'
  } finally {
    analyzing.value = false
  }
}

async function saveObjectSize(className: string, size: number) {
  try {
    await updateObjectSize(className, size)
    objectSizes.value[className] = size
    uiStore.showSuccess('Saved', `Size for ${className} updated`)
  } catch (e: any) {
    uiStore.showError('Error', e.message || 'Failed to save size')
  }
}

async function removeObjectSize(className: string) {
  try {
    await deleteObjectSize(className)
    delete objectSizes.value[className]
    uiStore.showSuccess('Deleted', `${className} removed`)
  } catch (e: any) {
    uiStore.showError('Error', e.message || 'Failed to delete size')
  }
}

async function addNewClass() {
  if (!newClassName.value.trim()) {
    uiStore.showError('Error', 'Please enter a class name')
    return
  }

  try {
    await updateObjectSize(newClassName.value.trim(), newClassSize.value)
    objectSizes.value[newClassName.value.trim()] = newClassSize.value
    uiStore.showSuccess('Added', `${newClassName.value} added`)
    newClassName.value = ''
    newClassSize.value = 0.3
  } catch (e: any) {
    uiStore.showError('Error', e.message || 'Failed to add class')
  }
}

async function applyPreset(presetKey: keyof typeof presets) {
  const preset = presets[presetKey]
  savingSizes.value = true

  try {
    await updateMultipleObjectSizes(preset.sizes)
    objectSizes.value = { ...objectSizes.value, ...preset.sizes }
    uiStore.showSuccess('Preset Applied', `${preset.name} sizes loaded`)
  } catch (e: any) {
    uiStore.showError('Error', e.message || 'Failed to apply preset')
  } finally {
    savingSizes.value = false
  }
}

async function saveAllSizes() {
  savingSizes.value = true
  try {
    await updateMultipleObjectSizes(objectSizes.value)
    uiStore.showSuccess('Saved', 'All object sizes saved')
  } catch (e: any) {
    uiStore.showError('Error', e.message || 'Failed to save sizes')
  } finally {
    savingSizes.value = false
  }
}

const datasetOptions = computed(() =>
  datasets.value.map(d => ({
    value: d.path,
    label: `${d.name} (${d.num_images} images)`,
  }))
)

// Calculate size statistics from analysis
const sizeStats = computed(() => {
  if (!analysis.value) return null

  const { image_sizes } = analysis.value as any
  if (!image_sizes?.widths?.length) return null

  const widths = image_sizes.widths
  const heights = image_sizes.heights

  return {
    minWidth: Math.min(...widths),
    maxWidth: Math.max(...widths),
    avgWidth: Math.round(widths.reduce((a: number, b: number) => a + b, 0) / widths.length),
    minHeight: Math.min(...heights),
    maxHeight: Math.max(...heights),
    avgHeight: Math.round(heights.reduce((a: number, b: number) => a + b, 0) / heights.length),
  }
})

const sortedObjectSizes = computed(() =>
  Object.entries(objectSizes.value).sort(([a], [b]) => a.localeCompare(b))
)

// Load data on mount
onMounted(() => {
  loadDatasets()
  loadObjectSizes()
})
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div>
      <h2 class="text-2xl font-bold text-white">Object Sizes</h2>
      <p class="mt-1 text-gray-400">
        Configure object size ratios for synthetic generation and analyze dataset dimensions.
      </p>
    </div>

    <!-- Error -->
    <AlertBox v-if="error" type="error" :title="error" dismissible @dismiss="error = null" />

    <!-- Object Size Configuration -->
    <div class="card p-6">
      <div class="flex items-center justify-between mb-6">
        <h3 class="text-lg font-semibold text-white">Object Size Configuration</h3>
        <div class="flex gap-2">
          <BaseButton variant="outline" size="sm" @click="loadObjectSizes" :disabled="loadingSizes">
            <RefreshCw :class="['h-4 w-4', loadingSizes ? 'animate-spin' : '']" />
            Refresh
          </BaseButton>
          <BaseButton size="sm" @click="saveAllSizes" :loading="savingSizes">
            <Save class="h-4 w-4" />
            Save All
          </BaseButton>
        </div>
      </div>

      <!-- Presets -->
      <div class="mb-6">
        <p class="text-sm text-gray-400 mb-3">Load preset sizes:</p>
        <div class="flex flex-wrap gap-2">
          <BaseButton
            v-for="(preset, key) in presets"
            :key="key"
            variant="outline"
            size="sm"
            @click="applyPreset(key)"
            :disabled="savingSizes"
          >
            <component :is="preset.icon" class="h-4 w-4" />
            {{ preset.name }}
          </BaseButton>
        </div>
      </div>

      <!-- Add New Class -->
      <div class="flex gap-3 mb-6 p-4 bg-background-tertiary rounded-lg">
        <BaseInput
          v-model="newClassName"
          placeholder="Class name (e.g., fish)"
          class="flex-1"
        />
        <div class="w-32">
          <input
            v-model.number="newClassSize"
            type="number"
            min="0.01"
            max="1.0"
            step="0.05"
            class="input w-full"
            placeholder="Size ratio"
          />
        </div>
        <BaseButton @click="addNewClass">
          <Plus class="h-4 w-4" />
          Add
        </BaseButton>
      </div>

      <!-- Object Sizes List -->
      <div v-if="loadingSizes" class="flex justify-center py-8">
        <LoadingSpinner message="Loading object sizes..." />
      </div>

      <div v-else-if="sortedObjectSizes.length === 0" class="text-center py-8 text-gray-400">
        <Settings class="h-12 w-12 mx-auto mb-4 opacity-50" />
        <p>No object sizes configured yet.</p>
        <p class="text-sm mt-2">Add a class above or load a preset to get started.</p>
      </div>

      <div v-else class="space-y-2 max-h-[400px] overflow-y-auto">
        <div
          v-for="[className, size] in sortedObjectSizes"
          :key="className"
          class="flex items-center gap-3 p-3 rounded-lg bg-background-tertiary hover:bg-gray-700/50 transition-colors"
        >
          <span class="w-40 text-sm font-medium text-white truncate" :title="className">
            {{ className }}
          </span>
          <input
            type="range"
            :value="size"
            @change="saveObjectSize(className, parseFloat(($event.target as HTMLInputElement).value))"
            min="0.01"
            max="1.0"
            step="0.01"
            class="flex-1 accent-primary"
          />
          <span class="w-16 text-sm text-gray-400 text-right font-mono">
            {{ (size * 100).toFixed(0) }}%
          </span>
          <button
            @click="removeObjectSize(className)"
            class="p-1.5 rounded text-gray-400 hover:text-red-400 hover:bg-red-900/20 transition-colors"
            title="Remove"
          >
            <Trash2 class="h-4 w-4" />
          </button>
        </div>
      </div>

      <p class="mt-4 text-xs text-gray-500">
        Size ratio represents the relative size of the object compared to the image (0.01 = 1%, 1.0 = 100%).
      </p>
    </div>

    <!-- Dataset Analysis Section -->
    <div class="card p-6">
      <h3 class="text-lg font-semibold text-white mb-4">Dataset Analysis</h3>

      <div v-if="loading" class="flex justify-center py-8">
        <LoadingSpinner message="Loading datasets..." />
      </div>

      <div v-else class="flex gap-4">
        <BaseSelect
          v-model="selectedDataset"
          :options="datasetOptions"
          placeholder="Choose a dataset to analyze..."
          class="flex-1"
        />
        <BaseButton
          :loading="analyzing"
          :disabled="!selectedDataset || analyzing"
          @click="analyzeSelected"
        >
          <BarChart3 class="h-5 w-5" />
          Analyze
        </BaseButton>
      </div>
    </div>

    <!-- Analysis Results -->
    <div v-if="analysis" class="space-y-6">
      <!-- Overview Metrics -->
      <div class="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Total Images"
          :value="analysis.total_images"
          :icon="Ruler"
        />
        <MetricCard
          title="Total Annotations"
          :value="analysis.total_annotations"
          :icon="BarChart3"
        />
        <MetricCard
          title="Categories"
          :value="analysis.categories.length"
        />
        <MetricCard
          title="Avg Annotations/Image"
          :value="analysis.annotations_per_image.mean.toFixed(1)"
        />
      </div>

      <!-- Image Size Statistics -->
      <div v-if="sizeStats" class="card p-6">
        <h3 class="text-lg font-semibold text-white mb-4">Image Dimensions</h3>

        <div class="grid gap-6 sm:grid-cols-2">
          <!-- Width Stats -->
          <div class="space-y-4">
            <h4 class="text-md font-medium text-gray-400">Width</h4>
            <div class="grid grid-cols-3 gap-4">
              <div class="text-center p-3 rounded-lg bg-background-tertiary">
                <Minimize class="h-5 w-5 mx-auto text-blue-400 mb-2" />
                <p class="text-2xl font-bold text-white">{{ sizeStats.minWidth }}</p>
                <p class="text-xs text-gray-500">Min</p>
              </div>
              <div class="text-center p-3 rounded-lg bg-background-tertiary">
                <Ruler class="h-5 w-5 mx-auto text-green-400 mb-2" />
                <p class="text-2xl font-bold text-white">{{ sizeStats.avgWidth }}</p>
                <p class="text-xs text-gray-500">Average</p>
              </div>
              <div class="text-center p-3 rounded-lg bg-background-tertiary">
                <Maximize class="h-5 w-5 mx-auto text-purple-400 mb-2" />
                <p class="text-2xl font-bold text-white">{{ sizeStats.maxWidth }}</p>
                <p class="text-xs text-gray-500">Max</p>
              </div>
            </div>
          </div>

          <!-- Height Stats -->
          <div class="space-y-4">
            <h4 class="text-md font-medium text-gray-400">Height</h4>
            <div class="grid grid-cols-3 gap-4">
              <div class="text-center p-3 rounded-lg bg-background-tertiary">
                <Minimize class="h-5 w-5 mx-auto text-blue-400 mb-2" />
                <p class="text-2xl font-bold text-white">{{ sizeStats.minHeight }}</p>
                <p class="text-xs text-gray-500">Min</p>
              </div>
              <div class="text-center p-3 rounded-lg bg-background-tertiary">
                <Ruler class="h-5 w-5 mx-auto text-green-400 mb-2" />
                <p class="text-2xl font-bold text-white">{{ sizeStats.avgHeight }}</p>
                <p class="text-xs text-gray-500">Average</p>
              </div>
              <div class="text-center p-3 rounded-lg bg-background-tertiary">
                <Maximize class="h-5 w-5 mx-auto text-purple-400 mb-2" />
                <p class="text-2xl font-bold text-white">{{ sizeStats.maxHeight }}</p>
                <p class="text-xs text-gray-500">Max</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Category Distribution -->
      <div class="card p-6">
        <h3 class="text-lg font-semibold text-white mb-4">Annotations per Category</h3>

        <div class="space-y-3">
          <div
            v-for="category in analysis.categories"
            :key="category.id"
            class="flex items-center gap-4"
          >
            <span class="w-32 text-sm text-gray-300 truncate">{{ category.name }}</span>
            <div class="flex-1 h-4 bg-background-tertiary rounded-full overflow-hidden">
              <div
                class="h-full bg-gradient-to-r from-primary to-blue-400"
                :style="{
                  width: `${(category.count / analysis.total_annotations) * 100}%`
                }"
              />
            </div>
            <span class="w-20 text-sm text-gray-400 text-right">{{ category.count }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- No analysis yet -->
    <EmptyState
      v-else-if="!analyzing"
      :icon="Ruler"
      title="No dataset analysis yet"
      description="Select a dataset above to analyze its image dimensions and category distribution."
    />
  </div>
</template>
