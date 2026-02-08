<script setup lang="ts">
import { ref, watch, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useWorkflowStore } from '@/stores/workflow'
import { useUiStore } from '@/stores/ui'
import { listDirectories, listFiles, getImageUrl } from '@/lib/api'
import BaseButton from '@/components/ui/BaseButton.vue'
import DirectoryBrowser from '@/components/common/DirectoryBrowser.vue'
import AlertBox from '@/components/common/AlertBox.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import {
  ArrowRight,
  ArrowLeft,
  Folder,
  Image,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
  Eye,
  EyeOff,
  Images,
} from 'lucide-vue-next'

const router = useRouter()
const workflowStore = useWorkflowStore()
const uiStore = useUiStore()

const loading = ref(false)
const loadingPreview = ref(false)
const directories = ref<string[]>([])
const backgroundsDir = ref(workflowStore.backgroundsDir || '')
const outputDir = ref(workflowStore.outputDir || '/app/output')
const error = ref<string | null>(null)

// Preview state
const previewImages = ref<string[]>([])
const currentPreviewPage = ref(0)
const previewPageSize = 8
const showPreviewPanel = ref(false)

// Computed for pagination
const totalPreviewPages = computed(() => Math.ceil(previewImages.value.length / previewPageSize))
const paginatedPreviewImages = computed(() => {
  const start = currentPreviewPage.value * previewPageSize
  return previewImages.value.slice(start, start + previewPageSize)
})

async function loadDirectories() {
  loading.value = true
  try {
    directories.value = await listDirectories()
  } catch (e: any) {
    error.value = e.message || 'Failed to load directories'
  } finally {
    loading.value = false
  }
}

async function loadPreviewImages(dir: string) {
  if (!dir) {
    previewImages.value = []
    return
  }

  loadingPreview.value = true
  try {
    // Get image files from the directory - try multiple patterns
    let files: string[] = []
    try {
      files = await listFiles(dir, '*.jpg')
    } catch { /* ignore */ }

    if (files.length === 0) {
      try {
        files = await listFiles(dir, '*.png')
      } catch { /* ignore */ }
    }

    if (files.length === 0) {
      try {
        files = await listFiles(dir, '*.jpeg')
      } catch { /* ignore */ }
    }

    previewImages.value = files.slice(0, 50) // Limit to 50 images for preview
    currentPreviewPage.value = 0
    showPreviewPanel.value = files.length > 0
  } catch (e) {
    previewImages.value = []
  } finally {
    loadingPreview.value = false
  }
}

function selectBackgroundDir(dir: string) {
  backgroundsDir.value = dir
  loadPreviewImages(dir)
}

function nextPreviewPage() {
  if (currentPreviewPage.value < totalPreviewPages.value - 1) {
    currentPreviewPage.value++
  }
}

function prevPreviewPage() {
  if (currentPreviewPage.value > 0) {
    currentPreviewPage.value--
  }
}

function togglePreview() {
  showPreviewPanel.value = !showPreviewPanel.value
  if (showPreviewPanel.value && previewImages.value.length === 0 && backgroundsDir.value) {
    loadPreviewImages(backgroundsDir.value)
  }
}

function getFileName(path: string): string {
  return path.split('/').pop() || path.split('\\').pop() || path
}

function saveAndContinue() {
  if (!outputDir.value) {
    uiStore.showError('Output Required', 'Please specify an output directory')
    return
  }

  workflowStore.setBackgroundsDir(backgroundsDir.value || null)
  workflowStore.setOutputDir(outputDir.value)
  workflowStore.markStepCompleted(3)
  uiStore.showSuccess('Configuration Saved', 'Source selection saved')
  router.push('/generation')
}

function goBack() {
  router.push('/configure')
}

// Load directories on mount
loadDirectories()
</script>

<template>
  <div class="space-y-8">
    <!-- Header -->
    <div>
      <h2 class="text-2xl font-bold text-white">Source Selection</h2>
      <p class="mt-2 text-gray-400">
        Select background images directory and configure output location.
      </p>
    </div>

    <!-- Error Alert -->
    <AlertBox v-if="error" type="error" :title="error" dismissible @dismiss="error = null" />

    <!-- Backgrounds Directory -->
    <div class="card p-6">
      <div class="flex items-center justify-between mb-4">
        <div>
          <h3 class="text-lg font-semibold text-white">Background Images</h3>
          <p class="text-sm text-gray-400">
            Select a directory containing background images for composition
          </p>
        </div>
        <button @click="loadDirectories" class="btn-ghost" :disabled="loading">
          <RefreshCw :class="['h-5 w-5', loading ? 'animate-spin' : '']" />
        </button>
      </div>

      <div v-if="loading" class="flex justify-center py-8">
        <LoadingSpinner message="Loading directories..." />
      </div>

      <div v-else-if="directories.length === 0" class="text-center py-8 text-gray-400">
        <Folder class="h-12 w-12 mx-auto mb-4 opacity-50" />
        <p>No directories found</p>
      </div>

      <div v-else class="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        <button
          v-for="dir in directories"
          :key="dir"
          @click="selectBackgroundDir(dir)"
          :class="[
            'flex items-center gap-3 rounded-lg p-4 text-left transition-colors',
            backgroundsDir === dir
              ? 'bg-primary/20 border-2 border-primary'
              : 'bg-background-tertiary hover:bg-gray-600 border-2 border-transparent',
          ]"
        >
          <Folder class="h-5 w-5 text-yellow-400 flex-shrink-0" />
          <span class="text-sm text-white truncate">{{ dir }}</span>
        </button>
      </div>

      <div v-if="backgroundsDir" class="mt-4 p-3 rounded-lg bg-green-900/20 border border-green-700/50">
        <div class="flex items-center justify-between">
          <p class="text-sm text-green-300">
            Selected: <span class="font-medium">{{ backgroundsDir }}</span>
          </p>
          <BaseButton variant="ghost" size="sm" @click="togglePreview">
            <component :is="showPreviewPanel ? EyeOff : Eye" class="h-4 w-4" />
            {{ showPreviewPanel ? 'Hide' : 'Preview' }}
          </BaseButton>
        </div>
      </div>

      <!-- Preview Panel -->
      <div v-if="showPreviewPanel && backgroundsDir" class="mt-4">
        <div class="flex items-center justify-between mb-3">
          <div class="flex items-center gap-2">
            <Images class="h-5 w-5 text-primary" />
            <span class="text-sm text-gray-300">
              {{ previewImages.length }} images found
            </span>
          </div>
          <div v-if="totalPreviewPages > 1" class="flex items-center gap-2">
            <BaseButton
              variant="ghost"
              size="sm"
              @click="prevPreviewPage"
              :disabled="currentPreviewPage === 0"
            >
              <ChevronLeft class="h-4 w-4" />
            </BaseButton>
            <span class="text-sm text-gray-400">
              {{ currentPreviewPage + 1 }} / {{ totalPreviewPages }}
            </span>
            <BaseButton
              variant="ghost"
              size="sm"
              @click="nextPreviewPage"
              :disabled="currentPreviewPage >= totalPreviewPages - 1"
            >
              <ChevronRight class="h-4 w-4" />
            </BaseButton>
          </div>
        </div>

        <div v-if="loadingPreview" class="flex justify-center py-8">
          <LoadingSpinner message="Loading previews..." />
        </div>

        <div v-else-if="previewImages.length === 0" class="text-center py-8 text-gray-400">
          <Image class="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>No images found in this directory</p>
        </div>

        <div v-else class="grid grid-cols-4 gap-2">
          <div
            v-for="imagePath in paginatedPreviewImages"
            :key="imagePath"
            class="relative aspect-square rounded-lg overflow-hidden bg-background-tertiary group"
          >
            <img
              :src="getImageUrl(imagePath)"
              :alt="getFileName(imagePath)"
              class="w-full h-full object-cover transition-transform group-hover:scale-110"
              @error="(e: any) => e.target.style.display = 'none'"
            />
            <div class="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-end p-2">
              <p class="text-xs text-white truncate w-full">{{ getFileName(imagePath) }}</p>
            </div>
          </div>
        </div>
      </div>

      <p class="mt-4 text-sm text-gray-500">
        Leave empty to use solid color backgrounds
      </p>
    </div>

    <!-- Output Directory -->
    <div class="card p-6">
      <h3 class="text-lg font-semibold text-white mb-4">Output Directory</h3>
      <DirectoryBrowser
        v-model="outputDir"
        label="Output Path"
        placeholder="/app/output/synthetic"
        path-mode="output"
      />
      <p class="text-sm text-gray-500 mt-2">Where the generated dataset will be saved</p>
    </div>

    <!-- Navigation Buttons -->
    <div class="flex justify-between">
      <BaseButton variant="outline" @click="goBack">
        <ArrowLeft class="h-5 w-5" />
        Back to Configure
      </BaseButton>
      <BaseButton @click="saveAndContinue">
        Continue to Generation
        <ArrowRight class="h-5 w-5" />
      </BaseButton>
    </div>
  </div>
</template>
