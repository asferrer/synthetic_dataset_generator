<script setup lang="ts">
import { ref, watch, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useWorkflowStore } from '@/stores/workflow'
import { useUiStore } from '@/stores/ui'
import { listFiles, listDirectories, getImageUrl } from '@/lib/api'
import BaseButton from '@/components/ui/BaseButton.vue'
import DirectoryBrowser from '@/components/common/DirectoryBrowser.vue'
import AlertBox from '@/components/common/AlertBox.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import {
  ArrowRight,
  ArrowLeft,
  Image,
  ChevronLeft,
  ChevronRight,
  Eye,
  EyeOff,
  Images,
  Package,
  Check,
  X,
  FolderSearch,
} from 'lucide-vue-next'
import { useI18n } from 'vue-i18n'

const router = useRouter()
const workflowStore = useWorkflowStore()
const uiStore = useUiStore()
const { t } = useI18n()

const loadingPreview = ref(false)
const backgroundsDir = ref(workflowStore.backgroundsDir || '')
const objectsDir = ref(workflowStore.objectsDir || '')

// Build a descriptive default output path based on context
function getDefaultOutputDir(): string {
  if (workflowStore.outputDir) return workflowStore.outputDir
  if (workflowStore.customObjectsMode) return '/app/output/synthetic_custom'
  if (workflowStore.metadata?.name && workflowStore.metadata.name !== 'Synthetic Dataset') {
    const safeName = workflowStore.metadata.name.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '')
    return `/app/output/${safeName}`
  }
  return '/app/output/synthetic'
}

const outputDir = ref(getDefaultOutputDir())
const error = ref<string | null>(null)

// Preview state
const previewImages = ref<string[]>([])
const currentPreviewPage = ref(0)
const previewPageSize = 8
const showPreviewPanel = ref(false)

// Object class detection state (custom objects mode)
const detectedClasses = ref<{ name: string; selected: boolean; count: number }[]>([])
const loadingClasses = ref(false)

// Computed for pagination
const totalPreviewPages = computed(() => Math.ceil(previewImages.value.length / previewPageSize))
const paginatedPreviewImages = computed(() => {
  const start = currentPreviewPage.value * previewPageSize
  return previewImages.value.slice(start, start + previewPageSize)
})

const selectedClasses = computed(() => detectedClasses.value.filter(c => c.selected))
const isCustomMode = computed(() => workflowStore.customObjectsMode)

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

// Auto-detect object classes from objects directory subdirectories
async function detectObjectClasses(dir: string) {
  if (!dir) {
    detectedClasses.value = []
    return
  }

  loadingClasses.value = true
  try {
    const subdirs = await listDirectories(dir)
    const classes: { name: string; selected: boolean; count: number }[] = []

    for (const subdir of subdirs) {
      // Extract class name from full path
      const className = subdir.split('/').pop() || subdir.split('\\').pop() || subdir
      // Count image files in each class directory
      let count = 0
      try {
        const files = await listFiles(subdir, '*.png')
        count = files.length
        if (count === 0) {
          const jpgFiles = await listFiles(subdir, '*.jpg')
          count = jpgFiles.length
        }
      } catch { /* ignore */ }

      if (count > 0) {
        // Pre-select classes that were previously selected
        const wasSelected = workflowStore.manualCategories.includes(className)
        classes.push({ name: className, selected: wasSelected || workflowStore.manualCategories.length === 0, count })
      }
    }

    classes.sort((a, b) => a.name.localeCompare(b.name))
    detectedClasses.value = classes

    // Auto-update manual categories in custom mode
    if (isCustomMode.value) {
      updateManualCategories()
    }
  } catch (e) {
    detectedClasses.value = []
  } finally {
    loadingClasses.value = false
  }
}

function toggleClass(className: string) {
  const cls = detectedClasses.value.find(c => c.name === className)
  if (cls) {
    cls.selected = !cls.selected
    updateManualCategories()
  }
}

function selectAllClasses() {
  detectedClasses.value.forEach(c => c.selected = true)
  updateManualCategories()
}

function deselectAllClasses() {
  detectedClasses.value.forEach(c => c.selected = false)
  updateManualCategories()
}

function updateManualCategories() {
  if (isCustomMode.value) {
    const selected = detectedClasses.value.filter(c => c.selected).map(c => c.name)
    workflowStore.setManualCategories(selected)
  }
}

// Watch backgroundsDir to load preview when changed via DirectoryBrowser
watch(backgroundsDir, (newDir) => {
  if (newDir) {
    loadPreviewImages(newDir)
  } else {
    previewImages.value = []
    showPreviewPanel.value = false
  }
})

// Watch objectsDir to detect classes when in custom objects mode
watch(objectsDir, (newDir) => {
  if (newDir && isCustomMode.value) {
    detectObjectClasses(newDir)
  } else {
    detectedClasses.value = []
  }
})

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
    uiStore.showError(t('workflow.sourceSelection.errors.outputRequired'), t('workflow.sourceSelection.errors.outputRequiredMsg'))
    return
  }

  if (isCustomMode.value && !objectsDir.value) {
    uiStore.showError(t('workflow.sourceSelection.errors.objectsRequired'), t('workflow.sourceSelection.errors.objectsRequiredMsg'))
    return
  }

  if (isCustomMode.value && selectedClasses.value.length === 0) {
    uiStore.showError(t('workflow.sourceSelection.errors.classesRequired'), t('workflow.sourceSelection.errors.classesRequiredMsg'))
    return
  }

  workflowStore.setBackgroundsDir(backgroundsDir.value || null)
  workflowStore.setObjectsDir(objectsDir.value || null)
  workflowStore.setOutputDir(outputDir.value)
  workflowStore.markStepCompleted(3)
  uiStore.showSuccess(t('workflow.sourceSelection.notifications.saved'), t('workflow.sourceSelection.notifications.savedMsg'))
  router.push('/generation')
}

function goBack() {
  router.push('/configure')
}

// Load preview if backgroundsDir is already set (e.g. navigating back)
if (backgroundsDir.value) {
  loadPreviewImages(backgroundsDir.value)
}

// Load classes if objectsDir is already set in custom mode
if (objectsDir.value && isCustomMode.value) {
  detectObjectClasses(objectsDir.value)
}
</script>

<template>
  <div class="space-y-8">
    <!-- Header -->
    <div>
      <h2 class="text-2xl font-bold text-white">{{ t('workflow.sourceSelection.title') }}</h2>
      <p class="mt-2 text-gray-400">
        {{ t('workflow.sourceSelection.subtitle') }}
      </p>
    </div>

    <!-- Custom mode indicator -->
    <AlertBox v-if="isCustomMode" type="info" :title="t('workflow.sourceSelection.customModeActive')">
      {{ t('workflow.sourceSelection.customModeActiveMsg') }}
    </AlertBox>

    <!-- Error Alert -->
    <AlertBox v-if="error" type="error" :title="error" dismissible @dismiss="error = null" />

    <!-- Backgrounds Directory -->
    <div class="card p-6">
      <div class="mb-4">
        <h3 class="text-lg font-semibold text-white">{{ t('workflow.sourceSelection.backgroundImages') }}</h3>
        <p class="text-sm text-gray-400">
          {{ t('workflow.sourceSelection.backgroundImagesDesc') }}
        </p>
      </div>

      <DirectoryBrowser
        v-model="backgroundsDir"
        :label="t('workflow.sourceSelection.backgroundsPath')"
        placeholder="/app/datasets/Backgrounds_filtered"
        path-mode="input"
      />

      <div v-if="backgroundsDir" class="mt-4 p-3 rounded-lg bg-green-900/20 border border-green-700/50">
        <div class="flex items-center justify-between">
          <p class="text-sm text-green-300">
            {{ t('workflow.sourceSelection.selected') }}: <span class="font-medium">{{ backgroundsDir }}</span>
          </p>
          <BaseButton variant="ghost" size="sm" @click="togglePreview">
            <component :is="showPreviewPanel ? EyeOff : Eye" class="h-4 w-4" />
            {{ showPreviewPanel ? t('workflow.sourceSelection.hidePreview') : t('workflow.sourceSelection.showPreview') }}
          </BaseButton>
        </div>
      </div>

      <!-- Preview Panel -->
      <div v-if="showPreviewPanel && backgroundsDir" class="mt-4">
        <div class="flex items-center justify-between mb-3">
          <div class="flex items-center gap-2">
            <Images class="h-5 w-5 text-primary" />
            <span class="text-sm text-gray-300">
              {{ t('workflow.sourceSelection.imagesFound', { count: previewImages.length }) }}
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
          <LoadingSpinner :message="t('workflow.sourceSelection.loadingPreviews')" />
        </div>

        <div v-else-if="previewImages.length === 0" class="text-center py-8 text-gray-400">
          <Image class="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>{{ t('workflow.sourceSelection.noImagesFound') }}</p>
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
        {{ t('workflow.sourceSelection.backgroundsHint') }}
      </p>
    </div>

    <!-- Object Sources Directory -->
    <div class="card p-6">
      <div class="flex items-center gap-3 mb-4">
        <Package class="h-5 w-5 text-primary" />
        <div>
          <h3 class="text-lg font-semibold text-white">
            {{ t('workflow.sourceSelection.objectSources') }}
            <span v-if="isCustomMode" class="ml-2 text-xs font-normal text-primary bg-primary/10 px-2 py-0.5 rounded-full">
              {{ t('workflow.sourceSelection.required') }}
            </span>
          </h3>
          <p class="text-sm text-gray-400">
            {{ t('workflow.sourceSelection.objectSourcesDesc') }}
          </p>
        </div>
      </div>
      <DirectoryBrowser
        v-model="objectsDir"
        :label="t('workflow.sourceSelection.selectDirectory')"
        placeholder="/app/datasets/my_dataset/Objects"
        path-mode="input"
      />
      <p class="text-sm text-gray-500 mt-2">
        {{ isCustomMode ? t('workflow.sourceSelection.objectSourcesHintCustom') : t('workflow.sourceSelection.objectSourcesHint') }}
      </p>
    </div>

    <!-- Object Class Selection (Custom Objects Mode) -->
    <div v-if="isCustomMode && objectsDir" class="card p-6">
      <div class="flex items-center gap-3 mb-4">
        <FolderSearch class="h-5 w-5 text-primary" />
        <div>
          <h3 class="text-lg font-semibold text-white">{{ t('workflow.sourceSelection.classSelection.title') }}</h3>
          <p class="text-sm text-gray-400">{{ t('workflow.sourceSelection.classSelection.description') }}</p>
        </div>
      </div>

      <div v-if="loadingClasses" class="flex justify-center py-8">
        <LoadingSpinner :message="t('workflow.sourceSelection.classSelection.detecting')" />
      </div>

      <div v-else-if="detectedClasses.length === 0" class="text-center py-8 text-gray-400">
        <Package class="h-12 w-12 mx-auto mb-4 opacity-50" />
        <p>{{ t('workflow.sourceSelection.classSelection.noClasses') }}</p>
      </div>

      <div v-else class="space-y-4">
        <!-- Bulk actions -->
        <div class="flex items-center justify-between">
          <span class="text-sm text-gray-400">
            {{ t('workflow.sourceSelection.classSelection.selectedCount', { selected: selectedClasses.length, total: detectedClasses.length }) }}
          </span>
          <div class="flex gap-2">
            <BaseButton variant="ghost" size="sm" @click="selectAllClasses">
              <Check class="h-4 w-4" />
              {{ t('workflow.sourceSelection.classSelection.selectAll') }}
            </BaseButton>
            <BaseButton variant="ghost" size="sm" @click="deselectAllClasses">
              <X class="h-4 w-4" />
              {{ t('workflow.sourceSelection.classSelection.deselectAll') }}
            </BaseButton>
          </div>
        </div>

        <!-- Class list -->
        <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
          <button
            v-for="cls in detectedClasses"
            :key="cls.name"
            @click="toggleClass(cls.name)"
            :class="[
              'flex items-center gap-3 p-3 rounded-lg border transition-all text-left',
              cls.selected
                ? 'border-primary bg-primary/10 text-white'
                : 'border-gray-700 bg-background-tertiary text-gray-400 hover:border-gray-500'
            ]"
          >
            <div
              :class="[
                'w-5 h-5 rounded border-2 flex items-center justify-center flex-shrink-0',
                cls.selected ? 'border-primary bg-primary' : 'border-gray-600'
              ]"
            >
              <Check v-if="cls.selected" class="h-3 w-3 text-white" />
            </div>
            <div class="min-w-0 flex-1">
              <p class="text-sm font-medium truncate">{{ cls.name }}</p>
              <p class="text-xs text-gray-500">{{ cls.count }} {{ t('workflow.sourceSelection.classSelection.images') }}</p>
            </div>
          </button>
        </div>
      </div>
    </div>

    <!-- Output Directory -->
    <div class="card p-6">
      <h3 class="text-lg font-semibold text-white mb-4">{{ t('workflow.sourceSelection.outputDirectory') }}</h3>
      <DirectoryBrowser
        v-model="outputDir"
        :label="t('workflow.sourceSelection.outputPath')"
        :placeholder="isCustomMode ? '/app/output/synthetic_custom' : '/app/output/synthetic'"
        path-mode="output"
      />
      <p class="text-sm text-gray-500 mt-2">{{ t('workflow.sourceSelection.outputHint') }}</p>
    </div>

    <!-- Navigation Buttons -->
    <div class="flex justify-between">
      <BaseButton variant="outline" @click="goBack">
        <ArrowLeft class="h-5 w-5" />
        {{ t('common.actions.back') }}
      </BaseButton>
      <BaseButton @click="saveAndContinue">
        {{ t('workflow.sourceSelection.continueButton') }}
        <ArrowRight class="h-5 w-5" />
      </BaseButton>
    </div>
  </div>
</template>
