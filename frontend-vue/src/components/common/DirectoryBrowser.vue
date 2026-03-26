<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { listDirectories, listFiles, checkPathExists } from '@/lib/api'
import { useMountPoints } from '@/composables/useMountPoints'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseInput from '@/components/ui/BaseInput.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import {
  Folder,
  FolderOpen,
  File,
  ChevronRight,
  ChevronUp,
  Home,
  RefreshCw,
  Check,
  Database,
  Image,
  Box,
  FolderOutput,
  AlertCircle,
} from 'lucide-vue-next'

const props = withDefaults(defineProps<{
  modelValue: string
  label?: string
  placeholder?: string
  showFiles?: boolean
  filePattern?: string
  basePath?: string
  /** Filter mount points by purpose: 'input', 'output', or 'both' */
  pathMode?: 'input' | 'output' | 'both'
  /** Show quick access buttons for mount points */
  showMountPoints?: boolean
  /** Restrict path selection to valid mount points only */
  restrictToMounts?: boolean
}>(), {
  label: 'Select Directory',
  placeholder: '/app/datasets/...',
  showFiles: false,
  filePattern: '*',
  basePath: '/app/datasets',
  pathMode: 'both',
  showMountPoints: true,
  restrictToMounts: true,
})

const emit = defineEmits<{
  (e: 'update:modelValue', value: string): void
  (e: 'select', path: string): void
}>()

// Mount points composable
const {
  mountPoints,
  loading: mountPointsLoading,
  loaded: mountPointsLoaded,
  getFilteredMountPoints,
  getDefaultPath,
  isPathValid,
  getMountPointForPath,
} = useMountPoints()

const isOpen = ref(false)
const loading = ref(false)
const currentPath = ref(props.basePath)
const directories = ref<string[]>([])
const files = ref<string[]>([])
const manualPath = ref(props.modelValue || '')
const error = ref<string | null>(null)
const validationError = ref<string | null>(null)

// Get filtered mount points based on pathMode
const filteredMountPoints = computed(() => getFilteredMountPoints(props.pathMode))

// Get current active mount point
const activeMountPoint = computed(() => getMountPointForPath(currentPath.value))

// Icon mapping for mount points
const mountPointIcons: Record<string, any> = {
  database: Database,
  image: Image,
  box: Box,
  'folder-output': FolderOutput,
}

// Breadcrumb parts
const breadcrumbs = computed(() => {
  const parts = currentPath.value.split('/').filter(Boolean)
  const crumbs: { name: string; path: string }[] = []
  let accumulated = ''

  for (const part of parts) {
    accumulated += '/' + part
    crumbs.push({ name: part, path: accumulated })
  }

  return crumbs
})

async function loadContents() {
  loading.value = true
  error.value = null

  try {
    directories.value = await listDirectories(currentPath.value)

    if (props.showFiles) {
      files.value = await listFiles(currentPath.value, props.filePattern)
    }
  } catch (e: any) {
    error.value = e.message || 'Failed to load directory'
    directories.value = []
    files.value = []
  } finally {
    loading.value = false
  }
}

function navigateTo(path: string) {
  currentPath.value = path
  loadContents()
}

function navigateUp() {
  const parts = currentPath.value.split('/').filter(Boolean)
  if (parts.length > 1) {
    parts.pop()
    currentPath.value = '/' + parts.join('/')
    loadContents()
  }
}

function navigateToRoot() {
  currentPath.value = props.basePath
  loadContents()
}

function navigateToMountPoint(path: string) {
  currentPath.value = path
  loadContents()
}

function validatePath(path: string): boolean {
  if (!props.restrictToMounts) return true

  const valid = isPathValid(path, props.pathMode)
  if (!valid) {
    validationError.value = 'Path must be within an allowed mount point'
  } else {
    validationError.value = null
  }
  return valid
}

function selectDirectory(dir: string) {
  if (!validatePath(dir)) return

  manualPath.value = dir
  emit('update:modelValue', dir)
  emit('select', dir)
  isOpen.value = false
}

function selectFile(file: string) {
  manualPath.value = file
  emit('update:modelValue', file)
  emit('select', file)
  isOpen.value = false
}

async function applyManualPath() {
  if (!manualPath.value) return

  // First validate against mount points
  if (!validatePath(manualPath.value)) {
    error.value = validationError.value
    return
  }

  const result = await checkPathExists(manualPath.value)
  if (result.exists) {
    emit('update:modelValue', manualPath.value)
    emit('select', manualPath.value)
    isOpen.value = false
    error.value = null
  } else {
    error.value = 'Path does not exist'
  }
}

function getDirectoryName(path: string): string {
  return path.split('/').pop() || path
}

function getFileName(path: string): string {
  return path.split('/').pop() || path
}

function toggleBrowser() {
  isOpen.value = !isOpen.value
  if (isOpen.value) {
    loadContents()
  }
}

// Update manual path when modelValue changes
watch(() => props.modelValue, (newVal) => {
  manualPath.value = newVal
})

// Initialize with default path based on mode when mount points are loaded
watch(mountPointsLoaded, (loaded) => {
  if (loaded && !props.modelValue) {
    const defaultPath = getDefaultPath(props.pathMode)
    if (defaultPath) {
      currentPath.value = defaultPath
    }
  }
})

onMounted(() => {
  if (props.modelValue) {
    manualPath.value = props.modelValue
    // Also set currentPath for browser navigation
    currentPath.value = props.modelValue
  }
})
</script>

<template>
  <div class="space-y-2">
    <label v-if="label" class="block text-sm font-medium text-gray-400">
      {{ label }}
    </label>

    <!-- Quick access mount points -->
    <div v-if="showMountPoints && filteredMountPoints.length > 0" class="flex flex-wrap gap-2 mb-2">
      <span class="text-xs text-gray-500 self-center mr-1">Quick access:</span>
      <button
        v-for="mp in filteredMountPoints"
        :key="mp.id"
        @click="navigateToMountPoint(mp.path); manualPath = mp.path"
        :class="[
          'flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium transition-colors',
          manualPath === mp.path || manualPath.startsWith(mp.path + '/')
            ? 'bg-primary text-white'
            : mp.exists
              ? 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              : 'bg-gray-800 text-gray-500 cursor-not-allowed'
        ]"
        :disabled="!mp.exists"
        :title="mp.description"
      >
        <component :is="mountPointIcons[mp.icon] || Folder" class="h-3 w-3" />
        {{ mp.name }}
      </button>
    </div>

    <!-- Input with browse button -->
    <div class="flex gap-2">
      <BaseInput
        v-model="manualPath"
        :placeholder="placeholder"
        class="flex-1"
        :class="{ 'border-red-500': validationError }"
        @keyup.enter="applyManualPath"
      />
      <BaseButton variant="outline" @click="toggleBrowser">
        <FolderOpen class="h-4 w-4" />
        Browse
      </BaseButton>
    </div>

    <!-- Validation error message -->
    <div v-if="validationError" class="flex items-center gap-1 text-xs text-red-400 mt-1">
      <AlertCircle class="h-3 w-3" />
      {{ validationError }}
    </div>

    <!-- Browser Panel -->
    <div
      v-if="isOpen"
      class="card p-4 mt-2 max-h-96 overflow-hidden flex flex-col"
    >
      <!-- Navigation bar -->
      <div class="flex items-center gap-2 mb-3 pb-3 border-b border-gray-700">
        <BaseButton variant="ghost" size="sm" @click="navigateToRoot" title="Go to root">
          <Home class="h-4 w-4" />
        </BaseButton>
        <BaseButton
          variant="ghost"
          size="sm"
          @click="navigateUp"
          :disabled="currentPath === props.basePath"
          title="Go up"
        >
          <ChevronUp class="h-4 w-4" />
        </BaseButton>
        <BaseButton variant="ghost" size="sm" @click="loadContents" :disabled="loading" title="Refresh">
          <RefreshCw :class="['h-4 w-4', loading ? 'animate-spin' : '']" />
        </BaseButton>

        <!-- Breadcrumbs -->
        <div class="flex-1 flex items-center gap-1 overflow-x-auto text-sm">
          <button
            v-for="(crumb, index) in breadcrumbs"
            :key="crumb.path"
            @click="navigateTo(crumb.path)"
            class="flex items-center text-gray-400 hover:text-white transition-colors"
          >
            <span v-if="index > 0" class="mx-1 text-gray-600">/</span>
            <span :class="index === breadcrumbs.length - 1 ? 'text-white' : ''">
              {{ crumb.name }}
            </span>
          </button>
        </div>
      </div>

      <!-- Select current directory button -->
      <button
        @click="selectDirectory(currentPath)"
        class="flex items-center gap-2 w-full px-3 py-2 mb-2 rounded-lg bg-primary/10 hover:bg-primary/20 text-primary transition-colors"
      >
        <Check class="h-4 w-4" />
        <span class="text-sm">Select this directory</span>
        <span class="text-xs text-gray-400 ml-auto truncate max-w-48">{{ currentPath }}</span>
      </button>

      <!-- Loading state -->
      <div v-if="loading" class="flex justify-center py-8">
        <LoadingSpinner size="sm" message="Loading..." />
      </div>

      <!-- Error state -->
      <div v-else-if="error" class="text-center py-4 text-red-400">
        {{ error }}
      </div>

      <!-- Empty state -->
      <div
        v-else-if="directories.length === 0 && files.length === 0"
        class="text-center py-8 text-gray-400"
      >
        <Folder class="h-8 w-8 mx-auto mb-2 opacity-50" />
        <p class="text-sm">Empty directory</p>
      </div>

      <!-- Directory and file list -->
      <div v-else class="overflow-y-auto flex-1 space-y-1">
        <!-- Directories -->
        <button
          v-for="dir in directories"
          :key="dir"
          @click="navigateTo(dir)"
          class="flex items-center gap-3 w-full px-3 py-2 rounded-lg hover:bg-gray-700 transition-colors text-left group"
        >
          <Folder class="h-4 w-4 text-yellow-400 flex-shrink-0" />
          <span class="text-sm text-gray-200 truncate flex-1">{{ getDirectoryName(dir) }}</span>
          <ChevronRight class="h-4 w-4 text-gray-500 opacity-0 group-hover:opacity-100 transition-opacity" />
        </button>

        <!-- Files (if showFiles is true) -->
        <button
          v-for="file in files"
          :key="file"
          @click="selectFile(file)"
          class="flex items-center gap-3 w-full px-3 py-2 rounded-lg hover:bg-gray-700 transition-colors text-left"
        >
          <File class="h-4 w-4 text-blue-400 flex-shrink-0" />
          <span class="text-sm text-gray-200 truncate">{{ getFileName(file) }}</span>
        </button>
      </div>
    </div>
  </div>
</template>
