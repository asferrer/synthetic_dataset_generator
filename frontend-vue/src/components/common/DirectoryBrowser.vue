<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { listDirectories, listFiles, checkPathExists } from '@/lib/api'
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
} from 'lucide-vue-next'

const props = withDefaults(defineProps<{
  modelValue: string
  label?: string
  placeholder?: string
  showFiles?: boolean
  filePattern?: string
  basePath?: string
}>(), {
  label: 'Select Directory',
  placeholder: '/data/...',
  showFiles: false,
  filePattern: '*',
  basePath: '/data',
})

const emit = defineEmits<{
  (e: 'update:modelValue', value: string): void
  (e: 'select', path: string): void
}>()

const isOpen = ref(false)
const loading = ref(false)
const currentPath = ref(props.basePath)
const directories = ref<string[]>([])
const files = ref<string[]>([])
const manualPath = ref(props.modelValue || '')
const error = ref<string | null>(null)

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

function selectDirectory(dir: string) {
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

  const result = await checkPathExists(manualPath.value)
  if (result.exists) {
    emit('update:modelValue', manualPath.value)
    emit('select', manualPath.value)
    isOpen.value = false
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

onMounted(() => {
  if (props.modelValue) {
    manualPath.value = props.modelValue
  }
})
</script>

<template>
  <div class="space-y-2">
    <label v-if="label" class="block text-sm font-medium text-gray-400">
      {{ label }}
    </label>

    <!-- Input with browse button -->
    <div class="flex gap-2">
      <BaseInput
        v-model="manualPath"
        :placeholder="placeholder"
        class="flex-1"
        @keyup.enter="applyManualPath"
      />
      <BaseButton variant="outline" @click="toggleBrowser">
        <FolderOpen class="h-4 w-4" />
        Browse
      </BaseButton>
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
