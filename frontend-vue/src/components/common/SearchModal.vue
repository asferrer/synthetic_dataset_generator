<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import {
  Search,
  X,
  Folder,
  FileText,
  Settings,
  Wrench,
  Play,
  BarChart,
  Layers,
  Server,
  Tag,
  Scissors,
  Download,
  GitMerge,
  Split,
  Command,
} from 'lucide-vue-next'

const router = useRouter()

const props = defineProps<{
  open: boolean
}>()

const emit = defineEmits<{
  close: []
}>()

const query = ref('')
const selectedIndex = ref(0)
const inputRef = ref<HTMLInputElement | null>(null)

// Navigation items
const navigationItems = [
  { id: 'analysis', label: 'Dataset Analysis', icon: BarChart, path: '/analysis', group: 'Workflow' },
  { id: 'configure', label: 'Configure Effects', icon: Settings, path: '/configure', group: 'Workflow' },
  { id: 'source', label: 'Source Selection', icon: Folder, path: '/source-selection', group: 'Workflow' },
  { id: 'generation', label: 'Generation', icon: Play, path: '/generation', group: 'Workflow' },
  { id: 'export', label: 'Export Dataset', icon: Download, path: '/export', group: 'Workflow' },
  { id: 'combine', label: 'Combine Datasets', icon: GitMerge, path: '/combine', group: 'Workflow' },
  { id: 'splits', label: 'Create Splits', icon: Split, path: '/splits', group: 'Workflow' },
  { id: 'job-monitor', label: 'Job Monitor', icon: Layers, path: '/tools/job-monitor', group: 'Tools' },
  { id: 'service-status', label: 'Service Status', icon: Server, path: '/tools/service-status', group: 'Tools' },
  { id: 'labeling', label: 'Labeling Tool', icon: Tag, path: '/tools/labeling', group: 'Tools' },
  { id: 'extraction', label: 'Object Extraction', icon: Scissors, path: '/tools/object-extraction', group: 'Tools' },
  { id: 'sam', label: 'SAM Segmentation', icon: Layers, path: '/tools/sam', group: 'Tools' },
  { id: 'label-manager', label: 'Label Manager', icon: Tag, path: '/tools/label-manager', group: 'Tools' },
  { id: 'object-sizes', label: 'Object Sizes', icon: Settings, path: '/tools/object-sizes', group: 'Tools' },
]

// Filter items based on query
const filteredItems = computed(() => {
  if (!query.value.trim()) {
    return navigationItems
  }

  const searchQuery = query.value.toLowerCase()
  return navigationItems.filter(item =>
    item.label.toLowerCase().includes(searchQuery) ||
    item.group.toLowerCase().includes(searchQuery)
  )
})

// Group items by category
const groupedItems = computed(() => {
  const groups: Record<string, typeof navigationItems> = {}
  for (const item of filteredItems.value) {
    if (!groups[item.group]) {
      groups[item.group] = []
    }
    groups[item.group].push(item)
  }
  return groups
})

// Flat list of all filtered items for keyboard navigation
const flatItems = computed(() => filteredItems.value)

function selectItem(item: typeof navigationItems[0]) {
  router.push(item.path)
  emit('close')
}

function handleKeyDown(event: KeyboardEvent) {
  switch (event.key) {
    case 'ArrowDown':
      event.preventDefault()
      selectedIndex.value = Math.min(selectedIndex.value + 1, flatItems.value.length - 1)
      break
    case 'ArrowUp':
      event.preventDefault()
      selectedIndex.value = Math.max(selectedIndex.value - 1, 0)
      break
    case 'Enter':
      event.preventDefault()
      if (flatItems.value[selectedIndex.value]) {
        selectItem(flatItems.value[selectedIndex.value])
      }
      break
    case 'Escape':
      emit('close')
      break
  }
}

// Reset selected index when query changes
watch(query, () => {
  selectedIndex.value = 0
})

// Focus input when modal opens
watch(() => props.open, (isOpen) => {
  if (isOpen) {
    query.value = ''
    selectedIndex.value = 0
    setTimeout(() => inputRef.value?.focus(), 50)
  }
})

// Global keyboard shortcut
function handleGlobalKeyDown(event: KeyboardEvent) {
  if ((event.ctrlKey || event.metaKey) && event.key === 'k') {
    event.preventDefault()
    if (props.open) {
      emit('close')
    }
  }
}

onMounted(() => {
  document.addEventListener('keydown', handleGlobalKeyDown)
})

onUnmounted(() => {
  document.removeEventListener('keydown', handleGlobalKeyDown)
})
</script>

<template>
  <Teleport to="body">
    <Transition name="modal">
      <div
        v-if="open"
        class="fixed inset-0 z-50 flex items-start justify-center pt-[15vh]"
      >
        <!-- Backdrop -->
        <div
          class="absolute inset-0 bg-black/60 backdrop-blur-sm"
          @click="$emit('close')"
        />

        <!-- Modal -->
        <div
          class="relative w-full max-w-xl rounded-xl border border-gray-700 bg-background-secondary shadow-2xl"
          @keydown="handleKeyDown"
        >
          <!-- Search Input -->
          <div class="flex items-center gap-3 border-b border-gray-700 p-4">
            <Search class="h-5 w-5 text-gray-400" />
            <input
              ref="inputRef"
              v-model="query"
              type="text"
              placeholder="Search pages, tools, and actions..."
              class="flex-1 bg-transparent text-white placeholder-gray-500 focus:outline-none"
            />
            <div class="flex items-center gap-1 text-xs text-gray-500">
              <kbd class="rounded bg-gray-700 px-1.5 py-0.5 font-mono">esc</kbd>
              <span>to close</span>
            </div>
          </div>

          <!-- Results -->
          <div class="max-h-[400px] overflow-y-auto p-2">
            <div v-if="flatItems.length === 0" class="py-8 text-center text-gray-400">
              No results found for "{{ query }}"
            </div>

            <template v-for="(items, group) in groupedItems" :key="group">
              <div class="px-3 py-2">
                <p class="text-xs font-medium text-gray-500 uppercase tracking-wider">
                  {{ group }}
                </p>
              </div>
              <div class="space-y-0.5">
                <button
                  v-for="(item, index) in items"
                  :key="item.id"
                  @click="selectItem(item)"
                  :class="[
                    'flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left transition-colors',
                    flatItems.indexOf(item) === selectedIndex
                      ? 'bg-primary/20 text-white'
                      : 'text-gray-300 hover:bg-gray-700/50'
                  ]"
                >
                  <component :is="item.icon" class="h-4 w-4 text-gray-400" />
                  <span>{{ item.label }}</span>
                </button>
              </div>
            </template>
          </div>

          <!-- Footer -->
          <div class="flex items-center justify-between border-t border-gray-700 px-4 py-2.5 text-xs text-gray-500">
            <div class="flex items-center gap-4">
              <span class="flex items-center gap-1">
                <kbd class="rounded bg-gray-700 px-1.5 py-0.5 font-mono">Enter</kbd>
                to select
              </span>
              <span class="flex items-center gap-1">
                <kbd class="rounded bg-gray-700 px-1.5 py-0.5 font-mono">&uarr;</kbd>
                <kbd class="rounded bg-gray-700 px-1.5 py-0.5 font-mono">&darr;</kbd>
                to navigate
              </span>
            </div>
            <div class="flex items-center gap-1">
              <Command class="h-3 w-3" />
              <kbd class="rounded bg-gray-700 px-1.5 py-0.5 font-mono">K</kbd>
              <span>to toggle</span>
            </div>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.modal-enter-active,
.modal-leave-active {
  transition: all 0.2s ease;
}

.modal-enter-from,
.modal-leave-to {
  opacity: 0;
}

.modal-enter-from > div:last-child,
.modal-leave-to > div:last-child {
  transform: scale(0.95);
}
</style>
