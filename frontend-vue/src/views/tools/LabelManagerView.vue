<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useUiStore } from '@/stores/ui'
import { listDatasets, analyzeDataset, renameCategory, deleteCategory } from '@/lib/api'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseSelect from '@/components/ui/BaseSelect.vue'
import BaseInput from '@/components/ui/BaseInput.vue'
import AlertBox from '@/components/common/AlertBox.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import {
  Tags,
  Edit,
  Trash2,
  Save,
  RefreshCw,
  X,
  AlertTriangle,
} from 'lucide-vue-next'
import type { DatasetInfo, DatasetAnalysis } from '@/types/api'

const uiStore = useUiStore()

const loading = ref(false)
const saving = ref(false)
const datasets = ref<DatasetInfo[]>([])
const selectedDataset = ref<string | null>(null)
const datasetAnalysis = ref<DatasetAnalysis | null>(null)
const editingCategory = ref<number | null>(null)
const newCategoryName = ref('')
const categoryToDelete = ref<{ id: number; name: string } | null>(null)
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

async function loadDatasetAnalysis() {
  if (!selectedDataset.value) return

  loading.value = true
  try {
    datasetAnalysis.value = await analyzeDataset(selectedDataset.value)
  } catch (e: any) {
    error.value = e.message || 'Failed to load dataset info'
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

function startEdit(categoryId: number, currentName: string) {
  editingCategory.value = categoryId
  newCategoryName.value = currentName
}

function cancelEdit() {
  editingCategory.value = null
  newCategoryName.value = ''
}

async function saveEdit() {
  if (!newCategoryName.value.trim() || !selectedDataset.value || editingCategory.value === null) return

  saving.value = true
  try {
    const result = await renameCategory(
      selectedDataset.value,
      editingCategory.value,
      newCategoryName.value.trim()
    )
    uiStore.showSuccess('Category Renamed', `Renamed to "${result.new_name}"`)
    cancelEdit()
    // Reload analysis to see updated categories
    await loadDatasetAnalysis()
  } catch (e: any) {
    uiStore.showError('Rename Failed', e.message || 'Failed to rename category')
  } finally {
    saving.value = false
  }
}

function confirmDelete(categoryId: number, categoryName: string) {
  categoryToDelete.value = { id: categoryId, name: categoryName }
}

function cancelDelete() {
  categoryToDelete.value = null
}

async function executeDelete() {
  if (!categoryToDelete.value || !selectedDataset.value) return

  saving.value = true
  try {
    const result = await deleteCategory(
      selectedDataset.value,
      categoryToDelete.value.id,
      true // delete annotations
    )
    uiStore.showSuccess(
      'Category Deleted',
      `Deleted "${result.category_name}" and ${result.annotations_deleted} annotations`
    )
    cancelDelete()
    // Reload analysis to see updated categories
    await loadDatasetAnalysis()
  } catch (e: any) {
    uiStore.showError('Delete Failed', e.message || 'Failed to delete category')
  } finally {
    saving.value = false
  }
}

// Watch for dataset selection changes
watch(selectedDataset, () => {
  loadDatasetAnalysis()
})

// Load datasets on mount
loadDatasets()
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div>
      <h2 class="text-2xl font-bold text-white">Label Manager</h2>
      <p class="mt-1 text-gray-400">
        View and manage category labels in your datasets.
      </p>
    </div>

    <!-- Error -->
    <AlertBox v-if="error" type="error" :title="error" dismissible @dismiss="error = null" />

    <!-- Delete Confirmation Modal -->
    <div
      v-if="categoryToDelete"
      class="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
      @click.self="cancelDelete"
    >
      <div class="card p-6 max-w-md mx-4">
        <div class="flex items-center gap-4 mb-4">
          <AlertTriangle class="h-8 w-8 text-red-400" />
          <div>
            <h3 class="text-lg font-semibold text-white">Delete Category?</h3>
            <p class="text-gray-400">This action cannot be undone.</p>
          </div>
        </div>

        <p class="text-gray-300 mb-6">
          Are you sure you want to delete the category <strong class="text-white">"{{ categoryToDelete.name }}"</strong>?
          This will also delete all annotations associated with this category.
        </p>

        <div class="flex justify-end gap-4">
          <BaseButton variant="outline" @click="cancelDelete" :disabled="saving">
            Cancel
          </BaseButton>
          <BaseButton variant="danger" @click="executeDelete" :loading="saving">
            <Trash2 class="h-4 w-4" />
            Delete Category
          </BaseButton>
        </div>
      </div>
    </div>

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
          placeholder="Choose a dataset..."
          class="flex-1"
        />
        <BaseButton variant="outline" @click="loadDatasets">
          <RefreshCw class="h-5 w-5" />
        </BaseButton>
      </div>
    </div>

    <!-- Categories -->
    <div v-if="datasetAnalysis" class="card p-6">
      <div class="flex items-center justify-between mb-4">
        <h3 class="text-lg font-semibold text-white">
          Categories ({{ datasetAnalysis.categories.length }})
        </h3>
        <BaseButton variant="outline" size="sm" @click="loadDatasetAnalysis" :disabled="loading">
          <RefreshCw :class="['h-4 w-4', loading ? 'animate-spin' : '']" />
          Refresh
        </BaseButton>
      </div>

      <div v-if="loading" class="flex justify-center py-8">
        <LoadingSpinner />
      </div>

      <EmptyState
        v-else-if="datasetAnalysis.categories.length === 0"
        :icon="Tags"
        title="No categories"
        description="This dataset has no categories defined."
      />

      <div v-else class="space-y-2">
        <div
          v-for="category in datasetAnalysis.categories"
          :key="category.id"
          class="flex items-center gap-4 p-3 rounded-lg bg-background-tertiary"
        >
          <div
            class="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/20 text-primary text-sm font-medium"
          >
            {{ category.id }}
          </div>

          <div v-if="editingCategory === category.id" class="flex-1 flex gap-2">
            <BaseInput
              v-model="newCategoryName"
              class="flex-1"
              placeholder="Enter new name..."
              @keyup.enter="saveEdit"
              @keyup.escape="cancelEdit"
            />
            <BaseButton size="sm" @click="saveEdit" :loading="saving" :disabled="!newCategoryName.trim()">
              <Save class="h-4 w-4" />
            </BaseButton>
            <BaseButton variant="ghost" size="sm" @click="cancelEdit" :disabled="saving">
              <X class="h-4 w-4" />
            </BaseButton>
          </div>

          <div v-else class="flex-1">
            <p class="font-medium text-white">{{ category.name }}</p>
            <p class="text-sm text-gray-400">{{ category.count }} annotations</p>
          </div>

          <div v-if="editingCategory !== category.id" class="flex gap-2">
            <BaseButton
              variant="ghost"
              size="sm"
              title="Rename category"
              @click="startEdit(category.id, category.name)"
            >
              <Edit class="h-4 w-4" />
            </BaseButton>
            <BaseButton
              variant="ghost"
              size="sm"
              title="Delete category"
              @click="confirmDelete(category.id, category.name)"
            >
              <Trash2 class="h-4 w-4 text-red-400" />
            </BaseButton>
          </div>
        </div>
      </div>

      <!-- Dataset Stats -->
      <div class="mt-6 pt-4 border-t border-gray-700">
        <div class="grid grid-cols-3 gap-4 text-center">
          <div>
            <p class="text-2xl font-bold text-primary">{{ datasetAnalysis.total_images }}</p>
            <p class="text-sm text-gray-400">Images</p>
          </div>
          <div>
            <p class="text-2xl font-bold text-primary">{{ datasetAnalysis.total_annotations }}</p>
            <p class="text-sm text-gray-400">Annotations</p>
          </div>
          <div>
            <p class="text-2xl font-bold text-primary">{{ datasetAnalysis.categories.length }}</p>
            <p class="text-sm text-gray-400">Categories</p>
          </div>
        </div>
      </div>
    </div>

    <!-- No dataset selected -->
    <EmptyState
      v-else-if="!loading"
      :icon="Tags"
      title="Select a dataset"
      description="Choose a dataset above to view and manage its categories."
    />
  </div>
</template>
