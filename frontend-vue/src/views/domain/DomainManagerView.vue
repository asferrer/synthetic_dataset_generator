<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useDomainStore } from '@/stores/domain'
import { useUiStore } from '@/stores/ui'
import BaseButton from '@/components/ui/BaseButton.vue'
import AlertBox from '@/components/common/AlertBox.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import {
  Globe,
  Plus,
  Edit,
  Trash2,
  Download,
  Upload,
  Check,
  Waves,
  Flame,
  Bird,
  Box,
  TreePine,
  Building2,
  Cloud,
  Droplets,
  Fish,
  Wind,
  MapPin,
  Package,
  Palette,
  Settings,
  ChevronRight,
} from 'lucide-vue-next'
import type { DomainSummary, Domain } from '@/types/api'

const router = useRouter()
const domainStore = useDomainStore()
const uiStore = useUiStore()

const showCreateModal = ref(false)
const showImportModal = ref(false)
const showDeleteConfirm = ref(false)
const domainToDelete = ref<DomainSummary | null>(null)
const importJsonText = ref('')
const importError = ref<string | null>(null)

// Icon mapping
const iconComponents: Record<string, any> = {
  Waves,
  Flame,
  Bird,
  Box,
  TreePine,
  Building2,
  Cloud,
  Droplets,
  Fish,
  Wind,
}

function getIconComponent(iconName: string) {
  return iconComponents[iconName] || Box
}

// New domain form
const newDomain = ref({
  domain_id: '',
  name: '',
  description: '',
  icon: 'Box',
})

async function handleCreateDomain() {
  if (!newDomain.value.domain_id || !newDomain.value.name) {
    uiStore.showError('Error', 'Domain ID and name are required')
    return
  }

  const domain = await domainStore.createDomain({
    domain_id: newDomain.value.domain_id,
    name: newDomain.value.name,
    description: newDomain.value.description,
    icon: newDomain.value.icon,
    regions: [
      {
        id: 'unknown',
        name: 'unknown',
        display_name: 'Unknown',
        color_rgb: [200, 200, 200],
        sam3_prompt: null,
      }
    ],
    objects: [],
    compatibility_matrix: {},
  })

  if (domain) {
    uiStore.showSuccess('Domain Created', `${domain.name} has been created`)
    showCreateModal.value = false
    newDomain.value = { domain_id: '', name: '', description: '', icon: 'Box' }
    router.push(`/domains/${domain.domain_id}/edit`)
  }
}

async function handleActivateDomain(domainId: string) {
  const success = await domainStore.setActiveDomain(domainId)
  if (success) {
    uiStore.showSuccess('Domain Activated', `${domainId} is now active`)
  }
}

async function handleDeleteDomain() {
  if (!domainToDelete.value) return

  const success = await domainStore.deleteDomain(domainToDelete.value.domain_id)
  if (success) {
    uiStore.showSuccess('Domain Deleted', `${domainToDelete.value.name} has been deleted`)
  }
  showDeleteConfirm.value = false
  domainToDelete.value = null
}

async function handleExportDomain(domainId: string) {
  const domain = await domainStore.exportDomain(domainId)
  if (domain) {
    const blob = new Blob([JSON.stringify(domain, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${domainId}.json`
    a.click()
    URL.revokeObjectURL(url)
    uiStore.showSuccess('Exported', `${domainId}.json downloaded`)
  }
}

async function handleImportDomain() {
  importError.value = null
  try {
    const domainData = JSON.parse(importJsonText.value) as Domain
    const domain = await domainStore.importDomain(domainData)
    if (domain) {
      uiStore.showSuccess('Domain Imported', `${domain.name} has been imported`)
      showImportModal.value = false
      importJsonText.value = ''
    }
  } catch (e: any) {
    importError.value = e.message || 'Invalid JSON format'
  }
}

function confirmDelete(domain: DomainSummary) {
  domainToDelete.value = domain
  showDeleteConfirm.value = true
}

const iconOptions = [
  { value: 'Box', label: 'Box' },
  { value: 'Waves', label: 'Waves' },
  { value: 'Flame', label: 'Flame' },
  { value: 'Bird', label: 'Bird' },
  { value: 'Fish', label: 'Fish' },
  { value: 'TreePine', label: 'Tree' },
  { value: 'Cloud', label: 'Cloud' },
  { value: 'Wind', label: 'Wind' },
  { value: 'Droplets', label: 'Droplets' },
  { value: 'Building2', label: 'Building' },
]

onMounted(() => {
  domainStore.fetchDomains()
})
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-start justify-between">
      <div>
        <h1 class="text-2xl font-bold text-white">Domain Manager</h1>
        <p class="mt-1 text-gray-400">
          Configure and manage domain configurations for different synthetic dataset types.
        </p>
      </div>
      <div class="flex gap-2">
        <BaseButton variant="outline" @click="showImportModal = true">
          <Upload class="h-4 w-4" />
          Import
        </BaseButton>
        <BaseButton @click="showCreateModal = true">
          <Plus class="h-4 w-4" />
          New Domain
        </BaseButton>
      </div>
    </div>

    <!-- Loading -->
    <div v-if="domainStore.isLoading" class="flex justify-center py-12">
      <LoadingSpinner size="lg" message="Loading domains..." />
    </div>

    <!-- Error -->
    <AlertBox
      v-else-if="domainStore.error"
      type="error"
      :title="domainStore.error"
      dismissible
      @dismiss="domainStore.error = null"
    />

    <!-- Domains List -->
    <template v-else>
      <!-- Active Domain Banner -->
      <div v-if="domainStore.activeDomain" class="card p-4 bg-primary/10 border border-primary/30">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <div class="p-2 rounded-lg bg-primary/20">
              <component :is="getIconComponent(domainStore.activeDomain.icon)" class="h-6 w-6 text-primary" />
            </div>
            <div>
              <p class="text-sm text-gray-400">Active Domain</p>
              <p class="font-semibold text-white">{{ domainStore.activeDomain.name }}</p>
            </div>
          </div>
          <BaseButton
            variant="outline"
            size="sm"
            @click="router.push(`/domains/${domainStore.activeDomainId}/edit`)"
          >
            <Settings class="h-4 w-4" />
            Configure
          </BaseButton>
        </div>
      </div>

      <!-- Built-in Domains -->
      <div>
        <h2 class="text-lg font-semibold text-white mb-4">Built-in Domains</h2>
        <div class="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <div
            v-for="domain in domainStore.builtinDomains"
            :key="domain.domain_id"
            :class="[
              'card p-5 cursor-pointer transition-all hover:ring-2',
              domainStore.activeDomainId === domain.domain_id
                ? 'ring-2 ring-primary bg-primary/5'
                : 'hover:ring-gray-500'
            ]"
            @click="handleActivateDomain(domain.domain_id)"
          >
            <div class="flex items-start justify-between mb-4">
              <div class="p-3 rounded-xl bg-background-tertiary">
                <component :is="getIconComponent(domain.icon)" class="h-8 w-8 text-primary" />
              </div>
              <Check
                v-if="domainStore.activeDomainId === domain.domain_id"
                class="h-5 w-5 text-primary"
              />
            </div>
            <h3 class="font-semibold text-white mb-1">{{ domain.name }}</h3>
            <p class="text-sm text-gray-400 mb-4 line-clamp-2">{{ domain.description }}</p>
            <div class="flex items-center gap-4 text-xs text-gray-500">
              <span class="flex items-center gap-1">
                <MapPin class="h-3.5 w-3.5" />
                {{ domain.region_count }} regions
              </span>
              <span class="flex items-center gap-1">
                <Package class="h-3.5 w-3.5" />
                {{ domain.object_count }} objects
              </span>
            </div>
            <div class="flex gap-2 mt-4 pt-4 border-t border-gray-700">
              <BaseButton
                variant="ghost"
                size="sm"
                @click.stop="router.push(`/domains/${domain.domain_id}`)"
              >
                View
              </BaseButton>
              <BaseButton
                variant="ghost"
                size="sm"
                @click.stop="handleExportDomain(domain.domain_id)"
              >
                <Download class="h-4 w-4" />
              </BaseButton>
            </div>
          </div>
        </div>
      </div>

      <!-- User Domains -->
      <div>
        <h2 class="text-lg font-semibold text-white mb-4">Custom Domains</h2>
        <div v-if="domainStore.userDomains.length === 0" class="card p-8">
          <EmptyState
            :icon="Globe"
            title="No custom domains yet"
            description="Create a new domain or import one to get started."
          >
            <template #actions>
              <BaseButton @click="showCreateModal = true">
                <Plus class="h-4 w-4" />
                Create Domain
              </BaseButton>
            </template>
          </EmptyState>
        </div>
        <div v-else class="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <div
            v-for="domain in domainStore.userDomains"
            :key="domain.domain_id"
            :class="[
              'card p-5 cursor-pointer transition-all hover:ring-2',
              domainStore.activeDomainId === domain.domain_id
                ? 'ring-2 ring-primary bg-primary/5'
                : 'hover:ring-gray-500'
            ]"
            @click="handleActivateDomain(domain.domain_id)"
          >
            <div class="flex items-start justify-between mb-4">
              <div class="p-3 rounded-xl bg-background-tertiary">
                <component :is="getIconComponent(domain.icon)" class="h-8 w-8 text-primary" />
              </div>
              <div class="flex items-center gap-2">
                <Check
                  v-if="domainStore.activeDomainId === domain.domain_id"
                  class="h-5 w-5 text-primary"
                />
                <span class="badge badge-info text-xs">Custom</span>
              </div>
            </div>
            <h3 class="font-semibold text-white mb-1">{{ domain.name }}</h3>
            <p class="text-sm text-gray-400 mb-4 line-clamp-2">{{ domain.description }}</p>
            <div class="flex items-center gap-4 text-xs text-gray-500">
              <span class="flex items-center gap-1">
                <MapPin class="h-3.5 w-3.5" />
                {{ domain.region_count }} regions
              </span>
              <span class="flex items-center gap-1">
                <Package class="h-3.5 w-3.5" />
                {{ domain.object_count }} objects
              </span>
            </div>
            <div class="flex gap-2 mt-4 pt-4 border-t border-gray-700">
              <BaseButton
                variant="ghost"
                size="sm"
                @click.stop="router.push(`/domains/${domain.domain_id}/edit`)"
              >
                <Edit class="h-4 w-4" />
                Edit
              </BaseButton>
              <BaseButton
                variant="ghost"
                size="sm"
                @click.stop="handleExportDomain(domain.domain_id)"
              >
                <Download class="h-4 w-4" />
              </BaseButton>
              <BaseButton
                variant="ghost"
                size="sm"
                class="text-red-400 hover:text-red-300"
                @click.stop="confirmDelete(domain)"
              >
                <Trash2 class="h-4 w-4" />
              </BaseButton>
            </div>
          </div>
        </div>
      </div>
    </template>

    <!-- Create Domain Modal -->
    <Teleport to="body">
      <div
        v-if="showCreateModal"
        class="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
        @click.self="showCreateModal = false"
      >
        <div class="card p-6 w-full max-w-md mx-4">
          <h2 class="text-xl font-semibold text-white mb-4">Create New Domain</h2>

          <div class="space-y-4">
            <div>
              <label class="block text-sm font-medium text-gray-300 mb-1">Domain ID</label>
              <input
                v-model="newDomain.domain_id"
                type="text"
                class="input w-full"
                placeholder="e.g., my_custom_domain"
                pattern="^[a-z][a-z0-9_]*$"
              />
              <p class="text-xs text-gray-500 mt-1">Lowercase letters, numbers, and underscores only</p>
            </div>

            <div>
              <label class="block text-sm font-medium text-gray-300 mb-1">Name</label>
              <input
                v-model="newDomain.name"
                type="text"
                class="input w-full"
                placeholder="e.g., My Custom Domain"
              />
            </div>

            <div>
              <label class="block text-sm font-medium text-gray-300 mb-1">Description</label>
              <textarea
                v-model="newDomain.description"
                class="input w-full"
                rows="2"
                placeholder="Describe your domain..."
              />
            </div>

            <div>
              <label class="block text-sm font-medium text-gray-300 mb-1">Icon</label>
              <div class="flex flex-wrap gap-2">
                <button
                  v-for="icon in iconOptions"
                  :key="icon.value"
                  :class="[
                    'p-2 rounded-lg transition-colors',
                    newDomain.icon === icon.value
                      ? 'bg-primary text-white'
                      : 'bg-background-tertiary text-gray-400 hover:text-white'
                  ]"
                  @click="newDomain.icon = icon.value"
                  :title="icon.label"
                >
                  <component :is="iconComponents[icon.value]" class="h-5 w-5" />
                </button>
              </div>
            </div>
          </div>

          <div class="flex justify-end gap-3 mt-6">
            <BaseButton variant="outline" @click="showCreateModal = false">Cancel</BaseButton>
            <BaseButton @click="handleCreateDomain" :loading="domainStore.isLoading">
              Create Domain
            </BaseButton>
          </div>
        </div>
      </div>
    </Teleport>

    <!-- Import Domain Modal -->
    <Teleport to="body">
      <div
        v-if="showImportModal"
        class="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
        @click.self="showImportModal = false"
      >
        <div class="card p-6 w-full max-w-2xl mx-4">
          <h2 class="text-xl font-semibold text-white mb-4">Import Domain</h2>

          <AlertBox v-if="importError" type="error" :title="importError" class="mb-4" />

          <div>
            <label class="block text-sm font-medium text-gray-300 mb-1">Domain JSON</label>
            <textarea
              v-model="importJsonText"
              class="input w-full font-mono text-sm"
              rows="12"
              placeholder="Paste your domain JSON here..."
            />
          </div>

          <div class="flex justify-end gap-3 mt-6">
            <BaseButton variant="outline" @click="showImportModal = false">Cancel</BaseButton>
            <BaseButton @click="handleImportDomain" :loading="domainStore.isLoading">
              <Upload class="h-4 w-4" />
              Import
            </BaseButton>
          </div>
        </div>
      </div>
    </Teleport>

    <!-- Delete Confirmation Modal -->
    <Teleport to="body">
      <div
        v-if="showDeleteConfirm"
        class="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
        @click.self="showDeleteConfirm = false"
      >
        <div class="card p-6 w-full max-w-md mx-4">
          <h2 class="text-xl font-semibold text-white mb-2">Delete Domain</h2>
          <p class="text-gray-400 mb-6">
            Are you sure you want to delete <strong class="text-white">{{ domainToDelete?.name }}</strong>?
            This action cannot be undone.
          </p>

          <div class="flex justify-end gap-3">
            <BaseButton variant="outline" @click="showDeleteConfirm = false">Cancel</BaseButton>
            <BaseButton
              variant="destructive"
              @click="handleDeleteDomain"
              :loading="domainStore.isLoading"
            >
              <Trash2 class="h-4 w-4" />
              Delete
            </BaseButton>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>
