<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useDomainStore } from '@/stores/domain'
import { useUiStore } from '@/stores/ui'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseInput from '@/components/ui/BaseInput.vue'
import AlertBox from '@/components/common/AlertBox.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import {
  ArrowLeft,
  Save,
  Plus,
  Trash2,
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
  GripVertical,
  ChevronDown,
  ChevronRight,
  RotateCcw,
} from 'lucide-vue-next'
import type {
  Domain,
  DomainRegion,
  DomainObject,
  DomainUpdateRequest,
} from '@/types/api'

const route = useRoute()
const router = useRouter()
const domainStore = useDomainStore()
const uiStore = useUiStore()

const loading = ref(true)
const saving = ref(false)
const resetting = ref(false)
const originalDomain = ref<Domain | null>(null)
const editedDomain = ref<Domain | null>(null)
const activeSection = ref<string>('general')
const expandedRegions = ref<Set<string>>(new Set())
const expandedObjects = ref<Set<string>>(new Set())
const isBuiltinDomain = ref(false)
const hasOverride = ref(false)

// Icon mapping
const iconComponents: Record<string, any> = {
  Waves, Flame, Bird, Box, TreePine, Building2, Cloud, Droplets, Fish, Wind,
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

const physicsTypes = [
  { value: 'buoyancy', label: 'Buoyancy (Water)' },
  { value: 'aerial', label: 'Aerial (Air)' },
  { value: 'gravity', label: 'Gravity (Ground)' },
  { value: 'neutral', label: 'Neutral (No physics)' },
]

const behaviors = [
  { value: 'float', label: 'Float' },
  { value: 'sink', label: 'Sink' },
  { value: 'neutral', label: 'Neutral' },
  { value: 'rise', label: 'Rise' },
  { value: 'fall', label: 'Fall' },
  { value: 'fly', label: 'Fly' },
  { value: 'static', label: 'Static' },
]

async function loadDomain() {
  const domainId = route.params.id as string
  loading.value = true

  const domain = await domainStore.fetchDomain(domainId)
  if (domain) {
    // Check if this is a built-in domain or has an override
    const overrideStatus = await domainStore.getOverrideStatus(domainId)
    isBuiltinDomain.value = domain.is_builtin || (overrideStatus?.is_builtin ?? false)
    hasOverride.value = overrideStatus?.has_override ?? false

    originalDomain.value = JSON.parse(JSON.stringify(domain))
    editedDomain.value = JSON.parse(JSON.stringify(domain))
  } else {
    uiStore.showError('Error', `Domain ${domainId} not found`)
    router.push('/domains')
  }

  loading.value = false
}

async function handleSave() {
  if (!editedDomain.value) return

  saving.value = true
  const updateRequest: DomainUpdateRequest = {
    name: editedDomain.value.name,
    description: editedDomain.value.description,
    version: editedDomain.value.version,
    icon: editedDomain.value.icon,
    regions: editedDomain.value.regions,
    objects: editedDomain.value.objects,
    compatibility_matrix: editedDomain.value.compatibility_matrix,
    effects: editedDomain.value.effects,
    physics: editedDomain.value.physics,
    presets: editedDomain.value.presets,
    labeling_templates: editedDomain.value.labeling_templates,
  }

  let result: Domain | null = null

  // Use createBuiltinOverride for built-in domains, updateDomain for user domains
  if (isBuiltinDomain.value) {
    result = await domainStore.createBuiltinOverride(editedDomain.value.domain_id, updateRequest)
    if (result) {
      hasOverride.value = true
      uiStore.showSuccess('Saved', 'Built-in domain customization saved successfully')
    }
  } else {
    result = await domainStore.updateDomain(editedDomain.value.domain_id, updateRequest)
    if (result) {
      uiStore.showSuccess('Saved', 'Domain updated successfully')
    }
  }

  if (result) {
    originalDomain.value = JSON.parse(JSON.stringify(result))
    editedDomain.value = JSON.parse(JSON.stringify(result))
  }
  saving.value = false
}

async function handleResetOverride() {
  if (!editedDomain.value || !hasOverride.value) return

  if (!confirm('Are you sure you want to reset this domain to its original built-in configuration? All your customizations will be lost.')) {
    return
  }

  resetting.value = true
  const result = await domainStore.resetBuiltinOverride(editedDomain.value.domain_id)
  if (result) {
    hasOverride.value = false
    originalDomain.value = JSON.parse(JSON.stringify(result))
    editedDomain.value = JSON.parse(JSON.stringify(result))
    uiStore.showSuccess('Reset', 'Domain restored to original built-in configuration')
  }
  resetting.value = false
}

function handleCancel() {
  if (hasChanges.value) {
    if (confirm('You have unsaved changes. Are you sure you want to leave?')) {
      router.push(`/domains/${editedDomain.value?.domain_id}`)
    }
  } else {
    router.push(`/domains/${editedDomain.value?.domain_id}`)
  }
}

// Region management
function addRegion() {
  if (!editedDomain.value) return
  const newRegion: DomainRegion = {
    id: `region_${Date.now()}`,
    name: 'new_region',
    display_name: 'New Region',
    color_rgb: [128, 128, 128],
    sam3_prompt: null,
  }
  editedDomain.value.regions.push(newRegion)
  expandedRegions.value.add(newRegion.id)
}

function removeRegion(index: number) {
  if (!editedDomain.value) return
  const region = editedDomain.value.regions[index]
  expandedRegions.value.delete(region.id)
  editedDomain.value.regions.splice(index, 1)
}

function toggleRegion(regionId: string) {
  if (expandedRegions.value.has(regionId)) {
    expandedRegions.value.delete(regionId)
  } else {
    expandedRegions.value.add(regionId)
  }
}

// Object management
function addObject() {
  if (!editedDomain.value) return
  const newObject: DomainObject = {
    class_name: `object_${Date.now()}`,
    display_name: 'New Object',
    real_world_size_meters: 1.0,
    keywords: [],
    physics_properties: {
      density_relative: 1.0,
      behavior: 'neutral',
    },
  }
  editedDomain.value.objects.push(newObject)
  expandedObjects.value.add(newObject.class_name)
}

function removeObject(index: number) {
  if (!editedDomain.value) return
  const obj = editedDomain.value.objects[index]
  expandedObjects.value.delete(obj.class_name)
  // Also remove from compatibility matrix
  delete editedDomain.value.compatibility_matrix[obj.class_name]
  editedDomain.value.objects.splice(index, 1)
}

function toggleObject(className: string) {
  if (expandedObjects.value.has(className)) {
    expandedObjects.value.delete(className)
  } else {
    expandedObjects.value.add(className)
  }
}

// Compatibility matrix
function setCompatibility(objectClass: string, regionId: string, score: number) {
  if (!editedDomain.value) return
  if (!editedDomain.value.compatibility_matrix[objectClass]) {
    editedDomain.value.compatibility_matrix[objectClass] = {}
  }
  if (score > 0) {
    editedDomain.value.compatibility_matrix[objectClass][regionId] = score
  } else {
    delete editedDomain.value.compatibility_matrix[objectClass][regionId]
  }
}

function getCompatibility(objectClass: string, regionId: string): number {
  return editedDomain.value?.compatibility_matrix[objectClass]?.[regionId] ?? 0
}

// Color helpers
function rgbToHex(rgb: number[]): string {
  return '#' + rgb.map(c => c.toString(16).padStart(2, '0')).join('')
}

function hexToRgb(hex: string): [number, number, number] {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)
  return result
    ? [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)]
    : [128, 128, 128]
}

function updateUniversalOverrides(value: string) {
  if (!editedDomain.value) return
  try {
    editedDomain.value.effects.universal_overrides = JSON.parse(value)
  } catch {
    // Invalid JSON, ignore
  }
}

const hasChanges = computed(() => {
  return JSON.stringify(originalDomain.value) !== JSON.stringify(editedDomain.value)
})

const sections = [
  { id: 'general', label: 'General' },
  { id: 'regions', label: 'Regions' },
  { id: 'objects', label: 'Objects' },
  { id: 'compatibility', label: 'Compatibility' },
  { id: 'effects', label: 'Effects' },
  { id: 'physics', label: 'Physics' },
]

watch(() => route.params.id, loadDomain)
onMounted(loadDomain)
</script>

<template>
  <div class="space-y-6">
    <!-- Loading -->
    <div v-if="loading" class="flex justify-center py-12">
      <LoadingSpinner size="lg" message="Loading domain..." />
    </div>

    <!-- Editor -->
    <template v-else-if="editedDomain">
      <!-- Header -->
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-4">
          <button
            @click="handleCancel"
            class="p-2 rounded-lg bg-background-tertiary hover:bg-gray-600 transition-colors"
          >
            <ArrowLeft class="h-5 w-5 text-gray-400" />
          </button>
          <div>
            <h1 class="text-2xl font-bold text-white">Edit Domain</h1>
            <p class="text-gray-400">{{ editedDomain.name }}</p>
          </div>
        </div>
        <div class="flex gap-2">
          <BaseButton variant="outline" @click="handleCancel">Cancel</BaseButton>
          <BaseButton
            @click="handleSave"
            :loading="saving"
            :disabled="!hasChanges"
          >
            <Save class="h-4 w-4" />
            Save Changes
          </BaseButton>
        </div>
      </div>

      <!-- Built-in Domain Notice -->
      <AlertBox
        v-if="isBuiltinDomain && !hasOverride"
        type="info"
        title="Editing Built-in Domain"
      >
        This is a built-in domain. Your changes will be saved as a custom override,
        allowing you to customize SAM3 prompts, regions, objects, and other settings
        while preserving the original configuration.
      </AlertBox>

      <!-- Override Active Notice -->
      <AlertBox
        v-if="isBuiltinDomain && hasOverride"
        type="info"
        title="Custom Override Active"
      >
        <div class="flex items-center justify-between">
          <span>This domain has been customized. You can reset to restore the original built-in configuration.</span>
          <BaseButton
            variant="outline"
            size="sm"
            @click="handleResetOverride"
            :loading="resetting"
            class="ml-4"
          >
            Reset to Original
          </BaseButton>
        </div>
      </AlertBox>

      <!-- Unsaved Changes Warning -->
      <AlertBox v-if="hasChanges" type="warning" title="You have unsaved changes" />

      <!-- Editor Layout -->
      <div class="flex gap-6">
        <!-- Sidebar Navigation -->
        <div class="w-48 flex-shrink-0">
          <nav class="sticky top-4 space-y-1">
            <button
              v-for="section in sections"
              :key="section.id"
              :class="[
                'w-full text-left px-4 py-2 rounded-lg transition-colors',
                activeSection === section.id
                  ? 'bg-primary text-white'
                  : 'text-gray-400 hover:text-white hover:bg-background-tertiary'
              ]"
              @click="activeSection = section.id"
            >
              {{ section.label }}
            </button>
          </nav>
        </div>

        <!-- Content -->
        <div class="flex-1 min-w-0">
          <!-- General Section -->
          <div v-show="activeSection === 'general'" class="card p-6 space-y-6">
            <h2 class="text-lg font-semibold text-white">General Information</h2>

            <div class="grid gap-6 sm:grid-cols-2">
              <div>
                <label class="block text-sm font-medium text-gray-300 mb-1">Domain ID</label>
                <input
                  :value="editedDomain.domain_id"
                  type="text"
                  class="input w-full bg-gray-800"
                  disabled
                />
                <p class="text-xs text-gray-500 mt-1">Cannot be changed after creation</p>
              </div>

              <div>
                <label class="block text-sm font-medium text-gray-300 mb-1">Version</label>
                <input
                  v-model="editedDomain.version"
                  type="text"
                  class="input w-full"
                  placeholder="1.0.0"
                />
              </div>

              <div class="sm:col-span-2">
                <label class="block text-sm font-medium text-gray-300 mb-1">Name</label>
                <input
                  v-model="editedDomain.name"
                  type="text"
                  class="input w-full"
                  placeholder="Domain Name"
                />
              </div>

              <div class="sm:col-span-2">
                <label class="block text-sm font-medium text-gray-300 mb-1">Description</label>
                <textarea
                  v-model="editedDomain.description"
                  class="input w-full"
                  rows="3"
                  placeholder="Describe your domain..."
                />
              </div>

              <div>
                <label class="block text-sm font-medium text-gray-300 mb-2">Icon</label>
                <div class="flex flex-wrap gap-2">
                  <button
                    v-for="icon in iconOptions"
                    :key="icon.value"
                    :class="[
                      'p-3 rounded-lg transition-colors',
                      editedDomain.icon === icon.value
                        ? 'bg-primary text-white'
                        : 'bg-background-tertiary text-gray-400 hover:text-white'
                    ]"
                    @click="editedDomain.icon = icon.value"
                    :title="icon.label"
                  >
                    <component :is="iconComponents[icon.value]" class="h-5 w-5" />
                  </button>
                </div>
              </div>
            </div>
          </div>

          <!-- Regions Section -->
          <div v-show="activeSection === 'regions'" class="card p-6 space-y-4">
            <div class="flex items-center justify-between">
              <h2 class="text-lg font-semibold text-white">Scene Regions</h2>
              <BaseButton size="sm" @click="addRegion">
                <Plus class="h-4 w-4" />
                Add Region
              </BaseButton>
            </div>

            <div class="space-y-2">
              <div
                v-for="(region, index) in editedDomain.regions"
                :key="region.id"
                class="border border-gray-700 rounded-lg overflow-hidden"
              >
                <button
                  class="w-full flex items-center gap-3 p-4 bg-background-tertiary hover:bg-gray-700 transition-colors"
                  @click="toggleRegion(region.id)"
                >
                  <GripVertical class="h-4 w-4 text-gray-500" />
                  <div
                    class="w-6 h-6 rounded"
                    :style="{ backgroundColor: rgbToHex(region.color_rgb) }"
                  />
                  <span class="flex-1 text-left font-medium text-white">{{ region.display_name }}</span>
                  <span class="text-sm text-gray-500 font-mono">{{ region.id }}</span>
                  <component
                    :is="expandedRegions.has(region.id) ? ChevronDown : ChevronRight"
                    class="h-5 w-5 text-gray-400"
                  />
                </button>

                <div v-if="expandedRegions.has(region.id)" class="p-4 border-t border-gray-700 space-y-4">
                  <div class="grid gap-4 sm:grid-cols-2">
                    <div>
                      <label class="block text-sm font-medium text-gray-300 mb-1">Region ID</label>
                      <input v-model="region.id" type="text" class="input w-full" />
                    </div>
                    <div>
                      <label class="block text-sm font-medium text-gray-300 mb-1">Display Name</label>
                      <input v-model="region.display_name" type="text" class="input w-full" />
                    </div>
                    <div>
                      <label class="block text-sm font-medium text-gray-300 mb-1">Color</label>
                      <div class="flex gap-2">
                        <input
                          type="color"
                          :value="rgbToHex(region.color_rgb)"
                          @input="region.color_rgb = hexToRgb(($event.target as HTMLInputElement).value)"
                          class="h-10 w-16 rounded cursor-pointer"
                        />
                        <input
                          :value="rgbToHex(region.color_rgb)"
                          type="text"
                          class="input flex-1"
                          @input="region.color_rgb = hexToRgb(($event.target as HTMLInputElement).value)"
                        />
                      </div>
                    </div>
                    <div>
                      <label class="block text-sm font-medium text-gray-300 mb-1">SAM3 Prompt</label>
                      <input
                        v-model="region.sam3_prompt"
                        type="text"
                        class="input w-full"
                        placeholder="Text prompt for segmentation"
                      />
                    </div>
                  </div>
                  <div class="flex justify-end">
                    <BaseButton
                      variant="destructive"
                      size="sm"
                      @click="removeRegion(index)"
                      :disabled="editedDomain.regions.length <= 1"
                    >
                      <Trash2 class="h-4 w-4" />
                      Remove
                    </BaseButton>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Objects Section -->
          <div v-show="activeSection === 'objects'" class="card p-6 space-y-4">
            <div class="flex items-center justify-between">
              <h2 class="text-lg font-semibold text-white">Object Types</h2>
              <BaseButton size="sm" @click="addObject">
                <Plus class="h-4 w-4" />
                Add Object
              </BaseButton>
            </div>

            <div v-if="editedDomain.objects.length === 0" class="text-center py-8 text-gray-400">
              No objects defined. Click "Add Object" to create one.
            </div>

            <div v-else class="space-y-2">
              <div
                v-for="(obj, index) in editedDomain.objects"
                :key="obj.class_name"
                class="border border-gray-700 rounded-lg overflow-hidden"
              >
                <button
                  class="w-full flex items-center gap-3 p-4 bg-background-tertiary hover:bg-gray-700 transition-colors"
                  @click="toggleObject(obj.class_name)"
                >
                  <GripVertical class="h-4 w-4 text-gray-500" />
                  <span class="flex-1 text-left font-medium text-white">{{ obj.display_name }}</span>
                  <span class="text-sm text-gray-500 font-mono">{{ obj.class_name }}</span>
                  <span class="badge badge-gray">{{ obj.real_world_size_meters }}m</span>
                  <component
                    :is="expandedObjects.has(obj.class_name) ? ChevronDown : ChevronRight"
                    class="h-5 w-5 text-gray-400"
                  />
                </button>

                <div v-if="expandedObjects.has(obj.class_name)" class="p-4 border-t border-gray-700 space-y-4">
                  <div class="grid gap-4 sm:grid-cols-2">
                    <div>
                      <label class="block text-sm font-medium text-gray-300 mb-1">Class Name</label>
                      <input v-model="obj.class_name" type="text" class="input w-full" />
                    </div>
                    <div>
                      <label class="block text-sm font-medium text-gray-300 mb-1">Display Name</label>
                      <input v-model="obj.display_name" type="text" class="input w-full" />
                    </div>
                    <div>
                      <label class="block text-sm font-medium text-gray-300 mb-1">Size (meters)</label>
                      <input
                        v-model.number="obj.real_world_size_meters"
                        type="number"
                        step="0.1"
                        min="0.01"
                        class="input w-full"
                      />
                    </div>
                    <div>
                      <label class="block text-sm font-medium text-gray-300 mb-1">Density (relative)</label>
                      <input
                        v-model.number="obj.physics_properties.density_relative"
                        type="number"
                        step="0.1"
                        min="0.001"
                        class="input w-full"
                      />
                    </div>
                    <div>
                      <label class="block text-sm font-medium text-gray-300 mb-1">Behavior</label>
                      <select v-model="obj.physics_properties.behavior" class="input w-full">
                        <option v-for="b in behaviors" :key="b.value" :value="b.value">
                          {{ b.label }}
                        </option>
                      </select>
                    </div>
                    <div>
                      <label class="block text-sm font-medium text-gray-300 mb-1">Keywords (comma-separated)</label>
                      <input
                        :value="obj.keywords.join(', ')"
                        @input="obj.keywords = ($event.target as HTMLInputElement).value.split(',').map(k => k.trim()).filter(k => k)"
                        type="text"
                        class="input w-full"
                        placeholder="keyword1, keyword2"
                      />
                    </div>
                  </div>
                  <div class="flex justify-end">
                    <BaseButton variant="destructive" size="sm" @click="removeObject(index)">
                      <Trash2 class="h-4 w-4" />
                      Remove
                    </BaseButton>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Compatibility Section -->
          <div v-show="activeSection === 'compatibility'" class="card p-6 space-y-4">
            <h2 class="text-lg font-semibold text-white">Object-Region Compatibility</h2>
            <p class="text-sm text-gray-400">
              Define how compatible each object is with each region. Values range from 0 (incompatible) to 1 (perfect match).
            </p>

            <div v-if="editedDomain.objects.length === 0" class="text-center py-8 text-gray-400">
              Add objects first to configure compatibility.
            </div>

            <div v-else class="overflow-x-auto">
              <table class="min-w-full">
                <thead>
                  <tr class="border-b border-gray-700">
                    <th class="text-left py-3 px-2 text-sm font-medium text-gray-400 sticky left-0 bg-background-secondary">Object</th>
                    <th
                      v-for="region in editedDomain.regions"
                      :key="region.id"
                      class="text-center py-3 px-2 text-xs font-medium text-gray-400 min-w-[80px]"
                    >
                      {{ region.display_name }}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <tr
                    v-for="obj in editedDomain.objects"
                    :key="obj.class_name"
                    class="border-b border-gray-800"
                  >
                    <td class="py-2 px-2 text-sm text-white sticky left-0 bg-background-secondary">
                      {{ obj.display_name }}
                    </td>
                    <td
                      v-for="region in editedDomain.regions"
                      :key="region.id"
                      class="text-center py-2 px-1"
                    >
                      <input
                        type="number"
                        :value="getCompatibility(obj.class_name, region.id)"
                        @input="setCompatibility(obj.class_name, region.id, parseFloat(($event.target as HTMLInputElement).value) || 0)"
                        min="0"
                        max="1"
                        step="0.1"
                        class="w-16 px-2 py-1 text-center text-sm rounded bg-background-tertiary border border-gray-600 text-white"
                      />
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <!-- Effects Section -->
          <div v-show="activeSection === 'effects'" class="card p-6 space-y-6">
            <h2 class="text-lg font-semibold text-white">Effects Configuration</h2>

            <div>
              <h3 class="font-medium text-white mb-2">Disabled Effects</h3>
              <p class="text-sm text-gray-400 mb-2">Enter effect IDs to disable (comma-separated)</p>
              <input
                :value="editedDomain.effects.disabled.join(', ')"
                @input="editedDomain.effects.disabled = ($event.target as HTMLInputElement).value.split(',').map(e => e.trim()).filter(e => e)"
                type="text"
                class="input w-full"
                placeholder="underwater, caustics"
              />
            </div>

            <div>
              <h3 class="font-medium text-white mb-2">Universal Overrides (JSON)</h3>
              <textarea
                :value="JSON.stringify(editedDomain.effects.universal_overrides, null, 2)"
                @input="updateUniversalOverrides(($event.target as HTMLTextAreaElement).value)"
                class="input w-full font-mono text-sm"
                rows="6"
                placeholder="{}"
              />
            </div>
          </div>

          <!-- Physics Section -->
          <div v-show="activeSection === 'physics'" class="card p-6 space-y-6">
            <h2 class="text-lg font-semibold text-white">Physics Configuration</h2>

            <div class="grid gap-6 sm:grid-cols-2">
              <div>
                <label class="block text-sm font-medium text-gray-300 mb-1">Physics Type</label>
                <select v-model="editedDomain.physics.physics_type" class="input w-full">
                  <option v-for="pt in physicsTypes" :key="pt.value" :value="pt.value">
                    {{ pt.label }}
                  </option>
                </select>
              </div>

              <div>
                <label class="block text-sm font-medium text-gray-300 mb-1">Medium Density</label>
                <input
                  v-model.number="editedDomain.physics.medium_density"
                  type="number"
                  step="0.001"
                  min="0"
                  class="input w-full"
                />
              </div>

              <div>
                <label class="block text-sm font-medium text-gray-300 mb-1">Float Threshold</label>
                <input
                  v-model.number="editedDomain.physics.float_threshold"
                  type="number"
                  step="0.1"
                  min="0"
                  max="1"
                  class="input w-full"
                />
              </div>

              <div>
                <label class="block text-sm font-medium text-gray-300 mb-1">Sink Threshold</label>
                <input
                  v-model.number="editedDomain.physics.sink_threshold"
                  type="number"
                  step="0.1"
                  min="0"
                  max="1"
                  class="input w-full"
                />
              </div>

              <div>
                <label class="block text-sm font-medium text-gray-300 mb-1">Surface Zone (Y position)</label>
                <input
                  v-model.number="editedDomain.physics.surface_zone"
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  class="input w-full"
                />
              </div>

              <div>
                <label class="block text-sm font-medium text-gray-300 mb-1">Bottom Zone (Y position)</label>
                <input
                  v-model.number="editedDomain.physics.bottom_zone"
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  class="input w-full"
                />
              </div>

              <div>
                <label class="block text-sm font-medium text-gray-300 mb-1">Gravity Direction</label>
                <select v-model="editedDomain.physics.gravity_direction" class="input w-full">
                  <option value="down">Down</option>
                  <option value="up">Up</option>
                </select>
              </div>
            </div>
          </div>
        </div>
      </div>
    </template>

    <!-- Not Found -->
    <AlertBox v-else type="error" title="Domain not found" />
  </div>
</template>
