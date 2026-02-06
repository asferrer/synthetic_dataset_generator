<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useDomainStore } from '@/stores/domain'
import { useUiStore } from '@/stores/ui'
import BaseButton from '@/components/ui/BaseButton.vue'
import AlertBox from '@/components/common/AlertBox.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import {
  ArrowLeft,
  Check,
  Edit,
  Download,
  MapPin,
  Package,
  Palette,
  Sparkles,
  Settings,
  Zap,
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
  Grid3x3,
} from 'lucide-vue-next'
import type { Domain } from '@/types/api'

const route = useRoute()
const router = useRouter()
const domainStore = useDomainStore()
const uiStore = useUiStore()

const loading = ref(true)
const domain = ref<Domain | null>(null)
const activeTab = ref<'regions' | 'objects' | 'effects' | 'physics' | 'matrix'>('regions')

// Icon mapping
const iconComponents: Record<string, any> = {
  Waves, Flame, Bird, Box, TreePine, Building2, Cloud, Droplets, Fish, Wind,
}

function getIconComponent(iconName: string) {
  return iconComponents[iconName] || Box
}

async function loadDomain() {
  const domainId = route.params.id as string
  loading.value = true

  const loadedDomain = await domainStore.fetchDomain(domainId)
  if (loadedDomain) {
    domain.value = loadedDomain
  } else {
    uiStore.showError('Error', `Domain ${domainId} not found`)
    router.push('/domains')
  }

  loading.value = false
}

async function handleActivate() {
  if (!domain.value) return
  const success = await domainStore.setActiveDomain(domain.value.domain_id)
  if (success) {
    uiStore.showSuccess('Domain Activated', `${domain.value.name} is now active`)
  }
}

async function handleExport() {
  if (!domain.value) return
  const exportedDomain = await domainStore.exportDomain(domain.value.domain_id)
  if (exportedDomain) {
    const blob = new Blob([JSON.stringify(exportedDomain, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${domain.value.domain_id}.json`
    a.click()
    URL.revokeObjectURL(url)
    uiStore.showSuccess('Exported', `${domain.value.domain_id}.json downloaded`)
  }
}

function rgbToHex(rgb: number[]): string {
  return '#' + rgb.map(c => c.toString(16).padStart(2, '0')).join('')
}

function getCompatibilityColor(score: number): string {
  if (score >= 0.8) return 'bg-green-500'
  if (score >= 0.5) return 'bg-yellow-500'
  if (score > 0) return 'bg-orange-500'
  return 'bg-gray-700'
}

const isActive = computed(() =>
  domainStore.activeDomainId === domain.value?.domain_id
)

watch(() => route.params.id, loadDomain)
onMounted(loadDomain)
</script>

<template>
  <div class="space-y-6">
    <!-- Loading -->
    <div v-if="loading" class="flex justify-center py-12">
      <LoadingSpinner size="lg" message="Loading domain..." />
    </div>

    <!-- Domain Content -->
    <template v-else-if="domain">
      <!-- Header -->
      <div class="flex items-start justify-between">
        <div class="flex items-center gap-4">
          <button
            @click="router.push('/domains')"
            class="p-2 rounded-lg bg-background-tertiary hover:bg-gray-600 transition-colors"
          >
            <ArrowLeft class="h-5 w-5 text-gray-400" />
          </button>
          <div class="p-3 rounded-xl bg-background-tertiary">
            <component :is="getIconComponent(domain.icon)" class="h-8 w-8 text-primary" />
          </div>
          <div>
            <div class="flex items-center gap-3">
              <h1 class="text-2xl font-bold text-white">{{ domain.name }}</h1>
              <span v-if="domain.is_builtin" class="badge badge-gray">Built-in</span>
              <span v-else class="badge badge-info">Custom</span>
              <span v-if="isActive" class="badge badge-success flex items-center gap-1">
                <Check class="h-3 w-3" />
                Active
              </span>
            </div>
            <p class="text-gray-400 mt-1">{{ domain.description }}</p>
          </div>
        </div>
        <div class="flex gap-2">
          <BaseButton variant="outline" @click="handleExport">
            <Download class="h-4 w-4" />
            Export
          </BaseButton>
          <BaseButton
            v-if="!domain.is_builtin"
            variant="outline"
            @click="router.push(`/domains/${domain.domain_id}/edit`)"
          >
            <Edit class="h-4 w-4" />
            Edit
          </BaseButton>
          <BaseButton v-if="!isActive" @click="handleActivate">
            <Check class="h-4 w-4" />
            Activate
          </BaseButton>
        </div>
      </div>

      <!-- Stats Overview -->
      <div class="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
        <div class="card p-4">
          <div class="flex items-center gap-3">
            <div class="p-2 rounded-lg bg-blue-500/20">
              <MapPin class="h-5 w-5 text-blue-400" />
            </div>
            <div>
              <p class="text-2xl font-bold text-white">{{ domain.regions.length }}</p>
              <p class="text-sm text-gray-400">Regions</p>
            </div>
          </div>
        </div>
        <div class="card p-4">
          <div class="flex items-center gap-3">
            <div class="p-2 rounded-lg bg-purple-500/20">
              <Package class="h-5 w-5 text-purple-400" />
            </div>
            <div>
              <p class="text-2xl font-bold text-white">{{ domain.objects.length }}</p>
              <p class="text-sm text-gray-400">Objects</p>
            </div>
          </div>
        </div>
        <div class="card p-4">
          <div class="flex items-center gap-3">
            <div class="p-2 rounded-lg bg-green-500/20">
              <Sparkles class="h-5 w-5 text-green-400" />
            </div>
            <div>
              <p class="text-2xl font-bold text-white">{{ domain.effects.domain_specific.length }}</p>
              <p class="text-sm text-gray-400">Effects</p>
            </div>
          </div>
        </div>
        <div class="card p-4">
          <div class="flex items-center gap-3">
            <div class="p-2 rounded-lg bg-orange-500/20">
              <Palette class="h-5 w-5 text-orange-400" />
            </div>
            <div>
              <p class="text-2xl font-bold text-white">{{ domain.presets.length }}</p>
              <p class="text-sm text-gray-400">Presets</p>
            </div>
          </div>
        </div>
        <div class="card p-4">
          <div class="flex items-center gap-3">
            <div class="p-2 rounded-lg bg-cyan-500/20">
              <Settings class="h-5 w-5 text-cyan-400" />
            </div>
            <div>
              <p class="text-2xl font-bold text-white capitalize">{{ domain.physics.physics_type }}</p>
              <p class="text-sm text-gray-400">Physics</p>
            </div>
          </div>
        </div>
      </div>

      <!-- Tabs -->
      <div class="border-b border-gray-700">
        <nav class="flex gap-4">
          <button
            v-for="tab in ['regions', 'objects', 'effects', 'physics', 'matrix'] as const"
            :key="tab"
            :class="[
              'px-4 py-3 text-sm font-medium border-b-2 transition-colors capitalize',
              activeTab === tab
                ? 'border-primary text-primary'
                : 'border-transparent text-gray-400 hover:text-white'
            ]"
            @click="activeTab = tab"
          >
            {{ tab === 'matrix' ? 'Compatibility Matrix' : tab }}
          </button>
        </nav>
      </div>

      <!-- Tab Content -->
      <div class="card p-6">
        <!-- Regions Tab -->
        <div v-if="activeTab === 'regions'" class="space-y-4">
          <h3 class="text-lg font-semibold text-white mb-4">Scene Regions</h3>
          <div class="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            <div
              v-for="region in domain.regions"
              :key="region.id"
              class="p-4 rounded-lg bg-background-tertiary"
            >
              <div class="flex items-start gap-3">
                <div
                  class="w-8 h-8 rounded-lg flex-shrink-0"
                  :style="{ backgroundColor: rgbToHex(region.color_rgb) }"
                />
                <div class="flex-1 min-w-0">
                  <p class="font-medium text-white">{{ region.display_name }}</p>
                  <p class="text-xs text-gray-500 font-mono">{{ region.id }}</p>
                  <p v-if="region.sam3_prompt" class="text-sm text-gray-400 mt-1 truncate">
                    SAM3: "{{ region.sam3_prompt }}"
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Objects Tab -->
        <div v-if="activeTab === 'objects'" class="space-y-4">
          <h3 class="text-lg font-semibold text-white mb-4">Object Types</h3>
          <div v-if="domain.objects.length === 0" class="text-center py-8 text-gray-400">
            No objects defined for this domain.
          </div>
          <div v-else class="overflow-x-auto">
            <table class="w-full">
              <thead>
                <tr class="border-b border-gray-700">
                  <th class="text-left py-3 px-4 text-sm font-medium text-gray-400">Object</th>
                  <th class="text-left py-3 px-4 text-sm font-medium text-gray-400">Size (m)</th>
                  <th class="text-left py-3 px-4 text-sm font-medium text-gray-400">Density</th>
                  <th class="text-left py-3 px-4 text-sm font-medium text-gray-400">Behavior</th>
                  <th class="text-left py-3 px-4 text-sm font-medium text-gray-400">Keywords</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="obj in domain.objects"
                  :key="obj.class_name"
                  class="border-b border-gray-800 hover:bg-background-tertiary"
                >
                  <td class="py-3 px-4">
                    <div>
                      <p class="font-medium text-white">{{ obj.display_name }}</p>
                      <p class="text-xs text-gray-500 font-mono">{{ obj.class_name }}</p>
                    </div>
                  </td>
                  <td class="py-3 px-4 text-gray-300">{{ obj.real_world_size_meters }}</td>
                  <td class="py-3 px-4 text-gray-300">{{ obj.physics_properties.density_relative }}</td>
                  <td class="py-3 px-4">
                    <span class="badge badge-gray">{{ obj.physics_properties.behavior || 'N/A' }}</span>
                  </td>
                  <td class="py-3 px-4 text-sm text-gray-400">
                    {{ obj.keywords.slice(0, 3).join(', ') }}
                    <span v-if="obj.keywords.length > 3">...</span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Effects Tab -->
        <div v-if="activeTab === 'effects'" class="space-y-6">
          <div>
            <h3 class="text-lg font-semibold text-white mb-4">Domain-Specific Effects</h3>
            <div v-if="domain.effects.domain_specific.length === 0" class="text-gray-400">
              No domain-specific effects configured.
            </div>
            <div v-else class="grid gap-3 sm:grid-cols-2">
              <div
                v-for="effect in domain.effects.domain_specific"
                :key="effect.effect_id"
                class="p-4 rounded-lg bg-background-tertiary"
              >
                <div class="flex items-center justify-between mb-2">
                  <p class="font-medium text-white">{{ effect.effect_id }}</p>
                  <span
                    :class="[
                      'badge',
                      effect.enabled_by_default ? 'badge-success' : 'badge-gray'
                    ]"
                  >
                    {{ effect.enabled_by_default ? 'Enabled' : 'Disabled' }}
                  </span>
                </div>
                <div class="text-sm text-gray-400">
                  <div v-for="(value, key) in effect.parameters" :key="key" class="flex justify-between">
                    <span>{{ key }}:</span>
                    <span class="text-gray-300">{{ JSON.stringify(value) }}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div v-if="domain.effects.disabled.length > 0">
            <h4 class="font-medium text-white mb-2">Disabled Effects</h4>
            <div class="flex flex-wrap gap-2">
              <span
                v-for="effect in domain.effects.disabled"
                :key="effect"
                class="badge badge-error"
              >
                {{ effect }}
              </span>
            </div>
          </div>

          <div v-if="Object.keys(domain.effects.universal_overrides).length > 0">
            <h4 class="font-medium text-white mb-2">Universal Overrides</h4>
            <pre class="p-4 rounded-lg bg-background-tertiary text-sm text-gray-300 overflow-x-auto">{{ JSON.stringify(domain.effects.universal_overrides, null, 2) }}</pre>
          </div>
        </div>

        <!-- Physics Tab -->
        <div v-if="activeTab === 'physics'" class="space-y-4">
          <h3 class="text-lg font-semibold text-white mb-4">Physics Configuration</h3>
          <div class="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <div class="p-4 rounded-lg bg-background-tertiary">
              <p class="text-sm text-gray-400">Physics Type</p>
              <p class="text-lg font-semibold text-white capitalize">{{ domain.physics.physics_type }}</p>
            </div>
            <div class="p-4 rounded-lg bg-background-tertiary">
              <p class="text-sm text-gray-400">Medium Density</p>
              <p class="text-lg font-semibold text-white">{{ domain.physics.medium_density }}</p>
            </div>
            <div class="p-4 rounded-lg bg-background-tertiary">
              <p class="text-sm text-gray-400">Gravity Direction</p>
              <p class="text-lg font-semibold text-white capitalize">{{ domain.physics.gravity_direction }}</p>
            </div>
            <div class="p-4 rounded-lg bg-background-tertiary">
              <p class="text-sm text-gray-400">Float Threshold</p>
              <p class="text-lg font-semibold text-white">{{ domain.physics.float_threshold ?? 'N/A' }}</p>
            </div>
            <div class="p-4 rounded-lg bg-background-tertiary">
              <p class="text-sm text-gray-400">Sink Threshold</p>
              <p class="text-lg font-semibold text-white">{{ domain.physics.sink_threshold ?? 'N/A' }}</p>
            </div>
            <div class="p-4 rounded-lg bg-background-tertiary">
              <p class="text-sm text-gray-400">Surface Zone</p>
              <p class="text-lg font-semibold text-white">{{ domain.physics.surface_zone ?? 'N/A' }}</p>
            </div>
          </div>
        </div>

        <!-- Compatibility Matrix Tab -->
        <div v-if="activeTab === 'matrix'" class="space-y-4">
          <h3 class="text-lg font-semibold text-white mb-4">Object-Region Compatibility</h3>
          <div v-if="Object.keys(domain.compatibility_matrix).length === 0" class="text-center py-8 text-gray-400">
            No compatibility matrix defined.
          </div>
          <div v-else class="overflow-x-auto">
            <table class="min-w-full">
              <thead>
                <tr class="border-b border-gray-700">
                  <th class="text-left py-3 px-2 text-sm font-medium text-gray-400 sticky left-0 bg-background-secondary">Object</th>
                  <th
                    v-for="region in domain.regions"
                    :key="region.id"
                    class="text-center py-3 px-2 text-xs font-medium text-gray-400 min-w-[60px]"
                  >
                    {{ region.display_name }}
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="(scores, objectClass) in domain.compatibility_matrix"
                  :key="objectClass"
                  class="border-b border-gray-800"
                >
                  <td class="py-2 px-2 text-sm text-white sticky left-0 bg-background-secondary">
                    {{ objectClass }}
                  </td>
                  <td
                    v-for="region in domain.regions"
                    :key="region.id"
                    class="text-center py-2 px-2"
                  >
                    <div
                      v-if="scores[region.id] !== undefined"
                      :class="[
                        'w-8 h-8 mx-auto rounded flex items-center justify-center text-xs font-medium',
                        getCompatibilityColor(scores[region.id])
                      ]"
                      :title="`${objectClass} + ${region.id}: ${scores[region.id]}`"
                    >
                      {{ (scores[region.id] * 100).toFixed(0) }}
                    </div>
                    <div v-else class="w-8 h-8 mx-auto rounded bg-gray-800 flex items-center justify-center text-gray-600">
                      -
                    </div>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <div class="flex items-center gap-4 text-xs text-gray-400 mt-4">
            <span class="flex items-center gap-1">
              <div class="w-3 h-3 rounded bg-green-500" /> High (80-100%)
            </span>
            <span class="flex items-center gap-1">
              <div class="w-3 h-3 rounded bg-yellow-500" /> Medium (50-79%)
            </span>
            <span class="flex items-center gap-1">
              <div class="w-3 h-3 rounded bg-orange-500" /> Low (1-49%)
            </span>
            <span class="flex items-center gap-1">
              <div class="w-3 h-3 rounded bg-gray-700" /> None (0%)
            </span>
          </div>
        </div>
      </div>
    </template>

    <!-- Not Found -->
    <AlertBox v-else type="error" title="Domain not found" />
  </div>
</template>
