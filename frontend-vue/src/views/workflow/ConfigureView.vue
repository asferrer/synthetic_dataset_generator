<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useWorkflowStore } from '@/stores/workflow'
import { useUiStore } from '@/stores/ui'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseSelect from '@/components/ui/BaseSelect.vue'
import {
  Switch,
  SwitchGroup,
  TabGroup,
  TabList,
  Tab,
  TabPanels,
  TabPanel,
} from '@headlessui/vue'
import {
  ArrowRight,
  ArrowLeft,
  RotateCcw,
  Sparkles,
  Sun,
  Contrast,
  Move,
  ZoomIn,
  FlipHorizontal,
  FlipVertical,
  Focus,
  Droplets,
  Waves,
  Eye,
  Layers,
  Box,
  Blend,
  Maximize2,
  Wind,
  CloudFog,
  Palette,
  Lightbulb,
} from 'lucide-vue-next'
import type { EffectsConfig, ObjectPlacementConfig } from '@/types/api'
import { useI18n } from 'vue-i18n'
import { useDomainGapStore } from '@/stores/domainGap'
import {
  Bookmark,
  Fish,
  Building2,
  Feather,
  Zap,
  Save,
  FolderOpen,
} from 'lucide-vue-next'

const router = useRouter()
const workflowStore = useWorkflowStore()
const uiStore = useUiStore()
const domainGapStore = useDomainGapStore()
const { t } = useI18n()

// ============================================
// PRESETS SYSTEM
// ============================================

interface EffectsPreset {
  id: string
  name: string
  icon: any
  description: string
  config: Partial<EffectsConfig>
}

const builtInPresets: EffectsPreset[] = [
  {
    id: 'underwater',
    name: 'Underwater Marine',
    icon: Fish,
    description: 'Optimized for underwater datasets with caustics and color tinting',
    config: {
      basic: {
        blur: { enabled: false, min_radius: 1, max_radius: 3 },
        noise: { enabled: false, min_intensity: 0.01, max_intensity: 0.03 },
        brightness: { enabled: true, min_factor: 0.85, max_factor: 1.15 },
        contrast: { enabled: true, min_factor: 0.9, max_factor: 1.1 },
        rotation: { enabled: true, min_angle: -15, max_angle: 15 },
        scale: { enabled: true, min_factor: 0.8, max_factor: 1.2 },
        flip_horizontal: { enabled: true, probability: 0.5 },
        flip_vertical: { enabled: false, probability: 0.3 },
      },
      color_correction: { enabled: true, color_intensity: 0.15 },
      blur_matching: { enabled: true, blur_strength: 0.4 },
      lighting: { enabled: true, lighting_type: 'ambient', lighting_intensity: 0.4 },
      underwater: { enabled: true, underwater_intensity: 0.2, water_color: [120, 80, 20], water_clarity: 'clear' },
      shadows: { enabled: true, shadow_opacity: 0.08, shadow_blur: 20 },
      caustics: { enabled: true, caustics_intensity: 0.12, caustics_deterministic: true },
      edge_smoothing: { enabled: true, edge_feather: 4 },
      motion_blur: { enabled: false, motion_blur_probability: 0.1, motion_blur_kernel: 9 },
      perspective: { enabled: false, perspective_magnitude: 0.05 },
    },
  },
  {
    id: 'urban',
    name: 'Urban Street',
    icon: Building2,
    description: 'For street scenes, traffic, and urban environments',
    config: {
      basic: {
        blur: { enabled: false, min_radius: 0.5, max_radius: 2 },
        noise: { enabled: true, min_intensity: 0.01, max_intensity: 0.04 },
        brightness: { enabled: true, min_factor: 0.8, max_factor: 1.2 },
        contrast: { enabled: true, min_factor: 0.85, max_factor: 1.15 },
        rotation: { enabled: true, min_angle: -5, max_angle: 5 },
        scale: { enabled: true, min_factor: 0.7, max_factor: 1.3 },
        flip_horizontal: { enabled: true, probability: 0.5 },
        flip_vertical: { enabled: false, probability: 0 },
      },
      color_correction: { enabled: true, color_intensity: 0.1 },
      blur_matching: { enabled: true, blur_strength: 0.3 },
      lighting: { enabled: true, lighting_type: 'gradient', lighting_intensity: 0.5 },
      underwater: { enabled: false, underwater_intensity: 0, water_color: [120, 80, 20], water_clarity: 'clear' },
      shadows: { enabled: true, shadow_opacity: 0.15, shadow_blur: 30 },
      caustics: { enabled: false, caustics_intensity: 0, caustics_deterministic: false },
      edge_smoothing: { enabled: true, edge_feather: 3 },
      motion_blur: { enabled: true, motion_blur_probability: 0.15, motion_blur_kernel: 11 },
      perspective: { enabled: true, perspective_magnitude: 0.06 },
    },
  },
  {
    id: 'minimal',
    name: 'Minimal',
    icon: Feather,
    description: 'Light augmentation for subtle variations',
    config: {
      basic: {
        blur: { enabled: false, min_radius: 0, max_radius: 1 },
        noise: { enabled: false, min_intensity: 0, max_intensity: 0.02 },
        brightness: { enabled: true, min_factor: 0.95, max_factor: 1.05 },
        contrast: { enabled: true, min_factor: 0.95, max_factor: 1.05 },
        rotation: { enabled: true, min_angle: -5, max_angle: 5 },
        scale: { enabled: true, min_factor: 0.95, max_factor: 1.05 },
        flip_horizontal: { enabled: true, probability: 0.5 },
        flip_vertical: { enabled: false, probability: 0 },
      },
      color_correction: { enabled: true, color_intensity: 0.08 },
      blur_matching: { enabled: true, blur_strength: 0.2 },
      lighting: { enabled: false, lighting_type: 'ambient', lighting_intensity: 0.3 },
      underwater: { enabled: false, underwater_intensity: 0, water_color: [120, 80, 20], water_clarity: 'clear' },
      shadows: { enabled: true, shadow_opacity: 0.05, shadow_blur: 15 },
      caustics: { enabled: false, caustics_intensity: 0, caustics_deterministic: false },
      edge_smoothing: { enabled: true, edge_feather: 2 },
      motion_blur: { enabled: false, motion_blur_probability: 0, motion_blur_kernel: 5 },
      perspective: { enabled: false, perspective_magnitude: 0 },
    },
  },
  {
    id: 'aggressive',
    name: 'Aggressive',
    icon: Zap,
    description: 'Heavy augmentation for maximum variety',
    config: {
      basic: {
        blur: { enabled: true, min_radius: 0.5, max_radius: 3 },
        noise: { enabled: true, min_intensity: 0.02, max_intensity: 0.08 },
        brightness: { enabled: true, min_factor: 0.7, max_factor: 1.3 },
        contrast: { enabled: true, min_factor: 0.7, max_factor: 1.3 },
        rotation: { enabled: true, min_angle: -30, max_angle: 30 },
        scale: { enabled: true, min_factor: 0.6, max_factor: 1.4 },
        flip_horizontal: { enabled: true, probability: 0.5 },
        flip_vertical: { enabled: true, probability: 0.3 },
      },
      color_correction: { enabled: true, color_intensity: 0.2 },
      blur_matching: { enabled: true, blur_strength: 0.6 },
      lighting: { enabled: true, lighting_type: 'spotlight', lighting_intensity: 0.6 },
      underwater: { enabled: false, underwater_intensity: 0, water_color: [120, 80, 20], water_clarity: 'clear' },
      shadows: { enabled: true, shadow_opacity: 0.2, shadow_blur: 35 },
      caustics: { enabled: false, caustics_intensity: 0, caustics_deterministic: false },
      edge_smoothing: { enabled: true, edge_feather: 5 },
      motion_blur: { enabled: true, motion_blur_probability: 0.25, motion_blur_kernel: 15 },
      perspective: { enabled: true, perspective_magnitude: 0.1 },
    },
  },
]

const selectedPreset = ref<string | null>(null)
const savedPresets = ref<EffectsPreset[]>([])
const showSavePresetModal = ref(false)
const newPresetName = ref('')

// Load saved presets from localStorage
function loadSavedPresets() {
  try {
    const saved = localStorage.getItem('effects_presets')
    if (saved) {
      savedPresets.value = JSON.parse(saved)
    }
  } catch (e) {
    console.error('Failed to load saved presets:', e)
  }
}

function savePresetsToStorage() {
  try {
    localStorage.setItem('effects_presets', JSON.stringify(savedPresets.value))
  } catch (e) {
    console.error('Failed to save presets:', e)
  }
}

function applyPreset(preset: EffectsPreset) {
  selectedPreset.value = preset.id
  // Deep merge the preset config with current config
  if (preset.config.basic) {
    effectsConfig.value.basic = { ...effectsConfig.value.basic, ...preset.config.basic }
  }
  if (preset.config.color_correction) {
    effectsConfig.value.color_correction = { ...effectsConfig.value.color_correction, ...preset.config.color_correction }
  }
  if (preset.config.blur_matching) {
    effectsConfig.value.blur_matching = { ...effectsConfig.value.blur_matching, ...preset.config.blur_matching }
  }
  if (preset.config.lighting) {
    effectsConfig.value.lighting = { ...effectsConfig.value.lighting, ...preset.config.lighting }
  }
  if (preset.config.underwater) {
    effectsConfig.value.underwater = { ...effectsConfig.value.underwater, ...preset.config.underwater }
  }
  if (preset.config.shadows) {
    effectsConfig.value.shadows = { ...effectsConfig.value.shadows, ...preset.config.shadows }
  }
  if (preset.config.caustics) {
    effectsConfig.value.caustics = { ...effectsConfig.value.caustics, ...preset.config.caustics }
  }
  if (preset.config.edge_smoothing) {
    effectsConfig.value.edge_smoothing = { ...effectsConfig.value.edge_smoothing, ...preset.config.edge_smoothing }
  }
  if (preset.config.motion_blur) {
    effectsConfig.value.motion_blur = { ...effectsConfig.value.motion_blur, ...preset.config.motion_blur }
  }
  if (preset.config.perspective) {
    effectsConfig.value.perspective = { ...effectsConfig.value.perspective, ...preset.config.perspective }
  }

  uiStore.showSuccess('Preset Applied', `Applied "${preset.name}" preset`)
}

function saveCurrentAsPreset() {
  if (!newPresetName.value.trim()) {
    uiStore.showError('Name Required', 'Please enter a name for the preset')
    return
  }

  const newPreset: EffectsPreset = {
    id: `custom_${Date.now()}`,
    name: newPresetName.value.trim(),
    icon: Bookmark,
    description: 'Custom saved preset',
    config: JSON.parse(JSON.stringify(effectsConfig.value)),
  }

  savedPresets.value.push(newPreset)
  savePresetsToStorage()
  showSavePresetModal.value = false
  newPresetName.value = ''
  uiStore.showSuccess('Preset Saved', `Saved "${newPreset.name}" preset`)
}

function deleteSavedPreset(presetId: string) {
  savedPresets.value = savedPresets.value.filter(p => p.id !== presetId)
  savePresetsToStorage()
  uiStore.showInfo('Preset Deleted', 'Custom preset has been deleted')
}

// Load saved presets on mount
loadSavedPresets()

// Load reference sets for auto-tune selector
domainGapStore.fetchReferenceSets()

// Deep copy of effects config
const effectsConfig = ref<EffectsConfig>(JSON.parse(JSON.stringify(workflowStore.effectsConfig)))
const placementConfig = ref<ObjectPlacementConfig>(JSON.parse(JSON.stringify(workflowStore.placementConfig)))

// Watch for external changes
watch(() => workflowStore.effectsConfig, (newVal) => {
  effectsConfig.value = JSON.parse(JSON.stringify(newVal))
}, { deep: true })

watch(() => workflowStore.placementConfig, (newVal) => {
  placementConfig.value = JSON.parse(JSON.stringify(newVal))
}, { deep: true })

// Tab options
const tabs = [
  { name: 'Basic Augmentation', icon: Sparkles },
  { name: 'Underwater & Realism', icon: Droplets },
  { name: 'Blending & Edges', icon: Blend },
  { name: 'Object Placement', icon: Box },
]

// Basic effects configuration
interface BasicEffectConfig {
  key: string
  label: string
  icon: any
  description: string
  hasRange: boolean
  rangeConfig?: {
    minKey: string
    maxKey: string
    min: number
    max: number
    step: number
    unit?: string
  }
  hasProbability?: boolean
}

const basicEffects: BasicEffectConfig[] = [
  {
    key: 'blur',
    label: 'Blur',
    icon: Focus,
    description: 'Apply Gaussian blur to images',
    hasRange: true,
    rangeConfig: { minKey: 'min_radius', maxKey: 'max_radius', min: 0, max: 10, step: 0.5, unit: 'px' },
  },
  {
    key: 'noise',
    label: 'Noise',
    icon: Sparkles,
    description: 'Add random noise to images',
    hasRange: true,
    rangeConfig: { minKey: 'min_intensity', maxKey: 'max_intensity', min: 0, max: 0.2, step: 0.01 },
  },
  {
    key: 'brightness',
    label: 'Brightness',
    icon: Sun,
    description: 'Adjust image brightness',
    hasRange: true,
    rangeConfig: { minKey: 'min_factor', maxKey: 'max_factor', min: 0.5, max: 1.5, step: 0.05 },
  },
  {
    key: 'contrast',
    label: 'Contrast',
    icon: Contrast,
    description: 'Adjust image contrast',
    hasRange: true,
    rangeConfig: { minKey: 'min_factor', maxKey: 'max_factor', min: 0.5, max: 1.5, step: 0.05 },
  },
  {
    key: 'rotation',
    label: 'Rotation',
    icon: Move,
    description: 'Rotate objects randomly',
    hasRange: true,
    rangeConfig: { minKey: 'min_angle', maxKey: 'max_angle', min: -180, max: 180, step: 5, unit: '°' },
  },
  {
    key: 'scale',
    label: 'Scale',
    icon: ZoomIn,
    description: 'Scale objects randomly',
    hasRange: true,
    rangeConfig: { minKey: 'min_factor', maxKey: 'max_factor', min: 0.5, max: 2, step: 0.1 },
  },
  {
    key: 'flip_horizontal',
    label: 'Horizontal Flip',
    icon: FlipHorizontal,
    description: 'Flip objects horizontally',
    hasRange: false,
    hasProbability: true,
  },
  {
    key: 'flip_vertical',
    label: 'Vertical Flip',
    icon: FlipVertical,
    description: 'Flip objects vertically',
    hasRange: false,
    hasProbability: true,
  },
]

// Lighting type options
const lightingTypeOptions = [
  { value: 'ambient', label: 'Ambient' },
  { value: 'spotlight', label: 'Spotlight' },
  { value: 'gradient', label: 'Gradient' },
]

// Water clarity options
const waterClarityOptions = [
  { value: 'clear', label: 'Clear' },
  { value: 'murky', label: 'Murky' },
  { value: 'very_murky', label: 'Very Murky' },
]

// Blend method options
const blendMethodOptions = [
  { value: 'alpha', label: 'Alpha Blend' },
  { value: 'poisson', label: 'Poisson Blend' },
  { value: 'laplacian', label: 'Laplacian Blend' },
]

function toggleBasicEffect(key: string) {
  const effect = effectsConfig.value.basic[key as keyof typeof effectsConfig.value.basic] as any
  effect.enabled = !effect.enabled
}

function saveConfig() {
  workflowStore.updateEffectsConfig(effectsConfig.value)
  workflowStore.updatePlacementConfig(placementConfig.value)
  workflowStore.markStepCompleted(2)
  uiStore.showSuccess('Configuration Saved', 'Effects and placement configuration has been saved')
  router.push('/source-selection')
}

function resetConfig() {
  workflowStore.resetEffectsConfig()
  workflowStore.resetPlacementConfig()
  effectsConfig.value = JSON.parse(JSON.stringify(workflowStore.effectsConfig))
  placementConfig.value = JSON.parse(JSON.stringify(workflowStore.placementConfig))
  uiStore.showInfo('Configuration Reset', 'All configurations have been reset to defaults')
}

function goBack() {
  router.push('/analysis')
}

const enabledBasicEffectsCount = computed(() =>
  basicEffects.filter(e => (effectsConfig.value.basic[e.key as keyof typeof effectsConfig.value.basic] as any).enabled).length
)

const enabledAdvancedEffectsCount = computed(() => {
  let count = 0
  if (effectsConfig.value.color_correction.enabled) count++
  if (effectsConfig.value.blur_matching.enabled) count++
  if (effectsConfig.value.lighting.enabled) count++
  if (effectsConfig.value.underwater.enabled) count++
  if (effectsConfig.value.motion_blur.enabled) count++
  if (effectsConfig.value.shadows.enabled) count++
  if (effectsConfig.value.caustics.enabled) count++
  if (effectsConfig.value.edge_smoothing.enabled) count++
  if (effectsConfig.value.perspective.enabled) count++
  return count
})
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-start justify-between">
      <div>
        <h2 class="text-2xl font-bold text-white">Configure Pipeline</h2>
        <p class="mt-2 text-gray-400">
          Configure augmentation effects and object placement for synthetic data generation.
          {{ enabledBasicEffectsCount }} basic effects, {{ enabledAdvancedEffectsCount }} advanced effects enabled.
        </p>
      </div>
      <button @click="resetConfig" class="btn-outline">
        <RotateCcw class="h-5 w-5" />
        Reset All
      </button>
    </div>

    <!-- Presets Section -->
    <div class="card p-6">
      <div class="flex items-center justify-between mb-4">
        <div>
          <h3 class="text-lg font-semibold text-white flex items-center gap-2">
            <Bookmark class="h-5 w-5 text-yellow-400" />
            Quick Presets
          </h3>
          <p class="text-sm text-gray-400">Select a preset or create your own configuration</p>
        </div>
        <BaseButton variant="outline" size="sm" @click="showSavePresetModal = true">
          <Save class="h-4 w-4" />
          Save Current
        </BaseButton>
      </div>

      <!-- Built-in Presets -->
      <div class="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 mb-4">
        <button
          v-for="preset in builtInPresets"
          :key="preset.id"
          @click="applyPreset(preset)"
          :class="[
            'flex flex-col items-center gap-2 p-4 rounded-lg transition-all text-center',
            selectedPreset === preset.id
              ? 'bg-primary/20 border-2 border-primary'
              : 'bg-background-tertiary hover:bg-gray-600 border-2 border-transparent'
          ]"
        >
          <component :is="preset.icon" class="h-8 w-8 text-primary" />
          <div>
            <p class="text-sm font-medium text-white">{{ preset.name }}</p>
            <p class="text-xs text-gray-400">{{ preset.description }}</p>
          </div>
        </button>
      </div>

      <!-- Saved Custom Presets -->
      <div v-if="savedPresets.length > 0">
        <p class="text-sm text-gray-400 mb-2">Your Saved Presets:</p>
        <div class="flex flex-wrap gap-2">
          <button
            v-for="preset in savedPresets"
            :key="preset.id"
            @click="applyPreset(preset)"
            :class="[
              'flex items-center gap-2 px-3 py-2 rounded-lg transition-all',
              selectedPreset === preset.id
                ? 'bg-primary/20 border border-primary'
                : 'bg-background-tertiary hover:bg-gray-600 border border-transparent'
            ]"
          >
            <FolderOpen class="h-4 w-4 text-primary" />
            <span class="text-sm text-white">{{ preset.name }}</span>
            <button
              @click.stop="deleteSavedPreset(preset.id)"
              class="text-gray-400 hover:text-red-400 ml-1"
            >
              ×
            </button>
          </button>
        </div>
      </div>
    </div>

    <!-- Save Preset Modal -->
    <div
      v-if="showSavePresetModal"
      class="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
      @click.self="showSavePresetModal = false"
    >
      <div class="card p-6 max-w-md mx-4">
        <h3 class="text-lg font-semibold text-white mb-4">Save Current Configuration</h3>
        <div class="space-y-4">
          <div>
            <label class="block text-sm text-gray-400 mb-1">Preset Name</label>
            <input
              v-model="newPresetName"
              type="text"
              class="input w-full"
              placeholder="My Custom Preset"
              @keyup.enter="saveCurrentAsPreset"
            />
          </div>
          <div class="flex justify-end gap-3">
            <BaseButton variant="outline" @click="showSavePresetModal = false">
              Cancel
            </BaseButton>
            <BaseButton @click="saveCurrentAsPreset">
              <Save class="h-4 w-4" />
              Save Preset
            </BaseButton>
          </div>
        </div>
      </div>
    </div>

    <!-- Tabs -->
    <TabGroup>
      <TabList class="flex space-x-1 rounded-xl bg-gray-800/50 p-1">
        <Tab
          v-for="tab in tabs"
          :key="tab.name"
          v-slot="{ selected }"
          class="w-full rounded-lg py-2.5 text-sm font-medium leading-5 transition-all focus:outline-none"
        >
          <div
            :class="[
              selected
                ? 'bg-primary text-white shadow'
                : 'text-gray-400 hover:bg-gray-700/50 hover:text-white',
              'flex items-center justify-center gap-2 rounded-lg px-3 py-2',
            ]"
          >
            <component :is="tab.icon" class="h-4 w-4" />
            {{ tab.name }}
          </div>
        </Tab>
      </TabList>

      <TabPanels class="mt-6">
        <!-- Basic Augmentation Tab -->
        <TabPanel class="space-y-6">
          <div class="grid gap-6 md:grid-cols-2">
            <div
              v-for="effect in basicEffects"
              :key="effect.key"
              :class="[
                'card p-5 transition-all',
                (effectsConfig.basic[effect.key as keyof typeof effectsConfig.basic] as any).enabled
                  ? 'border-primary/50 bg-primary/5'
                  : '',
              ]"
            >
              <div class="flex items-start justify-between mb-4">
                <div class="flex items-center gap-3">
                  <div
                    :class="[
                      'rounded-lg p-2',
                      (effectsConfig.basic[effect.key as keyof typeof effectsConfig.basic] as any).enabled
                        ? 'bg-primary/20 text-primary'
                        : 'bg-gray-700 text-gray-400',
                    ]"
                  >
                    <component :is="effect.icon" class="h-5 w-5" />
                  </div>
                  <div>
                    <h3 class="font-semibold text-white">{{ effect.label }}</h3>
                    <p class="text-sm text-gray-400">{{ effect.description }}</p>
                  </div>
                </div>
                <SwitchGroup>
                  <Switch
                    :model-value="(effectsConfig.basic[effect.key as keyof typeof effectsConfig.basic] as any).enabled"
                    @update:model-value="toggleBasicEffect(effect.key)"
                    :class="[
                      (effectsConfig.basic[effect.key as keyof typeof effectsConfig.basic] as any).enabled
                        ? 'bg-primary'
                        : 'bg-gray-600',
                      'relative inline-flex h-6 w-11 items-center rounded-full transition-colors',
                    ]"
                  >
                    <span
                      :class="[
                        (effectsConfig.basic[effect.key as keyof typeof effectsConfig.basic] as any).enabled
                          ? 'translate-x-6'
                          : 'translate-x-1',
                        'inline-block h-4 w-4 transform rounded-full bg-white transition-transform',
                      ]"
                    />
                  </Switch>
                </SwitchGroup>
              </div>

              <!-- Range Sliders -->
              <div
                v-if="effect.hasRange && (effectsConfig.basic[effect.key as keyof typeof effectsConfig.basic] as any).enabled"
                class="space-y-3 pt-4 border-t border-gray-700"
              >
                <div>
                  <label class="text-sm text-gray-400 flex justify-between mb-1">
                    <span>Min Value</span>
                    <span class="text-white font-mono">
                      {{ (effectsConfig.basic[effect.key as keyof typeof effectsConfig.basic] as any)[effect.rangeConfig!.minKey] }}{{ effect.rangeConfig!.unit || '' }}
                    </span>
                  </label>
                  <input
                    type="range"
                    v-model.number="(effectsConfig.basic[effect.key as keyof typeof effectsConfig.basic] as any)[effect.rangeConfig!.minKey]"
                    :min="effect.rangeConfig!.min"
                    :max="effect.rangeConfig!.max"
                    :step="effect.rangeConfig!.step"
                    class="w-full accent-primary"
                  />
                </div>
                <div>
                  <label class="text-sm text-gray-400 flex justify-between mb-1">
                    <span>Max Value</span>
                    <span class="text-white font-mono">
                      {{ (effectsConfig.basic[effect.key as keyof typeof effectsConfig.basic] as any)[effect.rangeConfig!.maxKey] }}{{ effect.rangeConfig!.unit || '' }}
                    </span>
                  </label>
                  <input
                    type="range"
                    v-model.number="(effectsConfig.basic[effect.key as keyof typeof effectsConfig.basic] as any)[effect.rangeConfig!.maxKey]"
                    :min="effect.rangeConfig!.min"
                    :max="effect.rangeConfig!.max"
                    :step="effect.rangeConfig!.step"
                    class="w-full accent-primary"
                  />
                </div>
              </div>

              <!-- Probability Slider -->
              <div
                v-if="effect.hasProbability && (effectsConfig.basic[effect.key as keyof typeof effectsConfig.basic] as any).enabled"
                class="pt-4 border-t border-gray-700"
              >
                <label class="text-sm text-gray-400 flex justify-between mb-1">
                  <span>Probability</span>
                  <span class="text-white font-mono">
                    {{ ((effectsConfig.basic[effect.key as keyof typeof effectsConfig.basic] as any).probability * 100).toFixed(0) }}%
                  </span>
                </label>
                <input
                  type="range"
                  v-model.number="(effectsConfig.basic[effect.key as keyof typeof effectsConfig.basic] as any).probability"
                  min="0"
                  max="1"
                  step="0.1"
                  class="w-full accent-primary"
                />
              </div>
            </div>
          </div>
        </TabPanel>

        <!-- Underwater & Realism Tab -->
        <TabPanel class="space-y-6">
          <div class="grid gap-6 md:grid-cols-2">
            <!-- Color Correction -->
            <div :class="['card p-5 transition-all', effectsConfig.color_correction.enabled ? 'border-primary/50 bg-primary/5' : '']">
              <div class="flex items-start justify-between mb-4">
                <div class="flex items-center gap-3">
                  <div :class="['rounded-lg p-2', effectsConfig.color_correction.enabled ? 'bg-primary/20 text-primary' : 'bg-gray-700 text-gray-400']">
                    <Palette class="h-5 w-5" />
                  </div>
                  <div>
                    <h3 class="font-semibold text-white">Color Correction</h3>
                    <p class="text-sm text-gray-400">Match object colors to background</p>
                  </div>
                </div>
                <Switch
                  v-model="effectsConfig.color_correction.enabled"
                  :class="[effectsConfig.color_correction.enabled ? 'bg-primary' : 'bg-gray-600', 'relative inline-flex h-6 w-11 items-center rounded-full transition-colors']"
                >
                  <span :class="[effectsConfig.color_correction.enabled ? 'translate-x-6' : 'translate-x-1', 'inline-block h-4 w-4 transform rounded-full bg-white transition-transform']" />
                </Switch>
              </div>
              <div v-if="effectsConfig.color_correction.enabled" class="pt-4 border-t border-gray-700">
                <label class="text-sm text-gray-400 flex justify-between mb-1">
                  <span>Color Intensity</span>
                  <span class="text-white font-mono">{{ effectsConfig.color_correction.color_intensity.toFixed(2) }}</span>
                </label>
                <input type="range" v-model.number="effectsConfig.color_correction.color_intensity" min="0" max="1" step="0.01" class="w-full accent-primary" />
              </div>
            </div>

            <!-- Blur Matching -->
            <div :class="['card p-5 transition-all', effectsConfig.blur_matching.enabled ? 'border-primary/50 bg-primary/5' : '']">
              <div class="flex items-start justify-between mb-4">
                <div class="flex items-center gap-3">
                  <div :class="['rounded-lg p-2', effectsConfig.blur_matching.enabled ? 'bg-primary/20 text-primary' : 'bg-gray-700 text-gray-400']">
                    <CloudFog class="h-5 w-5" />
                  </div>
                  <div>
                    <h3 class="font-semibold text-white">Blur Matching</h3>
                    <p class="text-sm text-gray-400">Match blur level to background</p>
                  </div>
                </div>
                <Switch
                  v-model="effectsConfig.blur_matching.enabled"
                  :class="[effectsConfig.blur_matching.enabled ? 'bg-primary' : 'bg-gray-600', 'relative inline-flex h-6 w-11 items-center rounded-full transition-colors']"
                >
                  <span :class="[effectsConfig.blur_matching.enabled ? 'translate-x-6' : 'translate-x-1', 'inline-block h-4 w-4 transform rounded-full bg-white transition-transform']" />
                </Switch>
              </div>
              <div v-if="effectsConfig.blur_matching.enabled" class="pt-4 border-t border-gray-700">
                <label class="text-sm text-gray-400 flex justify-between mb-1">
                  <span>Blur Strength</span>
                  <span class="text-white font-mono">{{ effectsConfig.blur_matching.blur_strength.toFixed(2) }}</span>
                </label>
                <input type="range" v-model.number="effectsConfig.blur_matching.blur_strength" min="0" max="1" step="0.05" class="w-full accent-primary" />
              </div>
            </div>

            <!-- Lighting -->
            <div :class="['card p-5 transition-all', effectsConfig.lighting.enabled ? 'border-primary/50 bg-primary/5' : '']">
              <div class="flex items-start justify-between mb-4">
                <div class="flex items-center gap-3">
                  <div :class="['rounded-lg p-2', effectsConfig.lighting.enabled ? 'bg-primary/20 text-primary' : 'bg-gray-700 text-gray-400']">
                    <Lightbulb class="h-5 w-5" />
                  </div>
                  <div>
                    <h3 class="font-semibold text-white">Lighting Effects</h3>
                    <p class="text-sm text-gray-400">Apply realistic lighting to objects</p>
                  </div>
                </div>
                <Switch
                  v-model="effectsConfig.lighting.enabled"
                  :class="[effectsConfig.lighting.enabled ? 'bg-primary' : 'bg-gray-600', 'relative inline-flex h-6 w-11 items-center rounded-full transition-colors']"
                >
                  <span :class="[effectsConfig.lighting.enabled ? 'translate-x-6' : 'translate-x-1', 'inline-block h-4 w-4 transform rounded-full bg-white transition-transform']" />
                </Switch>
              </div>
              <div v-if="effectsConfig.lighting.enabled" class="space-y-4 pt-4 border-t border-gray-700">
                <BaseSelect
                  v-model="effectsConfig.lighting.lighting_type"
                  label="Lighting Type"
                  :options="lightingTypeOptions"
                />
                <div>
                  <label class="text-sm text-gray-400 flex justify-between mb-1">
                    <span>Lighting Intensity</span>
                    <span class="text-white font-mono">{{ effectsConfig.lighting.lighting_intensity.toFixed(2) }}</span>
                  </label>
                  <input type="range" v-model.number="effectsConfig.lighting.lighting_intensity" min="0" max="1" step="0.05" class="w-full accent-primary" />
                </div>
              </div>
            </div>

            <!-- Underwater -->
            <div :class="['card p-5 transition-all', effectsConfig.underwater.enabled ? 'border-primary/50 bg-primary/5' : '']">
              <div class="flex items-start justify-between mb-4">
                <div class="flex items-center gap-3">
                  <div :class="['rounded-lg p-2', effectsConfig.underwater.enabled ? 'bg-primary/20 text-primary' : 'bg-gray-700 text-gray-400']">
                    <Waves class="h-5 w-5" />
                  </div>
                  <div>
                    <h3 class="font-semibold text-white">Underwater Tint</h3>
                    <p class="text-sm text-gray-400">Apply underwater color effects</p>
                  </div>
                </div>
                <Switch
                  v-model="effectsConfig.underwater.enabled"
                  :class="[effectsConfig.underwater.enabled ? 'bg-primary' : 'bg-gray-600', 'relative inline-flex h-6 w-11 items-center rounded-full transition-colors']"
                >
                  <span :class="[effectsConfig.underwater.enabled ? 'translate-x-6' : 'translate-x-1', 'inline-block h-4 w-4 transform rounded-full bg-white transition-transform']" />
                </Switch>
              </div>
              <div v-if="effectsConfig.underwater.enabled" class="space-y-4 pt-4 border-t border-gray-700">
                <div>
                  <label class="text-sm text-gray-400 flex justify-between mb-1">
                    <span>Underwater Intensity</span>
                    <span class="text-white font-mono">{{ effectsConfig.underwater.underwater_intensity.toFixed(2) }}</span>
                  </label>
                  <input type="range" v-model.number="effectsConfig.underwater.underwater_intensity" min="0" max="1" step="0.05" class="w-full accent-primary" />
                </div>
                <BaseSelect
                  v-model="effectsConfig.underwater.water_clarity"
                  label="Water Clarity"
                  :options="waterClarityOptions"
                />
                <div>
                  <label class="text-sm text-gray-400 mb-2 block">Water Color (RGB)</label>
                  <div class="flex gap-2">
                    <div class="flex-1">
                      <label class="text-xs text-gray-500">R</label>
                      <input type="number" v-model.number="effectsConfig.underwater.water_color[0]" min="0" max="255" class="input w-full text-sm" />
                    </div>
                    <div class="flex-1">
                      <label class="text-xs text-gray-500">G</label>
                      <input type="number" v-model.number="effectsConfig.underwater.water_color[1]" min="0" max="255" class="input w-full text-sm" />
                    </div>
                    <div class="flex-1">
                      <label class="text-xs text-gray-500">B</label>
                      <input type="number" v-model.number="effectsConfig.underwater.water_color[2]" min="0" max="255" class="input w-full text-sm" />
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <!-- Shadows -->
            <div :class="['card p-5 transition-all', effectsConfig.shadows.enabled ? 'border-primary/50 bg-primary/5' : '']">
              <div class="flex items-start justify-between mb-4">
                <div class="flex items-center gap-3">
                  <div :class="['rounded-lg p-2', effectsConfig.shadows.enabled ? 'bg-primary/20 text-primary' : 'bg-gray-700 text-gray-400']">
                    <Layers class="h-5 w-5" />
                  </div>
                  <div>
                    <h3 class="font-semibold text-white">Drop Shadows</h3>
                    <p class="text-sm text-gray-400">Add realistic shadows to objects</p>
                  </div>
                </div>
                <Switch
                  v-model="effectsConfig.shadows.enabled"
                  :class="[effectsConfig.shadows.enabled ? 'bg-primary' : 'bg-gray-600', 'relative inline-flex h-6 w-11 items-center rounded-full transition-colors']"
                >
                  <span :class="[effectsConfig.shadows.enabled ? 'translate-x-6' : 'translate-x-1', 'inline-block h-4 w-4 transform rounded-full bg-white transition-transform']" />
                </Switch>
              </div>
              <div v-if="effectsConfig.shadows.enabled" class="space-y-4 pt-4 border-t border-gray-700">
                <div>
                  <label class="text-sm text-gray-400 flex justify-between mb-1">
                    <span>Shadow Opacity</span>
                    <span class="text-white font-mono">{{ effectsConfig.shadows.shadow_opacity.toFixed(2) }}</span>
                  </label>
                  <input type="range" v-model.number="effectsConfig.shadows.shadow_opacity" min="0" max="1" step="0.05" class="w-full accent-primary" />
                </div>
                <div>
                  <label class="text-sm text-gray-400 flex justify-between mb-1">
                    <span>Shadow Blur</span>
                    <span class="text-white font-mono">{{ effectsConfig.shadows.shadow_blur }}px</span>
                  </label>
                  <input type="range" v-model.number="effectsConfig.shadows.shadow_blur" min="0" max="100" step="5" class="w-full accent-primary" />
                </div>
              </div>
            </div>

            <!-- Caustics -->
            <div :class="['card p-5 transition-all', effectsConfig.caustics.enabled ? 'border-primary/50 bg-primary/5' : '']">
              <div class="flex items-start justify-between mb-4">
                <div class="flex items-center gap-3">
                  <div :class="['rounded-lg p-2', effectsConfig.caustics.enabled ? 'bg-primary/20 text-primary' : 'bg-gray-700 text-gray-400']">
                    <Droplets class="h-5 w-5" />
                  </div>
                  <div>
                    <h3 class="font-semibold text-white">Caustics</h3>
                    <p class="text-sm text-gray-400">Light refraction patterns</p>
                  </div>
                </div>
                <Switch
                  v-model="effectsConfig.caustics.enabled"
                  :class="[effectsConfig.caustics.enabled ? 'bg-primary' : 'bg-gray-600', 'relative inline-flex h-6 w-11 items-center rounded-full transition-colors']"
                >
                  <span :class="[effectsConfig.caustics.enabled ? 'translate-x-6' : 'translate-x-1', 'inline-block h-4 w-4 transform rounded-full bg-white transition-transform']" />
                </Switch>
              </div>
              <div v-if="effectsConfig.caustics.enabled" class="space-y-4 pt-4 border-t border-gray-700">
                <div>
                  <label class="text-sm text-gray-400 flex justify-between mb-1">
                    <span>Caustics Intensity</span>
                    <span class="text-white font-mono">{{ effectsConfig.caustics.caustics_intensity.toFixed(2) }}</span>
                  </label>
                  <input type="range" v-model.number="effectsConfig.caustics.caustics_intensity" min="0" max="1" step="0.05" class="w-full accent-primary" />
                </div>
                <div class="flex items-center justify-between">
                  <span class="text-sm text-gray-400">Deterministic Pattern</span>
                  <Switch
                    v-model="effectsConfig.caustics.caustics_deterministic"
                    :class="[effectsConfig.caustics.caustics_deterministic ? 'bg-primary' : 'bg-gray-600', 'relative inline-flex h-5 w-9 items-center rounded-full transition-colors']"
                  >
                    <span :class="[effectsConfig.caustics.caustics_deterministic ? 'translate-x-5' : 'translate-x-1', 'inline-block h-3 w-3 transform rounded-full bg-white transition-transform']" />
                  </Switch>
                </div>
              </div>
            </div>

            <!-- Motion Blur -->
            <div :class="['card p-5 transition-all', effectsConfig.motion_blur.enabled ? 'border-primary/50 bg-primary/5' : '']">
              <div class="flex items-start justify-between mb-4">
                <div class="flex items-center gap-3">
                  <div :class="['rounded-lg p-2', effectsConfig.motion_blur.enabled ? 'bg-primary/20 text-primary' : 'bg-gray-700 text-gray-400']">
                    <Wind class="h-5 w-5" />
                  </div>
                  <div>
                    <h3 class="font-semibold text-white">Motion Blur</h3>
                    <p class="text-sm text-gray-400">Simulate movement blur</p>
                  </div>
                </div>
                <Switch
                  v-model="effectsConfig.motion_blur.enabled"
                  :class="[effectsConfig.motion_blur.enabled ? 'bg-primary' : 'bg-gray-600', 'relative inline-flex h-6 w-11 items-center rounded-full transition-colors']"
                >
                  <span :class="[effectsConfig.motion_blur.enabled ? 'translate-x-6' : 'translate-x-1', 'inline-block h-4 w-4 transform rounded-full bg-white transition-transform']" />
                </Switch>
              </div>
              <div v-if="effectsConfig.motion_blur.enabled" class="space-y-4 pt-4 border-t border-gray-700">
                <div>
                  <label class="text-sm text-gray-400 flex justify-between mb-1">
                    <span>Probability</span>
                    <span class="text-white font-mono">{{ (effectsConfig.motion_blur.motion_blur_probability * 100).toFixed(0) }}%</span>
                  </label>
                  <input type="range" v-model.number="effectsConfig.motion_blur.motion_blur_probability" min="0" max="1" step="0.05" class="w-full accent-primary" />
                </div>
                <div>
                  <label class="text-sm text-gray-400 flex justify-between mb-1">
                    <span>Kernel Size</span>
                    <span class="text-white font-mono">{{ effectsConfig.motion_blur.motion_blur_kernel }}px</span>
                  </label>
                  <input type="range" v-model.number="effectsConfig.motion_blur.motion_blur_kernel" min="3" max="31" step="2" class="w-full accent-primary" />
                </div>
              </div>
            </div>

            <!-- Perspective -->
            <div :class="['card p-5 transition-all', effectsConfig.perspective.enabled ? 'border-primary/50 bg-primary/5' : '']">
              <div class="flex items-start justify-between mb-4">
                <div class="flex items-center gap-3">
                  <div :class="['rounded-lg p-2', effectsConfig.perspective.enabled ? 'bg-primary/20 text-primary' : 'bg-gray-700 text-gray-400']">
                    <Eye class="h-5 w-5" />
                  </div>
                  <div>
                    <h3 class="font-semibold text-white">Perspective Warp</h3>
                    <p class="text-sm text-gray-400">Apply perspective distortion</p>
                  </div>
                </div>
                <Switch
                  v-model="effectsConfig.perspective.enabled"
                  :class="[effectsConfig.perspective.enabled ? 'bg-primary' : 'bg-gray-600', 'relative inline-flex h-6 w-11 items-center rounded-full transition-colors']"
                >
                  <span :class="[effectsConfig.perspective.enabled ? 'translate-x-6' : 'translate-x-1', 'inline-block h-4 w-4 transform rounded-full bg-white transition-transform']" />
                </Switch>
              </div>
              <div v-if="effectsConfig.perspective.enabled" class="pt-4 border-t border-gray-700">
                <label class="text-sm text-gray-400 flex justify-between mb-1">
                  <span>Magnitude</span>
                  <span class="text-white font-mono">{{ effectsConfig.perspective.perspective_magnitude.toFixed(2) }}</span>
                </label>
                <input type="range" v-model.number="effectsConfig.perspective.perspective_magnitude" min="0" max="0.3" step="0.01" class="w-full accent-primary" />
              </div>
            </div>
          </div>
        </TabPanel>

        <!-- Blending & Edges Tab -->
        <TabPanel class="space-y-6">
          <div class="grid gap-6 md:grid-cols-2">
            <!-- Blending Method -->
            <div class="card p-5">
              <div class="flex items-center gap-3 mb-4">
                <div class="rounded-lg p-2 bg-primary/20 text-primary">
                  <Blend class="h-5 w-5" />
                </div>
                <div>
                  <h3 class="font-semibold text-white">Blending Method</h3>
                  <p class="text-sm text-gray-400">How objects blend with background</p>
                </div>
              </div>
              <div class="space-y-4 pt-4 border-t border-gray-700">
                <BaseSelect
                  v-model="effectsConfig.blending.blend_method"
                  label="Method"
                  :options="blendMethodOptions"
                />
                <div class="flex items-center justify-between">
                  <span class="text-sm text-gray-400">Use Binary Alpha</span>
                  <Switch
                    v-model="effectsConfig.blending.use_binary_alpha"
                    :class="[effectsConfig.blending.use_binary_alpha ? 'bg-primary' : 'bg-gray-600', 'relative inline-flex h-5 w-9 items-center rounded-full transition-colors']"
                  >
                    <span :class="[effectsConfig.blending.use_binary_alpha ? 'translate-x-5' : 'translate-x-1', 'inline-block h-3 w-3 transform rounded-full bg-white transition-transform']" />
                  </Switch>
                </div>
                <div>
                  <label class="text-sm text-gray-400 flex justify-between mb-1">
                    <span>Alpha Feather Radius</span>
                    <span class="text-white font-mono">{{ effectsConfig.blending.alpha_feather_radius }}px</span>
                  </label>
                  <input type="range" v-model.number="effectsConfig.blending.alpha_feather_radius" min="0" max="10" step="1" class="w-full accent-primary" />
                </div>
              </div>
            </div>

            <!-- Edge Smoothing -->
            <div :class="['card p-5 transition-all', effectsConfig.edge_smoothing.enabled ? 'border-primary/50 bg-primary/5' : '']">
              <div class="flex items-start justify-between mb-4">
                <div class="flex items-center gap-3">
                  <div :class="['rounded-lg p-2', effectsConfig.edge_smoothing.enabled ? 'bg-primary/20 text-primary' : 'bg-gray-700 text-gray-400']">
                    <Maximize2 class="h-5 w-5" />
                  </div>
                  <div>
                    <h3 class="font-semibold text-white">Edge Smoothing</h3>
                    <p class="text-sm text-gray-400">Smooth object edges</p>
                  </div>
                </div>
                <Switch
                  v-model="effectsConfig.edge_smoothing.enabled"
                  :class="[effectsConfig.edge_smoothing.enabled ? 'bg-primary' : 'bg-gray-600', 'relative inline-flex h-6 w-11 items-center rounded-full transition-colors']"
                >
                  <span :class="[effectsConfig.edge_smoothing.enabled ? 'translate-x-6' : 'translate-x-1', 'inline-block h-4 w-4 transform rounded-full bg-white transition-transform']" />
                </Switch>
              </div>
              <div v-if="effectsConfig.edge_smoothing.enabled" class="pt-4 border-t border-gray-700">
                <label class="text-sm text-gray-400 flex justify-between mb-1">
                  <span>Edge Feather</span>
                  <span class="text-white font-mono">{{ effectsConfig.edge_smoothing.edge_feather }}px</span>
                </label>
                <input type="range" v-model.number="effectsConfig.edge_smoothing.edge_feather" min="0" max="20" step="1" class="w-full accent-primary" />
              </div>
            </div>

            <!-- Max Blur Budget -->
            <div class="card p-5">
              <div class="flex items-center gap-3 mb-4">
                <div class="rounded-lg p-2 bg-primary/20 text-primary">
                  <Focus class="h-5 w-5" />
                </div>
                <div>
                  <h3 class="font-semibold text-white">Max Blur Budget</h3>
                  <p class="text-sm text-gray-400">Maximum cumulative blur applied</p>
                </div>
              </div>
              <div class="pt-4 border-t border-gray-700">
                <label class="text-sm text-gray-400 flex justify-between mb-1">
                  <span>Blur Budget</span>
                  <span class="text-white font-mono">{{ effectsConfig.max_blur_budget.toFixed(1) }}</span>
                </label>
                <input type="range" v-model.number="effectsConfig.max_blur_budget" min="0" max="50" step="0.5" class="w-full accent-primary" />
              </div>
            </div>
          </div>
        </TabPanel>

        <!-- Object Placement Tab -->
        <TabPanel class="space-y-6">
          <div class="grid gap-6 md:grid-cols-2">
            <!-- Objects per Image -->
            <div class="card p-5">
              <div class="flex items-center gap-3 mb-4">
                <div class="rounded-lg p-2 bg-primary/20 text-primary">
                  <Box class="h-5 w-5" />
                </div>
                <div>
                  <h3 class="font-semibold text-white">Objects Per Image</h3>
                  <p class="text-sm text-gray-400">Number of objects to place</p>
                </div>
              </div>
              <div class="space-y-4 pt-4 border-t border-gray-700">
                <div>
                  <label class="text-sm text-gray-400 flex justify-between mb-1">
                    <span>Minimum</span>
                    <span class="text-white font-mono">{{ placementConfig.min_objects_per_image }}</span>
                  </label>
                  <input type="range" v-model.number="placementConfig.min_objects_per_image" min="1" max="10" step="1" class="w-full accent-primary" />
                </div>
                <div>
                  <label class="text-sm text-gray-400 flex justify-between mb-1">
                    <span>Maximum</span>
                    <span class="text-white font-mono">{{ placementConfig.max_objects_per_image }}</span>
                  </label>
                  <input type="range" v-model.number="placementConfig.max_objects_per_image" min="1" max="20" step="1" class="w-full accent-primary" />
                </div>
              </div>
            </div>

            <!-- Size Constraints -->
            <div class="card p-5">
              <div class="flex items-center gap-3 mb-4">
                <div class="rounded-lg p-2 bg-primary/20 text-primary">
                  <Maximize2 class="h-5 w-5" />
                </div>
                <div>
                  <h3 class="font-semibold text-white">Size Constraints</h3>
                  <p class="text-sm text-gray-400">Object size limits</p>
                </div>
              </div>
              <div class="space-y-4 pt-4 border-t border-gray-700">
                <div>
                  <label class="text-sm text-gray-400 flex justify-between mb-1">
                    <span>Min Size Ratio</span>
                    <span class="text-white font-mono">{{ (placementConfig.min_object_size_ratio * 100).toFixed(1) }}%</span>
                  </label>
                  <input type="range" v-model.number="placementConfig.min_object_size_ratio" min="0.01" max="0.2" step="0.005" class="w-full accent-primary" />
                </div>
                <div>
                  <label class="text-sm text-gray-400 flex justify-between mb-1">
                    <span>Absolute Min Size</span>
                    <span class="text-white font-mono">{{ placementConfig.absolute_min_size }}px</span>
                  </label>
                  <input type="range" v-model.number="placementConfig.absolute_min_size" min="5" max="50" step="5" class="w-full accent-primary" />
                </div>
              </div>
            </div>

            <!-- Area Ratios -->
            <div class="card p-5">
              <div class="flex items-center gap-3 mb-4">
                <div class="rounded-lg p-2 bg-primary/20 text-primary">
                  <Layers class="h-5 w-5" />
                </div>
                <div>
                  <h3 class="font-semibold text-white">Area Ratios</h3>
                  <p class="text-sm text-gray-400">Object area relative to image</p>
                </div>
              </div>
              <div class="space-y-4 pt-4 border-t border-gray-700">
                <div>
                  <label class="text-sm text-gray-400 flex justify-between mb-1">
                    <span>Min Area Ratio</span>
                    <span class="text-white font-mono">{{ (placementConfig.min_area_ratio * 100).toFixed(1) }}%</span>
                  </label>
                  <input type="range" v-model.number="placementConfig.min_area_ratio" min="0.001" max="0.1" step="0.005" class="w-full accent-primary" />
                </div>
                <div>
                  <label class="text-sm text-gray-400 flex justify-between mb-1">
                    <span>Max Area Ratio</span>
                    <span class="text-white font-mono">{{ (placementConfig.max_area_ratio * 100).toFixed(1) }}%</span>
                  </label>
                  <input type="range" v-model.number="placementConfig.max_area_ratio" min="0.1" max="0.8" step="0.05" class="w-full accent-primary" />
                </div>
              </div>
            </div>

            <!-- Overlap Threshold -->
            <div class="card p-5">
              <div class="flex items-center gap-3 mb-4">
                <div class="rounded-lg p-2 bg-primary/20 text-primary">
                  <Box class="h-5 w-5" />
                </div>
                <div>
                  <h3 class="font-semibold text-white">Overlap Threshold</h3>
                  <p class="text-sm text-gray-400">Maximum allowed object overlap</p>
                </div>
              </div>
              <div class="pt-4 border-t border-gray-700">
                <label class="text-sm text-gray-400 flex justify-between mb-1">
                  <span>Threshold</span>
                  <span class="text-white font-mono">{{ (placementConfig.overlap_threshold * 100).toFixed(0) }}%</span>
                </label>
                <input type="range" v-model.number="placementConfig.overlap_threshold" min="0" max="0.5" step="0.05" class="w-full accent-primary" />
              </div>
            </div>
          </div>
        </TabPanel>
      </TabPanels>
    </TabGroup>

    <!-- Auto-Tune Effects -->
    <div class="card overflow-hidden">
      <div class="flex items-center justify-between p-6">
        <div class="flex items-center gap-3">
          <Zap class="h-5 w-5 text-primary" />
          <div>
            <h3 class="text-lg font-semibold text-white">{{ t('workflow.configure.autoTune.title') }}</h3>
            <p class="text-sm text-gray-400">{{ t('workflow.configure.autoTune.description') }}</p>
          </div>
        </div>
        <Switch
          v-model="workflowStore.autoTuneEnabled"
          :class="[workflowStore.autoTuneEnabled ? 'bg-primary' : 'bg-gray-600', 'relative inline-flex h-6 w-11 items-center rounded-full transition-colors']"
        >
          <span :class="[workflowStore.autoTuneEnabled ? 'translate-x-6' : 'translate-x-1', 'inline-block h-4 w-4 transform rounded-full bg-white transition-transform']" />
        </Switch>
      </div>

      <div v-if="workflowStore.autoTuneEnabled" class="px-6 pb-6 space-y-4 border-t border-gray-700 pt-4">
        <!-- Reference Set -->
        <div>
          <label class="block text-sm font-medium text-gray-300 mb-1">{{ t('workflow.configure.autoTune.referenceSet') }}</label>
          <select
            v-model="workflowStore.autoTuneReferenceSetId"
            class="input w-full"
          >
            <option :value="null" disabled>{{ t('workflow.configure.autoTune.selectReferenceSet') }}</option>
            <option v-for="rs in domainGapStore.referenceSets" :key="rs.set_id" :value="rs.set_id">
              {{ rs.name }} ({{ rs.image_count }} {{ t('workflow.configure.autoTune.images') }})
            </option>
          </select>
          <p class="text-xs text-gray-500 mt-1">{{ t('workflow.configure.autoTune.referenceSetHint') }}</p>
        </div>

        <div class="grid gap-4 md:grid-cols-3">
          <!-- Target Score -->
          <div>
            <label class="text-sm text-gray-400 flex justify-between mb-1">
              <span>{{ t('workflow.configure.autoTune.targetScore') }}</span>
              <span class="text-white font-mono">{{ workflowStore.autoTuneTargetScore }}</span>
            </label>
            <input type="range" v-model.number="workflowStore.autoTuneTargetScore" min="5" max="50" step="5" class="w-full accent-primary" />
          </div>

          <!-- Max Iterations -->
          <div>
            <label class="text-sm text-gray-400 flex justify-between mb-1">
              <span>{{ t('workflow.configure.autoTune.maxIterations') }}</span>
              <span class="text-white font-mono">{{ workflowStore.autoTuneMaxIterations }}</span>
            </label>
            <input type="range" v-model.number="workflowStore.autoTuneMaxIterations" min="1" max="10" step="1" class="w-full accent-primary" />
          </div>

          <!-- Probe Size -->
          <div>
            <label class="text-sm text-gray-400 flex justify-between mb-1">
              <span>{{ t('workflow.configure.autoTune.probeSize') }}</span>
              <span class="text-white font-mono">{{ workflowStore.autoTuneProbeSize }}</span>
            </label>
            <input type="range" v-model.number="workflowStore.autoTuneProbeSize" min="10" max="100" step="10" class="w-full accent-primary" />
          </div>
        </div>
      </div>
    </div>

    <!-- Navigation Buttons -->
    <div class="flex justify-between pt-4">
      <BaseButton variant="outline" @click="goBack">
        <ArrowLeft class="h-5 w-5" />
        Back to Analysis
      </BaseButton>
      <BaseButton @click="saveConfig">
        Save & Continue
        <ArrowRight class="h-5 w-5" />
      </BaseButton>
    </div>
  </div>
</template>
