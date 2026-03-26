<script setup lang="ts">
import { ref, computed, reactive, onMounted, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useUiStore } from '@/stores/ui'
import { useDomainGapStore } from '@/stores/domainGap'
import { useDomainStore } from '@/stores/domain'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseInput from '@/components/ui/BaseInput.vue'
import BaseSelect from '@/components/ui/BaseSelect.vue'
import AlertBox from '@/components/common/AlertBox.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import MetricCard from '@/components/common/MetricCard.vue'
import DirectoryBrowser from '@/components/common/DirectoryBrowser.vue'
import {
  ScanSearch,
  Upload,
  Trash2,
  ImageIcon,
  BarChart3,
  Lightbulb,
  AlertTriangle,
  CheckCircle,
  Shuffle,
  ArrowRight,
  Info,
  Eye,
  FolderOpen,
  Palette,
  Zap,
  Sparkles,
} from 'lucide-vue-next'
import type { GapAnalysis, ReferenceSet, ParameterSuggestion, UploadProgress } from '@/stores/domainGap'

const { t } = useI18n()
const uiStore = useUiStore()
const store = useDomainGapStore()
const domainStore = useDomainStore()

// ============================================
// Tab management
// ============================================

const activeTab = ref<'references' | 'validation' | 'randomization' | 'styleTransfer' | 'diffusion'>('references')

const tabs = computed(() => [
  { id: 'references' as const, label: t('tools.domainGap.tabs.references'), icon: ImageIcon },
  { id: 'validation' as const, label: t('tools.domainGap.tabs.validation'), icon: BarChart3 },
  { id: 'randomization' as const, label: t('tools.domainGap.tabs.randomization'), icon: Shuffle },
  { id: 'styleTransfer' as const, label: t('tools.domainGap.tabs.styleTransfer'), icon: Palette },
  { id: 'diffusion' as const, label: t('tools.domainGap.tabs.diffusion'), icon: Sparkles },
])

// ============================================
// References tab state
// ============================================

const uploadName = ref('')
const uploadDescription = ref('')
const uploadDomainId = ref('')
const uploadFiles = ref<File[]>([])
const fileInputRef = ref<HTMLInputElement | null>(null)
const showUploadForm = ref(false)
const uploadMode = ref<'files' | 'directory'>('files')
const uploadDirectoryPath = ref('')

function onFileSelect(event: Event) {
  const target = event.target as HTMLInputElement
  if (target.files) {
    uploadFiles.value = Array.from(target.files)
  }
}

async function handleUpload() {
  if (!uploadName.value || !uploadDomainId.value) return

  let result: any = null

  if (uploadMode.value === 'directory') {
    if (!uploadDirectoryPath.value) return
    result = await store.createReferenceFromDir(
      uploadName.value,
      uploadDescription.value,
      uploadDomainId.value,
      uploadDirectoryPath.value,
    )
  } else {
    if (uploadFiles.value.length === 0) return
    result = await store.uploadReferenceSet(
      uploadName.value,
      uploadDescription.value,
      uploadDomainId.value,
      uploadFiles.value,
    )
  }

  if (result) {
    uiStore.showSuccess(
      t('tools.domainGap.notifications.uploadSuccess'),
      t('tools.domainGap.notifications.uploadSuccessMsg', {
        count: result.image_count ?? result.stats?.image_count ?? 0,
        name: result.name ?? uploadName.value,
      }),
    )
    uploadName.value = ''
    uploadDescription.value = ''
    uploadFiles.value = []
    uploadDirectoryPath.value = ''
    showUploadForm.value = false
  } else {
    uiStore.showError(
      t('tools.domainGap.notifications.uploadFailed'),
      store.error || undefined,
    )
  }
}

function getUploadDisabled(): boolean {
  if (!uploadName.value || !uploadDomainId.value || store.isUploading) return true
  if (uploadMode.value === 'files') return uploadFiles.value.length === 0
  return !uploadDirectoryPath.value
}

function getProgressPercent(progress: UploadProgress): number {
  if (progress.phase === 'creating') return 2
  if (progress.phase === 'finalizing') return 95
  // uploading: weight by files uploaded + current batch progress
  const batchWeight = 90 / progress.totalBatches
  const completedBatches = (progress.currentBatch - 1) * batchWeight
  const currentBatchPct = (progress.batchProgress / 100) * batchWeight
  return Math.round(5 + completedBatches + currentBatchPct)
}

async function handleDeleteSet(setId: string) {
  if (!confirm(t('tools.domainGap.references.deleteConfirm'))) return

  const success = await store.removeReferenceSet(setId)
  if (success) {
    uiStore.showSuccess(t('tools.domainGap.notifications.deleteSuccess'))
  } else {
    uiStore.showError(t('tools.domainGap.notifications.deleteFailed'))
  }
}

// ============================================
// Validation tab state
// ============================================

const syntheticDir = ref('')
const validationReferenceSetId = ref('')
const maxImages = ref(50)

async function handleAnalyze() {
  if (!syntheticDir.value || !validationReferenceSetId.value) return

  const result = await store.runAnalysis(
    syntheticDir.value,
    validationReferenceSetId.value,
    maxImages.value,
  )

  if (result) {
    uiStore.showSuccess(
      t('tools.domainGap.notifications.analysisComplete'),
      t('tools.domainGap.notifications.analysisCompleteMsg', {
        score: result.metrics.overall_gap_score.toFixed(1),
        level: result.metrics.gap_level,
      }),
    )
  } else {
    uiStore.showError(
      t('tools.domainGap.notifications.analysisFailed'),
      store.error || undefined,
    )
  }
}

function getGapScoreVariant(score: number): 'success' | 'warning' | 'error' {
  if (score < 30) return 'success'
  if (score < 60) return 'warning'
  return 'error'
}

function getGapLevelLabel(level: string): string {
  return t(`tools.domainGap.validation.results.levels.${level}`, level)
}

function getSeverityColor(severity: string): string {
  return severity === 'high' ? 'text-red-400' : 'text-yellow-400'
}

function getImpactColor(impact: string): string {
  if (impact === 'high') return 'text-green-400'
  if (impact === 'medium') return 'text-yellow-400'
  return 'text-gray-400'
}

// ============================================
// Randomization tab state
// ============================================

const randImagesDir = ref('')
const randOutputDir = ref('')
const randConfig = ref({
  num_variants: 3,
  intensity: 0.5,
  color_jitter: 0.3,
  noise_intensity: 0.02,
  histogram_match_strength: 0.5,
  preserve_annotations: true,
  reference_set_id: null as string | null,
})

async function handleRandomize() {
  if (!randImagesDir.value || !randOutputDir.value) return

  const result = await store.applyRandomizationBatch(
    randImagesDir.value,
    randOutputDir.value,
    randConfig.value,
  )

  if (result) {
    uiStore.showSuccess(
      t('tools.domainGap.notifications.randomizationStarted'),
      t('tools.domainGap.notifications.randomizationStartedMsg', {
        id: result.job_id || 'started',
      }),
    )
  } else {
    uiStore.showError(
      t('tools.domainGap.notifications.randomizationFailed'),
      store.error || undefined,
    )
  }
}

// ============================================
// Style Transfer tab state
// ============================================

const stImagesDir = ref('')
const stOutputDir = ref('')
const stReferenceSetId = ref('')
const stConfig = ref({
  style_weight: 0.6,
  content_weight: 1.0,
  preserve_structure: 0.8,
  color_only: false,
  depth_guided: true,
})

async function handleStyleTransfer() {
  if (!stImagesDir.value || !stOutputDir.value || !stReferenceSetId.value) return

  const result = await store.applyStyleTransferBatch(
    stImagesDir.value,
    stOutputDir.value,
    {
      reference_set_id: stReferenceSetId.value,
      ...stConfig.value,
    },
  )

  if (result) {
    uiStore.showSuccess(
      t('tools.domainGap.notifications.styleTransferStarted'),
      t('tools.domainGap.notifications.styleTransferStartedMsg', {
        id: result.job_id || 'started',
      }),
    )
  } else {
    uiStore.showError(
      t('tools.domainGap.notifications.styleTransferFailed'),
      store.error || undefined,
    )
  }
}

// ============================================
// Diffusion Refinement tab state
// ============================================

const selectedDiffusionMethod = ref<'controlnet' | 'ip_adapter' | 'lora'>('controlnet')
const diffusionReferenceSetId = ref('')
const selectedLoraModelId = ref('')
const diffusionSourceDir = ref('')
const diffusionOutputDir = ref('')
const showLoraTraining = ref(false)
const loraTrainingModelId = ref('')
const loraTrainingSteps = ref(500)
const loraTrainingRank = ref(4)
const loraTrainingLR = ref(0.0001)

const diffusionConfig = reactive({
  strength: 0.4,
  controlnet_conditioning_scale: 0.8,
  ip_adapter_scale: 0.5,
  lora_weight: 0.7,
  guidance_scale: 7.5,
  num_inference_steps: 30,
  use_lcm: false,
  prompt: '',
  negative_prompt: 'blurry, distorted, artifacts, watermark',
  seed: null as number | null,
})

const diffusionMethods = computed(() => [
  {
    id: 'controlnet' as const,
    label: t('tools.domainGap.diffusion.method.controlnet'),
    description: t('tools.domainGap.diffusion.method.controlnetDesc'),
  },
  {
    id: 'ip_adapter' as const,
    label: t('tools.domainGap.diffusion.method.ipAdapter'),
    description: t('tools.domainGap.diffusion.method.ipAdapterDesc'),
  },
  {
    id: 'lora' as const,
    label: t('tools.domainGap.diffusion.method.lora'),
    description: t('tools.domainGap.diffusion.method.loraDesc'),
  },
])

async function handleDiffusionRefineBatch() {
  if (!diffusionReferenceSetId.value || !diffusionSourceDir.value) return

  const config = {
    method: selectedDiffusionMethod.value,
    reference_set_id: diffusionReferenceSetId.value,
    lora_model_id: selectedDiffusionMethod.value === 'lora' ? selectedLoraModelId.value : null,
    strength: diffusionConfig.strength,
    controlnet_conditioning_scale: diffusionConfig.controlnet_conditioning_scale,
    use_depth_conditioning: selectedDiffusionMethod.value === 'controlnet',
    ip_adapter_scale: diffusionConfig.ip_adapter_scale,
    lora_weight: diffusionConfig.lora_weight,
    guidance_scale: diffusionConfig.guidance_scale,
    num_inference_steps: diffusionConfig.num_inference_steps,
    use_lcm: diffusionConfig.use_lcm,
    seed: diffusionConfig.seed,
    validate_annotations: true,
    annotation_threshold: 0.65,
    prompt: diffusionConfig.prompt,
    negative_prompt: diffusionConfig.negative_prompt,
  }

  const jobId = await store.applyDiffusionRefineBatch(
    diffusionSourceDir.value,
    diffusionOutputDir.value || diffusionSourceDir.value.replace(/\/?$/, '_refined/'),
    config as any,
  )

  if (jobId) {
    await store.fetchJobs()
    uiStore.showSuccess(
      t('tools.domainGap.diffusion.notifications.batchStarted'),
      jobId,
    )
  } else {
    uiStore.showError(
      t('tools.domainGap.diffusion.notifications.refineFailed'),
      store.lastDiffusionError || undefined,
    )
  }
}

async function handleTrainLora() {
  if (!diffusionReferenceSetId.value || !loraTrainingModelId.value) return

  const jobId = await store.trainLora(diffusionReferenceSetId.value, loraTrainingModelId.value, {
    training_steps: loraTrainingSteps.value,
    learning_rate: loraTrainingLR.value,
    lora_rank: loraTrainingRank.value,
  })

  if (jobId) {
    showLoraTraining.value = false
    await store.fetchJobs()
    uiStore.showSuccess(t('tools.domainGap.diffusion.notifications.loraTrainStarted'), jobId)
  } else {
    uiStore.showError(
      t('tools.domainGap.diffusion.notifications.refineFailed'),
      store.lastDiffusionError || undefined,
    )
  }
}

async function handleDeleteLora(modelId: string) {
  if (!confirm(t('tools.domainGap.diffusion.lora.deleteConfirm'))) return
  await store.removeLoraModel(modelId)
}

// ============================================
// Optimization state
// ============================================

const optOutputDir = ref('')
const optTargetScore = ref(30)
const optMaxIterations = ref(5)
const optTechniques = ref<string[]>(['randomization'])
const isOptimizing = ref(false)

async function handleOptimize() {
  if (!syntheticDir.value || !validationReferenceSetId.value || !optOutputDir.value) return

  isOptimizing.value = true
  const result = await store.runOptimization(
    syntheticDir.value,
    validationReferenceSetId.value,
    optOutputDir.value,
    {
      targetGapScore: optTargetScore.value,
      maxIterations: optMaxIterations.value,
      techniques: optTechniques.value,
    },
  )

  isOptimizing.value = false
  if (result) {
    uiStore.showSuccess(
      t('tools.domainGap.notifications.optimizationStarted'),
      t('tools.domainGap.notifications.optimizationStartedMsg', {
        id: result.job_id || 'started',
      }),
    )
  } else {
    uiStore.showError(
      t('tools.domainGap.notifications.optimizationFailed'),
      store.error || undefined,
    )
  }
}

// ============================================
// Domain options for selects
// ============================================

const domainOptions = computed(() =>
  domainStore.domains.map(d => ({
    value: d.domain_id,
    label: d.name,
  })),
)

const referenceSetOptions = computed(() =>
  store.referenceSets.map(s => ({
    value: s.set_id,
    label: `${s.name} (${s.image_count} imgs)`,
  })),
)

const randReferenceSetId = computed({
  get: () => randConfig.reference_set_id ?? '',
  set: (val: string) => {
    randConfig.reference_set_id = val === '' ? null : val
  },
})

// ============================================
// Lifecycle
// ============================================

onMounted(async () => {
  await Promise.all([
    store.fetchReferenceSets(),
    domainStore.fetchDomains(),
    store.fetchLoraModels(),
  ])
})

onUnmounted(() => {
  store.reset()
})
</script>

<template>
  <div class="min-h-screen bg-background p-6 lg:p-8">
    <!-- Header -->
    <div class="mb-8">
      <div class="flex items-center gap-3 mb-2">
        <div class="flex h-10 w-10 items-center justify-center rounded-xl bg-purple-600/20">
          <ScanSearch class="h-6 w-6 text-purple-400" />
        </div>
        <div>
          <h1 class="text-2xl font-bold text-white">{{ t('tools.domainGap.title') }}</h1>
          <p class="text-sm text-gray-400">{{ t('tools.domainGap.subtitle') }}</p>
        </div>
      </div>
    </div>

    <!-- Error alert -->
    <AlertBox v-if="store.error" type="error" class="mb-6" dismissible @dismiss="store.error = null">
      {{ store.error }}
    </AlertBox>

    <!-- Tab navigation -->
    <div class="mb-6 flex gap-1 rounded-xl bg-background-secondary p-1">
      <button
        v-for="tab in tabs"
        :key="tab.id"
        @click="activeTab = tab.id"
        :class="[
          'flex items-center gap-2 rounded-lg px-4 py-2.5 text-sm font-medium transition-colors',
          activeTab === tab.id
            ? 'bg-primary text-white'
            : 'text-gray-400 hover:text-white hover:bg-gray-700/50',
        ]"
      >
        <component :is="tab.icon" class="h-4 w-4" />
        {{ tab.label }}
      </button>
    </div>

    <!-- ============================================ -->
    <!-- TAB: References -->
    <!-- ============================================ -->
    <div v-if="activeTab === 'references'">
      <!-- Upload section -->
      <div class="mb-6">
        <BaseButton
          v-if="!showUploadForm"
          @click="showUploadForm = true"
          variant="primary"
        >
          <Upload class="h-4 w-4 mr-2" />
          {{ t('tools.domainGap.references.uploadTitle') }}
        </BaseButton>

        <!-- Upload form -->
        <div v-if="showUploadForm" class="rounded-xl border border-gray-700/50 bg-background-card p-6">
          <h3 class="text-lg font-semibold text-white mb-4">
            {{ t('tools.domainGap.references.uploadTitle') }}
          </h3>

          <!-- Upload mode toggle -->
          <div class="flex gap-1 rounded-lg bg-background-secondary p-1 mb-4">
            <button
              @click="uploadMode = 'files'"
              :class="[
                'flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors flex-1 justify-center',
                uploadMode === 'files'
                  ? 'bg-primary text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700/50',
              ]"
            >
              <Upload class="h-4 w-4" />
              {{ t('tools.domainGap.references.uploadMode.files') }}
            </button>
            <button
              @click="uploadMode = 'directory'"
              :class="[
                'flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors flex-1 justify-center',
                uploadMode === 'directory'
                  ? 'bg-primary text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700/50',
              ]"
            >
              <FolderOpen class="h-4 w-4" />
              {{ t('tools.domainGap.references.uploadMode.directory') }}
            </button>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <BaseInput
              v-model="uploadName"
              :label="t('tools.domainGap.references.name')"
              :placeholder="t('tools.domainGap.references.namePlaceholder')"
            />
            <BaseSelect
              v-model="uploadDomainId"
              :label="t('tools.domainGap.references.domainId')"
              :options="domainOptions"
              placeholder="Select domain..."
            />
          </div>
          <BaseInput
            v-model="uploadDescription"
            :label="t('tools.domainGap.references.setDescription')"
            :placeholder="t('tools.domainGap.references.descriptionPlaceholder')"
            class="mb-4"
          />

          <!-- File upload mode -->
          <div v-if="uploadMode === 'files'" class="mb-4">
            <label class="block text-sm font-medium text-gray-300 mb-2">
              {{ t('tools.domainGap.references.selectImages') }}
            </label>
            <input
              ref="fileInputRef"
              type="file"
              accept="image/*"
              multiple
              @change="onFileSelect"
              class="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-medium file:bg-primary file:text-white hover:file:bg-primary/80"
            />
            <p v-if="uploadFiles.length > 0" class="mt-2 text-sm text-gray-400">
              {{ uploadFiles.length }} {{ t('tools.domainGap.references.filesSelected') }}
            </p>
          </div>

          <!-- Directory mode -->
          <div v-else class="mb-4">
            <DirectoryBrowser
              v-model="uploadDirectoryPath"
              :label="t('tools.domainGap.references.directoryLabel')"
              placeholder="/app/datasets/references"
              path-mode="input"
            />
            <p class="mt-2 text-xs text-gray-500">
              {{ t('tools.domainGap.references.directoryHint') }}
            </p>
          </div>

          <!-- Upload progress -->
          <div v-if="store.uploadProgress" class="mb-4 rounded-lg bg-background-tertiary p-4">
            <div class="flex items-center justify-between mb-2">
              <span class="text-sm font-medium text-gray-300">
                <template v-if="store.uploadProgress.phase === 'creating'">
                  {{ t('tools.domainGap.references.progress.creating') }}
                </template>
                <template v-else-if="store.uploadProgress.phase === 'uploading'">
                  {{ t('tools.domainGap.references.progress.uploading') }}
                  ({{ store.uploadProgress.currentBatch }}/{{ store.uploadProgress.totalBatches }})
                </template>
                <template v-else>
                  {{ t('tools.domainGap.references.progress.finalizing') }}
                </template>
              </span>
              <span class="text-sm text-gray-400">
                {{ store.uploadProgress.filesUploaded }}/{{ store.uploadProgress.totalFiles }}
                {{ t('tools.domainGap.references.progress.files') }}
              </span>
            </div>
            <div class="h-2.5 bg-gray-700 rounded-full overflow-hidden">
              <div
                class="h-full bg-primary transition-all duration-300"
                :style="{ width: `${getProgressPercent(store.uploadProgress)}%` }"
              />
            </div>
          </div>

          <div class="flex gap-3">
            <BaseButton
              @click="handleUpload"
              variant="primary"
              :disabled="getUploadDisabled()"
              :loading="store.isUploading"
            >
              {{ uploadMode === 'directory' ? t('tools.domainGap.references.createFromDir') : t('tools.domainGap.references.upload') }}
            </BaseButton>
            <BaseButton @click="showUploadForm = false" variant="ghost" :disabled="store.isUploading">
              {{ t('common.actions.cancel') }}
            </BaseButton>
          </div>
        </div>
      </div>

      <!-- Reference sets list -->
      <div v-if="store.isLoading && store.referenceSets.length === 0" class="flex justify-center py-12">
        <LoadingSpinner />
      </div>

      <EmptyState
        v-else-if="!store.hasReferenceSets"
        :icon="ImageIcon"
        :title="t('tools.domainGap.references.noSets')"
      />

      <div v-else class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <div
          v-for="refSet in store.referenceSets"
          :key="refSet.set_id"
          class="rounded-xl border border-gray-700/50 bg-background-card p-5 hover:border-gray-600/50 transition-colors"
        >
          <div class="flex items-start justify-between mb-3">
            <div>
              <h3 class="font-semibold text-white">{{ refSet.name }}</h3>
              <p v-if="refSet.description" class="text-sm text-gray-400 mt-1">{{ refSet.description }}</p>
            </div>
            <button
              @click="handleDeleteSet(refSet.set_id)"
              class="p-1.5 text-gray-500 hover:text-red-400 transition-colors rounded-lg hover:bg-red-900/20"
            >
              <Trash2 class="h-4 w-4" />
            </button>
          </div>

          <div class="flex flex-wrap gap-2 text-xs text-gray-400 mb-3">
            <span class="flex items-center gap-1 bg-gray-700/50 px-2 py-1 rounded">
              <ImageIcon class="h-3 w-3" />
              {{ t('tools.domainGap.references.imageCount', { count: refSet.image_count }) }}
            </span>
            <span class="bg-gray-700/50 px-2 py-1 rounded">
              {{ refSet.domain_id }}
            </span>
          </div>

          <!-- Stats preview -->
          <div v-if="refSet.stats" class="grid grid-cols-2 gap-2 text-xs">
            <div class="bg-gray-800/50 rounded p-2">
              <span class="text-gray-500">{{ t('tools.domainGap.references.stats.brightness') }}</span>
              <p class="text-white font-medium">{{ refSet.stats.avg_brightness.toFixed(1) }}</p>
            </div>
            <div class="bg-gray-800/50 rounded p-2">
              <span class="text-gray-500">{{ t('tools.domainGap.references.stats.edgeVariance') }}</span>
              <p class="text-white font-medium">{{ refSet.stats.avg_edge_variance.toFixed(1) }}</p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ============================================ -->
    <!-- TAB: Validation -->
    <!-- ============================================ -->
    <div v-if="activeTab === 'validation'">
      <!-- Config form -->
      <div class="rounded-xl border border-gray-700/50 bg-background-card p-6 mb-6">
        <h3 class="text-lg font-semibold text-white mb-4">
          {{ t('tools.domainGap.validation.title') }}
        </h3>
        <p class="text-sm text-gray-400 mb-4">
          {{ t('tools.domainGap.validation.description') }}
        </p>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <DirectoryBrowser
            v-model="syntheticDir"
            :label="t('tools.domainGap.validation.syntheticDir')"
            path-mode="both"
          />
          <BaseSelect
            v-model="validationReferenceSetId"
            :label="t('tools.domainGap.validation.referenceSet')"
            :options="referenceSetOptions"
            :placeholder="t('tools.domainGap.validation.selectReferenceSet')"
          />
        </div>
        <div class="mb-4">
          <BaseInput
            v-model.number="maxImages"
            :label="t('tools.domainGap.validation.maxImages')"
            type="number"
            :min="5"
            :max="500"
          />
        </div>
        <BaseButton
          @click="handleAnalyze"
          variant="primary"
          :disabled="!syntheticDir || !validationReferenceSetId || store.isAnalyzing"
          :loading="store.isAnalyzing"
        >
          <BarChart3 class="h-4 w-4 mr-2" />
          {{ store.isAnalyzing ? t('tools.domainGap.validation.computing') : t('tools.domainGap.validation.analyze') }}
        </BaseButton>

        <!-- Analysis progress bar -->
        <div v-if="store.analysisProgress" class="mt-4 rounded-lg bg-background-tertiary p-4">
          <div class="flex items-center justify-between mb-2">
            <span class="text-sm font-medium text-gray-300">
              {{ t(`tools.domainGap.validation.progress.phases.${store.analysisProgress.phase}`, store.analysisProgress.phase) }}
            </span>
            <span class="text-sm text-gray-400">
              {{ Math.round(store.analysisProgress.globalProgress) }}%
            </span>
          </div>
          <div class="h-2.5 bg-gray-700 rounded-full overflow-hidden">
            <div
              class="h-full bg-primary rounded-full transition-all duration-500"
              :style="{ width: `${store.analysisProgress.globalProgress}%` }"
            />
          </div>
        </div>
      </div>

      <!-- Analysis results -->
      <div v-if="store.latestAnalysis">
        <!-- Sample size / PCA warnings -->
        <AlertBox
          v-if="store.latestAnalysis.metrics.sample_size_warning"
          type="warning"
          class="mb-4"
        >
          {{ store.latestAnalysis.metrics.sample_size_warning }}
        </AlertBox>
        <AlertBox
          v-if="store.latestAnalysis.metrics.pca_applied"
          type="info"
          class="mb-4"
        >
          {{ t('tools.domainGap.validation.results.pcaApplied') }}
        </AlertBox>

        <!-- Gap score overview -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <MetricCard
            :title="t('tools.domainGap.validation.results.gapScore')"
            :value="`${store.latestAnalysis.metrics.overall_gap_score.toFixed(1)}/100`"
            :icon="BarChart3"
            :variant="getGapScoreVariant(store.latestAnalysis.metrics.overall_gap_score)"
          />
          <MetricCard
            :title="t('tools.domainGap.validation.results.gapLevel')"
            :value="getGapLevelLabel(store.latestAnalysis.metrics.gap_level)"
            :icon="Info"
          />
          <MetricCard
            v-if="store.latestAnalysis.metrics.radio_mmd_score !== null"
            :title="t('tools.domainGap.validation.results.radioMmdScore')"
            :value="store.latestAnalysis.metrics.radio_mmd_score.toFixed(4)"
            :icon="BarChart3"
          />
          <MetricCard
            v-if="store.latestAnalysis.metrics.fd_radio_score !== null"
            :title="t('tools.domainGap.validation.results.fdRadioScore')"
            :value="store.latestAnalysis.metrics.fd_radio_score.toFixed(2)"
            :icon="BarChart3"
          />
          <MetricCard
            v-if="store.latestAnalysis.metrics.fid_score !== null"
            :title="t('tools.domainGap.validation.results.fidScore')"
            :value="store.latestAnalysis.metrics.fid_score.toFixed(2)"
            :icon="BarChart3"
          />
          <MetricCard
            v-if="store.latestAnalysis.metrics.kid_score !== null"
            :title="t('tools.domainGap.validation.results.kidScore')"
            :value="store.latestAnalysis.metrics.kid_score.toFixed(4)"
            :icon="BarChart3"
          />
        </div>

        <!-- Extended metrics: CMMD and PRDC -->
        <div
          v-if="store.latestAnalysis.metrics.cmmd_score !== null || store.latestAnalysis.metrics.precision !== null"
          class="grid grid-cols-1 md:grid-cols-5 gap-4 mb-6"
        >
          <MetricCard
            v-if="store.latestAnalysis.metrics.cmmd_score !== null"
            :title="t('tools.domainGap.validation.results.cmmdScore')"
            :value="store.latestAnalysis.metrics.cmmd_score.toFixed(4)"
            :icon="BarChart3"
          />
          <MetricCard
            v-if="store.latestAnalysis.metrics.precision !== null"
            :title="t('tools.domainGap.validation.results.precision')"
            :value="store.latestAnalysis.metrics.precision.toFixed(3)"
            :icon="BarChart3"
            :variant="store.latestAnalysis.metrics.precision >= 0.7 ? 'success' : store.latestAnalysis.metrics.precision >= 0.4 ? 'warning' : 'error'"
          />
          <MetricCard
            v-if="store.latestAnalysis.metrics.recall !== null"
            :title="t('tools.domainGap.validation.results.recall')"
            :value="store.latestAnalysis.metrics.recall.toFixed(3)"
            :icon="BarChart3"
            :variant="store.latestAnalysis.metrics.recall >= 0.7 ? 'success' : store.latestAnalysis.metrics.recall >= 0.4 ? 'warning' : 'error'"
          />
          <MetricCard
            v-if="store.latestAnalysis.metrics.density !== null"
            :title="t('tools.domainGap.validation.results.density')"
            :value="store.latestAnalysis.metrics.density.toFixed(3)"
            :icon="BarChart3"
          />
          <MetricCard
            v-if="store.latestAnalysis.metrics.coverage !== null"
            :title="t('tools.domainGap.validation.results.coverage')"
            :value="store.latestAnalysis.metrics.coverage.toFixed(3)"
            :icon="BarChart3"
            :variant="store.latestAnalysis.metrics.coverage >= 0.7 ? 'success' : store.latestAnalysis.metrics.coverage >= 0.4 ? 'warning' : 'error'"
          />
          <MetricCard
            v-if="store.latestAnalysis.metrics.sharpness_ratio !== null"
            :title="t('tools.domainGap.validation.results.sharpnessRatio')"
            :value="store.latestAnalysis.metrics.sharpness_ratio.toFixed(3)"
            :icon="Eye"
            :variant="Math.abs(1 - store.latestAnalysis.metrics.sharpness_ratio) < 0.3 ? 'success' : Math.abs(1 - store.latestAnalysis.metrics.sharpness_ratio) < 0.6 ? 'warning' : 'error'"
          />
        </div>

        <!-- Issues -->
        <div class="rounded-xl border border-gray-700/50 bg-background-card p-6 mb-6">
          <h3 class="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <AlertTriangle class="h-5 w-5 text-yellow-400" />
            {{ t('tools.domainGap.validation.issues.title') }}
          </h3>

          <div v-if="store.latestAnalysis.issues.length === 0" class="text-gray-400 text-sm">
            <div class="flex items-center gap-2">
              <CheckCircle class="h-4 w-4 text-green-400" />
              {{ t('tools.domainGap.validation.issues.noIssues') }}
            </div>
          </div>

          <div v-else class="space-y-3">
            <div
              v-for="(issue, idx) in store.latestAnalysis.issues"
              :key="idx"
              class="flex items-start gap-3 bg-gray-800/50 rounded-lg p-3"
            >
              <AlertTriangle :class="['h-4 w-4 mt-0.5 flex-shrink-0', getSeverityColor(issue.severity)]" />
              <div class="flex-1">
                <div class="flex items-center gap-2 mb-1">
                  <span class="text-xs font-medium text-gray-300 bg-gray-700 px-2 py-0.5 rounded uppercase">
                    {{ issue.category }}
                  </span>
                  <span :class="['text-xs font-medium', getSeverityColor(issue.severity)]">
                    {{ issue.severity }}
                  </span>
                </div>
                <p class="text-sm text-gray-300">{{ issue.description }}</p>
                <p class="text-xs text-gray-500 mt-1">
                  {{ issue.metric_name }}: {{ issue.metric_value.toFixed(2) }} (ref: {{ issue.reference_value.toFixed(2) }})
                </p>
              </div>
            </div>
          </div>
        </div>

        <!-- Suggestions -->
        <div class="rounded-xl border border-gray-700/50 bg-background-card p-6">
          <h3 class="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Lightbulb class="h-5 w-5 text-yellow-400" />
            {{ t('tools.domainGap.validation.suggestions.title') }}
          </h3>

          <div v-if="store.latestAnalysis.suggestions.length === 0" class="text-gray-400 text-sm">
            {{ t('tools.domainGap.validation.suggestions.noSuggestions') }}
          </div>

          <div v-else class="space-y-3">
            <div
              v-for="(suggestion, idx) in store.latestAnalysis.suggestions"
              :key="idx"
              class="bg-gray-800/50 rounded-lg p-4"
            >
              <div class="flex items-start justify-between mb-2">
                <code class="text-sm text-purple-400 bg-purple-900/20 px-2 py-0.5 rounded">
                  {{ suggestion.parameter_path }}
                </code>
                <span :class="['text-xs font-medium', getImpactColor(suggestion.expected_impact)]">
                  {{ t('tools.domainGap.validation.suggestions.impact') }}: {{ suggestion.expected_impact }}
                </span>
              </div>
              <p class="text-sm text-gray-300 mb-3">{{ suggestion.reason }}</p>
              <div class="flex items-center gap-4 text-sm">
                <span class="text-gray-400">
                  {{ t('tools.domainGap.validation.suggestions.currentValue') }}:
                  <span class="text-white font-medium">{{ suggestion.current_value ?? 'N/A' }}</span>
                </span>
                <ArrowRight class="h-4 w-4 text-gray-500" />
                <span class="text-gray-400">
                  {{ t('tools.domainGap.validation.suggestions.suggestedValue') }}:
                  <span class="text-green-400 font-medium">{{ suggestion.suggested_value }}</span>
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Optimization panel (within validation tab results) -->
    <div v-if="activeTab === 'validation' && store.latestAnalysis" class="mt-6">
      <div class="rounded-xl border border-purple-700/30 bg-background-card p-6">
        <h3 class="text-lg font-semibold text-white mb-2 flex items-center gap-2">
          <Zap class="h-5 w-5 text-purple-400" />
          {{ t('tools.domainGap.optimization.title') }}
        </h3>
        <p class="text-sm text-gray-400 mb-4">
          {{ t('tools.domainGap.optimization.description') }}
        </p>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <DirectoryBrowser
            v-model="optOutputDir"
            :label="t('tools.domainGap.optimization.outputDir')"
            path-mode="output"
          />
          <div>
            <label class="block text-sm font-medium text-gray-300 mb-2">
              {{ t('tools.domainGap.optimization.targetScore') }}: {{ optTargetScore }}
            </label>
            <input
              v-model.number="optTargetScore"
              type="range"
              min="5"
              max="80"
              step="5"
              class="w-full accent-purple-500"
            />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-300 mb-2">
              {{ t('tools.domainGap.optimization.maxIterations') }}: {{ optMaxIterations }}
            </label>
            <input
              v-model.number="optMaxIterations"
              type="range"
              min="1"
              max="20"
              step="1"
              class="w-full accent-purple-500"
            />
          </div>
        </div>

        <div class="mb-4">
          <label class="block text-sm font-medium text-gray-300 mb-2">
            {{ t('tools.domainGap.optimization.techniques') }}
          </label>
          <div class="flex gap-4">
            <label class="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
              <input
                type="checkbox"
                value="randomization"
                v-model="optTechniques"
                class="h-4 w-4 accent-purple-500"
              />
              {{ t('tools.domainGap.tabs.randomization') }}
            </label>
            <label class="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
              <input
                type="checkbox"
                value="style_transfer"
                v-model="optTechniques"
                class="h-4 w-4 accent-purple-500"
              />
              {{ t('tools.domainGap.tabs.styleTransfer') }}
            </label>
          </div>
        </div>

        <BaseButton
          @click="handleOptimize"
          variant="primary"
          :disabled="!optOutputDir || isOptimizing || optTechniques.length === 0"
          :loading="isOptimizing"
          class="bg-purple-600 hover:bg-purple-700"
        >
          <Zap class="h-4 w-4 mr-2" />
          {{ t('tools.domainGap.optimization.run') }}
        </BaseButton>
      </div>
    </div>

    <!-- ============================================ -->
    <!-- TAB: Randomization -->
    <!-- ============================================ -->
    <div v-if="activeTab === 'randomization'">
      <div class="rounded-xl border border-gray-700/50 bg-background-card p-6">
        <h3 class="text-lg font-semibold text-white mb-2">
          {{ t('tools.domainGap.randomization.title') }}
        </h3>
        <p class="text-sm text-gray-400 mb-6">
          {{ t('tools.domainGap.randomization.description') }}
        </p>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <DirectoryBrowser
            v-model="randImagesDir"
            :label="t('tools.domainGap.randomization.imagesDir')"
            path-mode="both"
          />
          <DirectoryBrowser
            v-model="randOutputDir"
            :label="t('tools.domainGap.randomization.outputDir')"
            path-mode="output"
          />
        </div>

        <!-- Config sliders -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
          <div>
            <label class="block text-sm font-medium text-gray-300 mb-2">
              {{ t('tools.domainGap.randomization.config.numVariants') }}: {{ randConfig.num_variants }}
            </label>
            <input
              v-model.number="randConfig.num_variants"
              type="range"
              min="1"
              max="10"
              step="1"
              class="w-full accent-primary"
            />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-300 mb-2">
              {{ t('tools.domainGap.randomization.config.intensity') }}: {{ randConfig.intensity.toFixed(2) }}
            </label>
            <input
              v-model.number="randConfig.intensity"
              type="range"
              min="0"
              max="1"
              step="0.05"
              class="w-full accent-primary"
            />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-300 mb-2">
              {{ t('tools.domainGap.randomization.config.colorJitter') }}: {{ randConfig.color_jitter.toFixed(2) }}
            </label>
            <input
              v-model.number="randConfig.color_jitter"
              type="range"
              min="0"
              max="1"
              step="0.05"
              class="w-full accent-primary"
            />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-300 mb-2">
              {{ t('tools.domainGap.randomization.config.noiseIntensity') }}: {{ randConfig.noise_intensity.toFixed(3) }}
            </label>
            <input
              v-model.number="randConfig.noise_intensity"
              type="range"
              min="0"
              max="0.1"
              step="0.005"
              class="w-full accent-primary"
            />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-300 mb-2">
              {{ t('tools.domainGap.randomization.config.histogramMatch') }}: {{ randConfig.histogram_match_strength.toFixed(2) }}
            </label>
            <input
              v-model.number="randConfig.histogram_match_strength"
              type="range"
              min="0"
              max="1"
              step="0.05"
              class="w-full accent-primary"
            />
          </div>
          <div class="flex items-center gap-3">
            <input
              v-model="randConfig.preserve_annotations"
              type="checkbox"
              id="preserve-annotations"
              class="h-4 w-4 accent-primary"
            />
            <label for="preserve-annotations" class="text-sm font-medium text-gray-300">
              {{ t('tools.domainGap.randomization.config.preserveAnnotations') }}
            </label>
          </div>
        </div>

        <!-- Reference set for histogram matching -->
        <div class="mb-6">
          <BaseSelect
            v-model="randReferenceSetId"
            :label="t('tools.domainGap.validation.referenceSet')"
            :options="[{ value: '', label: 'None (no histogram matching)' }, ...referenceSetOptions]"
          />
        </div>

        <BaseButton
          @click="handleRandomize"
          variant="primary"
          :disabled="!randImagesDir || !randOutputDir || store.isLoading"
          :loading="store.isLoading"
        >
          <Shuffle class="h-4 w-4 mr-2" />
          {{ t('tools.domainGap.randomization.applyBatch') }}
        </BaseButton>
      </div>
    </div>

    <!-- ============================================ -->
    <!-- TAB: Diffusion Refinement -->
    <!-- ============================================ -->
    <div v-if="activeTab === 'diffusion'" class="space-y-6">

      <!-- Method Selection -->
      <div class="rounded-xl border border-gray-700/50 bg-background-card p-6">
        <h3 class="text-lg font-semibold text-white mb-2">
          {{ t('tools.domainGap.diffusion.title') }}
        </h3>
        <p class="text-sm text-gray-400 mb-6">
          {{ t('tools.domainGap.diffusion.description') }}
        </p>

        <!-- Method Radio Group -->
        <div class="mb-6">
          <label class="block text-sm font-medium text-gray-300 mb-3">
            {{ t('tools.domainGap.diffusion.method.label') }}
          </label>
          <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
            <button
              v-for="method in diffusionMethods"
              :key="method.id"
              @click="selectedDiffusionMethod = method.id"
              :class="[
                'p-4 rounded-xl border-2 text-left transition-colors',
                selectedDiffusionMethod === method.id
                  ? 'border-primary bg-primary/10 text-white'
                  : 'border-gray-700/50 hover:border-gray-600/50 text-gray-300',
              ]"
            >
              <div class="font-medium text-sm">{{ method.label }}</div>
              <div class="text-xs text-gray-500 mt-1">{{ method.description }}</div>
            </button>
          </div>
        </div>

        <!-- Reference Set Selector -->
        <div class="mb-4">
          <BaseSelect
            v-model="diffusionReferenceSetId"
            :label="t('tools.domainGap.diffusion.actions.selectReferenceSet')"
            :options="referenceSetOptions"
            :placeholder="t('tools.domainGap.diffusion.actions.selectReferenceSet')"
          />
        </div>

        <!-- LoRA Model Selector (visible only when method is lora) -->
        <div v-if="selectedDiffusionMethod === 'lora'" class="mb-4">
          <BaseSelect
            v-model="selectedLoraModelId"
            :label="t('tools.domainGap.diffusion.actions.selectLoraModel')"
            :options="store.loraModels.map(m => ({ value: m.model_id, label: `${m.model_id} (${m.model_size_mb.toFixed(1)} MB)` }))"
            :placeholder="t('tools.domainGap.diffusion.actions.selectLoraModel')"
          />
          <p v-if="!store.hasLoraModels" class="text-xs text-gray-500 mt-1">
            {{ t('tools.domainGap.diffusion.lora.noModels') }}
          </p>
        </div>
      </div>

      <!-- Parameters Panel -->
      <div class="rounded-xl border border-gray-700/50 bg-background-card p-6">
        <h4 class="font-semibold text-white mb-4">{{ t('tools.domainGap.diffusion.params.strength') }}</h4>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <!-- Strength Slider -->
          <div>
            <label class="block text-sm font-medium text-gray-300 mb-2">
              {{ t('tools.domainGap.diffusion.params.strength') }}: {{ diffusionConfig.strength.toFixed(2) }}
            </label>
            <input
              v-model.number="diffusionConfig.strength"
              type="range"
              min="0.1"
              max="0.7"
              step="0.05"
              class="w-full accent-primary"
            />
            <p class="text-xs text-gray-500 mt-1">{{ t('tools.domainGap.diffusion.params.strengthHint') }}</p>
          </div>

          <!-- ControlNet Scale (visible for controlnet method) -->
          <div v-if="selectedDiffusionMethod === 'controlnet'">
            <label class="block text-sm font-medium text-gray-300 mb-2">
              {{ t('tools.domainGap.diffusion.params.conditioningScale') }}: {{ diffusionConfig.controlnet_conditioning_scale.toFixed(2) }}
            </label>
            <input
              v-model.number="diffusionConfig.controlnet_conditioning_scale"
              type="range"
              min="0.3"
              max="1.0"
              step="0.05"
              class="w-full accent-primary"
            />
            <p class="text-xs text-gray-500 mt-1">{{ t('tools.domainGap.diffusion.params.conditioningScaleHint') }}</p>
          </div>

          <!-- IP-Adapter Scale (visible for ip_adapter method) -->
          <div v-if="selectedDiffusionMethod === 'ip_adapter'">
            <label class="block text-sm font-medium text-gray-300 mb-2">
              {{ t('tools.domainGap.diffusion.params.ipAdapterScale') }}: {{ diffusionConfig.ip_adapter_scale.toFixed(2) }}
            </label>
            <input
              v-model.number="diffusionConfig.ip_adapter_scale"
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              class="w-full accent-primary"
            />
            <p class="text-xs text-gray-500 mt-1">{{ t('tools.domainGap.diffusion.params.ipAdapterScaleHint') }}</p>
          </div>

          <!-- LoRA Weight (visible for lora method) -->
          <div v-if="selectedDiffusionMethod === 'lora'">
            <label class="block text-sm font-medium text-gray-300 mb-2">
              {{ t('tools.domainGap.diffusion.params.loraWeight') }}: {{ diffusionConfig.lora_weight.toFixed(2) }}
            </label>
            <input
              v-model.number="diffusionConfig.lora_weight"
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              class="w-full accent-primary"
            />
            <p class="text-xs text-gray-500 mt-1">{{ t('tools.domainGap.diffusion.params.loraWeightHint') }}</p>
          </div>

          <!-- Guidance Scale -->
          <div>
            <label class="block text-sm font-medium text-gray-300 mb-2">
              {{ t('tools.domainGap.diffusion.params.guidanceScale') }}: {{ diffusionConfig.guidance_scale.toFixed(1) }}
            </label>
            <input
              v-model.number="diffusionConfig.guidance_scale"
              type="range"
              min="1"
              max="20"
              step="0.5"
              class="w-full accent-primary"
            />
          </div>

          <!-- Inference Steps -->
          <div>
            <label class="block text-sm font-medium text-gray-300 mb-2">
              {{ t('tools.domainGap.diffusion.params.steps') }}: {{ diffusionConfig.num_inference_steps }}
            </label>
            <input
              v-model.number="diffusionConfig.num_inference_steps"
              type="range"
              min="4"
              max="100"
              step="1"
              class="w-full accent-primary"
            />
          </div>

          <!-- LCM Toggle -->
          <div class="flex items-center gap-3">
            <input
              v-model="diffusionConfig.use_lcm"
              type="checkbox"
              id="diffusion-use-lcm"
              class="h-4 w-4 accent-primary"
            />
            <label for="diffusion-use-lcm" class="text-sm font-medium text-gray-300">
              {{ t('tools.domainGap.diffusion.params.useLcm') }}
            </label>
            <span class="text-xs text-gray-500">{{ t('tools.domainGap.diffusion.params.useLcmHint') }}</span>
          </div>

          <!-- Prompt -->
          <div class="md:col-span-2">
            <BaseInput
              v-model="diffusionConfig.prompt"
              :label="t('tools.domainGap.diffusion.params.prompt')"
              :placeholder="t('tools.domainGap.diffusion.params.promptPlaceholder')"
            />
          </div>

          <!-- Negative Prompt -->
          <div class="md:col-span-2">
            <BaseInput
              v-model="diffusionConfig.negative_prompt"
              :label="t('tools.domainGap.diffusion.params.negativePrompt')"
            />
          </div>

          <!-- Seed -->
          <div>
            <BaseInput
              v-model.number="diffusionConfig.seed"
              :label="t('tools.domainGap.diffusion.params.seed')"
              type="number"
              :placeholder="t('tools.domainGap.diffusion.params.seedPlaceholder')"
            />
          </div>
        </div>
      </div>

      <!-- Directory Inputs & Actions -->
      <div class="rounded-xl border border-gray-700/50 bg-background-card p-6">
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <DirectoryBrowser
            v-model="diffusionSourceDir"
            :label="t('tools.domainGap.diffusion.sourceDirectory')"
            path-mode="both"
          />
          <DirectoryBrowser
            v-model="diffusionOutputDir"
            :label="t('tools.domainGap.diffusion.outputDirectory')"
            path-mode="output"
          />
        </div>

        <BaseButton
          @click="handleDiffusionRefineBatch"
          variant="primary"
          :disabled="store.isDiffusionProcessing || !diffusionReferenceSetId || !diffusionSourceDir"
          :loading="store.isDiffusionProcessing"
        >
          <Sparkles class="h-4 w-4 mr-2" />
          <span v-if="store.isDiffusionProcessing">{{ t('tools.domainGap.diffusion.actions.refining') }}</span>
          <span v-else>{{ t('tools.domainGap.diffusion.actions.refineBatch') }}</span>
        </BaseButton>

        <!-- Error Display -->
        <AlertBox
          v-if="store.lastDiffusionError"
          type="error"
          class="mt-4"
        >
          {{ store.lastDiffusionError }}
        </AlertBox>
      </div>

      <!-- Annotation Preservation Metrics -->
      <div
        v-if="store.lastPreservationMetrics"
        class="rounded-xl border border-gray-700/50 bg-background-card p-6"
      >
        <h4 class="font-semibold text-white mb-4">
          {{ t('tools.domainGap.diffusion.annotations.title') }}
        </h4>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <MetricCard
            :title="t('tools.domainGap.diffusion.annotations.edgeIou')"
            :value="`${(store.lastPreservationMetrics.edge_iou * 100).toFixed(1)}%`"
            :icon="BarChart3"
            :variant="store.lastPreservationMetrics.edge_iou >= 0.65 ? 'success' : 'error'"
          />
          <MetricCard
            :title="t('tools.domainGap.diffusion.annotations.ssim')"
            :value="`${(store.lastPreservationMetrics.ssim * 100).toFixed(1)}%`"
            :icon="BarChart3"
            :variant="store.lastPreservationMetrics.ssim >= 0.6 ? 'success' : 'error'"
          />
          <MetricCard
            :title="t('tools.domainGap.diffusion.annotations.keypointDisplacement')"
            :value="`${store.lastPreservationMetrics.mean_keypoint_displacement.toFixed(1)}px`"
            :icon="BarChart3"
            :variant="store.lastPreservationMetrics.mean_keypoint_displacement <= 10 ? 'success' : 'error'"
          />
        </div>

        <div class="text-center">
          <span
            v-if="store.lastPreservationMetrics.annotations_valid"
            class="inline-flex items-center gap-1.5 text-green-400 text-sm font-medium"
          >
            <CheckCircle class="h-4 w-4" />
            {{ t('tools.domainGap.diffusion.annotations.valid') }}
          </span>
          <span
            v-else
            class="inline-flex items-center gap-1.5 text-red-400 text-sm font-medium"
          >
            <AlertTriangle class="h-4 w-4" />
            {{ t('tools.domainGap.diffusion.annotations.invalid') }}
          </span>
        </div>
      </div>

      <!-- LoRA Management Section -->
      <div class="rounded-xl border border-gray-700/50 bg-background-card p-6">
        <div class="flex items-center justify-between mb-4">
          <h4 class="font-semibold text-white">{{ t('tools.domainGap.diffusion.lora.title') }}</h4>
          <BaseButton
            @click="showLoraTraining = !showLoraTraining"
            variant="ghost"
            class="text-purple-400 hover:text-purple-300 hover:bg-purple-900/20"
          >
            <Sparkles class="h-4 w-4 mr-2" />
            {{ t('tools.domainGap.diffusion.lora.train') }}
          </BaseButton>
        </div>

        <!-- LoRA Training Form -->
        <div v-if="showLoraTraining" class="mb-6 rounded-xl border border-purple-700/30 bg-purple-900/10 p-4">
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <BaseInput
              v-model="loraTrainingModelId"
              :label="t('tools.domainGap.diffusion.lora.modelId')"
              :placeholder="t('tools.domainGap.diffusion.lora.modelIdPlaceholder')"
            />
            <BaseInput
              v-model.number="loraTrainingSteps"
              :label="t('tools.domainGap.diffusion.lora.trainingSteps')"
              type="number"
              :min="100"
              :max="5000"
            />
            <div>
              <label class="block text-sm font-medium text-gray-300 mb-2">
                {{ t('tools.domainGap.diffusion.lora.rank') }}
              </label>
              <BaseSelect
                :modelValue="loraTrainingRank"
                @update:modelValue="loraTrainingRank = Number($event)"
                :options="[{ value: 4, label: '4' }, { value: 8, label: '8' }, { value: 16, label: '16' }, { value: 32, label: '32' }]"
              />
            </div>
            <BaseInput
              v-model.number="loraTrainingLR"
              :label="t('tools.domainGap.diffusion.lora.learningRate')"
              type="number"
              :min="0.00001"
              :max="0.01"
              :step="0.0001"
            />
          </div>
          <div class="flex gap-3">
            <BaseButton
              @click="handleTrainLora"
              variant="primary"
              :disabled="store.isLoraTraining || !diffusionReferenceSetId || !loraTrainingModelId"
              :loading="store.isLoraTraining"
              class="bg-purple-600 hover:bg-purple-700"
            >
              <span v-if="store.isLoraTraining">{{ t('tools.domainGap.diffusion.lora.training') }}</span>
              <span v-else>{{ t('tools.domainGap.diffusion.lora.train') }}</span>
            </BaseButton>
            <BaseButton @click="showLoraTraining = false" variant="ghost">
              {{ t('common.actions.cancel') }}
            </BaseButton>
          </div>
        </div>

        <!-- LoRA Models List -->
        <div v-if="store.hasLoraModels" class="space-y-2">
          <div
            v-for="lora in store.loraModels"
            :key="lora.model_id"
            class="flex items-center justify-between rounded-lg border border-gray-700/50 bg-background-secondary p-3"
          >
            <div>
              <div class="font-medium text-sm text-white">{{ lora.model_id }}</div>
              <div class="text-xs text-gray-500 mt-0.5">
                {{ lora.training_steps }} steps &middot; rank {{ lora.lora_rank }} &middot; {{ lora.model_size_mb.toFixed(1) }} MB
              </div>
            </div>
            <button
              @click="handleDeleteLora(lora.model_id)"
              class="p-1.5 text-gray-500 hover:text-red-400 transition-colors rounded-lg hover:bg-red-900/20"
            >
              <Trash2 class="h-4 w-4" />
            </button>
          </div>
        </div>
        <div v-else class="text-sm text-gray-500">
          {{ t('tools.domainGap.diffusion.lora.noModels') }}
        </div>
      </div>
    </div>

    <!-- ============================================ -->
    <!-- TAB: Style Transfer -->
    <!-- ============================================ -->
    <div v-if="activeTab === 'styleTransfer'">
      <div class="rounded-xl border border-gray-700/50 bg-background-card p-6">
        <h3 class="text-lg font-semibold text-white mb-2">
          {{ t('tools.domainGap.styleTransfer.title') }}
        </h3>
        <p class="text-sm text-gray-400 mb-6">
          {{ t('tools.domainGap.styleTransfer.description') }}
        </p>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <DirectoryBrowser
            v-model="stImagesDir"
            :label="t('tools.domainGap.styleTransfer.imagesDir')"
            path-mode="both"
          />
          <DirectoryBrowser
            v-model="stOutputDir"
            :label="t('tools.domainGap.styleTransfer.outputDir')"
            path-mode="output"
          />
        </div>

        <div class="mb-6">
          <BaseSelect
            v-model="stReferenceSetId"
            :label="t('tools.domainGap.styleTransfer.referenceSet')"
            :options="referenceSetOptions"
            :placeholder="t('tools.domainGap.validation.selectReferenceSet')"
          />
          <p class="text-xs text-gray-500 mt-1">
            {{ t('tools.domainGap.styleTransfer.referenceSetHint') }}
          </p>
        </div>

        <!-- Config sliders -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
          <div>
            <label class="block text-sm font-medium text-gray-300 mb-2">
              {{ t('tools.domainGap.styleTransfer.config.styleWeight') }}: {{ stConfig.style_weight.toFixed(2) }}
            </label>
            <input
              v-model.number="stConfig.style_weight"
              type="range"
              min="0"
              max="1"
              step="0.05"
              class="w-full accent-primary"
            />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-300 mb-2">
              {{ t('tools.domainGap.styleTransfer.config.contentWeight') }}: {{ stConfig.content_weight.toFixed(2) }}
            </label>
            <input
              v-model.number="stConfig.content_weight"
              type="range"
              min="0"
              max="2"
              step="0.1"
              class="w-full accent-primary"
            />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-300 mb-2">
              {{ t('tools.domainGap.styleTransfer.config.preserveStructure') }}: {{ stConfig.preserve_structure.toFixed(2) }}
            </label>
            <input
              v-model.number="stConfig.preserve_structure"
              type="range"
              min="0"
              max="1"
              step="0.05"
              class="w-full accent-primary"
            />
          </div>
          <div class="flex items-center gap-3">
            <input
              v-model="stConfig.color_only"
              type="checkbox"
              id="st-color-only"
              class="h-4 w-4 accent-primary"
            />
            <label for="st-color-only" class="text-sm font-medium text-gray-300">
              {{ t('tools.domainGap.styleTransfer.config.colorOnly') }}
            </label>
          </div>
          <div class="flex items-center gap-3">
            <input
              v-model="stConfig.depth_guided"
              type="checkbox"
              id="st-depth-guided"
              class="h-4 w-4 accent-primary"
            />
            <label for="st-depth-guided" class="text-sm font-medium text-gray-300">
              {{ t('tools.domainGap.styleTransfer.config.depthGuided') }}
            </label>
          </div>
        </div>

        <BaseButton
          @click="handleStyleTransfer"
          variant="primary"
          :disabled="!stImagesDir || !stOutputDir || !stReferenceSetId || store.isLoading"
          :loading="store.isLoading"
        >
          <Palette class="h-4 w-4 mr-2" />
          {{ t('tools.domainGap.styleTransfer.applyBatch') }}
        </BaseButton>
      </div>
    </div>
  </div>
</template>
