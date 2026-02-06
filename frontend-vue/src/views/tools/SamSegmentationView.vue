<script setup lang="ts">
import { ref, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useUiStore } from '@/stores/ui'
import {
  segmentWithText,
  sam3ConvertDataset,
  getSam3JobStatus,
} from '@/lib/api'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseInput from '@/components/ui/BaseInput.vue'
import AlertBox from '@/components/common/AlertBox.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import DirectoryBrowser from '@/components/common/DirectoryBrowser.vue'
import {
  Box,
  Wand2,
  Database,
  CheckCircle,
  Play,
} from 'lucide-vue-next'
import type { SegmentationResult, Job } from '@/types/api'

const uiStore = useUiStore()
const { t } = useI18n()

// Mode
const mode = ref<'text' | 'convert'>('text')

// Text segmentation state
const imagePath = ref('')
const textPrompt = ref('')
const textLoading = ref(false)
const textResult = ref<SegmentationResult | null>(null)

// Dataset conversion state
const cocoJsonPath = ref('')
const imagesDir = ref('')
const outputDir = ref('/data/sam3_converted')
const minArea = ref(100)
const confidenceThreshold = ref(0.8)
const convertLoading = ref(false)
const currentJob = ref<Job | null>(null)
let pollingInterval: ReturnType<typeof setInterval> | null = null

const error = ref<string | null>(null)

async function runTextSegmentation() {
  if (!imagePath.value || !textPrompt.value) {
    uiStore.showError(t('tools.samSegmentation.notifications.missingInput'), t('tools.samSegmentation.notifications.missingInputMsg'))
    return
  }

  textLoading.value = true
  error.value = null
  textResult.value = null

  try {
    textResult.value = await segmentWithText(imagePath.value, textPrompt.value)
    uiStore.showSuccess(t('tools.samSegmentation.notifications.segmentationComplete'), t('tools.samSegmentation.notifications.segmentationCompleteMsg', { count: textResult.value.masks.length }))
  } catch (e: any) {
    error.value = e.message || t('tools.samSegmentation.notifications.segmentationFailed')
    uiStore.showError(t('tools.samSegmentation.notifications.segmentationFailed'), error.value)
  } finally {
    textLoading.value = false
  }
}

async function startConversion() {
  if (!cocoJsonPath.value || !imagesDir.value || !outputDir.value) {
    uiStore.showError(t('tools.samSegmentation.notifications.missingInput'), t('tools.samSegmentation.notifications.missingInputMsg'))
    return
  }

  convertLoading.value = true
  error.value = null
  currentJob.value = null

  try {
    const response = await sam3ConvertDataset(
      cocoJsonPath.value,
      imagesDir.value,
      outputDir.value,
      { minArea: minArea.value, confidenceThreshold: confidenceThreshold.value }
    )

    uiStore.showSuccess(t('tools.samSegmentation.notifications.conversionStarted'), t('tools.samSegmentation.notifications.conversionStartedMsg', { id: response.job_id.slice(0, 8) }))
    startPolling(response.job_id)
  } catch (e: any) {
    error.value = e.message || t('tools.samSegmentation.notifications.conversionFailed')
    uiStore.showError(t('tools.samSegmentation.notifications.conversionFailed'), error.value)
    convertLoading.value = false
  }
}

async function pollJobStatus(jobId: string) {
  try {
    const job = await getSam3JobStatus(jobId)
    currentJob.value = job

    if (job.status === 'completed') {
      stopPolling()
      convertLoading.value = false
      uiStore.showSuccess(t('tools.samSegmentation.notifications.conversionComplete'), t('tools.samSegmentation.notifications.conversionCompleteMsg'))
    } else if (job.status === 'failed') {
      stopPolling()
      convertLoading.value = false
      error.value = job.error || t('tools.samSegmentation.notifications.conversionFailed')
      uiStore.showError(t('tools.samSegmentation.notifications.conversionFailed'), error.value)
    }
  } catch (e) {
    // Ignore polling errors
  }
}

function startPolling(jobId: string) {
  pollingInterval = setInterval(() => pollJobStatus(jobId), 2000)
}

function stopPolling() {
  if (pollingInterval) {
    clearInterval(pollingInterval)
    pollingInterval = null
  }
}

onUnmounted(() => {
  stopPolling()
})
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div>
      <h2 class="text-2xl font-bold text-white">{{ t('tools.samSegmentation.title') }}</h2>
      <p class="mt-1 text-gray-400">
        {{ t('tools.samSegmentation.subtitle') }}
      </p>
    </div>

    <!-- Error -->
    <AlertBox v-if="error" type="error" :title="error" dismissible @dismiss="error = null" />

    <!-- Mode Toggle -->
    <div class="flex gap-2">
      <BaseButton
        :variant="mode === 'text' ? 'primary' : 'outline'"
        @click="mode = 'text'"
      >
        <Wand2 class="h-5 w-5" />
        {{ t('tools.samSegmentation.modes.text') }}
      </BaseButton>
      <BaseButton
        :variant="mode === 'convert' ? 'primary' : 'outline'"
        @click="mode = 'convert'"
      >
        <Database class="h-5 w-5" />
        {{ t('tools.samSegmentation.modes.convert') }}
      </BaseButton>
    </div>

    <!-- Text Segmentation Mode -->
    <div v-if="mode === 'text'" class="grid gap-6 lg:grid-cols-2">
      <!-- Input -->
      <div class="card p-6">
        <h3 class="text-lg font-semibold text-white mb-4">{{ t('tools.samSegmentation.textMode.title') }}</h3>

        <div class="space-y-4">
          <DirectoryBrowser
            v-model="imagePath"
            :label="t('tools.samSegmentation.textMode.imagePath')"
            placeholder="/data/images/example.jpg"
            :show-files="true"
            file-pattern="*.jpg,*.png,*.jpeg"
            path-mode="input"
          />
          <BaseInput
            v-model="textPrompt"
            :label="t('tools.samSegmentation.textMode.textPrompt')"
            :placeholder="t('tools.samSegmentation.textMode.textPromptPlaceholder')"
            :hint="t('tools.samSegmentation.textMode.textPromptHint')"
          />
        </div>

        <BaseButton
          class="mt-6 w-full"
          :loading="textLoading"
          :disabled="textLoading || !imagePath || !textPrompt"
          @click="runTextSegmentation"
        >
          <Wand2 class="h-5 w-5" />
          {{ t('tools.samSegmentation.textMode.runSegmentation') }}
        </BaseButton>
      </div>

      <!-- Results -->
      <div class="card p-6">
        <h3 class="text-lg font-semibold text-white mb-4">{{ t('tools.samSegmentation.textMode.results') }}</h3>

        <div v-if="textLoading" class="flex justify-center py-12">
          <LoadingSpinner :message="t('common.status.processing')" />
        </div>

        <div v-else-if="!textResult" class="text-center py-12 text-gray-400">
          <Box class="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>{{ t('tools.samSegmentation.textMode.runSegmentation') }}</p>
        </div>

        <div v-else class="space-y-4">
          <div class="flex items-center justify-between">
            <span class="text-gray-400">{{ t('tools.samSegmentation.textMode.objectsFound') }}</span>
            <span class="text-2xl font-bold text-primary">{{ textResult.masks.length }}</span>
          </div>

          <div class="space-y-2">
            <div
              v-for="(score, index) in textResult.scores"
              :key="index"
              class="flex items-center gap-3"
            >
              <span class="w-20 text-sm text-gray-400">{{ t('tools.samSegmentation.textMode.object') }} {{ index + 1 }}</span>
              <div class="flex-1 h-2 bg-background-tertiary rounded-full overflow-hidden">
                <div
                  class="h-full bg-primary"
                  :style="{ width: `${score * 100}%` }"
                />
              </div>
              <span class="text-sm text-white">{{ (score * 100).toFixed(1) }}%</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Dataset Conversion Mode -->
    <div v-if="mode === 'convert'" class="space-y-6">
      <div class="grid gap-6 lg:grid-cols-2">
        <!-- Input -->
        <div class="card p-6">
          <h3 class="text-lg font-semibold text-white mb-4">{{ t('tools.samSegmentation.convertMode.sourceTitle') }}</h3>
          <p class="text-sm text-gray-400 mb-4">
            {{ t('tools.samSegmentation.convertMode.sourceDescription') }}
          </p>

          <div class="space-y-4">
            <DirectoryBrowser
              v-model="cocoJsonPath"
              :label="t('tools.samSegmentation.convertMode.cocoJsonPath')"
              placeholder="/data/dataset/annotations.json"
              :show-files="true"
              file-pattern="*.json"
              path-mode="input"
            />
            <DirectoryBrowser
              v-model="imagesDir"
              :label="t('tools.samSegmentation.convertMode.imagesDirectory')"
              placeholder="/data/dataset/images"
              path-mode="input"
            />
          </div>
        </div>

        <!-- Output -->
        <div class="card p-6">
          <h3 class="text-lg font-semibold text-white mb-4">{{ t('tools.samSegmentation.convertMode.outputTitle') }}</h3>

          <div class="space-y-4">
            <DirectoryBrowser
              v-model="outputDir"
              :label="t('tools.samSegmentation.convertMode.outputDirectory')"
              placeholder="/data/sam3_converted"
              path-mode="output"
            />

            <div>
              <label class="text-sm text-gray-400 flex justify-between mb-2">
                <span>{{ t('tools.samSegmentation.convertMode.minArea') }}</span>
                <span class="text-white">{{ minArea }}{{ t('tools.samSegmentation.convertMode.minAreaUnit') }}</span>
              </label>
              <input
                type="range"
                v-model.number="minArea"
                min="10"
                max="1000"
                step="10"
                class="w-full accent-primary"
              />
              <p class="text-xs text-gray-500 mt-1">{{ t('tools.samSegmentation.convertMode.minAreaHint') }}</p>
            </div>

            <div>
              <label class="text-sm text-gray-400 flex justify-between mb-2">
                <span>{{ t('tools.samSegmentation.convertMode.confidenceThreshold') }}</span>
                <span class="text-white">{{ (confidenceThreshold * 100).toFixed(0) }}%</span>
              </label>
              <input
                type="range"
                v-model.number="confidenceThreshold"
                min="0.1"
                max="1"
                step="0.05"
                class="w-full accent-primary"
              />
              <p class="text-xs text-gray-500 mt-1">{{ t('tools.samSegmentation.convertMode.confidenceHint') }}</p>
            </div>
          </div>
        </div>
      </div>

      <!-- Conversion Progress -->
      <div v-if="currentJob" class="card p-6">
        <h3 class="text-lg font-semibold text-white mb-4">{{ t('tools.samSegmentation.progress.title') }}</h3>

        <div class="flex items-center gap-4 mb-4">
          <component
            :is="currentJob.status === 'completed' ? CheckCircle : Database"
            :class="[
              'h-8 w-8',
              currentJob.status === 'completed' ? 'text-green-400' : 'text-primary animate-pulse'
            ]"
          />
          <div class="flex-1">
            <p class="font-medium text-white">
              {{ currentJob.status === 'running' ? t('tools.samSegmentation.progress.converting') : currentJob.status }}
            </p>
            <p class="text-sm text-gray-400">{{ t('tools.samSegmentation.progress.jobId', { id: currentJob.job_id.slice(0, 8) }) }}...</p>
          </div>
          <span class="text-2xl font-bold text-primary">{{ currentJob.progress }}%</span>
        </div>

        <div class="h-3 bg-background-tertiary rounded-full overflow-hidden">
          <div
            class="h-full bg-primary transition-all"
            :style="{ width: `${currentJob.progress}%` }"
          />
        </div>
      </div>

      <!-- Start Button -->
      <div class="flex justify-end">
        <BaseButton
          :loading="convertLoading"
          :disabled="convertLoading || !cocoJsonPath || !imagesDir || !outputDir"
          @click="startConversion"
          size="lg"
        >
          <Play class="h-5 w-5" />
          {{ t('tools.samSegmentation.actions.startConversion') }}
        </BaseButton>
      </div>
    </div>
  </div>
</template>
