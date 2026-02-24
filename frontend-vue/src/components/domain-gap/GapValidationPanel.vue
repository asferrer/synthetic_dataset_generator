<script setup lang="ts">
/**
 * GapValidationPanel - Domain Gap validation step 4.5
 *
 * Shown in the Generation view after a generation job completes.
 * Allows users to quickly validate the domain gap of generated images
 * against real reference images before proceeding to Export.
 */
import { ref, computed, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useUiStore } from '@/stores/ui'
import { useDomainGapStore } from '@/stores/domainGap'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseSelect from '@/components/ui/BaseSelect.vue'
import MetricCard from '@/components/common/MetricCard.vue'
import {
  ScanSearch,
  BarChart3,
  CheckCircle,
  AlertTriangle,
  Lightbulb,
  ArrowRight,
  ChevronDown,
  ChevronUp,
  SkipForward,
  Info,
} from 'lucide-vue-next'

const props = defineProps<{
  outputDir: string
}>()

const emit = defineEmits<{
  (e: 'skip'): void
  (e: 'apply-suggestions', suggestions: any[]): void
}>()

const { t } = useI18n()
const uiStore = useUiStore()
const store = useDomainGapStore()

const expanded = ref(false)
const showDetails = ref(false)

const referenceSetOptions = computed(() =>
  store.referenceSets.map(s => ({
    value: s.set_id,
    label: `${s.name} (${s.image_count} imgs)`,
  })),
)

const selectedRefSet = ref('')

const hasResults = computed(() => store.latestAnalysis !== null)

const gapScoreVariant = computed((): 'success' | 'warning' | 'error' => {
  const score = store.gapScore
  if (score === null) return 'success'
  if (score < 30) return 'success'
  if (score < 60) return 'warning'
  return 'error'
})

const gapIcon = computed(() => {
  const score = store.gapScore
  if (score === null) return Info
  if (score < 30) return CheckCircle
  if (score < 60) return AlertTriangle
  return AlertTriangle
})

const gapStatusText = computed(() => {
  const score = store.gapScore
  if (score === null) return ''
  if (score < 30) return t('tools.domainGap.validation.results.levels.low')
  if (score < 60) return t('tools.domainGap.validation.results.levels.medium')
  if (score < 80) return t('tools.domainGap.validation.results.levels.high')
  return t('tools.domainGap.validation.results.levels.critical')
})

async function handleAnalyze() {
  if (!selectedRefSet.value) return

  const result = await store.runAnalysis(
    props.outputDir,
    selectedRefSet.value,
    50,
  )

  if (result) {
    showDetails.value = true
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

function handleApplySuggestions() {
  if (store.latestAnalysis?.suggestions) {
    emit('apply-suggestions', store.latestAnalysis.suggestions)
  }
}

onMounted(async () => {
  store.clearAnalysis()
  await store.fetchReferenceSets()
  if (store.hasReferenceSets) {
    expanded.value = true
  }
})
</script>

<template>
  <div class="rounded-xl border border-purple-700/30 bg-purple-900/10 overflow-hidden">
    <!-- Header (always visible) -->
    <button
      @click="expanded = !expanded"
      class="w-full flex items-center justify-between p-4 text-left hover:bg-purple-900/20 transition-colors"
    >
      <div class="flex items-center gap-3">
        <div class="flex h-8 w-8 items-center justify-center rounded-lg bg-purple-600/20">
          <ScanSearch class="h-5 w-5 text-purple-400" />
        </div>
        <div>
          <h3 class="text-sm font-semibold text-white">
            {{ t('tools.domainGap.validation.title') }}
          </h3>
          <p class="text-xs text-gray-400">
            {{ t('tools.domainGap.validation.description') }}
          </p>
        </div>
      </div>
      <div class="flex items-center gap-2">
        <!-- Gap score badge (if available) -->
        <span
          v-if="hasResults"
          :class="[
            'text-xs font-medium px-2.5 py-1 rounded-full',
            gapScoreVariant === 'success' ? 'bg-green-900/50 text-green-400' : '',
            gapScoreVariant === 'warning' ? 'bg-yellow-900/50 text-yellow-400' : '',
            gapScoreVariant === 'error' ? 'bg-red-900/50 text-red-400' : '',
          ]"
        >
          {{ store.gapScore?.toFixed(1) }}/100 - {{ gapStatusText }}
        </span>
        <ChevronDown v-if="!expanded" class="h-5 w-5 text-gray-400" />
        <ChevronUp v-else class="h-5 w-5 text-gray-400" />
      </div>
    </button>

    <!-- Expanded content -->
    <div v-if="expanded" class="border-t border-purple-700/30 p-4 space-y-4">
      <!-- No reference sets -->
      <div v-if="!store.hasReferenceSets" class="text-sm text-gray-400 flex items-center gap-2">
        <Info class="h-4 w-4" />
        {{ t('tools.domainGap.references.noSets') }}
        <router-link to="/tools/domain-gap" class="text-purple-400 hover:text-purple-300 underline">
          {{ t('tools.domainGap.references.uploadTitle') }}
        </router-link>
      </div>

      <!-- Analysis form -->
      <div v-else>
        <div class="flex items-end gap-3">
          <div class="flex-1">
            <BaseSelect
              v-model="selectedRefSet"
              :label="t('tools.domainGap.validation.referenceSet')"
              :options="referenceSetOptions"
              :placeholder="t('tools.domainGap.validation.selectReferenceSet')"
            />
          </div>
          <BaseButton
            @click="handleAnalyze"
            variant="primary"
            size="sm"
            :disabled="!selectedRefSet || store.isAnalyzing"
            :loading="store.isAnalyzing"
          >
            <BarChart3 class="h-4 w-4 mr-1" />
            {{ store.isAnalyzing ? t('tools.domainGap.validation.computing') : t('tools.domainGap.validation.analyze') }}
          </BaseButton>
          <BaseButton
            @click="emit('skip')"
            variant="ghost"
            size="sm"
          >
            <SkipForward class="h-4 w-4 mr-1" />
            {{ t('tools.domainGap.validation.skip') }}
          </BaseButton>
        </div>
      </div>

      <!-- Results (compact) -->
      <div v-if="hasResults && store.latestAnalysis" class="space-y-3">
        <!-- Score cards -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
          <MetricCard
            :title="t('tools.domainGap.validation.results.gapScore')"
            :value="`${store.latestAnalysis.metrics.overall_gap_score.toFixed(1)}`"
            :icon="gapIcon"
            :variant="gapScoreVariant"
          />
          <MetricCard
            v-if="store.latestAnalysis.metrics.fid_score !== null"
            :title="t('tools.domainGap.validation.results.fidScore')"
            :value="store.latestAnalysis.metrics.fid_score.toFixed(1)"
            :icon="BarChart3"
          />
          <MetricCard
            v-if="store.latestAnalysis.metrics.kid_score !== null"
            :title="t('tools.domainGap.validation.results.kidScore')"
            :value="store.latestAnalysis.metrics.kid_score.toFixed(4)"
            :icon="BarChart3"
          />
          <MetricCard
            :title="t('tools.domainGap.validation.results.gapLevel')"
            :value="gapStatusText"
            :icon="Info"
          />
        </div>

        <!-- Issues summary -->
        <div v-if="store.latestAnalysis.issues.length > 0">
          <button
            @click="showDetails = !showDetails"
            class="text-sm text-gray-400 hover:text-white flex items-center gap-1"
          >
            <AlertTriangle class="h-3.5 w-3.5 text-yellow-400" />
            {{ store.latestAnalysis.issues.length }} {{ t('tools.domainGap.validation.issuesDetected') }}
            <ChevronDown v-if="!showDetails" class="h-3.5 w-3.5" />
            <ChevronUp v-else class="h-3.5 w-3.5" />
          </button>

          <div v-if="showDetails" class="mt-2 space-y-2">
            <div
              v-for="(issue, idx) in store.latestAnalysis.issues.slice(0, 5)"
              :key="idx"
              class="text-xs bg-gray-800/50 rounded-lg p-2 flex items-start gap-2"
            >
              <AlertTriangle
                :class="[
                  'h-3.5 w-3.5 mt-0.5 flex-shrink-0',
                  issue.severity === 'high' ? 'text-red-400' : 'text-yellow-400',
                ]"
              />
              <div>
                <span class="text-gray-300">{{ issue.description }}</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Suggestions (compact) -->
        <div v-if="store.latestAnalysis.suggestions.length > 0" class="space-y-2">
          <div class="flex items-center justify-between">
            <span class="text-sm text-gray-400 flex items-center gap-1">
              <Lightbulb class="h-3.5 w-3.5 text-yellow-400" />
              {{ store.latestAnalysis.suggestions.length }} {{ t('tools.domainGap.validation.parameterSuggestions') }}
            </span>
            <BaseButton
              @click="handleApplySuggestions"
              variant="ghost"
              size="sm"
            >
              {{ t('tools.domainGap.validation.suggestions.applyAll') }}
            </BaseButton>
          </div>
          <div class="flex flex-wrap gap-2">
            <div
              v-for="(s, idx) in store.latestAnalysis.suggestions.slice(0, 4)"
              :key="idx"
              class="text-xs bg-gray-800/50 rounded-lg px-2.5 py-1.5"
            >
              <code class="text-purple-400">{{ s.parameter_path }}</code>
              <span class="text-gray-500 mx-1">→</span>
              <span class="text-green-400">{{ s.suggested_value }}</span>
            </div>
          </div>
        </div>

        <!-- Recommendation -->
        <div
          :class="[
            'rounded-lg p-3 text-sm flex items-center gap-2',
            gapScoreVariant === 'success' ? 'bg-green-900/20 text-green-400' : '',
            gapScoreVariant === 'warning' ? 'bg-yellow-900/20 text-yellow-400' : '',
            gapScoreVariant === 'error' ? 'bg-red-900/20 text-red-400' : '',
          ]"
        >
          <component :is="gapIcon" class="h-4 w-4 flex-shrink-0" />
          <span v-if="gapScoreVariant === 'success'">
            Domain gap is low. Your synthetic data should work well for training.
          </span>
          <span v-else-if="gapScoreVariant === 'warning'">
            Moderate domain gap detected. Consider applying suggestions to improve quality.
          </span>
          <span v-else>
            High domain gap detected. It is recommended to adjust parameters and regenerate before proceeding.
          </span>
        </div>
      </div>
    </div>
  </div>
</template>
