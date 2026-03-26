<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import TabContainer from '@/components/common/TabContainer.vue'
import AnalysisTab from './tabs/AnalysisTab.vue'
import SourceSelectionTab from './tabs/SourceSelectionTab.vue'
import { BarChart3, FolderInput } from 'lucide-vue-next'

const router = useRouter()
const { t } = useI18n()

const activeTab = ref('analyze')

const tabs = [
  { id: 'analyze', label: t('workflow.prepare.tabs.analyze'), icon: BarChart3 },
  { id: 'sources', label: t('workflow.prepare.tabs.sources'), icon: FolderInput },
]

function onAnalysisNext() {
  activeTab.value = 'sources'
}

function onSourcesNext() {
  router.push('/configure')
}

function onSourcesBack() {
  activeTab.value = 'analyze'
}
</script>

<template>
  <div class="space-y-6">
    <div>
      <h2 class="text-2xl font-bold text-white">{{ t('workflow.prepare.title') }}</h2>
      <p class="mt-2 text-gray-400">{{ t('workflow.prepare.subtitle') }}</p>
    </div>

    <TabContainer v-model="activeTab" :tabs="tabs">
      <template #analyze>
        <AnalysisTab @navigate-next="onAnalysisNext" />
      </template>
      <template #sources>
        <SourceSelectionTab @navigate-next="onSourcesNext" @navigate-back="onSourcesBack" />
      </template>
    </TabContainer>
  </div>
</template>
