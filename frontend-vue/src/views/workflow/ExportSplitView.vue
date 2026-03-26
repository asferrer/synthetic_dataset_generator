<script setup lang="ts">
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import TabContainer from '@/components/common/TabContainer.vue'
import ExportTab from './tabs/ExportTab.vue'
import CombineTab from './tabs/CombineTab.vue'
import SplitTab from './tabs/SplitTab.vue'
import BalanceTab from './tabs/BalanceTab.vue'
import { Download, Combine, Split, Scale } from 'lucide-vue-next'

const { t } = useI18n()

const activeTab = ref('export')

const tabs = [
  { id: 'export', label: t('workflow.exportSplit.tabs.export'), icon: Download },
  { id: 'combine', label: t('workflow.exportSplit.tabs.combine'), icon: Combine },
  { id: 'split', label: t('workflow.exportSplit.tabs.split'), icon: Split },
  { id: 'balance', label: t('workflow.exportSplit.tabs.balance'), icon: Scale },
]
</script>

<template>
  <div class="space-y-6">
    <div>
      <h2 class="text-2xl font-bold text-white">{{ t('workflow.exportSplit.title') }}</h2>
      <p class="mt-2 text-gray-400">{{ t('workflow.exportSplit.subtitle') }}</p>
    </div>

    <TabContainer v-model="activeTab" :tabs="tabs">
      <template #export>
        <ExportTab />
      </template>
      <template #combine>
        <CombineTab />
      </template>
      <template #split>
        <SplitTab />
      </template>
      <template #balance>
        <BalanceTab />
      </template>
    </TabContainer>
  </div>
</template>
