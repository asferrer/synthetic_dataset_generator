<script setup lang="ts">
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import TabContainer from '@/components/common/TabContainer.vue'
import JobMonitorTab from './tabs/JobMonitorTab.vue'
import ServiceStatusTab from './tabs/ServiceStatusTab.vue'
import { Activity, Server } from 'lucide-vue-next'

const { t } = useI18n()

const activeTab = ref('jobs')

const tabs = [
  { id: 'jobs', label: t('nav.tools.tabs.jobs'), icon: Activity },
  { id: 'services', label: t('nav.tools.tabs.services'), icon: Server },
]
</script>

<template>
  <div class="space-y-6">
    <div>
      <h2 class="text-2xl font-bold text-white">{{ t('nav.titles.system') }}</h2>
      <p class="mt-2 text-gray-400">{{ t('nav.tools.systemDescription') }}</p>
    </div>

    <TabContainer v-model="activeTab" :tabs="tabs">
      <template #jobs>
        <JobMonitorTab />
      </template>
      <template #services>
        <ServiceStatusTab />
      </template>
    </TabContainer>
  </div>
</template>
