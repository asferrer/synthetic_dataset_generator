<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import TabContainer from '@/components/common/TabContainer.vue'
import EffectsTab from './tabs/EffectsTab.vue'
import ObjectSizesTab from './tabs/ObjectSizesTab.vue'
import BaseButton from '@/components/ui/BaseButton.vue'
import { Settings, Ruler, ArrowLeft, ArrowRight } from 'lucide-vue-next'

const router = useRouter()
const { t } = useI18n()

const activeTab = ref('effects')

const tabs = [
  { id: 'effects', label: t('workflow.configure.tabs.effects'), icon: Settings },
  { id: 'sizes', label: t('workflow.configure.tabs.objectSizes'), icon: Ruler },
]

function onEffectsNext() {
  router.push('/prepare?tab=sources')
}

function onEffectsBack() {
  router.push('/prepare')
}

function goBack() {
  router.push('/prepare')
}

function goNext() {
  router.push('/generate')
}
</script>

<template>
  <div class="space-y-6">
    <div>
      <h2 class="text-2xl font-bold text-white">{{ t('workflow.configure.title') }}</h2>
      <p class="mt-2 text-gray-400">{{ t('workflow.configure.subtitle') }}</p>
    </div>

    <TabContainer v-model="activeTab" :tabs="tabs">
      <template #effects>
        <EffectsTab @navigate-next="goNext" @navigate-back="goBack" />
      </template>
      <template #sizes>
        <ObjectSizesTab />
      </template>
    </TabContainer>

    <!-- Navigation Buttons -->
    <div class="flex justify-between pt-4">
      <BaseButton variant="outline" @click="goBack">
        <ArrowLeft class="h-5 w-5" />
        {{ t('common.actions.back') }}
      </BaseButton>
      <BaseButton @click="goNext">
        {{ t('common.actions.continue') }}
        <ArrowRight class="h-5 w-5" />
      </BaseButton>
    </div>
  </div>
</template>
