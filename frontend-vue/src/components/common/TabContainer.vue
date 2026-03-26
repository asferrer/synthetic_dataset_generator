<script setup lang="ts">
import { computed, watch, type Component } from 'vue'
import { useRoute, useRouter } from 'vue-router'

interface Tab {
  id: string
  label: string
  icon?: Component
}

const props = defineProps<{
  tabs: Tab[]
  modelValue: string
}>()

const emit = defineEmits<{
  'update:modelValue': [value: string]
}>()

const route = useRoute()
const router = useRouter()

const activeTab = computed({
  get: () => props.modelValue,
  set: (val: string) => emit('update:modelValue', val),
})

// Sync tab with URL query param on mount
watch(
  () => route.query.tab,
  (tab) => {
    if (tab && typeof tab === 'string' && props.tabs.some((t) => t.id === tab)) {
      activeTab.value = tab
    }
  },
  { immediate: true },
)

function selectTab(id: string) {
  activeTab.value = id
  router.replace({ query: { ...route.query, tab: id } })
}
</script>

<template>
  <div>
    <!-- Tab bar -->
    <div class="mb-6 flex gap-1 rounded-lg bg-background-secondary p-1 border border-gray-700/50">
      <button
        v-for="tab in tabs"
        :key="tab.id"
        @click="selectTab(tab.id)"
        :class="[
          'flex items-center gap-2 rounded-md px-4 py-2.5 text-sm font-medium transition-all duration-200',
          activeTab === tab.id
            ? 'bg-primary text-white shadow-sm'
            : 'text-gray-400 hover:bg-gray-700/50 hover:text-white',
        ]"
      >
        <component v-if="tab.icon" :is="tab.icon" class="h-4 w-4" />
        {{ tab.label }}
      </button>
    </div>

    <!-- Tab content -->
    <div>
      <template v-for="tab in tabs" :key="tab.id">
        <div v-show="activeTab === tab.id">
          <slot :name="tab.id" />
        </div>
      </template>
    </div>
  </div>
</template>
