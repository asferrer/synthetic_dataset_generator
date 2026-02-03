<script setup lang="ts">
import { computed } from 'vue'

interface Props {
  title: string
  value: string | number
  icon?: any
  trend?: number
  trendLabel?: string
  variant?: 'default' | 'success' | 'warning' | 'error' | 'info'
}

const props = withDefaults(defineProps<Props>(), {
  variant: 'default',
})

const bgClasses = computed(() => {
  const variants = {
    default: 'bg-background-card',
    success: 'bg-green-900/20 border-green-700/50',
    warning: 'bg-yellow-900/20 border-yellow-700/50',
    error: 'bg-red-900/20 border-red-700/50',
    info: 'bg-blue-900/20 border-blue-700/50',
  }
  return variants[props.variant]
})

const iconBgClasses = computed(() => {
  const variants = {
    default: 'bg-gray-700',
    success: 'bg-green-700/50',
    warning: 'bg-yellow-700/50',
    error: 'bg-red-700/50',
    info: 'bg-blue-700/50',
  }
  return variants[props.variant]
})
</script>

<template>
  <div :class="['rounded-xl border border-gray-700/50 p-5', bgClasses]">
    <div class="flex items-start justify-between">
      <div class="flex-1">
        <p class="text-sm font-medium text-gray-400">{{ title }}</p>
        <p class="mt-2 text-3xl font-bold text-white">{{ value }}</p>
        <div v-if="trend !== undefined" class="mt-2 flex items-center gap-1">
          <span
            :class="[
              'text-sm font-medium',
              trend >= 0 ? 'text-green-400' : 'text-red-400',
            ]"
          >
            {{ trend >= 0 ? '+' : '' }}{{ trend }}%
          </span>
          <span v-if="trendLabel" class="text-sm text-gray-500">
            {{ trendLabel }}
          </span>
        </div>
      </div>
      <div
        v-if="icon"
        :class="['rounded-lg p-3', iconBgClasses]"
      >
        <component :is="icon" class="h-6 w-6 text-white" />
      </div>
    </div>
  </div>
</template>
