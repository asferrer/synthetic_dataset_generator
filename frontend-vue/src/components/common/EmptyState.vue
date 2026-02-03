<script setup lang="ts">
import { Inbox } from 'lucide-vue-next'

interface Props {
  icon?: any
  title: string
  description?: string
}

withDefaults(defineProps<Props>(), {
  icon: Inbox,
})
</script>

<template>
  <div class="flex flex-col items-center justify-center py-12 text-center">
    <div class="mb-4 rounded-full bg-gray-700/50 p-4">
      <component :is="icon" class="h-10 w-10 text-gray-400" />
    </div>
    <h3 class="text-lg font-medium text-white">{{ title }}</h3>
    <p v-if="description" class="mt-2 max-w-sm text-sm text-gray-400">
      {{ description }}
    </p>
    <div v-if="$slots.action" class="mt-6">
      <slot name="action" />
    </div>
  </div>
</template>
