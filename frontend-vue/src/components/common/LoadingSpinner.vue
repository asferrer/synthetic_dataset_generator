<script setup lang="ts">
interface Props {
  size?: 'sm' | 'md' | 'lg'
  message?: string
}

withDefaults(defineProps<Props>(), {
  size: 'md',
})

const sizeClasses = {
  sm: 'h-4 w-4 border-2',
  md: 'h-8 w-8 border-2',
  lg: 'h-12 w-12 border-3',
}
</script>

<template>
  <div class="flex flex-col items-center justify-center gap-3">
    <div
      :class="[
        'animate-spin rounded-full border-primary border-t-transparent',
        sizeClasses[size],
      ]"
    />
    <p v-if="message" class="text-sm text-gray-400">{{ message }}</p>
  </div>
</template>
