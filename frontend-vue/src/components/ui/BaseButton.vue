<script setup lang="ts">
import { computed } from 'vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'

interface Props {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger' | 'success'
  size?: 'sm' | 'md' | 'lg'
  disabled?: boolean
  loading?: boolean
  type?: 'button' | 'submit' | 'reset'
}

const props = withDefaults(defineProps<Props>(), {
  variant: 'primary',
  size: 'md',
  disabled: false,
  loading: false,
  type: 'button',
})

const variantClasses = {
  primary: 'bg-primary hover:bg-primary-hover text-white',
  secondary: 'bg-gray-700 hover:bg-gray-600 text-white',
  outline: 'border border-gray-600 hover:bg-gray-700/50 text-gray-300',
  ghost: 'hover:bg-gray-700/50 text-gray-300',
  danger: 'bg-red-600 hover:bg-red-700 text-white',
  success: 'bg-green-600 hover:bg-green-700 text-white',
}

const sizeClasses = {
  sm: 'px-3 py-1.5 text-sm',
  md: 'px-4 py-2',
  lg: 'px-6 py-3 text-lg',
}

const buttonClasses = computed(() => [
  'inline-flex items-center justify-center gap-2 rounded-lg font-medium',
  'transition-all duration-200',
  'disabled:opacity-50 disabled:cursor-not-allowed',
  variantClasses[props.variant],
  sizeClasses[props.size],
])
</script>

<template>
  <button
    :type="type"
    :disabled="disabled || loading"
    :class="buttonClasses"
  >
    <LoadingSpinner v-if="loading" size="sm" />
    <slot />
  </button>
</template>
