<script setup lang="ts">
import { computed } from 'vue'

interface Props {
  modelValue: string | number
  type?: 'text' | 'number' | 'email' | 'password' | 'search'
  placeholder?: string
  disabled?: boolean
  error?: string
  label?: string
  hint?: string
}

const props = withDefaults(defineProps<Props>(), {
  type: 'text',
  disabled: false,
})

const emit = defineEmits<{
  'update:modelValue': [value: string | number]
}>()

const inputClasses = computed(() => [
  'w-full px-4 py-2.5 bg-background-tertiary border rounded-lg',
  'text-white placeholder-gray-400',
  'focus:outline-none focus:ring-1 transition-colors duration-200',
  props.error
    ? 'border-red-500 focus:border-red-500 focus:ring-red-500'
    : 'border-gray-600 focus:border-primary focus:ring-primary',
  props.disabled ? 'opacity-50 cursor-not-allowed' : '',
])

function handleInput(event: Event) {
  const target = event.target as HTMLInputElement
  const value = props.type === 'number' ? parseFloat(target.value) || 0 : target.value
  emit('update:modelValue', value)
}
</script>

<template>
  <div class="space-y-1.5">
    <label v-if="label" class="block text-sm font-medium text-gray-300">
      {{ label }}
    </label>
    <input
      :type="type"
      :value="modelValue"
      :placeholder="placeholder"
      :disabled="disabled"
      :class="inputClasses"
      @input="handleInput"
    />
    <p v-if="hint && !error" class="text-sm text-gray-500">{{ hint }}</p>
    <p v-if="error" class="text-sm text-red-400">{{ error }}</p>
  </div>
</template>
