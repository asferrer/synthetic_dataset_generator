<script setup lang="ts">
import { computed } from 'vue'
import { Info, CheckCircle, AlertTriangle, XCircle, X } from 'lucide-vue-next'

interface Props {
  type?: 'info' | 'success' | 'warning' | 'error'
  title?: string
  dismissible?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  type: 'info',
  dismissible: false,
})

const emit = defineEmits<{
  dismiss: []
}>()

const iconMap = {
  info: Info,
  success: CheckCircle,
  warning: AlertTriangle,
  error: XCircle,
}

const colorClasses = computed(() => {
  const variants = {
    info: 'bg-blue-900/30 border-blue-700/50 text-blue-300',
    success: 'bg-green-900/30 border-green-700/50 text-green-300',
    warning: 'bg-yellow-900/30 border-yellow-700/50 text-yellow-300',
    error: 'bg-red-900/30 border-red-700/50 text-red-300',
  }
  return variants[props.type]
})

const iconColorClasses = computed(() => {
  const variants = {
    info: 'text-blue-400',
    success: 'text-green-400',
    warning: 'text-yellow-400',
    error: 'text-red-400',
  }
  return variants[props.type]
})
</script>

<template>
  <div :class="['rounded-lg border p-4', colorClasses]">
    <div class="flex items-start gap-3">
      <component
        :is="iconMap[type]"
        :class="['h-5 w-5 flex-shrink-0 mt-0.5', iconColorClasses]"
      />
      <div class="flex-1">
        <p v-if="title" class="font-medium text-white">{{ title }}</p>
        <div :class="[title ? 'mt-1' : '', 'text-sm']">
          <slot />
        </div>
      </div>
      <button
        v-if="dismissible"
        @click="emit('dismiss')"
        class="rounded-lg p-1 hover:bg-white/10"
      >
        <X class="h-4 w-4" />
      </button>
    </div>
  </div>
</template>
