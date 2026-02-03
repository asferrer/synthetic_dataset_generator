<script setup lang="ts">
import { useUiStore } from '@/stores/ui'
import { X, CheckCircle, XCircle, AlertTriangle, Info } from 'lucide-vue-next'

const uiStore = useUiStore()

const iconMap = {
  success: CheckCircle,
  error: XCircle,
  warning: AlertTriangle,
  info: Info,
}

const colorMap = {
  success: 'bg-green-900/90 border-green-700',
  error: 'bg-red-900/90 border-red-700',
  warning: 'bg-yellow-900/90 border-yellow-700',
  info: 'bg-blue-900/90 border-blue-700',
}

const iconColorMap = {
  success: 'text-green-400',
  error: 'text-red-400',
  warning: 'text-yellow-400',
  info: 'text-blue-400',
}
</script>

<template>
  <div class="fixed bottom-4 right-4 z-[100] flex flex-col gap-2">
    <TransitionGroup name="toast">
      <div
        v-for="toast in uiStore.toasts"
        :key="toast.id"
        :class="[
          'flex items-start gap-3 rounded-lg border p-4 shadow-lg backdrop-blur-sm',
          'min-w-[320px] max-w-md',
          colorMap[toast.type],
        ]"
      >
        <component
          :is="iconMap[toast.type]"
          :class="['h-5 w-5 flex-shrink-0', iconColorMap[toast.type]]"
        />
        <div class="flex-1">
          <p class="font-medium text-white">{{ toast.title }}</p>
          <p v-if="toast.message" class="mt-1 text-sm text-gray-300">
            {{ toast.message }}
          </p>
        </div>
        <button
          @click="uiStore.removeToast(toast.id)"
          class="rounded-lg p-1 text-gray-400 hover:bg-white/10 hover:text-white"
        >
          <X class="h-4 w-4" />
        </button>
      </div>
    </TransitionGroup>
  </div>
</template>

<style scoped>
.toast-enter-active,
.toast-leave-active {
  transition: all 0.3s ease;
}

.toast-enter-from {
  opacity: 0;
  transform: translateX(100px);
}

.toast-leave-to {
  opacity: 0;
  transform: translateX(100px);
}

.toast-move {
  transition: transform 0.3s ease;
}
</style>
