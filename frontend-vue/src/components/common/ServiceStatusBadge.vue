<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { getHealthStatus } from '@/lib/api'
import { Server, CheckCircle, AlertCircle, XCircle } from 'lucide-vue-next'
import type { HealthStatus } from '@/types/api'

const health = ref<HealthStatus | null>(null)
const loading = ref(true)
const error = ref(false)
let intervalId: ReturnType<typeof setInterval>

async function checkHealth() {
  try {
    health.value = await getHealthStatus()
    error.value = false
  } catch (e) {
    error.value = true
  } finally {
    loading.value = false
  }
}

// Count healthy services from the services array
const healthyCount = computed(() => {
  if (!health.value?.services) return 0
  return health.value.services.filter(s => s.status === 'healthy').length
})

// Total services from the services array
const totalServices = computed(() => {
  if (!health.value?.services) return 0
  return health.value.services.length
})

const statusColor = computed(() => {
  if (error.value || !health.value) return 'text-red-400'
  if (health.value.status === 'healthy') return 'text-green-400'
  if (health.value.status === 'degraded') return 'text-yellow-400'
  return 'text-red-400'
})

const statusIcon = computed(() => {
  if (error.value || !health.value) return XCircle
  if (health.value.status === 'healthy') return CheckCircle
  if (health.value.status === 'degraded') return AlertCircle
  return XCircle
})

const statusText = computed(() => {
  if (loading.value) return 'Checking...'
  if (error.value) return 'Offline'
  if (totalServices.value === 0) return 'No services'
  return `${healthyCount.value}/${totalServices.value} Services`
})

onMounted(() => {
  checkHealth()
  intervalId = setInterval(checkHealth, 30000) // Check every 30 seconds
})

onUnmounted(() => {
  clearInterval(intervalId)
})
</script>

<template>
  <router-link
    to="/tools/service-status"
    class="flex items-center gap-2 rounded-lg bg-background-tertiary px-3 py-1.5 text-sm transition-colors hover:bg-gray-600"
  >
    <Server class="h-4 w-4 text-gray-400" />
    <span class="text-gray-300">{{ statusText }}</span>
    <component :is="statusIcon" :class="['h-4 w-4', statusColor]" />
  </router-link>
</template>
