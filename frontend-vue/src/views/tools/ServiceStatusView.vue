<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { getHealthStatus } from '@/lib/api'
import BaseButton from '@/components/ui/BaseButton.vue'
import AlertBox from '@/components/common/AlertBox.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import {
  Server,
  CheckCircle,
  XCircle,
  AlertTriangle,
  RefreshCw,
  Wifi,
  Clock,
  Activity,
} from 'lucide-vue-next'
import type { HealthStatus, ServiceHealth } from '@/types/api'

const loading = ref(true)
const health = ref<HealthStatus | null>(null)
const error = ref<string | null>(null)
const lastChecked = ref<Date | null>(null)
let pollingInterval: ReturnType<typeof setInterval> | null = null

// Service metadata for display
const serviceInfo: Record<string, { description: string; port: number }> = {
  gateway: { description: 'Main API router and orchestrator', port: 8000 },
  depth: { description: 'Depth estimation for realistic placement', port: 8001 },
  segmentation: { description: 'SAM-based object segmentation', port: 8002 },
  effects: { description: 'Image augmentation and effects', port: 8003 },
  augmentor: { description: 'Dataset augmentation pipeline', port: 8004 },
}

async function checkHealth() {
  loading.value = true
  error.value = null

  try {
    health.value = await getHealthStatus()
    lastChecked.value = new Date()
  } catch (e: any) {
    error.value = e.message || 'Failed to check service health'
  } finally {
    loading.value = false
  }
}

function getStatusIcon(status?: string) {
  switch (status) {
    case 'healthy': return CheckCircle
    case 'degraded': return AlertTriangle
    default: return XCircle
  }
}

function getStatusColor(status?: string) {
  switch (status) {
    case 'healthy': return 'text-green-400'
    case 'degraded': return 'text-yellow-400'
    default: return 'text-red-400'
  }
}

function getStatusBg(status?: string) {
  switch (status) {
    case 'healthy': return 'bg-green-900/20 border-green-700/50'
    case 'degraded': return 'bg-yellow-900/20 border-yellow-700/50'
    default: return 'bg-red-900/20 border-red-700/50'
  }
}

function formatLatency(ms?: number) {
  if (!ms) return 'N/A'
  return `${ms.toFixed(0)}ms`
}

function getServiceDescription(name: string): string {
  return serviceInfo[name]?.description || 'Microservice'
}

function getServicePort(name: string): number {
  return serviceInfo[name]?.port || 0
}

function formatServiceName(name: string): string {
  return name.split('_').map(word =>
    word.charAt(0).toUpperCase() + word.slice(1)
  ).join(' ')
}

// Computed counts for summary
const healthyCount = computed(() =>
  health.value?.services?.filter(s => s.status === 'healthy').length || 0
)

const degradedCount = computed(() =>
  health.value?.services?.filter(s => s.status === 'degraded').length || 0
)

const offlineCount = computed(() =>
  health.value?.services?.filter(s => s.status === 'unhealthy').length || 0
)

onMounted(() => {
  checkHealth()
  pollingInterval = setInterval(checkHealth, 30000)
})

onUnmounted(() => {
  if (pollingInterval) clearInterval(pollingInterval)
})
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h2 class="text-2xl font-bold text-white">Service Status</h2>
        <p class="mt-1 text-gray-400">
          Monitor the health of all backend services.
          <span v-if="lastChecked" class="text-gray-500">
            Last checked: {{ lastChecked.toLocaleTimeString() }}
          </span>
        </p>
      </div>
      <BaseButton variant="outline" @click="checkHealth" :disabled="loading">
        <RefreshCw :class="['h-5 w-5', loading ? 'animate-spin' : '']" />
        Refresh
      </BaseButton>
    </div>

    <!-- Overall Status Banner -->
    <div v-if="health" :class="[
      'card border p-4 flex items-center gap-4',
      getStatusBg(health.status)
    ]">
      <component
        :is="getStatusIcon(health.status)"
        :class="['h-8 w-8', getStatusColor(health.status)]"
      />
      <div>
        <p class="font-semibold text-white">
          System Status: <span :class="getStatusColor(health.status)" class="capitalize">{{ health.status }}</span>
        </p>
        <p class="text-sm text-gray-400">
          {{ health.all_healthy ? 'All services operational' : 'Some services may be experiencing issues' }}
        </p>
      </div>
    </div>

    <!-- Error -->
    <AlertBox v-if="error" type="error" title="Connection Error">
      {{ error }}
      <template #action>
        <BaseButton size="sm" @click="checkHealth">Retry</BaseButton>
      </template>
    </AlertBox>

    <!-- Loading -->
    <div v-if="loading && !health" class="flex justify-center py-12">
      <LoadingSpinner size="lg" message="Checking services..." />
    </div>

    <!-- Services Grid -->
    <div v-else-if="health?.services" class="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
      <div
        v-for="service in health.services"
        :key="service.name"
        :class="[
          'card border p-6 transition-all',
          getStatusBg(service.status),
        ]"
      >
        <div class="flex items-start justify-between mb-4">
          <div class="flex items-center gap-3">
            <div class="rounded-lg bg-gray-700/50 p-3">
              <Server class="h-6 w-6 text-gray-300" />
            </div>
            <div>
              <h3 class="font-semibold text-white">{{ formatServiceName(service.name) }}</h3>
              <p class="text-sm text-gray-400">Port {{ getServicePort(service.name) }}</p>
            </div>
          </div>
          <component
            :is="getStatusIcon(service.status)"
            :class="['h-6 w-6', getStatusColor(service.status)]"
          />
        </div>

        <p class="text-sm text-gray-400 mb-4">{{ getServiceDescription(service.name) }}</p>

        <div class="flex items-center justify-between text-sm">
          <div class="flex items-center gap-2">
            <Wifi class="h-4 w-4 text-gray-500" />
            <span class="text-gray-400">Status:</span>
            <span
              :class="[
                'font-medium capitalize',
                getStatusColor(service.status),
              ]"
            >
              {{ service.status || 'offline' }}
            </span>
          </div>
          <div class="flex items-center gap-2">
            <Clock class="h-4 w-4 text-gray-500" />
            <span class="text-gray-400">
              {{ formatLatency(service.latency_ms) }}
            </span>
          </div>
        </div>

        <!-- URL if available -->
        <p v-if="service.url" class="mt-2 text-xs text-gray-500 font-mono truncate">
          {{ service.url }}
        </p>

        <!-- Error message if any -->
        <p
          v-if="service.error"
          class="mt-3 text-sm text-red-400 bg-red-900/20 rounded p-2"
        >
          {{ service.error }}
        </p>
      </div>
    </div>

    <!-- Fallback if no services -->
    <div v-else-if="!loading" class="text-center py-12">
      <Activity class="h-12 w-12 mx-auto mb-4 text-gray-500" />
      <p class="text-gray-400">No service information available</p>
      <BaseButton variant="outline" class="mt-4" @click="checkHealth">
        Retry Connection
      </BaseButton>
    </div>

    <!-- Overall Status Summary -->
    <div v-if="health?.services" class="card p-6">
      <h3 class="text-lg font-semibold text-white mb-4">System Summary</h3>
      <div class="grid gap-4 sm:grid-cols-3">
        <div class="text-center p-4 rounded-lg bg-green-900/20">
          <p class="text-3xl font-bold text-green-400">{{ healthyCount }}</p>
          <p class="text-sm text-gray-400">Healthy</p>
        </div>
        <div class="text-center p-4 rounded-lg bg-yellow-900/20">
          <p class="text-3xl font-bold text-yellow-400">{{ degradedCount }}</p>
          <p class="text-sm text-gray-400">Degraded</p>
        </div>
        <div class="text-center p-4 rounded-lg bg-red-900/20">
          <p class="text-3xl font-bold text-red-400">{{ offlineCount }}</p>
          <p class="text-sm text-gray-400">Offline</p>
        </div>
      </div>
    </div>
  </div>
</template>
