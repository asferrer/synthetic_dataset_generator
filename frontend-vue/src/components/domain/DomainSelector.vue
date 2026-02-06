<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { useDomainStore } from '@/stores/domain'
import {
  ChevronDown,
  Check,
  Settings,
  Waves,
  Flame,
  Bird,
  Box,
  TreePine,
  Building2,
  Cloud,
  Droplets,
  Fish,
  Wind,
} from 'lucide-vue-next'

const props = defineProps<{
  compact?: boolean
  showManageLink?: boolean
}>()

const emit = defineEmits<{
  (e: 'change', domainId: string): void
}>()

const router = useRouter()
const domainStore = useDomainStore()
const isOpen = ref(false)

// Icon mapping
const iconComponents: Record<string, any> = {
  Waves, Flame, Bird, Box, TreePine, Building2, Cloud, Droplets, Fish, Wind,
}

function getIconComponent(iconName: string) {
  return iconComponents[iconName] || Box
}

async function selectDomain(domainId: string) {
  if (domainStore.activeDomainId === domainId) {
    isOpen.value = false
    return
  }

  const success = await domainStore.setActiveDomain(domainId)
  if (success) {
    emit('change', domainId)
  }
  isOpen.value = false
}

function handleClickOutside(event: MouseEvent) {
  const target = event.target as HTMLElement
  if (!target.closest('.domain-selector')) {
    isOpen.value = false
  }
}

onMounted(() => {
  document.addEventListener('click', handleClickOutside)
  if (domainStore.domains.length === 0) {
    domainStore.fetchDomains()
  }
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
})
</script>

<template>
  <div class="domain-selector relative">
    <!-- Trigger Button -->
    <button
      @click.stop="isOpen = !isOpen"
      :class="[
        'flex items-center gap-3 rounded-lg transition-colors w-full',
        compact
          ? 'px-3 py-2 bg-background-tertiary hover:bg-gray-600'
          : 'px-4 py-3 bg-background-tertiary hover:bg-gray-600'
      ]"
    >
      <div
        v-if="domainStore.activeDomain"
        :class="[compact ? 'p-1.5' : 'p-2', 'rounded-lg bg-primary/20']"
      >
        <component
          :is="getIconComponent(domainStore.activeDomain.icon)"
          :class="[compact ? 'h-4 w-4' : 'h-5 w-5', 'text-primary']"
        />
      </div>
      <div v-else class="p-2 rounded-lg bg-gray-600">
        <Box :class="[compact ? 'h-4 w-4' : 'h-5 w-5', 'text-gray-400']" />
      </div>

      <div class="flex-1 text-left min-w-0">
        <p v-if="!compact" class="text-xs text-gray-500">Active Domain</p>
        <p :class="[compact ? 'text-sm' : '', 'font-medium text-white truncate']">
          {{ domainStore.activeDomain?.name || 'Select Domain' }}
        </p>
      </div>

      <ChevronDown
        :class="[
          'h-4 w-4 text-gray-400 transition-transform',
          isOpen && 'rotate-180'
        ]"
      />
    </button>

    <!-- Dropdown -->
    <Transition
      enter-active-class="transition ease-out duration-100"
      enter-from-class="transform opacity-0 scale-95"
      enter-to-class="transform opacity-100 scale-100"
      leave-active-class="transition ease-in duration-75"
      leave-from-class="transform opacity-100 scale-100"
      leave-to-class="transform opacity-0 scale-95"
    >
      <div
        v-if="isOpen"
        class="absolute top-full left-0 right-0 mt-2 z-50 py-2 bg-background-secondary border border-gray-700 rounded-xl shadow-xl max-h-80 overflow-y-auto"
      >
        <!-- Built-in Domains -->
        <div v-if="domainStore.builtinDomains.length > 0">
          <p class="px-3 py-1.5 text-xs font-medium text-gray-500 uppercase">Built-in</p>
          <button
            v-for="domain in domainStore.builtinDomains"
            :key="domain.domain_id"
            @click="selectDomain(domain.domain_id)"
            :class="[
              'w-full flex items-center gap-3 px-3 py-2 transition-colors',
              domainStore.activeDomainId === domain.domain_id
                ? 'bg-primary/10 text-primary'
                : 'text-gray-300 hover:bg-gray-700'
            ]"
          >
            <component :is="getIconComponent(domain.icon)" class="h-5 w-5 flex-shrink-0" />
            <span class="flex-1 text-left text-sm">{{ domain.name }}</span>
            <Check
              v-if="domainStore.activeDomainId === domain.domain_id"
              class="h-4 w-4"
            />
          </button>
        </div>

        <!-- User Domains -->
        <div v-if="domainStore.userDomains.length > 0">
          <p class="px-3 py-1.5 text-xs font-medium text-gray-500 uppercase mt-2">Custom</p>
          <button
            v-for="domain in domainStore.userDomains"
            :key="domain.domain_id"
            @click="selectDomain(domain.domain_id)"
            :class="[
              'w-full flex items-center gap-3 px-3 py-2 transition-colors',
              domainStore.activeDomainId === domain.domain_id
                ? 'bg-primary/10 text-primary'
                : 'text-gray-300 hover:bg-gray-700'
            ]"
          >
            <component :is="getIconComponent(domain.icon)" class="h-5 w-5 flex-shrink-0" />
            <span class="flex-1 text-left text-sm">{{ domain.name }}</span>
            <Check
              v-if="domainStore.activeDomainId === domain.domain_id"
              class="h-4 w-4"
            />
          </button>
        </div>

        <!-- Manage Link -->
        <div v-if="showManageLink" class="border-t border-gray-700 mt-2 pt-2">
          <button
            @click="router.push('/domains'); isOpen = false"
            class="w-full flex items-center gap-3 px-3 py-2 text-gray-400 hover:text-white hover:bg-gray-700 transition-colors"
          >
            <Settings class="h-5 w-5" />
            <span class="text-sm">Manage Domains</span>
          </button>
        </div>
      </div>
    </Transition>
  </div>
</template>
