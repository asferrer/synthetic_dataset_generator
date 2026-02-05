<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { useUiStore } from '@/stores/ui'
import { Menu, Bell, Search } from 'lucide-vue-next'
import ServiceStatusBadge from '@/components/common/ServiceStatusBadge.vue'
import NotificationPanel from '@/components/common/NotificationPanel.vue'
import SearchModal from '@/components/common/SearchModal.vue'
import LanguageSwitcher from '@/components/common/LanguageSwitcher.vue'

const route = useRoute()
const uiStore = useUiStore()
const { t } = useI18n()
const searchOpen = ref(false)

const pageTitle = computed(() => {
  // Try to get translated title from route meta
  const titleKey = route.meta.titleKey as string | undefined
  if (titleKey) {
    const translated = t(titleKey)
    if (translated !== titleKey) {
      return translated
    }
  }
  return (route.meta.title as string) || t('common.app.name')
})

// Close panel when clicking outside
function handleClickOutside(event: MouseEvent) {
  const target = event.target as HTMLElement
  if (!target.closest('.notification-container')) {
    uiStore.closeNotificationPanel()
  }
}

// Global keyboard shortcut for search
function handleKeyDown(event: KeyboardEvent) {
  if ((event.ctrlKey || event.metaKey) && event.key === 'k') {
    event.preventDefault()
    searchOpen.value = !searchOpen.value
  }
}

onMounted(() => {
  document.addEventListener('click', handleClickOutside)
  document.addEventListener('keydown', handleKeyDown)
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
  document.removeEventListener('keydown', handleKeyDown)
})
</script>

<template>
  <header
    class="sticky top-0 z-30 flex h-16 items-center justify-between border-b border-gray-700/50 bg-background-secondary/80 px-6 backdrop-blur-md"
    :class="{ 'lg:ml-64': !uiStore.sidebarCollapsed, 'lg:ml-20': uiStore.sidebarCollapsed }"
  >
    <!-- Left side -->
    <div class="flex items-center gap-4">
      <!-- Mobile menu button -->
      <button
        @click="uiStore.toggleMobileSidebar"
        class="rounded-lg p-2 text-gray-400 hover:bg-gray-700/50 hover:text-white lg:hidden"
      >
        <Menu class="h-6 w-6" />
      </button>

      <!-- Page title -->
      <h1 class="text-xl font-semibold text-white">
        {{ pageTitle }}
      </h1>
    </div>

    <!-- Right side -->
    <div class="flex items-center gap-4">
      <!-- Service status badge -->
      <ServiceStatusBadge />

      <!-- Language switcher -->
      <LanguageSwitcher />

      <!-- Search button -->
      <button
        @click="searchOpen = true"
        class="hidden items-center gap-2 rounded-lg bg-background-tertiary px-3 py-1.5 text-sm text-gray-400 transition-colors hover:bg-gray-600 hover:text-white sm:flex"
      >
        <Search class="h-4 w-4" />
        <span>{{ t('common.actions.search') }}</span>
        <kbd class="rounded bg-gray-700 px-1.5 py-0.5 text-xs font-mono">Ctrl+K</kbd>
      </button>

      <!-- Search Modal -->
      <SearchModal :open="searchOpen" @close="searchOpen = false" />

      <!-- Notifications -->
      <div class="relative notification-container">
        <button
          @click.stop="uiStore.toggleNotificationPanel()"
          class="relative rounded-lg p-2 text-gray-400 hover:bg-gray-700/50 hover:text-white"
          :class="{ 'bg-gray-700/50 text-white': uiStore.notificationPanelOpen }"
        >
          <Bell class="h-5 w-5" />
          <span
            v-if="uiStore.hasUnread"
            class="absolute right-1 top-1 flex h-4 w-4 items-center justify-center rounded-full bg-primary text-[10px] font-bold text-white"
          >
            {{ uiStore.unreadCount > 9 ? '9+' : uiStore.unreadCount }}
          </span>
        </button>
        <NotificationPanel />
      </div>
    </div>
  </header>
</template>
