<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { useUiStore } from '@/stores/ui'
import {
  LayoutDashboard,
  Globe,
  FolderInput,
  Settings,
  Sparkles,
  Download,
  Scissors,
  ScanSearch,
  Activity,
  ChevronLeft,
  ChevronRight,
} from 'lucide-vue-next'

const route = useRoute()
const uiStore = useUiStore()
const { t } = useI18n()

interface NavItem {
  nameKey: string
  path: string
  icon: any
  step?: number
}

const workflowItems: NavItem[] = [
  { nameKey: 'nav.workflow.prepare', path: '/prepare', icon: FolderInput, step: 1 },
  { nameKey: 'nav.workflow.configure', path: '/configure', icon: Settings, step: 2 },
  { nameKey: 'nav.workflow.generate', path: '/generate', icon: Sparkles, step: 3 },
  { nameKey: 'nav.workflow.export', path: '/export', icon: Download, step: 4 },
]

const toolItems: NavItem[] = [
  { nameKey: 'nav.tools.objectExtraction', path: '/tools/object-extraction', icon: Scissors },
  { nameKey: 'nav.tools.domainGap', path: '/tools/domain-gap', icon: ScanSearch },
]

const systemItems: NavItem[] = [
  { nameKey: 'nav.tools.system', path: '/tools/system', icon: Activity },
]

const isActive = (path: string) => route.path === path || route.path.startsWith(path + '?')

const sidebarClasses = computed(() => [
  'fixed inset-y-0 left-0 z-50 flex flex-col bg-background-secondary border-r border-gray-700/50',
  'transition-all duration-300 ease-in-out',
  uiStore.sidebarCollapsed ? 'w-20' : 'w-64',
  uiStore.sidebarMobileOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0',
])
</script>

<template>
  <aside :class="sidebarClasses">
    <!-- Logo -->
    <div class="flex h-16 items-center justify-between border-b border-gray-700/50 px-4">
      <router-link to="/" class="flex items-center gap-3">
        <div class="flex h-10 w-10 items-center justify-center rounded-xl bg-primary">
          <Sparkles class="h-6 w-6 text-white" />
        </div>
        <span v-if="!uiStore.sidebarCollapsed" class="font-semibold text-white">
          {{ t('common.app.shortName') }}
        </span>
      </router-link>
    </div>

    <!-- Navigation -->
    <nav class="flex-1 overflow-y-auto px-3 py-4">
      <!-- Dashboard -->
      <div class="mb-6">
        <router-link
          to="/"
          :class="[
            'flex items-center gap-3 rounded-lg px-3 py-2.5 transition-colors',
            isActive('/') && route.path === '/' ? 'bg-primary text-white' : 'text-gray-400 hover:bg-gray-700/50 hover:text-white',
          ]"
        >
          <LayoutDashboard class="h-5 w-5 flex-shrink-0" />
          <span v-if="!uiStore.sidebarCollapsed">{{ t('nav.sidebar.dashboard') }}</span>
        </router-link>
      </div>

      <!-- Domain Manager -->
      <div class="mb-6">
        <router-link
          to="/domains"
          :class="[
            'flex items-center gap-3 rounded-lg px-3 py-2.5 transition-colors',
            route.path.startsWith('/domains') ? 'bg-primary text-white' : 'text-gray-400 hover:bg-gray-700/50 hover:text-white',
          ]"
        >
          <Globe class="h-5 w-5 flex-shrink-0" />
          <span v-if="!uiStore.sidebarCollapsed">{{ t('nav.sidebar.domains') }}</span>
        </router-link>
      </div>

      <!-- Workflow Section -->
      <div class="mb-6">
        <h3
          v-if="!uiStore.sidebarCollapsed"
          class="mb-2 px-3 text-xs font-semibold uppercase tracking-wider text-gray-500"
        >
          {{ t('nav.sidebar.workflow') }}
        </h3>
        <div class="space-y-1">
          <router-link
            v-for="item in workflowItems"
            :key="item.path"
            :to="item.path"
            :class="[
              'flex items-center gap-3 rounded-lg px-3 py-2.5 transition-colors',
              isActive(item.path) ? 'bg-primary text-white' : 'text-gray-400 hover:bg-gray-700/50 hover:text-white',
            ]"
          >
            <div class="relative flex-shrink-0">
              <component :is="item.icon" class="h-5 w-5" />
              <span
                v-if="item.step && !uiStore.sidebarCollapsed"
                class="absolute -right-1 -top-1 flex h-4 w-4 items-center justify-center rounded-full bg-gray-600 text-[10px] font-medium text-white"
              >
                {{ item.step }}
              </span>
            </div>
            <span v-if="!uiStore.sidebarCollapsed">{{ t(item.nameKey) }}</span>
          </router-link>
        </div>
      </div>

      <!-- Tools Section -->
      <div class="mb-6">
        <h3
          v-if="!uiStore.sidebarCollapsed"
          class="mb-2 px-3 text-xs font-semibold uppercase tracking-wider text-gray-500"
        >
          {{ t('nav.sidebar.tools') }}
        </h3>
        <div class="space-y-1">
          <router-link
            v-for="item in toolItems"
            :key="item.path"
            :to="item.path"
            :class="[
              'flex items-center gap-3 rounded-lg px-3 py-2.5 transition-colors',
              isActive(item.path) ? 'bg-primary text-white' : 'text-gray-400 hover:bg-gray-700/50 hover:text-white',
            ]"
          >
            <component :is="item.icon" class="h-5 w-5 flex-shrink-0" />
            <span v-if="!uiStore.sidebarCollapsed">{{ t(item.nameKey) }}</span>
          </router-link>
        </div>
      </div>

      <!-- System Section -->
      <div>
        <h3
          v-if="!uiStore.sidebarCollapsed"
          class="mb-2 px-3 text-xs font-semibold uppercase tracking-wider text-gray-500"
        >
          {{ t('nav.sidebar.system') }}
        </h3>
        <div class="space-y-1">
          <router-link
            v-for="item in systemItems"
            :key="item.path"
            :to="item.path"
            :class="[
              'flex items-center gap-3 rounded-lg px-3 py-2.5 transition-colors',
              isActive(item.path) ? 'bg-primary text-white' : 'text-gray-400 hover:bg-gray-700/50 hover:text-white',
            ]"
          >
            <component :is="item.icon" class="h-5 w-5 flex-shrink-0" />
            <span v-if="!uiStore.sidebarCollapsed">{{ t(item.nameKey) }}</span>
          </router-link>
        </div>
      </div>
    </nav>

    <!-- Collapse toggle -->
    <div class="border-t border-gray-700/50 p-3">
      <button
        @click="uiStore.toggleSidebar"
        class="flex w-full items-center justify-center gap-2 rounded-lg px-3 py-2 text-gray-400 hover:bg-gray-700/50 hover:text-white transition-colors"
      >
        <ChevronLeft v-if="!uiStore.sidebarCollapsed" class="h-5 w-5" />
        <ChevronRight v-else class="h-5 w-5" />
        <span v-if="!uiStore.sidebarCollapsed">{{ t('common.actions.collapse') }}</span>
      </button>
    </div>
  </aside>
</template>
