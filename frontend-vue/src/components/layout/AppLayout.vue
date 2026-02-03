<script setup lang="ts">
import { useUiStore } from '@/stores/ui'
import Sidebar from './Sidebar.vue'
import Header from './Header.vue'

const uiStore = useUiStore()
</script>

<template>
  <div class="flex h-screen bg-background-primary">
    <!-- Sidebar -->
    <Sidebar />

    <!-- Main content area -->
    <div class="flex flex-1 flex-col overflow-hidden">
      <!-- Header -->
      <Header />

      <!-- Page content -->
      <main
        class="flex-1 overflow-auto p-6"
        :class="{ 'lg:ml-64': !uiStore.sidebarCollapsed, 'lg:ml-20': uiStore.sidebarCollapsed }"
      >
        <div class="mx-auto max-w-7xl">
          <slot />
        </div>
      </main>
    </div>

    <!-- Mobile sidebar overlay -->
    <transition name="fade">
      <div
        v-if="uiStore.sidebarMobileOpen"
        class="fixed inset-0 z-40 bg-black/50 lg:hidden"
        @click="uiStore.closeMobileSidebar"
      />
    </transition>
  </div>
</template>
