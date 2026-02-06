<script setup lang="ts">
import { useUiStore } from '@/stores/ui'
import { X, CheckCircle, XCircle, AlertTriangle, Info, Trash2, CheckCheck } from 'lucide-vue-next'

const uiStore = useUiStore()

const iconMap = {
  success: CheckCircle,
  error: XCircle,
  warning: AlertTriangle,
  info: Info,
}

const iconColorMap = {
  success: 'text-green-400',
  error: 'text-red-400',
  warning: 'text-yellow-400',
  info: 'text-blue-400',
}

const bgColorMap = {
  success: 'bg-green-900/30',
  error: 'bg-red-900/30',
  warning: 'bg-yellow-900/30',
  info: 'bg-blue-900/30',
}

function formatTime(date: Date | string): string {
  const now = new Date()
  const dateObj = typeof date === 'string' ? new Date(date) : date
  const diff = now.getTime() - dateObj.getTime()
  const minutes = Math.floor(diff / 60000)
  const hours = Math.floor(diff / 3600000)
  const days = Math.floor(diff / 86400000)

  if (minutes < 1) return 'Just now'
  if (minutes < 60) return `${minutes}m ago`
  if (hours < 24) return `${hours}h ago`
  return `${days}d ago`
}

function handleNotificationClick(id: string) {
  uiStore.markNotificationRead(id)
}
</script>

<template>
  <Transition name="panel">
    <div
      v-if="uiStore.notificationPanelOpen"
      class="absolute right-0 top-full mt-2 w-96 max-h-[500px] overflow-hidden rounded-lg border border-gray-700 bg-background-secondary shadow-xl"
    >
      <!-- Header -->
      <div class="flex items-center justify-between border-b border-gray-700 p-4">
        <h3 class="font-semibold text-white">Notifications</h3>
        <div class="flex items-center gap-2">
          <button
            v-if="uiStore.hasUnread"
            @click="uiStore.markAllNotificationsRead()"
            class="rounded-lg p-1.5 text-gray-400 hover:bg-gray-700 hover:text-white"
            title="Mark all as read"
          >
            <CheckCheck class="h-4 w-4" />
          </button>
          <button
            v-if="uiStore.notifications.length > 0"
            @click="uiStore.clearNotifications()"
            class="rounded-lg p-1.5 text-gray-400 hover:bg-gray-700 hover:text-white"
            title="Clear all"
          >
            <Trash2 class="h-4 w-4" />
          </button>
          <button
            @click="uiStore.closeNotificationPanel()"
            class="rounded-lg p-1.5 text-gray-400 hover:bg-gray-700 hover:text-white"
          >
            <X class="h-4 w-4" />
          </button>
        </div>
      </div>

      <!-- Notification List -->
      <div class="max-h-[400px] overflow-y-auto">
        <div
          v-if="uiStore.notifications.length === 0"
          class="flex flex-col items-center justify-center py-12 text-gray-400"
        >
          <Info class="h-12 w-12 mb-3 opacity-50" />
          <p>No notifications yet</p>
        </div>

        <div v-else class="divide-y divide-gray-700/50">
          <div
            v-for="notification in uiStore.notifications"
            :key="notification.id"
            @click="handleNotificationClick(notification.id)"
            :class="[
              'flex gap-3 p-4 cursor-pointer transition-colors hover:bg-gray-700/30',
              !notification.read ? bgColorMap[notification.type] : '',
            ]"
          >
            <component
              :is="iconMap[notification.type]"
              :class="['h-5 w-5 flex-shrink-0 mt-0.5', iconColorMap[notification.type]]"
            />
            <div class="flex-1 min-w-0">
              <div class="flex items-start justify-between gap-2">
                <p :class="['font-medium truncate', notification.read ? 'text-gray-300' : 'text-white']">
                  {{ notification.title }}
                </p>
                <button
                  @click.stop="uiStore.removeNotification(notification.id)"
                  class="rounded p-1 text-gray-500 hover:bg-gray-600 hover:text-white flex-shrink-0"
                >
                  <X class="h-3 w-3" />
                </button>
              </div>
              <p v-if="notification.message" class="mt-1 text-sm text-gray-400 line-clamp-2">
                {{ notification.message }}
              </p>
              <p class="mt-1 text-xs text-gray-500">
                {{ formatTime(notification.timestamp) }}
              </p>
            </div>
            <!-- Unread indicator -->
            <div
              v-if="!notification.read"
              class="h-2 w-2 rounded-full bg-primary flex-shrink-0 mt-2"
            />
          </div>
        </div>
      </div>
    </div>
  </Transition>
</template>

<style scoped>
.panel-enter-active,
.panel-leave-active {
  transition: all 0.2s ease;
}

.panel-enter-from,
.panel-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}

.line-clamp-2 {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
</style>
