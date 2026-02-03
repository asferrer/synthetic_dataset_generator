import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export interface Toast {
  id: string
  type: 'success' | 'error' | 'warning' | 'info'
  title: string
  message?: string
  duration?: number
}

export interface Notification {
  id: string
  type: 'success' | 'error' | 'warning' | 'info'
  title: string
  message?: string
  timestamp: Date
  read: boolean
}

export const useUiStore = defineStore('ui', () => {
  // State
  const sidebarCollapsed = ref(false)
  const sidebarMobileOpen = ref(false)
  const toasts = ref<Toast[]>([])
  const notifications = ref<Notification[]>([])
  const notificationPanelOpen = ref(false)
  const isLoading = ref(false)
  const loadingMessage = ref('')

  // Getters
  const hasToasts = computed(() => toasts.value.length > 0)
  const unreadCount = computed(() => notifications.value.filter(n => !n.read).length)
  const hasUnread = computed(() => unreadCount.value > 0)

  // Actions
  function toggleSidebar() {
    sidebarCollapsed.value = !sidebarCollapsed.value
  }

  function setSidebarCollapsed(collapsed: boolean) {
    sidebarCollapsed.value = collapsed
  }

  function toggleMobileSidebar() {
    sidebarMobileOpen.value = !sidebarMobileOpen.value
  }

  function closeMobileSidebar() {
    sidebarMobileOpen.value = false
  }

  function addToast(toast: Omit<Toast, 'id'>) {
    const id = `toast-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    const newToast: Toast = { ...toast, id }
    toasts.value.push(newToast)

    // Also add to notification history
    addNotification({
      type: toast.type,
      title: toast.title,
      message: toast.message,
    })

    // Auto-remove after duration
    const duration = toast.duration ?? 5000
    if (duration > 0) {
      setTimeout(() => {
        removeToast(id)
      }, duration)
    }

    return id
  }

  function removeToast(id: string) {
    const index = toasts.value.findIndex(t => t.id === id)
    if (index > -1) {
      toasts.value.splice(index, 1)
    }
  }

  function clearToasts() {
    toasts.value = []
  }

  function showSuccess(title: string, message?: string) {
    return addToast({ type: 'success', title, message })
  }

  function showError(title: string, message?: string) {
    return addToast({ type: 'error', title, message, duration: 8000 })
  }

  function showWarning(title: string, message?: string) {
    return addToast({ type: 'warning', title, message })
  }

  function showInfo(title: string, message?: string) {
    return addToast({ type: 'info', title, message })
  }

  function setLoading(loading: boolean, message = '') {
    isLoading.value = loading
    loadingMessage.value = message
  }

  // Notification panel actions
  function toggleNotificationPanel() {
    notificationPanelOpen.value = !notificationPanelOpen.value
  }

  function closeNotificationPanel() {
    notificationPanelOpen.value = false
  }

  function addNotification(notification: Omit<Notification, 'id' | 'timestamp' | 'read'>) {
    const id = `notif-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    notifications.value.unshift({
      ...notification,
      id,
      timestamp: new Date(),
      read: false,
    })
    // Keep only last 50 notifications
    if (notifications.value.length > 50) {
      notifications.value = notifications.value.slice(0, 50)
    }
  }

  function markNotificationRead(id: string) {
    const notification = notifications.value.find(n => n.id === id)
    if (notification) {
      notification.read = true
    }
  }

  function markAllNotificationsRead() {
    notifications.value.forEach(n => { n.read = true })
  }

  function clearNotifications() {
    notifications.value = []
  }

  function removeNotification(id: string) {
    const index = notifications.value.findIndex(n => n.id === id)
    if (index > -1) {
      notifications.value.splice(index, 1)
    }
  }

  return {
    // State
    sidebarCollapsed,
    sidebarMobileOpen,
    toasts,
    notifications,
    notificationPanelOpen,
    isLoading,
    loadingMessage,
    // Getters
    hasToasts,
    unreadCount,
    hasUnread,
    // Actions
    toggleSidebar,
    setSidebarCollapsed,
    toggleMobileSidebar,
    closeMobileSidebar,
    addToast,
    removeToast,
    clearToasts,
    showSuccess,
    showError,
    showWarning,
    showInfo,
    setLoading,
    // Notification actions
    toggleNotificationPanel,
    closeNotificationPanel,
    addNotification,
    markNotificationRead,
    markAllNotificationsRead,
    clearNotifications,
    removeNotification,
  }
})
