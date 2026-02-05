<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useUiStore } from '@/stores/ui'
import { SUPPORTED_LOCALES, LOCALE_NAMES, type SupportedLocale } from '@/i18n'
import { Globe, Check } from 'lucide-vue-next'

const { locale } = useI18n()
const uiStore = useUiStore()

const isOpen = ref(false)

import { ref, onMounted, onUnmounted } from 'vue'

const currentLocale = computed(() => locale.value as SupportedLocale)

function selectLocale(newLocale: SupportedLocale) {
  locale.value = newLocale
  uiStore.setLocale(newLocale)
  isOpen.value = false
}

function toggleDropdown() {
  isOpen.value = !isOpen.value
}

function handleClickOutside(event: MouseEvent) {
  const target = event.target as HTMLElement
  if (!target.closest('.language-switcher')) {
    isOpen.value = false
  }
}

onMounted(() => {
  document.addEventListener('click', handleClickOutside)
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
})
</script>

<template>
  <div class="relative language-switcher">
    <button
      @click.stop="toggleDropdown"
      class="flex items-center gap-2 rounded-lg bg-background-tertiary px-3 py-1.5 text-sm text-gray-400 transition-colors hover:bg-gray-600 hover:text-white"
    >
      <Globe class="h-4 w-4" />
      <span class="hidden sm:inline">{{ LOCALE_NAMES[currentLocale] }}</span>
    </button>

    <Transition
      enter-active-class="transition duration-100 ease-out"
      enter-from-class="transform scale-95 opacity-0"
      enter-to-class="transform scale-100 opacity-100"
      leave-active-class="transition duration-75 ease-in"
      leave-from-class="transform scale-100 opacity-100"
      leave-to-class="transform scale-95 opacity-0"
    >
      <div
        v-if="isOpen"
        class="absolute right-0 z-50 mt-2 w-36 origin-top-right rounded-lg border border-gray-700 bg-background-secondary py-1 shadow-lg"
      >
        <button
          v-for="loc in SUPPORTED_LOCALES"
          :key="loc"
          @click="selectLocale(loc)"
          class="flex w-full items-center justify-between px-3 py-2 text-sm transition-colors"
          :class="[
            currentLocale === loc
              ? 'bg-primary/10 text-primary'
              : 'text-gray-300 hover:bg-gray-700/50 hover:text-white'
          ]"
        >
          <span>{{ LOCALE_NAMES[loc] }}</span>
          <Check v-if="currentLocale === loc" class="h-4 w-4" />
        </button>
      </div>
    </Transition>
  </div>
</template>
