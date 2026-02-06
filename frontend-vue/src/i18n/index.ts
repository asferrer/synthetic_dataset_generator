import { createI18n } from 'vue-i18n'
import en from './locales/en'
import es from './locales/es'

export type SupportedLocale = 'en' | 'es'

export const SUPPORTED_LOCALES: SupportedLocale[] = ['en', 'es']

export const LOCALE_NAMES: Record<SupportedLocale, string> = {
  en: 'English',
  es: 'Español'
}

// Detect initial locale from localStorage or browser
function getInitialLocale(): SupportedLocale {
  // Check localStorage first
  const stored = localStorage.getItem('ui-locale')
  if (stored && SUPPORTED_LOCALES.includes(stored as SupportedLocale)) {
    return stored as SupportedLocale
  }

  // Fall back to browser language
  const browserLang = navigator.language.split('-')[0] as SupportedLocale
  if (SUPPORTED_LOCALES.includes(browserLang)) {
    return browserLang
  }

  return 'en'
}

const i18n = createI18n({
  legacy: false,
  locale: getInitialLocale(),
  fallbackLocale: 'en',
  messages: {
    en,
    es,
  },
  missingWarn: import.meta.env.DEV,
  fallbackWarn: import.meta.env.DEV,
})

export default i18n

// Export global composer for use outside components
export const i18nGlobal = i18n.global
