/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#3B82F6',
          hover: '#60A5FA',
          light: '#1E3A5F',
          50: '#EFF6FF',
          100: '#DBEAFE',
          200: '#BFDBFE',
          300: '#93C5FD',
          400: '#60A5FA',
          500: '#3B82F6',
          600: '#2563EB',
          700: '#1D4ED8',
          800: '#1E40AF',
          900: '#1E3A8A',
        },
        background: {
          primary: '#0F172A',
          secondary: '#1E293B',
          tertiary: '#334155',
          card: '#1E293B',
        },
        surface: {
          DEFAULT: '#1E293B',
          hover: '#334155',
        },
        success: {
          DEFAULT: '#34D399',
          light: '#D1FAE5',
          dark: '#059669',
        },
        warning: {
          DEFAULT: '#FBBF24',
          light: '#FEF3C7',
          dark: '#D97706',
        },
        error: {
          DEFAULT: '#F87171',
          light: '#FEE2E2',
          dark: '#DC2626',
        },
        info: {
          DEFAULT: '#60A5FA',
          light: '#DBEAFE',
          dark: '#2563EB',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Consolas', 'monospace'],
      },
      animation: {
        'spin-slow': 'spin 3s linear infinite',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
    },
  },
  plugins: [],
}
