<script setup lang="ts">
import { useRouter } from 'vue-router'
import { Home, ArrowLeft } from 'lucide-vue-next'

const router = useRouter()
</script>

<template>
  <div class="flex min-h-[60vh] flex-col items-center justify-center text-center">
    <div class="text-8xl font-bold text-gray-700">404</div>
    <h1 class="mt-4 text-2xl font-semibold text-white">Page Not Found</h1>
    <p class="mt-2 text-gray-400">
      The page you're looking for doesn't exist or has been moved.
    </p>
    <div class="mt-8 flex gap-4">
      <button @click="router.back()" class="btn-outline">
        <ArrowLeft class="h-5 w-5" />
        Go Back
      </button>
      <button @click="router.push('/')" class="btn-primary">
        <Home class="h-5 w-5" />
        Go Home
      </button>
    </div>
  </div>
</template>
