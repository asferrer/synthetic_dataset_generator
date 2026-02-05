/**
 * Composable for managing mount points / Docker volumes
 *
 * Provides access to available mount points for directory selection,
 * with filtering by purpose (input/output) and path validation.
 */

import { ref, computed, onMounted } from 'vue'
import { getMountPoints, type MountPoint, type MountPointsResponse } from '@/lib/api'

export function useMountPoints() {
  const mountPoints = ref<MountPoint[]>([])
  const defaultInput = ref('/data')
  const defaultOutput = ref('/data/output')
  const loading = ref(false)
  const error = ref<string | null>(null)
  const loaded = ref(false)

  /**
   * Mount points filtered for input operations
   */
  const inputMountPoints = computed(() =>
    mountPoints.value.filter(mp => mp.purpose === 'input' || mp.purpose === 'both')
  )

  /**
   * Mount points filtered for output operations
   */
  const outputMountPoints = computed(() =>
    mountPoints.value.filter(mp => mp.purpose === 'output' || mp.purpose === 'both')
  )

  /**
   * Mount points that exist on the filesystem
   */
  const existingMountPoints = computed(() =>
    mountPoints.value.filter(mp => mp.exists)
  )

  /**
   * Load mount points from the API
   */
  async function loadMountPoints(): Promise<void> {
    if (loading.value) return

    loading.value = true
    error.value = null

    try {
      const response = await getMountPoints()
      mountPoints.value = response.mount_points
      defaultInput.value = response.default_input
      defaultOutput.value = response.default_output
      loaded.value = true
    } catch (e: any) {
      error.value = e.message || 'Failed to load mount points'
      console.error('Failed to load mount points:', e)
    } finally {
      loading.value = false
    }
  }

  /**
   * Check if a path is within an allowed mount point
   */
  function isPathValid(path: string, mode: 'input' | 'output' | 'both' = 'both'): boolean {
    if (!path) return false

    const relevantMounts = mode === 'both'
      ? mountPoints.value
      : mode === 'input'
        ? inputMountPoints.value
        : outputMountPoints.value

    return relevantMounts.some(mp =>
      path === mp.path || path.startsWith(mp.path + '/')
    )
  }

  /**
   * Get the mount point that contains the given path
   */
  function getMountPointForPath(path: string): MountPoint | undefined {
    if (!path) return undefined

    // Find the most specific mount point (longest matching path)
    return mountPoints.value
      .filter(mp => path === mp.path || path.startsWith(mp.path + '/'))
      .sort((a, b) => b.path.length - a.path.length)[0]
  }

  /**
   * Get filtered mount points based on purpose
   */
  function getFilteredMountPoints(mode: 'input' | 'output' | 'both' = 'both'): MountPoint[] {
    switch (mode) {
      case 'input':
        return inputMountPoints.value
      case 'output':
        return outputMountPoints.value
      default:
        return mountPoints.value
    }
  }

  /**
   * Get the default path for a given mode
   */
  function getDefaultPath(mode: 'input' | 'output' | 'both' = 'both'): string {
    if (mode === 'output') return defaultOutput.value
    return defaultInput.value
  }

  // Auto-load mount points on first use
  onMounted(() => {
    if (!loaded.value) {
      loadMountPoints()
    }
  })

  return {
    // State
    mountPoints,
    inputMountPoints,
    outputMountPoints,
    existingMountPoints,
    defaultInput,
    defaultOutput,
    loading,
    error,
    loaded,

    // Methods
    loadMountPoints,
    isPathValid,
    getMountPointForPath,
    getFilteredMountPoints,
    getDefaultPath,
  }
}
