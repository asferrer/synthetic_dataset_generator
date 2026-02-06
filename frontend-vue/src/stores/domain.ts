import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type {
  Domain,
  DomainSummary,
  DomainCreateRequest,
  DomainUpdateRequest,
  DomainRegion,
  DomainObject,
  DomainPreset,
  DomainLabelingTemplate,
} from '@/types/api'
import {
  listDomains,
  getDomain,
  getActiveDomain,
  activateDomain as activateDomainApi,
  createDomain as createDomainApi,
  updateDomain as updateDomainApi,
  deleteDomain as deleteDomainApi,
  exportDomain as exportDomainApi,
  importDomain as importDomainApi,
  checkCompatibility as checkCompatibilityApi,
} from '@/lib/api'

// ============================================
// STORE DEFINITION
// ============================================

export const useDomainStore = defineStore('domain', () => {
  // ============================================
  // STATE
  // ============================================

  // List of all available domains (summary info)
  const domains = ref<DomainSummary[]>([])

  // Currently active domain (full data)
  const activeDomain = ref<Domain | null>(null)

  // Active domain ID (persisted)
  const activeDomainId = ref<string | null>(null)

  // Loading states
  const isLoading = ref(false)
  const isLoadingDomain = ref(false)
  const error = ref<string | null>(null)

  // Cache for full domain data
  const domainCache = ref<Record<string, Domain>>({})

  // ============================================
  // GETTERS
  // ============================================

  const hasActiveDomain = computed(() => activeDomain.value !== null)

  const builtinDomains = computed(() =>
    domains.value.filter(d => d.is_builtin)
  )

  const userDomains = computed(() =>
    domains.value.filter(d => !d.is_builtin)
  )

  const activeRegions = computed(() =>
    activeDomain.value?.regions || []
  )

  const activeObjects = computed(() =>
    activeDomain.value?.objects || []
  )

  const activeEffects = computed(() =>
    activeDomain.value?.effects || null
  )

  const activePhysics = computed(() =>
    activeDomain.value?.physics || null
  )

  const activePresets = computed(() =>
    activeDomain.value?.presets || []
  )

  const activeLabelingTemplates = computed(() =>
    activeDomain.value?.labeling_templates || []
  )

  const activeCompatibilityMatrix = computed(() =>
    activeDomain.value?.compatibility_matrix || {}
  )

  // Get domain icon component name based on icon string
  const getDomainIcon = computed(() => (iconName: string) => {
    const iconMap: Record<string, string> = {
      'Waves': 'Waves',
      'Flame': 'Flame',
      'Bird': 'Bird',
      'Box': 'Box',
      'TreePine': 'TreePine',
      'Building2': 'Building2',
      'Cloud': 'Cloud',
      'Droplets': 'Droplets',
      'Fish': 'Fish',
      'Wind': 'Wind',
    }
    return iconMap[iconName] || 'Box'
  })

  // ============================================
  // ACTIONS
  // ============================================

  /**
   * Fetch all available domains from the API
   */
  async function fetchDomains() {
    isLoading.value = true
    error.value = null
    try {
      domains.value = await listDomains()

      // If we have an active domain ID but no active domain loaded, load it
      if (activeDomainId.value && !activeDomain.value) {
        await loadActiveDomain()
      }
    } catch (e: any) {
      error.value = e.message || 'Failed to fetch domains'
      console.error('Error fetching domains:', e)
    } finally {
      isLoading.value = false
    }
  }

  /**
   * Load the currently active domain from the API
   */
  async function loadActiveDomain() {
    isLoadingDomain.value = true
    error.value = null
    try {
      const response = await getActiveDomain()
      activeDomain.value = response.domain
      activeDomainId.value = response.active_domain_id
      // Cache it
      domainCache.value[response.active_domain_id] = response.domain
    } catch (e: any) {
      // If no active domain, try to activate the first available
      if (domains.value.length > 0 && !activeDomainId.value) {
        await setActiveDomain(domains.value[0].domain_id)
      } else {
        error.value = e.message || 'Failed to load active domain'
      }
    } finally {
      isLoadingDomain.value = false
    }
  }

  /**
   * Get a specific domain by ID (uses cache if available)
   */
  async function fetchDomain(domainId: string): Promise<Domain | null> {
    // Check cache first
    if (domainCache.value[domainId]) {
      return domainCache.value[domainId]
    }

    isLoadingDomain.value = true
    error.value = null
    try {
      const domain = await getDomain(domainId)
      domainCache.value[domainId] = domain
      return domain
    } catch (e: any) {
      error.value = e.message || `Failed to fetch domain: ${domainId}`
      return null
    } finally {
      isLoadingDomain.value = false
    }
  }

  /**
   * Set the active domain
   */
  async function setActiveDomain(domainId: string) {
    isLoadingDomain.value = true
    error.value = null
    try {
      await activateDomainApi(domainId)

      // Load the full domain data
      const domain = await fetchDomain(domainId)
      if (domain) {
        activeDomain.value = domain
        activeDomainId.value = domainId
      }
      return true
    } catch (e: any) {
      error.value = e.message || `Failed to activate domain: ${domainId}`
      return false
    } finally {
      isLoadingDomain.value = false
    }
  }

  /**
   * Create a new domain
   */
  async function createDomain(request: DomainCreateRequest): Promise<Domain | null> {
    isLoading.value = true
    error.value = null
    try {
      const domain = await createDomainApi(request)
      // Refresh the domains list
      await fetchDomains()
      // Cache the new domain
      domainCache.value[domain.domain_id] = domain
      return domain
    } catch (e: any) {
      error.value = e.message || 'Failed to create domain'
      return null
    } finally {
      isLoading.value = false
    }
  }

  /**
   * Update an existing domain
   */
  async function updateDomain(domainId: string, request: DomainUpdateRequest): Promise<Domain | null> {
    isLoading.value = true
    error.value = null
    try {
      const domain = await updateDomainApi(domainId, request)
      // Update cache
      domainCache.value[domainId] = domain
      // Refresh domains list
      await fetchDomains()
      // If this is the active domain, update it
      if (activeDomainId.value === domainId) {
        activeDomain.value = domain
      }
      return domain
    } catch (e: any) {
      error.value = e.message || `Failed to update domain: ${domainId}`
      return null
    } finally {
      isLoading.value = false
    }
  }

  /**
   * Delete a domain
   */
  async function deleteDomain(domainId: string): Promise<boolean> {
    isLoading.value = true
    error.value = null
    try {
      await deleteDomainApi(domainId)
      // Remove from cache
      delete domainCache.value[domainId]
      // Refresh domains list
      await fetchDomains()
      // If this was the active domain, clear it
      if (activeDomainId.value === domainId) {
        activeDomain.value = null
        activeDomainId.value = null
        // Try to activate another domain
        if (domains.value.length > 0) {
          await setActiveDomain(domains.value[0].domain_id)
        }
      }
      return true
    } catch (e: any) {
      error.value = e.message || `Failed to delete domain: ${domainId}`
      return false
    } finally {
      isLoading.value = false
    }
  }

  /**
   * Export a domain as JSON
   */
  async function exportDomain(domainId: string): Promise<Domain | null> {
    try {
      return await exportDomainApi(domainId)
    } catch (e: any) {
      error.value = e.message || `Failed to export domain: ${domainId}`
      return null
    }
  }

  /**
   * Import a domain from JSON
   */
  async function importDomain(domainData: Domain, overwrite = false): Promise<Domain | null> {
    isLoading.value = true
    error.value = null
    try {
      const domain = await importDomainApi(domainData, overwrite)
      // Refresh domains list
      await fetchDomains()
      // Cache the imported domain
      domainCache.value[domain.domain_id] = domain
      return domain
    } catch (e: any) {
      error.value = e.message || 'Failed to import domain'
      return null
    } finally {
      isLoading.value = false
    }
  }

  /**
   * Check compatibility score for object-region pair
   */
  async function checkCompatibility(objectClass: string, regionId: string, domainId?: string): Promise<number> {
    try {
      const response = await checkCompatibilityApi({
        object_class: objectClass,
        region_id: regionId,
        domain_id: domainId || activeDomainId.value || undefined,
      })
      return response.score
    } catch {
      // Return default score on error
      return 0.5
    }
  }

  /**
   * Get compatibility score from cache (no API call)
   */
  function getCompatibilityScore(objectClass: string, regionId: string): number {
    const matrix = activeCompatibilityMatrix.value
    if (matrix[objectClass] && typeof matrix[objectClass][regionId] === 'number') {
      return matrix[objectClass][regionId]
    }
    return 0.5 // Default score
  }

  /**
   * Get regions compatible with an object class
   */
  function getCompatibleRegions(objectClass: string, minScore = 0.5): DomainRegion[] {
    const matrix = activeCompatibilityMatrix.value
    const objectScores = matrix[objectClass] || {}

    return activeRegions.value.filter(region =>
      (objectScores[region.id] || 0) >= minScore
    )
  }

  /**
   * Get objects compatible with a region
   */
  function getCompatibleObjects(regionId: string, minScore = 0.5): DomainObject[] {
    const matrix = activeCompatibilityMatrix.value

    return activeObjects.value.filter(obj => {
      const score = matrix[obj.class_name]?.[regionId] || 0
      return score >= minScore
    })
  }

  /**
   * Get preset by ID
   */
  function getPreset(presetId: string): DomainPreset | undefined {
    return activePresets.value.find(p => p.id === presetId)
  }

  /**
   * Get labeling template by ID
   */
  function getLabelingTemplate(templateId: string): DomainLabelingTemplate | undefined {
    return activeLabelingTemplates.value.find(t => t.id === templateId)
  }

  /**
   * Clear cache and refresh
   */
  async function refresh() {
    domainCache.value = {}
    await fetchDomains()
    if (activeDomainId.value) {
      await loadActiveDomain()
    }
  }

  /**
   * Reset store to initial state
   */
  function reset() {
    domains.value = []
    activeDomain.value = null
    activeDomainId.value = null
    domainCache.value = {}
    isLoading.value = false
    isLoadingDomain.value = false
    error.value = null
  }

  // ============================================
  // RETURN
  // ============================================

  return {
    // State
    domains,
    activeDomain,
    activeDomainId,
    isLoading,
    isLoadingDomain,
    error,
    domainCache,

    // Getters
    hasActiveDomain,
    builtinDomains,
    userDomains,
    activeRegions,
    activeObjects,
    activeEffects,
    activePhysics,
    activePresets,
    activeLabelingTemplates,
    activeCompatibilityMatrix,
    getDomainIcon,

    // Actions
    fetchDomains,
    loadActiveDomain,
    fetchDomain,
    setActiveDomain,
    createDomain,
    updateDomain,
    deleteDomain,
    exportDomain,
    importDomain,
    checkCompatibility,
    getCompatibilityScore,
    getCompatibleRegions,
    getCompatibleObjects,
    getPreset,
    getLabelingTemplate,
    refresh,
    reset,
  }
}, {
  persist: {
    // Only persist the active domain ID, not the full data
    pick: ['activeDomainId'],
  },
})
