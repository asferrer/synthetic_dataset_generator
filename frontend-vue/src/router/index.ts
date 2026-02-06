import { createRouter, createWebHistory, RouteRecordRaw } from 'vue-router'
import { i18nGlobal } from '@/i18n'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'home',
    component: () => import('@/views/HomeView.vue'),
    meta: { titleKey: 'nav.titles.dashboard' },
  },
  // Workflow routes
  {
    path: '/analysis',
    name: 'analysis',
    component: () => import('@/views/workflow/AnalysisView.vue'),
    meta: { titleKey: 'nav.titles.analysis', step: 1 },
  },
  {
    path: '/configure',
    name: 'configure',
    component: () => import('@/views/workflow/ConfigureView.vue'),
    meta: { titleKey: 'nav.titles.configure', step: 2 },
  },
  {
    path: '/source-selection',
    name: 'source-selection',
    component: () => import('@/views/workflow/SourceSelectionView.vue'),
    meta: { titleKey: 'nav.titles.sourceSelection', step: 3 },
  },
  {
    path: '/generation',
    name: 'generation',
    component: () => import('@/views/workflow/GenerationView.vue'),
    meta: { titleKey: 'nav.titles.generation', step: 4 },
  },
  {
    path: '/export',
    name: 'export',
    component: () => import('@/views/workflow/ExportView.vue'),
    meta: { titleKey: 'nav.titles.export', step: 5 },
  },
  {
    path: '/combine',
    name: 'combine',
    component: () => import('@/views/workflow/CombineView.vue'),
    meta: { titleKey: 'nav.titles.combine', step: 6 },
  },
  {
    path: '/splits',
    name: 'splits',
    component: () => import('@/views/workflow/SplitsView.vue'),
    meta: { titleKey: 'nav.titles.splits', step: 7 },
  },
  // Domain routes
  {
    path: '/domains',
    name: 'domains',
    component: () => import('@/views/domain/DomainManagerView.vue'),
    meta: { titleKey: 'nav.titles.domains' },
  },
  {
    path: '/domains/:id',
    name: 'domain-detail',
    component: () => import('@/views/domain/DomainDetailView.vue'),
    meta: { titleKey: 'nav.titles.domainDetail' },
  },
  {
    path: '/domains/:id/edit',
    name: 'domain-edit',
    component: () => import('@/views/domain/DomainEditorView.vue'),
    meta: { titleKey: 'nav.titles.domainEdit' },
  },
  // Tools routes
  {
    path: '/tools/job-monitor',
    name: 'job-monitor',
    component: () => import('@/views/tools/JobMonitorView.vue'),
    meta: { titleKey: 'nav.titles.jobMonitor' },
  },
  {
    path: '/tools/service-status',
    name: 'service-status',
    component: () => import('@/views/tools/ServiceStatusView.vue'),
    meta: { titleKey: 'nav.titles.serviceStatus' },
  },
  {
    path: '/tools/object-extraction',
    name: 'object-extraction',
    component: () => import('@/views/tools/ObjectExtractionView.vue'),
    meta: { titleKey: 'nav.titles.objectExtraction' },
  },
  {
    path: '/tools/labeling',
    name: 'labeling',
    component: () => import('@/views/tools/LabelingView.vue'),
    meta: { titleKey: 'nav.titles.autoLabeling' },
  },
  {
    path: '/tools/sam-segmentation',
    name: 'sam-segmentation',
    component: () => import('@/views/tools/SamSegmentationView.vue'),
    meta: { titleKey: 'nav.titles.samSegmentation' },
  },
  {
    path: '/tools/label-manager',
    name: 'label-manager',
    component: () => import('@/views/tools/LabelManagerView.vue'),
    meta: { titleKey: 'nav.titles.labelManager' },
  },
  {
    path: '/tools/object-sizes',
    name: 'object-sizes',
    component: () => import('@/views/tools/ObjectSizesView.vue'),
    meta: { titleKey: 'nav.titles.objectSizes' },
  },
  {
    path: '/tools/post-processing',
    name: 'post-processing',
    component: () => import('@/views/tools/PostProcessingView.vue'),
    meta: { titleKey: 'nav.titles.postProcessing' },
  },
  // Catch-all 404
  {
    path: '/:pathMatch(.*)*',
    name: 'not-found',
    component: () => import('@/views/NotFoundView.vue'),
    meta: { titleKey: 'nav.titles.notFound' },
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior(to, from, savedPosition) {
    if (savedPosition) {
      return savedPosition
    }
    return { top: 0 }
  },
})

// Update document title on navigation with i18n
router.beforeEach((to, from, next) => {
  const titleKey = to.meta.titleKey as string | undefined
  const appName = i18nGlobal.t('common.app.name')

  if (titleKey) {
    const translatedTitle = i18nGlobal.t(titleKey)
    document.title = `${translatedTitle} | ${appName}`
  } else {
    document.title = appName
  }

  next()
})

export default router

// Augment route meta types
declare module 'vue-router' {
  interface RouteMeta {
    titleKey?: string
    title?: string
    step?: number
  }
}
