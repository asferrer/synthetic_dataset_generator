import { createRouter, createWebHistory, RouteRecordRaw } from 'vue-router'
import { i18nGlobal } from '@/i18n'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'home',
    component: () => import('@/views/HomeView.vue'),
    meta: { titleKey: 'nav.titles.dashboard' },
  },
  // Workflow routes (4 steps)
  {
    path: '/prepare',
    name: 'prepare',
    component: () => import('@/views/workflow/PrepareView.vue'),
    meta: { titleKey: 'nav.titles.prepare', step: 1 },
  },
  {
    path: '/configure',
    name: 'configure',
    component: () => import('@/views/workflow/ConfigureView.vue'),
    meta: { titleKey: 'nav.titles.configure', step: 2 },
  },
  {
    path: '/generate',
    name: 'generate',
    component: () => import('@/views/workflow/GenerationView.vue'),
    meta: { titleKey: 'nav.titles.generate', step: 3 },
  },
  {
    path: '/export',
    name: 'export',
    component: () => import('@/views/workflow/ExportSplitView.vue'),
    meta: { titleKey: 'nav.titles.export', step: 4 },
  },
  // Workflow redirects (backward compatibility)
  { path: '/analysis', redirect: '/prepare?tab=analyze' },
  { path: '/source-selection', redirect: '/prepare?tab=sources' },
  { path: '/generation', redirect: '/generate' },
  { path: '/combine', redirect: '/export?tab=combine' },
  { path: '/splits', redirect: '/export?tab=split' },
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
  // Tools routes (2 tools + system)
  {
    path: '/tools/object-extraction',
    name: 'object-extraction',
    component: () => import('@/views/tools/ObjectExtractionView.vue'),
    meta: { titleKey: 'nav.titles.objectExtraction' },
  },
  {
    path: '/tools/domain-gap',
    name: 'domain-gap',
    component: () => import('@/views/tools/DomainGapView.vue'),
    meta: { titleKey: 'nav.titles.domainGap' },
  },
  {
    path: '/tools/system',
    name: 'system',
    component: () => import('@/views/tools/SystemView.vue'),
    meta: { titleKey: 'nav.titles.system' },
  },
  // Tools redirects (backward compatibility)
  { path: '/tools/job-monitor', redirect: '/tools/system?tab=jobs' },
  { path: '/tools/service-status', redirect: '/tools/system?tab=services' },
  { path: '/tools/object-sizes', redirect: '/configure?tab=sizes' },
  { path: '/tools/post-processing', redirect: '/export?tab=balance' },
  // Labeling tools removed - redirect to home
  { path: '/tools/labeling', redirect: '/' },
  { path: '/tools/sam-segmentation', redirect: '/' },
  { path: '/tools/label-manager', redirect: '/' },
  { path: '/tools/annotation-review', redirect: '/' },
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
