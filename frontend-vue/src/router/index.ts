import { createRouter, createWebHistory, RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'home',
    component: () => import('@/views/HomeView.vue'),
    meta: { title: 'Dashboard' },
  },
  // Workflow routes
  {
    path: '/analysis',
    name: 'analysis',
    component: () => import('@/views/workflow/AnalysisView.vue'),
    meta: { title: 'Dataset Analysis', step: 1 },
  },
  {
    path: '/configure',
    name: 'configure',
    component: () => import('@/views/workflow/ConfigureView.vue'),
    meta: { title: 'Configure Effects', step: 2 },
  },
  {
    path: '/source-selection',
    name: 'source-selection',
    component: () => import('@/views/workflow/SourceSelectionView.vue'),
    meta: { title: 'Source Selection', step: 3 },
  },
  {
    path: '/generation',
    name: 'generation',
    component: () => import('@/views/workflow/GenerationView.vue'),
    meta: { title: 'Generation', step: 4 },
  },
  {
    path: '/export',
    name: 'export',
    component: () => import('@/views/workflow/ExportView.vue'),
    meta: { title: 'Export', step: 5 },
  },
  {
    path: '/combine',
    name: 'combine',
    component: () => import('@/views/workflow/CombineView.vue'),
    meta: { title: 'Combine Datasets', step: 6 },
  },
  {
    path: '/splits',
    name: 'splits',
    component: () => import('@/views/workflow/SplitsView.vue'),
    meta: { title: 'Dataset Splits', step: 7 },
  },
  // Domain routes
  {
    path: '/domains',
    name: 'domains',
    component: () => import('@/views/domain/DomainManagerView.vue'),
    meta: { title: 'Domain Manager' },
  },
  {
    path: '/domains/:id',
    name: 'domain-detail',
    component: () => import('@/views/domain/DomainDetailView.vue'),
    meta: { title: 'Domain Details' },
  },
  {
    path: '/domains/:id/edit',
    name: 'domain-edit',
    component: () => import('@/views/domain/DomainEditorView.vue'),
    meta: { title: 'Edit Domain' },
  },
  // Tools routes
  {
    path: '/tools/job-monitor',
    name: 'job-monitor',
    component: () => import('@/views/tools/JobMonitorView.vue'),
    meta: { title: 'Job Monitor' },
  },
  {
    path: '/tools/service-status',
    name: 'service-status',
    component: () => import('@/views/tools/ServiceStatusView.vue'),
    meta: { title: 'Service Status' },
  },
  {
    path: '/tools/object-extraction',
    name: 'object-extraction',
    component: () => import('@/views/tools/ObjectExtractionView.vue'),
    meta: { title: 'Object Extraction' },
  },
  {
    path: '/tools/labeling',
    name: 'labeling',
    component: () => import('@/views/tools/LabelingView.vue'),
    meta: { title: 'Auto Labeling' },
  },
  {
    path: '/tools/sam-segmentation',
    name: 'sam-segmentation',
    component: () => import('@/views/tools/SamSegmentationView.vue'),
    meta: { title: 'SAM Segmentation' },
  },
  {
    path: '/tools/label-manager',
    name: 'label-manager',
    component: () => import('@/views/tools/LabelManagerView.vue'),
    meta: { title: 'Label Manager' },
  },
  {
    path: '/tools/object-sizes',
    name: 'object-sizes',
    component: () => import('@/views/tools/ObjectSizesView.vue'),
    meta: { title: 'Object Sizes' },
  },
  {
    path: '/tools/post-processing',
    name: 'post-processing',
    component: () => import('@/views/tools/PostProcessingView.vue'),
    meta: { title: 'Post-Processing' },
  },
  // Catch-all 404
  {
    path: '/:pathMatch(.*)*',
    name: 'not-found',
    component: () => import('@/views/NotFoundView.vue'),
    meta: { title: 'Page Not Found' },
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

// Update document title on navigation
router.beforeEach((to, from, next) => {
  const title = to.meta.title as string | undefined
  document.title = title ? `${title} | Synthetic Dataset Generator` : 'Synthetic Dataset Generator'
  next()
})

export default router
