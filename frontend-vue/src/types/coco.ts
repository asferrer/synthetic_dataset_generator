// COCO Format Types

export interface COCOInfo {
  year?: number
  version?: string
  description?: string
  contributor?: string
  url?: string
  date_created?: string
}

export interface COCOLicense {
  id: number
  name: string
  url?: string
}

export interface COCOCategory {
  id: number
  name: string
  supercategory?: string
}

export interface COCOImage {
  id: number
  file_name: string
  width: number
  height: number
  date_captured?: string
  license?: number
  coco_url?: string
  flickr_url?: string
}

export interface COCOAnnotation {
  id: number
  image_id: number
  category_id: number
  bbox: [number, number, number, number]  // [x, y, width, height]
  area: number
  segmentation?: number[][] | { counts: number[]; size: [number, number] }
  iscrowd: 0 | 1
  attributes?: Record<string, unknown>
}

export interface COCODataset {
  info?: COCOInfo
  licenses?: COCOLicense[]
  categories: COCOCategory[]
  images: COCOImage[]
  annotations: COCOAnnotation[]
}

// Analysis types derived from COCO
export interface CategoryStats {
  id: number
  name: string
  count: number
  percentage: number
  avgArea: number
  minArea: number
  maxArea: number
}

export interface DatasetStats {
  totalImages: number
  totalAnnotations: number
  totalCategories: number
  annotationsPerImage: {
    min: number
    max: number
    mean: number
    median: number
  }
  categoryCounts: Record<string, number>
  categoryStats: CategoryStats[]
}
