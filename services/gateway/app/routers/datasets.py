"""
Datasets Router
===============
Endpoints for dataset management, analysis, and file system operations.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Datasets"])

# Default data directories - can be overridden via environment variables
DATA_DIR = os.environ.get("DATA_DIR", "/data")
DATASETS_DIR = os.environ.get("DATASETS_DIR", os.path.join(DATA_DIR, "datasets"))
UPLOADS_DIR = os.environ.get("UPLOADS_DIR", os.path.join(DATA_DIR, "uploads"))


# =============================================================================
# Request/Response Models
# =============================================================================

class CategoryInfo(BaseModel):
    """Category information with statistics."""
    id: int
    name: str
    count: int


class AnnotationsPerImage(BaseModel):
    """Statistics about annotations per image."""
    mean: float
    median: float
    min: int
    max: int
    std: float


class DatasetAnalysis(BaseModel):
    """Complete dataset analysis result."""
    path: str
    total_images: int
    total_annotations: int
    categories: List[CategoryInfo]
    annotations_per_image: AnnotationsPerImage
    has_segmentation: bool = False
    has_keypoints: bool = False


class AnalyzeRequest(BaseModel):
    """Request to analyze a dataset."""
    path: str = Field(..., description="Path to COCO JSON file")


class UploadResponse(BaseModel):
    """Response from dataset upload."""
    path: str
    filename: str
    size: int


# =============================================================================
# Dataset Analysis Endpoints
# =============================================================================

@router.post("/datasets/analyze", response_model=DatasetAnalysis)
async def analyze_dataset(request: AnalyzeRequest):
    """
    Analyze a COCO format dataset.

    Returns statistics about images, annotations, and categories.
    """
    logger.info(f"Analyzing dataset: {request.path}")

    try:
        # Check if file exists
        if not os.path.exists(request.path):
            raise HTTPException(status_code=404, detail=f"Dataset not found: {request.path}")

        # Load COCO JSON
        with open(request.path, 'r') as f:
            coco_data = json.load(f)

        # Extract basic counts
        images = coco_data.get("images", [])
        annotations = coco_data.get("annotations", [])
        categories = coco_data.get("categories", [])

        total_images = len(images)
        total_annotations = len(annotations)

        # Count annotations per category
        category_counts: Dict[int, int] = {}
        for ann in annotations:
            cat_id = ann.get("category_id")
            if cat_id is not None:
                category_counts[cat_id] = category_counts.get(cat_id, 0) + 1

        # Build category info list
        category_info = []
        for cat in categories:
            cat_id = cat.get("id")
            cat_name = cat.get("name", f"category_{cat_id}")
            count = category_counts.get(cat_id, 0)
            category_info.append(CategoryInfo(id=cat_id, name=cat_name, count=count))

        # Calculate annotations per image statistics
        image_ann_counts: Dict[int, int] = {}
        for ann in annotations:
            img_id = ann.get("image_id")
            if img_id is not None:
                image_ann_counts[img_id] = image_ann_counts.get(img_id, 0) + 1

        # Include images with 0 annotations
        for img in images:
            img_id = img.get("id")
            if img_id not in image_ann_counts:
                image_ann_counts[img_id] = 0

        counts = list(image_ann_counts.values()) if image_ann_counts else [0]

        import statistics
        mean_ann = statistics.mean(counts) if counts else 0
        median_ann = statistics.median(counts) if counts else 0
        min_ann = min(counts) if counts else 0
        max_ann = max(counts) if counts else 0
        std_ann = statistics.stdev(counts) if len(counts) > 1 else 0

        # Check for segmentation and keypoints
        has_segmentation = any(
            ann.get("segmentation") and len(ann.get("segmentation", [])) > 0
            for ann in annotations[:100]  # Check first 100 for performance
        )
        has_keypoints = any(
            ann.get("keypoints") and len(ann.get("keypoints", [])) > 0
            for ann in annotations[:100]
        )

        return DatasetAnalysis(
            path=request.path,
            total_images=total_images,
            total_annotations=total_annotations,
            categories=category_info,
            annotations_per_image=AnnotationsPerImage(
                mean=mean_ann,
                median=median_ann,
                min=min_ann,
                max=max_ann,
                std=std_ann,
            ),
            has_segmentation=has_segmentation,
            has_keypoints=has_keypoints,
        )

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {e}")
    except Exception as e:
        logger.error(f"Dataset analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasets/upload", response_model=UploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    directory: Optional[str] = Form(None)
):
    """
    Upload a COCO JSON dataset file.

    Returns the path where the file was saved.
    """
    logger.info(f"Uploading dataset: {file.filename}")

    try:
        # Validate file extension
        if not file.filename or not file.filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="File must be a JSON file")

        # Determine target directory
        target_dir = directory if directory else UPLOADS_DIR
        os.makedirs(target_dir, exist_ok=True)

        # Generate unique filename if file exists
        base_name = file.filename
        target_path = os.path.join(target_dir, base_name)

        counter = 1
        while os.path.exists(target_path):
            name, ext = os.path.splitext(base_name)
            target_path = os.path.join(target_dir, f"{name}_{counter}{ext}")
            counter += 1

        # Read and save file
        content = await file.read()

        # Validate JSON format
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON format: {e}")

        # Save file
        with open(target_path, 'wb') as f:
            f.write(content)

        return UploadResponse(
            path=target_path,
            filename=os.path.basename(target_path),
            size=len(content),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# File System Endpoints
# =============================================================================

@router.get("/fs/directories")
async def list_directories(path: Optional[str] = None) -> Dict[str, List[str]]:
    """
    List directories in a given path.

    Returns list of directory paths.
    """
    try:
        base_path = path if path else DATA_DIR

        if not os.path.exists(base_path):
            return {"directories": []}

        if not os.path.isdir(base_path):
            raise HTTPException(status_code=400, detail=f"Path is not a directory: {base_path}")

        directories = []
        for entry in os.listdir(base_path):
            full_path = os.path.join(base_path, entry)
            if os.path.isdir(full_path):
                directories.append(full_path)

        return {"directories": sorted(directories)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"List directories failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fs/files")
async def list_files(
    path: str,
    extension: Optional[str] = None
) -> Dict[str, List[str]]:
    """
    List files in a given path.

    Args:
        path: Directory path to list files from
        extension: Optional file extension filter (e.g., ".json", ".png")

    Returns list of file paths.
    """
    try:
        if not os.path.exists(path):
            return {"files": []}

        if not os.path.isdir(path):
            raise HTTPException(status_code=400, detail=f"Path is not a directory: {path}")

        files = []
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)
            if os.path.isfile(full_path):
                if extension is None or entry.endswith(extension):
                    files.append(full_path)

        return {"files": sorted(files)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"List files failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Dataset Combination and Splitting (Proxied to Augmentor when needed)
# =============================================================================

class CombineRequest(BaseModel):
    """Request to combine multiple datasets."""
    datasets: List[str] = Field(..., min_length=2, description="Paths to COCO JSON files")
    output_path: str = Field(..., description="Output path for combined dataset")
    merge_categories: bool = Field(True, description="Merge categories with same name")
    deduplicate: bool = Field(False, description="Remove duplicate images")


class SplitRequest(BaseModel):
    """Request to split a dataset."""
    dataset_path: str = Field(..., description="Path to COCO JSON file")
    output_dir: str = Field(..., description="Output directory for splits")
    train_ratio: float = Field(0.7, ge=0, le=1)
    val_ratio: float = Field(0.2, ge=0, le=1)
    test_ratio: float = Field(0.1, ge=0, le=1)
    stratified: bool = Field(True, description="Stratified split by category")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class ExportFormat(str):
    """Supported export formats."""
    YOLO = "yolo"
    VOC = "voc"
    COCO = "coco"


class ExportRequest(BaseModel):
    """Request to export a dataset to a specific format."""
    source_path: str = Field(..., description="Path to COCO JSON file")
    images_dir: str = Field(..., description="Directory containing images")
    output_dir: str = Field(..., description="Output directory for exported dataset")
    format: str = Field("yolo", description="Export format: yolo, voc, coco")
    include_images: bool = Field(True, description="Copy images to output directory")
    split_name: Optional[str] = Field(None, description="Optional split name (train, val, test)")


@router.post("/datasets/combine")
async def combine_datasets(request: CombineRequest):
    """
    Combine multiple COCO datasets into one.

    Merges images, annotations, and categories from multiple datasets.
    """
    logger.info(f"Combining {len(request.datasets)} datasets")

    try:
        combined = {
            "info": {"description": "Combined dataset", "version": "1.0"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [],
        }

        image_id_offset = 0
        annotation_id_offset = 0
        category_mapping: Dict[str, int] = {}  # name -> new_id
        next_category_id = 1

        for dataset_path in request.datasets:
            if not os.path.exists(dataset_path):
                raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_path}")

            with open(dataset_path, 'r') as f:
                dataset = json.load(f)

            # Process categories
            old_to_new_cat: Dict[int, int] = {}
            for cat in dataset.get("categories", []):
                cat_name = cat.get("name")
                if request.merge_categories and cat_name in category_mapping:
                    old_to_new_cat[cat["id"]] = category_mapping[cat_name]
                else:
                    new_cat = {**cat, "id": next_category_id}
                    combined["categories"].append(new_cat)
                    old_to_new_cat[cat["id"]] = next_category_id
                    category_mapping[cat_name] = next_category_id
                    next_category_id += 1

            # Process images
            old_to_new_img: Dict[int, int] = {}
            for img in dataset.get("images", []):
                new_img_id = img["id"] + image_id_offset
                old_to_new_img[img["id"]] = new_img_id
                new_img = {**img, "id": new_img_id}
                combined["images"].append(new_img)

            # Process annotations
            for ann in dataset.get("annotations", []):
                new_ann = {
                    **ann,
                    "id": ann["id"] + annotation_id_offset,
                    "image_id": old_to_new_img[ann["image_id"]],
                    "category_id": old_to_new_cat[ann["category_id"]],
                }
                combined["annotations"].append(new_ann)

            # Update offsets
            if dataset.get("images"):
                image_id_offset = max(img["id"] for img in combined["images"]) + 1
            if dataset.get("annotations"):
                annotation_id_offset = max(ann["id"] for ann in combined["annotations"]) + 1

        # Save combined dataset
        os.makedirs(os.path.dirname(request.output_path), exist_ok=True)
        with open(request.output_path, 'w') as f:
            json.dump(combined, f, indent=2)

        return {
            "success": True,
            "output_path": request.output_path,
            "total_images": len(combined["images"]),
            "total_annotations": len(combined["annotations"]),
            "total_categories": len(combined["categories"]),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Combine datasets failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasets/split")
async def split_dataset(request: SplitRequest):
    """
    Split a COCO dataset into train/val/test splits.
    """
    import random

    logger.info(f"Splitting dataset: {request.dataset_path}")

    try:
        # Validate ratios
        total_ratio = request.train_ratio + request.val_ratio + request.test_ratio
        if abs(total_ratio - 1.0) > 0.01:
            raise HTTPException(status_code=400, detail=f"Ratios must sum to 1.0, got {total_ratio}")

        if not os.path.exists(request.dataset_path):
            raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_path}")

        with open(request.dataset_path, 'r') as f:
            dataset = json.load(f)

        images = dataset.get("images", [])
        annotations = dataset.get("annotations", [])
        categories = dataset.get("categories", [])

        if request.seed is not None:
            random.seed(request.seed)

        # Shuffle images
        shuffled_images = images.copy()
        random.shuffle(shuffled_images)

        # Calculate split sizes
        n = len(shuffled_images)
        train_end = int(n * request.train_ratio)
        val_end = train_end + int(n * request.val_ratio)

        train_images = shuffled_images[:train_end]
        val_images = shuffled_images[train_end:val_end]
        test_images = shuffled_images[val_end:]

        # Create annotation lookup
        img_to_anns: Dict[int, List] = {}
        for ann in annotations:
            img_id = ann["image_id"]
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)

        # Create splits
        os.makedirs(request.output_dir, exist_ok=True)

        splits = {
            "train": train_images,
            "val": val_images,
            "test": test_images,
        }

        result = {"success": True, "splits": {}}

        for split_name, split_images in splits.items():
            if len(split_images) == 0:
                continue

            split_img_ids = {img["id"] for img in split_images}
            split_anns = [ann for ann in annotations if ann["image_id"] in split_img_ids]

            split_dataset = {
                "info": dataset.get("info", {}),
                "licenses": dataset.get("licenses", []),
                "images": split_images,
                "annotations": split_anns,
                "categories": categories,
            }

            split_path = os.path.join(request.output_dir, f"{split_name}.json")
            with open(split_path, 'w') as f:
                json.dump(split_dataset, f, indent=2)

            result["splits"][split_name] = {
                "path": split_path,
                "images": len(split_images),
                "annotations": len(split_anns),
            }

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Split dataset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class KFoldRequest(BaseModel):
    """Request for K-Fold cross validation split."""
    dataset_path: str = Field(..., description="Path to COCO JSON file")
    output_dir: str = Field(..., description="Output directory for fold files")
    num_folds: int = Field(5, ge=2, le=10, description="Number of folds (K)")
    val_fold: int = Field(0, ge=0, description="Which fold to use as validation (0-indexed)")
    stratified: bool = Field(True, description="Stratified split by category")
    random_seed: Optional[int] = Field(None, description="Random seed for reproducibility")


@router.post("/datasets/kfold")
async def kfold_split(request: KFoldRequest):
    """
    Create K-Fold cross validation splits from a COCO dataset.

    Divides the dataset into K equal folds. One fold is designated as validation,
    the rest as training. This allows for K different train/val configurations.
    """
    import random

    logger.info(f"Creating {request.num_folds}-fold split: {request.dataset_path}")

    try:
        # Validate val_fold
        if request.val_fold >= request.num_folds:
            raise HTTPException(
                status_code=400,
                detail=f"val_fold ({request.val_fold}) must be less than num_folds ({request.num_folds})"
            )

        if not os.path.exists(request.dataset_path):
            raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_path}")

        with open(request.dataset_path, 'r') as f:
            dataset = json.load(f)

        images = dataset.get("images", [])
        annotations = dataset.get("annotations", [])
        categories = dataset.get("categories", [])

        if len(images) < request.num_folds:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset has only {len(images)} images, need at least {request.num_folds} for {request.num_folds}-fold split"
            )

        if request.random_seed is not None:
            random.seed(request.random_seed)

        # Shuffle images
        shuffled_images = images.copy()
        random.shuffle(shuffled_images)

        # Divide into K folds
        fold_size = len(shuffled_images) // request.num_folds
        remainder = len(shuffled_images) % request.num_folds

        folds = []
        start_idx = 0
        for i in range(request.num_folds):
            # Distribute remainder across first folds
            size = fold_size + (1 if i < remainder else 0)
            folds.append(shuffled_images[start_idx:start_idx + size])
            start_idx += size

        # Create annotation lookup
        img_to_anns: Dict[int, List] = {}
        for ann in annotations:
            img_id = ann["image_id"]
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)

        # Create output directory
        os.makedirs(request.output_dir, exist_ok=True)

        # Save each fold
        result_folds = []
        for fold_idx, fold_images in enumerate(folds):
            fold_img_ids = {img["id"] for img in fold_images}
            fold_anns = [ann for ann in annotations if ann["image_id"] in fold_img_ids]

            fold_dataset = {
                "info": {
                    **dataset.get("info", {}),
                    "fold": fold_idx + 1,
                    "total_folds": request.num_folds,
                },
                "licenses": dataset.get("licenses", []),
                "images": fold_images,
                "annotations": fold_anns,
                "categories": categories,
            }

            fold_path = os.path.join(request.output_dir, f"fold_{fold_idx + 1}.json")
            with open(fold_path, 'w') as f:
                json.dump(fold_dataset, f, indent=2)

            result_folds.append({
                "images": len(fold_images),
                "annotations": len(fold_anns),
                "path": fold_path,
            })

        # Also create combined train/val files for the selected validation fold
        val_images = folds[request.val_fold]
        train_images = []
        for i, fold in enumerate(folds):
            if i != request.val_fold:
                train_images.extend(fold)

        val_img_ids = {img["id"] for img in val_images}
        train_img_ids = {img["id"] for img in train_images}

        val_anns = [ann for ann in annotations if ann["image_id"] in val_img_ids]
        train_anns = [ann for ann in annotations if ann["image_id"] in train_img_ids]

        # Save train split
        train_dataset = {
            "info": {
                **dataset.get("info", {}),
                "split": "train",
                "val_fold": request.val_fold + 1,
            },
            "licenses": dataset.get("licenses", []),
            "images": train_images,
            "annotations": train_anns,
            "categories": categories,
        }
        train_path = os.path.join(request.output_dir, "train.json")
        with open(train_path, 'w') as f:
            json.dump(train_dataset, f, indent=2)

        # Save val split
        val_dataset = {
            "info": {
                **dataset.get("info", {}),
                "split": "val",
                "val_fold": request.val_fold + 1,
            },
            "licenses": dataset.get("licenses", []),
            "images": val_images,
            "annotations": val_anns,
            "categories": categories,
        }
        val_path = os.path.join(request.output_dir, "val.json")
        with open(val_path, 'w') as f:
            json.dump(val_dataset, f, indent=2)

        return {
            "success": True,
            "num_folds": request.num_folds,
            "val_fold": request.val_fold,
            "output_dir": request.output_dir,
            "folds": result_folds,
            "train_path": train_path,
            "val_path": val_path,
            "train_images": len(train_images),
            "val_images": len(val_images),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"K-Fold split failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Dataset Export Endpoints
# =============================================================================

@router.post("/datasets/export")
async def export_dataset(request: ExportRequest):
    """
    Export a COCO dataset to another format (YOLO, VOC, etc.).
    """
    import shutil

    logger.info(f"Exporting dataset to {request.format}: {request.source_path}")

    try:
        if not os.path.exists(request.source_path):
            raise HTTPException(status_code=404, detail=f"Dataset not found: {request.source_path}")

        with open(request.source_path, 'r') as f:
            coco_data = json.load(f)

        images = coco_data.get("images", [])
        annotations = coco_data.get("annotations", [])
        categories = coco_data.get("categories", [])

        # Create output directory
        os.makedirs(request.output_dir, exist_ok=True)

        if request.format.lower() == "yolo":
            return await _export_to_yolo(
                images, annotations, categories,
                request.images_dir, request.output_dir,
                request.include_images, request.split_name
            )
        elif request.format.lower() == "voc":
            return await _export_to_voc(
                images, annotations, categories,
                request.images_dir, request.output_dir,
                request.include_images
            )
        elif request.format.lower() == "coco":
            # Just copy the COCO JSON
            output_path = os.path.join(request.output_dir, "annotations.json")
            shutil.copy2(request.source_path, output_path)

            if request.include_images:
                images_out = os.path.join(request.output_dir, "images")
                os.makedirs(images_out, exist_ok=True)
                for img in images:
                    src = os.path.join(request.images_dir, img["file_name"])
                    if os.path.exists(src):
                        shutil.copy2(src, images_out)

            return {
                "success": True,
                "format": "coco",
                "output_dir": request.output_dir,
                "images_exported": len(images),
                "annotations_exported": len(annotations),
            }
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export dataset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _export_to_yolo(
    images: List[Dict],
    annotations: List[Dict],
    categories: List[Dict],
    images_dir: str,
    output_dir: str,
    include_images: bool,
    split_name: Optional[str] = None
) -> Dict:
    """Export to YOLO format (txt files with normalized coordinates)."""
    import shutil

    # Create category mapping (YOLO uses 0-indexed classes)
    cat_id_to_idx = {cat["id"]: idx for idx, cat in enumerate(categories)}

    # Create YOLO directory structure
    if split_name:
        labels_dir = os.path.join(output_dir, "labels", split_name)
        images_out_dir = os.path.join(output_dir, "images", split_name)
    else:
        labels_dir = os.path.join(output_dir, "labels")
        images_out_dir = os.path.join(output_dir, "images")

    os.makedirs(labels_dir, exist_ok=True)
    if include_images:
        os.makedirs(images_out_dir, exist_ok=True)

    # Create image id to info mapping
    img_id_to_info = {img["id"]: img for img in images}

    # Group annotations by image
    img_to_anns: Dict[int, List] = {}
    for ann in annotations:
        img_id = ann["image_id"]
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    exported_count = 0
    for img in images:
        img_id = img["id"]
        img_width = img.get("width", 1)
        img_height = img.get("height", 1)
        file_name = img["file_name"]
        base_name = os.path.splitext(file_name)[0]

        # Get annotations for this image
        img_anns = img_to_anns.get(img_id, [])

        # Write YOLO format labels
        label_path = os.path.join(labels_dir, f"{base_name}.txt")
        with open(label_path, 'w') as f:
            for ann in img_anns:
                cat_id = ann["category_id"]
                if cat_id not in cat_id_to_idx:
                    continue

                class_idx = cat_id_to_idx[cat_id]

                # Convert COCO bbox [x, y, width, height] to YOLO format
                # YOLO: [class, x_center, y_center, width, height] normalized
                bbox = ann.get("bbox", [0, 0, 0, 0])
                x, y, w, h = bbox

                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                norm_w = w / img_width
                norm_h = h / img_height

                # Clamp values to [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                norm_w = max(0, min(1, norm_w))
                norm_h = max(0, min(1, norm_h))

                f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        # Copy image if requested
        if include_images:
            src_path = os.path.join(images_dir, file_name)
            if os.path.exists(src_path):
                shutil.copy2(src_path, images_out_dir)

        exported_count += 1

    # Write classes.txt
    classes_path = os.path.join(output_dir, "classes.txt")
    with open(classes_path, 'w') as f:
        for cat in categories:
            f.write(f"{cat['name']}\n")

    # Write data.yaml for YOLO training
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(f"# YOLO Dataset Configuration\n")
        f.write(f"path: {output_dir}\n")
        if split_name:
            f.write(f"train: images/{split_name}\n")
            f.write(f"val: images/{split_name}\n")
        else:
            f.write(f"train: images\n")
            f.write(f"val: images\n")
        f.write(f"\n# Classes\n")
        f.write(f"nc: {len(categories)}\n")
        f.write(f"names: [{', '.join(repr(cat['name']) for cat in categories)}]\n")

    return {
        "success": True,
        "format": "yolo",
        "output_dir": output_dir,
        "images_exported": exported_count,
        "annotations_exported": len(annotations),
        "classes_file": classes_path,
        "yaml_file": yaml_path,
    }


async def _export_to_voc(
    images: List[Dict],
    annotations: List[Dict],
    categories: List[Dict],
    images_dir: str,
    output_dir: str,
    include_images: bool
) -> Dict:
    """Export to Pascal VOC format (XML files)."""
    import shutil
    from xml.etree.ElementTree import Element, SubElement, tostring
    from xml.dom import minidom

    # Create VOC directory structure
    annotations_dir = os.path.join(output_dir, "Annotations")
    images_out_dir = os.path.join(output_dir, "JPEGImages")
    imagesets_dir = os.path.join(output_dir, "ImageSets", "Main")

    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(imagesets_dir, exist_ok=True)
    if include_images:
        os.makedirs(images_out_dir, exist_ok=True)

    # Create category mapping
    cat_id_to_name = {cat["id"]: cat["name"] for cat in categories}

    # Group annotations by image
    img_to_anns: Dict[int, List] = {}
    for ann in annotations:
        img_id = ann["image_id"]
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    exported_count = 0
    image_names = []

    for img in images:
        img_id = img["id"]
        file_name = img["file_name"]
        base_name = os.path.splitext(file_name)[0]
        image_names.append(base_name)

        # Create VOC XML annotation
        root = Element("annotation")

        folder = SubElement(root, "folder")
        folder.text = "JPEGImages"

        filename = SubElement(root, "filename")
        filename.text = file_name

        size = SubElement(root, "size")
        width_elem = SubElement(size, "width")
        width_elem.text = str(img.get("width", 0))
        height_elem = SubElement(size, "height")
        height_elem.text = str(img.get("height", 0))
        depth = SubElement(size, "depth")
        depth.text = "3"

        segmented = SubElement(root, "segmented")
        segmented.text = "0"

        # Add objects
        img_anns = img_to_anns.get(img_id, [])
        for ann in img_anns:
            cat_id = ann["category_id"]
            if cat_id not in cat_id_to_name:
                continue

            obj = SubElement(root, "object")

            name = SubElement(obj, "name")
            name.text = cat_id_to_name[cat_id]

            pose = SubElement(obj, "pose")
            pose.text = "Unspecified"

            truncated = SubElement(obj, "truncated")
            truncated.text = "0"

            difficult = SubElement(obj, "difficult")
            difficult.text = "0"

            # Add bounding box
            bbox = ann.get("bbox", [0, 0, 0, 0])
            x, y, w, h = bbox

            bndbox = SubElement(obj, "bndbox")
            xmin = SubElement(bndbox, "xmin")
            xmin.text = str(int(x))
            ymin = SubElement(bndbox, "ymin")
            ymin.text = str(int(y))
            xmax = SubElement(bndbox, "xmax")
            xmax.text = str(int(x + w))
            ymax = SubElement(bndbox, "ymax")
            ymax.text = str(int(y + h))

        # Write XML file
        xml_str = minidom.parseString(tostring(root)).toprettyxml(indent="  ")
        xml_path = os.path.join(annotations_dir, f"{base_name}.xml")
        with open(xml_path, 'w') as f:
            f.write(xml_str)

        # Copy image if requested
        if include_images:
            src_path = os.path.join(images_dir, file_name)
            if os.path.exists(src_path):
                shutil.copy2(src_path, images_out_dir)

        exported_count += 1

    # Write ImageSets file
    imageset_path = os.path.join(imagesets_dir, "trainval.txt")
    with open(imageset_path, 'w') as f:
        for name in image_names:
            f.write(f"{name}\n")

    return {
        "success": True,
        "format": "voc",
        "output_dir": output_dir,
        "images_exported": exported_count,
        "annotations_exported": len(annotations),
        "imageset_file": imageset_path,
    }


# =============================================================================
# Object Size Configuration Endpoints
# =============================================================================

# In-memory storage for object sizes (in production, use database)
_object_sizes: Dict[str, float] = {}


@router.get("/config/object-sizes")
async def get_object_sizes() -> Dict[str, float]:
    """Get all configured object sizes."""
    return _object_sizes


@router.get("/config/object-sizes/{class_name}")
async def get_object_size(class_name: str) -> Dict[str, Any]:
    """Get the size for a specific class."""
    if class_name not in _object_sizes:
        raise HTTPException(status_code=404, detail=f"Class not found: {class_name}")
    return {"class_name": class_name, "size": _object_sizes[class_name]}


@router.put("/config/object-sizes/{class_name}")
async def update_object_size(class_name: str, size: float) -> Dict[str, Any]:
    """Update the size for a specific class."""
    if size < 0.01 or size > 1.0:
        raise HTTPException(status_code=400, detail="Size must be between 0.01 and 1.0")
    _object_sizes[class_name] = size
    return {"success": True, "class_name": class_name, "size": size}


@router.post("/config/object-sizes/batch")
async def update_multiple_object_sizes(sizes: Dict[str, float]) -> Dict[str, Any]:
    """Update multiple object sizes at once."""
    for name, size in sizes.items():
        if size < 0.01 or size > 1.0:
            raise HTTPException(status_code=400, detail=f"Size for {name} must be between 0.01 and 1.0")
        _object_sizes[name] = size
    return {"success": True, "updated": len(sizes)}


@router.delete("/config/object-sizes/{class_name}")
async def delete_object_size(class_name: str) -> Dict[str, Any]:
    """Delete a class from object sizes."""
    if class_name not in _object_sizes:
        raise HTTPException(status_code=404, detail=f"Class not found: {class_name}")
    del _object_sizes[class_name]
    return {"success": True, "deleted": class_name}


# =============================================================================
# Category Management Endpoints
# =============================================================================

class RenameCategoryRequest(BaseModel):
    """Request to rename a category."""
    dataset_path: str = Field(..., description="Path to COCO JSON file")
    category_id: int = Field(..., description="Category ID to rename")
    new_name: str = Field(..., min_length=1, description="New category name")


class DeleteCategoryRequest(BaseModel):
    """Request to delete a category."""
    dataset_path: str = Field(..., description="Path to COCO JSON file")
    category_id: int = Field(..., description="Category ID to delete")
    delete_annotations: bool = Field(True, description="Also delete annotations for this category")


@router.put("/datasets/categories/rename")
async def rename_category(request: RenameCategoryRequest):
    """
    Rename a category in a COCO dataset.

    This modifies the dataset file in place.
    """
    logger.info(f"Renaming category {request.category_id} to '{request.new_name}' in {request.dataset_path}")

    try:
        if not os.path.exists(request.dataset_path):
            raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_path}")

        # Load dataset
        with open(request.dataset_path, 'r') as f:
            coco_data = json.load(f)

        categories = coco_data.get("categories", [])

        # Find and rename the category
        found = False
        old_name = None
        for cat in categories:
            if cat.get("id") == request.category_id:
                old_name = cat.get("name")
                cat["name"] = request.new_name
                found = True
                break

        if not found:
            raise HTTPException(status_code=404, detail=f"Category ID {request.category_id} not found")

        # Check for duplicate names
        names = [cat.get("name") for cat in categories]
        if names.count(request.new_name) > 1:
            raise HTTPException(status_code=400, detail=f"Category name '{request.new_name}' already exists")

        # Save updated dataset
        with open(request.dataset_path, 'w') as f:
            json.dump(coco_data, f, indent=2)

        return {
            "success": True,
            "category_id": request.category_id,
            "old_name": old_name,
            "new_name": request.new_name,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rename category failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/datasets/categories/delete")
async def delete_category(request: DeleteCategoryRequest):
    """
    Delete a category from a COCO dataset.

    Optionally deletes all annotations with this category ID.
    This modifies the dataset file in place.
    """
    logger.info(f"Deleting category {request.category_id} from {request.dataset_path}")

    try:
        if not os.path.exists(request.dataset_path):
            raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_path}")

        # Load dataset
        with open(request.dataset_path, 'r') as f:
            coco_data = json.load(f)

        categories = coco_data.get("categories", [])
        annotations = coco_data.get("annotations", [])

        # Find the category
        category_to_delete = None
        for cat in categories:
            if cat.get("id") == request.category_id:
                category_to_delete = cat
                break

        if not category_to_delete:
            raise HTTPException(status_code=404, detail=f"Category ID {request.category_id} not found")

        # Remove the category
        coco_data["categories"] = [cat for cat in categories if cat.get("id") != request.category_id]

        # Optionally remove annotations
        deleted_annotations = 0
        if request.delete_annotations:
            original_count = len(annotations)
            coco_data["annotations"] = [
                ann for ann in annotations
                if ann.get("category_id") != request.category_id
            ]
            deleted_annotations = original_count - len(coco_data["annotations"])

        # Save updated dataset
        with open(request.dataset_path, 'w') as f:
            json.dump(coco_data, f, indent=2)

        return {
            "success": True,
            "category_id": request.category_id,
            "category_name": category_to_delete.get("name"),
            "annotations_deleted": deleted_annotations,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete category failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
