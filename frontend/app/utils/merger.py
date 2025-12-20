"""
Dataset Merger Utility
======================
Merge multiple COCO datasets into one.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from copy import deepcopy


@dataclass
class MergeResult:
    """Result of a merge operation"""
    success: bool
    merged_dataset: Optional[Dict]
    total_images: int
    total_annotations: int
    total_categories: int
    source_counts: Dict[str, Dict[str, int]]  # per-source statistics
    warnings: List[str]
    error: Optional[str] = None


class DatasetMerger:
    """
    Utility class for merging multiple COCO datasets.

    Supports:
    - ID offset strategy (add offset to prevent collisions)
    - ID reassign strategy (reassign all IDs from 1)
    - Category unification (merge categories with same name)
    - Category separation (keep categories separate with suffix)
    """

    @staticmethod
    def merge_datasets(
        datasets: List[Dict],
        dataset_names: Optional[List[str]] = None,
        id_strategy: str = "offset",
        category_strategy: str = "unify"
    ) -> MergeResult:
        """
        Merge multiple COCO datasets into one.

        Args:
            datasets: List of COCO dataset dictionaries
            dataset_names: Optional names for each dataset (for reporting)
            id_strategy: How to handle ID collisions
                - "offset": Add offset to IDs based on source dataset
                - "reassign": Reassign all IDs sequentially from 1
            category_strategy: How to handle categories
                - "unify": Merge categories with the same name
                - "separate": Keep categories separate (add suffix)

        Returns:
            MergeResult with merged dataset and statistics
        """
        if not datasets:
            return MergeResult(
                success=False,
                merged_dataset=None,
                total_images=0,
                total_annotations=0,
                total_categories=0,
                source_counts={},
                warnings=[],
                error="No datasets provided"
            )

        if len(datasets) == 1:
            # Single dataset, just return a copy
            ds = deepcopy(datasets[0])
            return MergeResult(
                success=True,
                merged_dataset=ds,
                total_images=len(ds.get("images", [])),
                total_annotations=len(ds.get("annotations", [])),
                total_categories=len(ds.get("categories", [])),
                source_counts={"dataset_0": {
                    "images": len(ds.get("images", [])),
                    "annotations": len(ds.get("annotations", []))
                }},
                warnings=[]
            )

        # Generate names if not provided
        if dataset_names is None:
            dataset_names = [f"dataset_{i}" for i in range(len(datasets))]

        warnings = []
        source_counts = {}

        try:
            if id_strategy == "offset":
                merged = DatasetMerger._merge_with_offset(
                    datasets, dataset_names, category_strategy, warnings, source_counts
                )
            else:
                merged = DatasetMerger._merge_with_reassign(
                    datasets, dataset_names, category_strategy, warnings, source_counts
                )

            return MergeResult(
                success=True,
                merged_dataset=merged,
                total_images=len(merged.get("images", [])),
                total_annotations=len(merged.get("annotations", [])),
                total_categories=len(merged.get("categories", [])),
                source_counts=source_counts,
                warnings=warnings
            )

        except Exception as e:
            return MergeResult(
                success=False,
                merged_dataset=None,
                total_images=0,
                total_annotations=0,
                total_categories=0,
                source_counts=source_counts,
                warnings=warnings,
                error=str(e)
            )

    @staticmethod
    def _merge_with_offset(
        datasets: List[Dict],
        names: List[str],
        category_strategy: str,
        warnings: List[str],
        source_counts: Dict
    ) -> Dict:
        """Merge using ID offset strategy"""
        merged = {
            "images": [],
            "annotations": [],
            "categories": []
        }

        # Copy info from first dataset if present
        if "info" in datasets[0]:
            merged["info"] = deepcopy(datasets[0]["info"])
        if "licenses" in datasets[0]:
            merged["licenses"] = deepcopy(datasets[0]["licenses"])

        # Build unified category mapping
        category_map = {}  # (dataset_idx, old_cat_id) -> new_cat_id
        category_name_to_id = {}  # category_name -> new_cat_id
        next_cat_id = 1

        for ds_idx, ds in enumerate(datasets):
            for cat in ds.get("categories", []):
                old_id = cat["id"]
                name = cat["name"]

                if category_strategy == "unify":
                    if name in category_name_to_id:
                        # Use existing category ID
                        category_map[(ds_idx, old_id)] = category_name_to_id[name]
                    else:
                        # Create new category
                        category_name_to_id[name] = next_cat_id
                        category_map[(ds_idx, old_id)] = next_cat_id
                        merged["categories"].append({
                            "id": next_cat_id,
                            "name": name,
                            "supercategory": cat.get("supercategory", "")
                        })
                        next_cat_id += 1
                else:  # separate
                    # Add suffix to category name
                    new_name = f"{name}_{names[ds_idx]}"
                    category_name_to_id[new_name] = next_cat_id
                    category_map[(ds_idx, old_id)] = next_cat_id
                    merged["categories"].append({
                        "id": next_cat_id,
                        "name": new_name,
                        "supercategory": cat.get("supercategory", "")
                    })
                    next_cat_id += 1

        # Merge images and annotations with offset
        image_offset = 0
        annotation_offset = 0

        for ds_idx, ds in enumerate(datasets):
            ds_images = ds.get("images", [])
            ds_annotations = ds.get("annotations", [])

            # Track source counts
            source_counts[names[ds_idx]] = {
                "images": len(ds_images),
                "annotations": len(ds_annotations)
            }

            # Build image ID mapping for this dataset
            image_id_map = {}  # old_id -> new_id

            for img in ds_images:
                old_id = img["id"]
                new_id = old_id + image_offset
                image_id_map[old_id] = new_id

                new_img = deepcopy(img)
                new_img["id"] = new_id
                # Add source tracking
                new_img["_source"] = names[ds_idx]
                merged["images"].append(new_img)

            # Process annotations
            for ann in ds_annotations:
                new_ann = deepcopy(ann)
                new_ann["id"] = ann["id"] + annotation_offset
                new_ann["image_id"] = image_id_map.get(ann["image_id"], ann["image_id"])
                new_ann["category_id"] = category_map.get(
                    (ds_idx, ann["category_id"]),
                    ann["category_id"]
                )
                merged["annotations"].append(new_ann)

            # Update offsets for next dataset
            if ds_images:
                image_offset = max(img["id"] for img in merged["images"]) + 1
            if ds_annotations:
                annotation_offset = max(ann["id"] for ann in merged["annotations"]) + 1

        return merged

    @staticmethod
    def _merge_with_reassign(
        datasets: List[Dict],
        names: List[str],
        category_strategy: str,
        warnings: List[str],
        source_counts: Dict
    ) -> Dict:
        """Merge with complete ID reassignment"""
        merged = {
            "images": [],
            "annotations": [],
            "categories": []
        }

        # Copy info from first dataset if present
        if "info" in datasets[0]:
            merged["info"] = deepcopy(datasets[0]["info"])
        if "licenses" in datasets[0]:
            merged["licenses"] = deepcopy(datasets[0]["licenses"])

        # Build unified category mapping
        category_map = {}  # (dataset_idx, old_cat_id) -> new_cat_id
        category_name_to_id = {}
        next_cat_id = 1

        for ds_idx, ds in enumerate(datasets):
            for cat in ds.get("categories", []):
                old_id = cat["id"]
                name = cat["name"]

                if category_strategy == "unify":
                    if name in category_name_to_id:
                        category_map[(ds_idx, old_id)] = category_name_to_id[name]
                    else:
                        category_name_to_id[name] = next_cat_id
                        category_map[(ds_idx, old_id)] = next_cat_id
                        merged["categories"].append({
                            "id": next_cat_id,
                            "name": name,
                            "supercategory": cat.get("supercategory", "")
                        })
                        next_cat_id += 1
                else:
                    new_name = f"{name}_{names[ds_idx]}"
                    category_name_to_id[new_name] = next_cat_id
                    category_map[(ds_idx, old_id)] = next_cat_id
                    merged["categories"].append({
                        "id": next_cat_id,
                        "name": new_name,
                        "supercategory": cat.get("supercategory", "")
                    })
                    next_cat_id += 1

        # Collect and reassign all images and annotations
        next_image_id = 1
        next_ann_id = 1

        for ds_idx, ds in enumerate(datasets):
            ds_images = ds.get("images", [])
            ds_annotations = ds.get("annotations", [])

            source_counts[names[ds_idx]] = {
                "images": len(ds_images),
                "annotations": len(ds_annotations)
            }

            image_id_map = {}

            for img in ds_images:
                old_id = img["id"]
                image_id_map[old_id] = next_image_id

                new_img = deepcopy(img)
                new_img["id"] = next_image_id
                new_img["_source"] = names[ds_idx]
                merged["images"].append(new_img)
                next_image_id += 1

            for ann in ds_annotations:
                new_ann = deepcopy(ann)
                new_ann["id"] = next_ann_id
                new_ann["image_id"] = image_id_map.get(ann["image_id"], ann["image_id"])
                new_ann["category_id"] = category_map.get(
                    (ds_idx, ann["category_id"]),
                    ann["category_id"]
                )
                merged["annotations"].append(new_ann)
                next_ann_id += 1

        return merged

    @staticmethod
    def preview_merge(datasets: List[Dict], dataset_names: Optional[List[str]] = None) -> Dict:
        """
        Preview statistics of a potential merge without executing it.

        Args:
            datasets: List of COCO datasets
            dataset_names: Optional names for datasets

        Returns:
            Dictionary with merge preview statistics
        """
        if not datasets:
            return {"error": "No datasets provided"}

        if dataset_names is None:
            dataset_names = [f"dataset_{i}" for i in range(len(datasets))]

        total_images = 0
        total_annotations = 0
        all_categories = set()
        sources = []

        for idx, ds in enumerate(datasets):
            n_images = len(ds.get("images", []))
            n_anns = len(ds.get("annotations", []))
            cats = [c["name"] for c in ds.get("categories", [])]

            total_images += n_images
            total_annotations += n_anns
            all_categories.update(cats)

            sources.append({
                "name": dataset_names[idx],
                "images": n_images,
                "annotations": n_anns,
                "categories": len(cats)
            })

        return {
            "total_images": total_images,
            "total_annotations": total_annotations,
            "unique_categories": len(all_categories),
            "all_categories": sorted(list(all_categories)),
            "sources": sources
        }
