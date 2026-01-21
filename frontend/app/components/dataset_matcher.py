"""
Dataset Matcher Component
=========================
Analyzes compatibility between existing datasets and balancing requirements.
Provides filtering and hybrid target calculation for dataset reuse.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import random


@dataclass
class MatchResult:
    """Result of compatibility analysis between existing dataset and targets."""
    is_compatible: bool
    coverage_percentage: float  # 0-100
    matched_annotations: Dict[str, List[int]] = field(default_factory=dict)
    missing_per_class: Dict[str, int] = field(default_factory=dict)
    excess_per_class: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    # Additional metadata
    total_needed: int = 0
    total_available: int = 0
    total_covered: int = 0


class DatasetMatcher:
    """Analyzes and filters existing datasets for balancing purposes."""

    def __init__(self):
        """Initialize the dataset matcher."""
        pass

    def _normalize_category_name(self, name: str) -> str:
        """Normalize category name for flexible matching."""
        return name.lower().strip().replace("_", " ").replace("-", " ")

    def _find_matching_category(
        self,
        target_name: str,
        available_names: List[str]
    ) -> Optional[str]:
        """Find a category with a similar name in the available categories."""
        normalized_target = self._normalize_category_name(target_name)

        for name in available_names:
            if self._normalize_category_name(name) == normalized_target:
                return name
        return None

    def analyze_compatibility(
        self,
        existing_dataset: Dict[str, Any],
        targets_per_class: Dict[str, int],
        required_categories: Optional[List[str]] = None
    ) -> MatchResult:
        """
        Analyze how much of the existing dataset can be used for balancing.

        Args:
            existing_dataset: COCO format dataset
            targets_per_class: Required images per class for balancing
            required_categories: List of category names needed (defaults to targets keys)

        Returns:
            MatchResult with detailed compatibility analysis
        """
        if required_categories is None:
            required_categories = list(targets_per_class.keys())

        # Validate dataset structure
        validation_errors = self._validate_dataset_structure(existing_dataset)
        if validation_errors:
            return MatchResult(
                is_compatible=False,
                coverage_percentage=0.0,
                warnings=validation_errors,
                total_needed=sum(targets_per_class.values())
            )

        # Build category name mapping
        cat_id_to_name = {
            cat["id"]: cat["name"]
            for cat in existing_dataset.get("categories", [])
        }
        cat_name_to_id = {v: k for k, v in cat_id_to_name.items()}
        available_names = list(cat_name_to_id.keys())

        # Count annotations per class (by image, not total annotations)
        # We need to count unique images per class, not annotations
        annotations_by_class: Dict[str, List[int]] = {}
        images_per_class: Dict[str, set] = {}

        for ann in existing_dataset.get("annotations", []):
            cat_name = cat_id_to_name.get(ann["category_id"])
            if cat_name:
                if cat_name not in annotations_by_class:
                    annotations_by_class[cat_name] = []
                    images_per_class[cat_name] = set()
                annotations_by_class[cat_name].append(ann["id"])
                images_per_class[cat_name].add(ann["image_id"])

        # Count available images per class
        existing_counts = {cls: len(imgs) for cls, imgs in images_per_class.items()}

        # Calculate matches and gaps
        matched: Dict[str, List[int]] = {}
        missing: Dict[str, int] = {}
        excess: Dict[str, int] = {}
        warnings: List[str] = []
        total_needed = 0
        total_covered = 0
        total_available = 0

        for cls, needed in targets_per_class.items():
            total_needed += needed

            # Try to find matching category (with flexible name matching)
            matching_name = self._find_matching_category(cls, available_names)

            if matching_name is None:
                warnings.append(f"Clase '{cls}' no encontrada en dataset existente")
                missing[cls] = needed
                matched[cls] = []
            else:
                available = existing_counts.get(matching_name, 0)
                total_available += available
                ann_ids = annotations_by_class.get(matching_name, [])

                if available >= needed:
                    # Full coverage - take only what we need
                    matched[cls] = ann_ids[:needed]
                    total_covered += needed
                    if available > needed:
                        excess[cls] = available - needed
                else:
                    # Partial coverage
                    matched[cls] = ann_ids
                    total_covered += available
                    missing[cls] = needed - available

        coverage = (total_covered / total_needed * 100) if total_needed > 0 else 100.0

        return MatchResult(
            is_compatible=coverage >= 100.0,
            coverage_percentage=min(coverage, 100.0),
            matched_annotations=matched,
            missing_per_class=missing,
            excess_per_class=excess,
            warnings=warnings,
            total_needed=total_needed,
            total_available=total_available,
            total_covered=total_covered
        )

    def filter_for_balancing(
        self,
        existing_dataset: Dict[str, Any],
        targets_per_class: Dict[str, int],
        strategy: str = "random"
    ) -> Dict[str, Any]:
        """
        Filter existing dataset to take only what's needed for balancing.

        Args:
            existing_dataset: Full COCO dataset
            targets_per_class: Required counts per class
            strategy: Selection strategy ('random', 'newest', 'first')

        Returns:
            Filtered COCO dataset with only necessary images/annotations
        """
        # Build category mapping
        cat_id_to_name = {
            cat["id"]: cat["name"]
            for cat in existing_dataset.get("categories", [])
        }
        cat_name_to_id = {v: k for k, v in cat_id_to_name.items()}
        available_names = list(cat_name_to_id.keys())

        # Group annotations by class
        anns_by_class: Dict[str, List[Dict]] = {}
        for ann in existing_dataset.get("annotations", []):
            cat_name = cat_id_to_name.get(ann["category_id"])
            if cat_name:
                # Find if this matches any target class
                for target_cls in targets_per_class.keys():
                    if self._find_matching_category(target_cls, [cat_name]):
                        if target_cls not in anns_by_class:
                            anns_by_class[target_cls] = []
                        anns_by_class[target_cls].append(ann)
                        break

        # Select annotations based on strategy
        selected_anns: List[Dict] = []
        selected_img_ids: set = set()

        for cls, needed in targets_per_class.items():
            available = anns_by_class.get(cls, [])

            if strategy == "random":
                random.shuffle(available)
            elif strategy == "newest":
                # Sort by ID descending (assuming higher IDs are newer)
                available = sorted(available, key=lambda x: x.get("id", 0), reverse=True)
            # 'first' strategy keeps original order

            # Take up to 'needed' annotations
            for ann in available[:needed]:
                selected_anns.append(ann)
                selected_img_ids.add(ann["image_id"])

        # Filter images
        filtered_images = [
            img for img in existing_dataset.get("images", [])
            if img["id"] in selected_img_ids
        ]

        # Filter categories (only those with selected annotations)
        selected_cat_ids = {ann["category_id"] for ann in selected_anns}
        filtered_categories = [
            cat for cat in existing_dataset.get("categories", [])
            if cat["id"] in selected_cat_ids
        ]

        return {
            "images": filtered_images,
            "annotations": selected_anns,
            "categories": filtered_categories,
            "info": existing_dataset.get("info", {
                "description": "Filtered dataset for balancing",
                "version": "1.0"
            }),
            "licenses": existing_dataset.get("licenses", [])
        }

    def calculate_hybrid_targets(
        self,
        targets_per_class: Dict[str, int],
        matched_annotations: Dict[str, List[int]]
    ) -> Dict[str, int]:
        """
        Calculate how many new images to generate for hybrid mode.

        Args:
            targets_per_class: Original targets from balancing analysis
            matched_annotations: What the existing dataset provides (annotation IDs per class)

        Returns:
            Dict of class -> number of new images to generate
        """
        hybrid_targets: Dict[str, int] = {}

        for cls, target in targets_per_class.items():
            existing_count = len(matched_annotations.get(cls, []))
            gap = max(0, target - existing_count)

            if gap > 0:
                hybrid_targets[cls] = gap

        return hybrid_targets

    def get_coverage_summary(
        self,
        match_result: MatchResult,
        targets_per_class: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Get a human-readable summary of coverage analysis.

        Args:
            match_result: Result from analyze_compatibility
            targets_per_class: Original targets

        Returns:
            Dict with summary information for UI display
        """
        summary = {
            "coverage_percentage": match_result.coverage_percentage,
            "is_fully_compatible": match_result.is_compatible,
            "total_needed": match_result.total_needed,
            "total_covered": match_result.total_covered,
            "total_missing": sum(match_result.missing_per_class.values()),
            "classes_with_full_coverage": [],
            "classes_with_partial_coverage": [],
            "classes_missing": [],
        }

        for cls, needed in targets_per_class.items():
            available = len(match_result.matched_annotations.get(cls, []))
            missing = match_result.missing_per_class.get(cls, 0)

            if missing == 0:
                summary["classes_with_full_coverage"].append({
                    "name": cls,
                    "needed": needed,
                    "available": available
                })
            elif available > 0:
                summary["classes_with_partial_coverage"].append({
                    "name": cls,
                    "needed": needed,
                    "available": available,
                    "missing": missing,
                    "coverage_pct": (available / needed * 100) if needed > 0 else 0
                })
            else:
                summary["classes_missing"].append({
                    "name": cls,
                    "needed": needed
                })

        return summary

    def _validate_dataset_structure(self, dataset: Dict) -> List[str]:
        """Validate basic COCO dataset structure."""
        errors = []

        if not isinstance(dataset, dict):
            errors.append("Dataset debe ser un diccionario")
            return errors

        if "images" not in dataset:
            errors.append("Falta la seccion 'images'")
        elif not isinstance(dataset["images"], list):
            errors.append("'images' debe ser una lista")

        if "annotations" not in dataset:
            errors.append("Falta la seccion 'annotations'")
        elif not isinstance(dataset["annotations"], list):
            errors.append("'annotations' debe ser una lista")

        if "categories" not in dataset:
            errors.append("Falta la seccion 'categories'")
        elif not isinstance(dataset["categories"], list):
            errors.append("'categories' debe ser una lista")

        return errors

    def merge_with_generated(
        self,
        existing_filtered: Dict[str, Any],
        newly_generated: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge filtered existing dataset with newly generated images (for hybrid mode).

        Args:
            existing_filtered: Filtered dataset from filter_for_balancing
            newly_generated: Dataset from generation service

        Returns:
            Merged COCO dataset
        """
        # Get max IDs from existing
        max_img_id = max([img["id"] for img in existing_filtered.get("images", [])], default=0)
        max_ann_id = max([ann["id"] for ann in existing_filtered.get("annotations", [])], default=0)

        # Offset new dataset IDs
        new_images = []
        img_id_map = {}
        for img in newly_generated.get("images", []):
            old_id = img["id"]
            new_id = max_img_id + old_id + 1
            img_id_map[old_id] = new_id
            new_img = img.copy()
            new_img["id"] = new_id
            new_images.append(new_img)

        new_annotations = []
        for ann in newly_generated.get("annotations", []):
            new_ann = ann.copy()
            new_ann["id"] = max_ann_id + ann["id"] + 1
            new_ann["image_id"] = img_id_map.get(ann["image_id"], ann["image_id"])
            new_annotations.append(new_ann)

        # Unify categories
        existing_cats = {cat["name"]: cat for cat in existing_filtered.get("categories", [])}
        for cat in newly_generated.get("categories", []):
            if cat["name"] not in existing_cats:
                existing_cats[cat["name"]] = cat

        return {
            "images": existing_filtered.get("images", []) + new_images,
            "annotations": existing_filtered.get("annotations", []) + new_annotations,
            "categories": list(existing_cats.values()),
            "info": {
                "description": "Merged dataset (existing + generated)",
                "version": "1.0"
            },
            "licenses": existing_filtered.get("licenses", [])
        }
