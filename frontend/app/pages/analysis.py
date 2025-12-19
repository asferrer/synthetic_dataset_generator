"""
Analysis Page
=============
COCO dataset analysis and visualization.
"""

import json
import streamlit as st
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any, Optional


def analyze_coco_dataset(coco_data: Dict) -> Dict[str, Any]:
    """
    Analyze a COCO dataset and return statistics.

    Args:
        coco_data: Parsed COCO JSON data

    Returns:
        Dictionary with analysis results
    """
    # Build category mapping
    categories = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}

    # Count annotations per class
    class_counts = Counter()
    for ann in coco_data.get("annotations", []):
        cat_id = ann.get("category_id")
        if cat_id in categories:
            class_counts[categories[cat_id]] += 1

    # Image statistics
    num_images = len(coco_data.get("images", []))
    num_annotations = len(coco_data.get("annotations", []))

    # Annotations per image
    anns_per_image = Counter()
    for ann in coco_data.get("annotations", []):
        anns_per_image[ann.get("image_id")] += 1

    ann_counts = list(anns_per_image.values()) if anns_per_image else [0]

    # Bounding box statistics
    bbox_areas = []
    for ann in coco_data.get("annotations", []):
        bbox = ann.get("bbox", [0, 0, 0, 0])
        if len(bbox) >= 4:
            area = bbox[2] * bbox[3]
            bbox_areas.append(area)

    import numpy as np
    bbox_areas = np.array(bbox_areas) if bbox_areas else np.array([0])

    return {
        "num_images": num_images,
        "num_annotations": num_annotations,
        "num_classes": len(categories),
        "class_counts": dict(class_counts),
        "categories": list(categories.values()),
        "stats": {
            "mean_anns_per_image": np.mean(ann_counts),
            "median_anns_per_image": np.median(ann_counts),
            "std_anns_per_image": np.std(ann_counts),
            "min_anns_per_image": np.min(ann_counts),
            "max_anns_per_image": np.max(ann_counts),
            "mean_bbox_area": np.mean(bbox_areas),
            "median_bbox_area": np.median(bbox_areas),
        },
        "anns_per_image": dict(anns_per_image),
    }


def calculate_balancing_targets(
    analysis: Dict,
    strategy: str,
    selected_classes: List[str],
) -> Dict[str, int]:
    """
    Calculate synthetic instances needed per class.

    Args:
        analysis: Analysis results from analyze_coco_dataset
        strategy: Balancing strategy (complete, partial, minority)
        selected_classes: Classes to balance

    Returns:
        Dictionary mapping class name to required synthetic instances
    """
    class_counts = analysis.get("class_counts", {})
    max_count = max(class_counts.values()) if class_counts else 0

    targets = {}

    for cls in selected_classes:
        current = class_counts.get(cls, 0)

        if strategy == "complete":
            # Balance all to maximum
            targets[cls] = max(0, max_count - current)

        elif strategy == "partial":
            # Balance to 75% of maximum
            target = int(max_count * 0.75)
            targets[cls] = max(0, target - current)

        elif strategy == "minority":
            # Only balance classes below median
            import numpy as np
            median = np.median(list(class_counts.values())) if class_counts else 0
            if current < median:
                targets[cls] = max(0, int(median) - current)
            else:
                targets[cls] = 0

        else:
            targets[cls] = 0

    return targets


def render_analysis_page():
    """Render the COCO analysis page"""
    st.header("Dataset Analysis")
    st.markdown("Analyze your COCO dataset and plan synthetic data generation.")

    # File upload section
    col1, col2 = st.columns([2, 1])

    with col1:
        upload_method = st.radio(
            "Input method",
            ["Upload JSON file", "Enter path"],
            horizontal=True,
            key="analysis_input_method"
        )

        coco_data = None

        if upload_method == "Upload JSON file":
            uploaded = st.file_uploader(
                "Upload COCO JSON",
                type=["json"],
                key="coco_upload"
            )

            if uploaded:
                try:
                    coco_data = json.load(uploaded)
                    st.success(f"Loaded: {uploaded.name}")
                except Exception as e:
                    st.error(f"Failed to parse JSON: {e}")

        else:
            json_path = st.text_input(
                "COCO JSON path",
                placeholder="/app/datasets/annotations.json",
                key="coco_path"
            )

            if json_path and Path(json_path).exists():
                try:
                    with open(json_path) as f:
                        coco_data = json.load(f)
                    st.success(f"Loaded: {json_path}")
                except Exception as e:
                    st.error(f"Failed to load: {e}")

    with col2:
        st.info(
            "**Supported formats:**\n"
            "- COCO JSON format\n"
            "- Required keys: images, annotations, categories"
        )

    if coco_data is None:
        st.warning("Upload or specify a COCO JSON file to analyze.")
        return

    # Run analysis
    analysis = analyze_coco_dataset(coco_data)

    # Store in session
    st.session_state["coco_analysis"] = analysis
    st.session_state["coco_data"] = coco_data

    st.divider()

    # Display results
    st.subheader("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Images", analysis["num_images"])

    with col2:
        st.metric("Annotations", analysis["num_annotations"])

    with col3:
        st.metric("Classes", analysis["num_classes"])

    with col4:
        st.metric(
            "Avg. per image",
            f"{analysis['stats']['mean_anns_per_image']:.1f}"
        )

    st.divider()

    # Class distribution
    st.subheader("Class Distribution")

    class_counts = analysis["class_counts"]

    if class_counts:
        # Sort by count descending
        sorted_classes = sorted(
            class_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Create bar chart data
        import pandas as pd

        df = pd.DataFrame(sorted_classes, columns=["Class", "Count"])

        # Calculate imbalance ratio
        max_count = df["Count"].max()
        min_count = df["Count"].min()
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

        col1, col2 = st.columns([3, 1])

        with col1:
            st.bar_chart(df.set_index("Class"))

        with col2:
            st.metric("Max count", max_count)
            st.metric("Min count", min_count)
            st.metric(
                "Imbalance ratio",
                f"{imbalance_ratio:.1f}x",
                delta="High" if imbalance_ratio > 3 else "OK",
                delta_color="inverse" if imbalance_ratio > 3 else "normal"
            )

        # Detailed table
        with st.expander("Class Details"):
            df["Percentage"] = (df["Count"] / df["Count"].sum() * 100).round(1)
            df["Gap to Max"] = max_count - df["Count"]
            st.dataframe(df, width="stretch")

    st.divider()

    # Balancing section
    st.subheader("Balancing Configuration")

    col1, col2 = st.columns(2)

    with col1:
        strategy = st.selectbox(
            "Balancing strategy",
            ["complete", "partial", "minority"],
            format_func=lambda x: {
                "complete": "Complete - Balance all to maximum",
                "partial": "Partial - Balance to 75% of max",
                "minority": "Minority - Only balance underrepresented",
            }.get(x, x),
            key="balancing_strategy"
        )

        selected_classes = st.multiselect(
            "Classes to balance",
            options=analysis["categories"],
            default=analysis["categories"],
            key="selected_classes"
        )

    with col2:
        if selected_classes:
            targets = calculate_balancing_targets(
                analysis, strategy, selected_classes
            )

            total_synthetic = sum(targets.values())

            st.metric("Total synthetic needed", total_synthetic)

            # Show breakdown
            target_df = pd.DataFrame([
                {"Class": cls, "Current": class_counts.get(cls, 0), "Synthetic": targets.get(cls, 0)}
                for cls in selected_classes
            ])
            target_df["Total"] = target_df["Current"] + target_df["Synthetic"]

            st.dataframe(target_df, width="stretch")

            # Store targets
            st.session_state["balancing_targets"] = targets

    st.divider()

    # Export section
    st.subheader("Export Analysis")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Export Analysis Report", key="export_analysis"):
            report = {
                "summary": {
                    "num_images": analysis["num_images"],
                    "num_annotations": analysis["num_annotations"],
                    "num_classes": analysis["num_classes"],
                },
                "class_distribution": analysis["class_counts"],
                "statistics": analysis["stats"],
                "balancing": {
                    "strategy": strategy,
                    "selected_classes": selected_classes,
                    "targets": targets if selected_classes else {},
                },
            }

            report_json = json.dumps(report, indent=2)

            st.download_button(
                "Download Report (JSON)",
                data=report_json,
                file_name="analysis_report.json",
                mime="application/json"
            )

    with col2:
        if selected_classes and sum(targets.values()) > 0:
            if st.button("Proceed to Generation", type="primary", key="proceed_gen"):
                st.session_state["proceed_to_generation"] = True
                st.rerun()
