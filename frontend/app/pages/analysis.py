"""
Analysis Page
=============
COCO dataset analysis and visualization with enhanced UI.
"""

import json
import streamlit as st
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any, Optional

from app.components.ui import (
    page_header, section_header, metric_card, metric_row,
    alert_box, empty_state, spacer, divider_with_text
)


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
    """Render the COCO analysis page with enhanced UI"""

    page_header(
        title="Dataset Analysis",
        subtitle="Analyze your COCO dataset and plan synthetic data generation",
        icon="📊"
    )

    # File upload section
    section_header("Data Source", icon="📁")

    col1, col2 = st.columns([3, 1])

    with col1:
        upload_method = st.radio(
            "Select input method",
            ["Upload JSON file", "Enter file path"],
            horizontal=True,
            key="analysis_input_method",
            help="Choose how to provide your COCO annotation file"
        )

        coco_data = None

        if upload_method == "Upload JSON file":
            uploaded = st.file_uploader(
                "Drop your COCO JSON file here",
                type=["json"],
                key="coco_upload",
                help="Upload a COCO format annotation file"
            )

            if uploaded:
                try:
                    coco_data = json.load(uploaded)
                    st.success(f"✅ Successfully loaded: **{uploaded.name}**")
                except Exception as e:
                    alert_box(f"Failed to parse JSON: {e}", type="error")

        else:
            json_path = st.text_input(
                "COCO JSON path",
                placeholder="/app/datasets/annotations.json",
                key="coco_path",
                help="Enter the full path to your COCO annotation file"
            )

            if json_path:
                if Path(json_path).exists():
                    try:
                        with open(json_path) as f:
                            coco_data = json.load(f)
                        st.success(f"✅ Successfully loaded: **{json_path}**")
                    except Exception as e:
                        alert_box(f"Failed to load file: {e}", type="error")
                else:
                    alert_box(f"File not found: {json_path}", type="warning")

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-weight: 600; margin-bottom: 0.5rem;">📋 Supported Formats</div>
            <ul style="margin: 0; padding-left: 1.2rem; color: var(--color-text-secondary); font-size: 0.875rem;">
                <li>COCO JSON format</li>
                <li>Required: images, annotations, categories</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    if coco_data is None:
        spacer(32)
        empty_state(
            title="No Dataset Loaded",
            message="Upload a COCO JSON file or enter a file path to analyze your dataset.",
            icon="📁"
        )
        return

    # Run analysis
    with st.spinner("Analyzing dataset..."):
        analysis = analyze_coco_dataset(coco_data)

    # Store in session
    st.session_state["coco_analysis"] = analysis
    st.session_state["coco_data"] = coco_data

    spacer(24)

    # Dataset Overview
    section_header("Dataset Overview", icon="📈")

    metric_row([
        {"title": "Total Images", "value": f"{analysis['num_images']:,}", "icon": "🖼️"},
        {"title": "Total Annotations", "value": f"{analysis['num_annotations']:,}", "icon": "🏷️"},
        {"title": "Classes", "value": analysis["num_classes"], "icon": "📦"},
        {"title": "Avg. per Image", "value": f"{analysis['stats']['mean_anns_per_image']:.1f}", "icon": "📊"},
    ])

    spacer(24)

    # Class Distribution
    section_header("Class Distribution", icon="📊")

    class_counts = analysis["class_counts"]

    if class_counts:
        import pandas as pd
        import numpy as np

        # Sort by count descending
        sorted_classes = sorted(
            class_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        df = pd.DataFrame(sorted_classes, columns=["Class", "Count"])

        # Calculate imbalance metrics
        max_count = df["Count"].max()
        min_count = df["Count"].min()
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

        col1, col2 = st.columns([3, 1])

        with col1:
            # Enhanced bar chart
            st.markdown("""
            <div style="background: var(--color-bg-card); padding: 1rem; border-radius: var(--radius-lg); border: 1px solid var(--color-border);">
            """, unsafe_allow_html=True)

            st.bar_chart(
                df.set_index("Class"),
                use_container_width=True,
                height=300
            )

            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            # Imbalance metrics
            metric_card(
                title="Max Count",
                value=f"{max_count:,}",
                icon="📈"
            )
            spacer(8)
            metric_card(
                title="Min Count",
                value=f"{min_count:,}",
                icon="📉"
            )
            spacer(8)

            # Imbalance ratio with color coding
            if imbalance_ratio > 10:
                imbalance_color = "error"
                imbalance_status = "Critical"
            elif imbalance_ratio > 3:
                imbalance_color = "warning"
                imbalance_status = "High"
            else:
                imbalance_color = "success"
                imbalance_status = "OK"

            metric_card(
                title="Imbalance Ratio",
                value=f"{imbalance_ratio:.1f}x",
                icon="⚖️",
                delta=imbalance_status,
                delta_color="inverse" if imbalance_ratio > 3 else "normal",
                color=imbalance_color
            )

        # Detailed table
        with st.expander("📋 View Detailed Class Statistics", expanded=False):
            df["Percentage"] = (df["Count"] / df["Count"].sum() * 100).round(1).astype(str) + "%"
            df["Gap to Max"] = max_count - df["Count"]
            df["Gap %"] = ((max_count - df["Count"]) / max_count * 100).round(1).astype(str) + "%"

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Class": st.column_config.TextColumn("Class", width="medium"),
                    "Count": st.column_config.NumberColumn("Count", format="%d"),
                    "Percentage": st.column_config.TextColumn("% of Total"),
                    "Gap to Max": st.column_config.NumberColumn("Gap to Max", format="%d"),
                    "Gap %": st.column_config.TextColumn("Gap %"),
                }
            )

    spacer(24)

    # Balancing Configuration
    section_header("Balancing Configuration", icon="⚖️")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background: var(--color-bg-secondary); padding: 1rem; border-radius: var(--radius-md); margin-bottom: 1rem;">
            <div style="font-size: 0.875rem; color: var(--color-text-muted); margin-bottom: 0.5rem;">
                Select a strategy to determine how synthetic data will be distributed across classes.
            </div>
        </div>
        """, unsafe_allow_html=True)

        strategy = st.selectbox(
            "Balancing Strategy",
            ["complete", "partial", "minority"],
            format_func=lambda x: {
                "complete": "🎯 Complete - Balance all classes to maximum count",
                "partial": "📊 Partial - Balance to 75% of maximum",
                "minority": "📉 Minority - Only balance underrepresented classes",
            }.get(x, x),
            key="balancing_strategy",
            help="Choose how to balance class distribution"
        )

        spacer(8)

        selected_classes = st.multiselect(
            "Classes to Balance",
            options=analysis["categories"],
            default=analysis["categories"],
            key="selected_classes",
            help="Select which classes to include in balancing"
        )

    with col2:
        if selected_classes:
            targets = calculate_balancing_targets(
                analysis, strategy, selected_classes
            )

            total_synthetic = sum(targets.values())

            # Summary card
            st.markdown(f"""
            <div class="metric-card" style="text-align: center; background: linear-gradient(135deg, var(--color-primary-light), var(--color-bg-card));">
                <div style="font-size: 0.75rem; color: var(--color-text-muted); text-transform: uppercase; letter-spacing: 0.05em;">
                    Total Synthetic Images Needed
                </div>
                <div style="font-size: 2.5rem; font-weight: 700; color: var(--color-primary); margin: 0.5rem 0;">
                    {total_synthetic:,}
                </div>
                <div style="font-size: 0.875rem; color: var(--color-text-secondary);">
                    across {len([c for c in selected_classes if targets.get(c, 0) > 0])} classes
                </div>
            </div>
            """, unsafe_allow_html=True)

            spacer(16)

            # Target breakdown table
            import pandas as pd
            target_df = pd.DataFrame([
                {
                    "Class": cls,
                    "Current": class_counts.get(cls, 0),
                    "Synthetic": targets.get(cls, 0),
                    "Final": class_counts.get(cls, 0) + targets.get(cls, 0)
                }
                for cls in selected_classes
            ])

            st.dataframe(
                target_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Class": st.column_config.TextColumn("Class"),
                    "Current": st.column_config.NumberColumn("Current", format="%d"),
                    "Synthetic": st.column_config.NumberColumn("To Generate", format="%d"),
                    "Final": st.column_config.NumberColumn("Final Total", format="%d"),
                }
            )

            # Store targets
            st.session_state["balancing_targets"] = targets
        else:
            alert_box("Select at least one class to configure balancing.", type="info")

    spacer(32)

    # Export & Actions
    section_header("Export & Actions", icon="🚀")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📄</div>
            <div style="font-weight: 600;">Export Report</div>
            <div style="font-size: 0.875rem; color: var(--color-text-muted); margin-bottom: 1rem;">
                Download analysis as JSON
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("📥 Download Report", key="export_analysis", use_container_width=True):
            report = {
                "summary": {
                    "num_images": analysis["num_images"],
                    "num_annotations": analysis["num_annotations"],
                    "num_classes": analysis["num_classes"],
                },
                "class_distribution": analysis["class_counts"],
                "statistics": {k: float(v) if hasattr(v, 'item') else v for k, v in analysis["stats"].items()},
                "balancing": {
                    "strategy": strategy,
                    "selected_classes": selected_classes,
                    "targets": targets if selected_classes else {},
                },
            }

            report_json = json.dumps(report, indent=2)

            st.download_button(
                "💾 Save Report",
                data=report_json,
                file_name="analysis_report.json",
                mime="application/json",
                use_container_width=True
            )

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📊</div>
            <div style="font-weight: 600;">Detailed Stats</div>
            <div style="font-size: 0.875rem; color: var(--color-text-muted); margin-bottom: 1rem;">
                View annotation statistics
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📊 View Statistics"):
            stats = analysis["stats"]
            st.markdown(f"""
            | Metric | Value |
            |--------|-------|
            | Mean annotations/image | {stats['mean_anns_per_image']:.2f} |
            | Median annotations/image | {stats['median_anns_per_image']:.2f} |
            | Std. dev. | {stats['std_anns_per_image']:.2f} |
            | Min annotations | {stats['min_anns_per_image']:.0f} |
            | Max annotations | {stats['max_anns_per_image']:.0f} |
            | Mean bbox area | {stats['mean_bbox_area']:.0f} px² |
            """)

    with col3:
        st.markdown("""
        <div class="metric-card" style="border: 2px solid var(--color-primary);">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🚀</div>
            <div style="font-weight: 600;">Start Generation</div>
            <div style="font-size: 0.875rem; color: var(--color-text-muted); margin-bottom: 1rem;">
                Proceed to image generation
            </div>
        </div>
        """, unsafe_allow_html=True)

        if selected_classes and sum(targets.values()) > 0:
            if st.button(
                "🎨 Proceed to Generation",
                type="primary",
                key="proceed_gen",
                use_container_width=True
            ):
                st.session_state["proceed_to_generation"] = True
                st.success("✅ Configuration saved! Navigate to the **Generation** tab to start.")
        else:
            st.button(
                "🎨 Proceed to Generation",
                disabled=True,
                key="proceed_gen_disabled",
                use_container_width=True,
                help="Select classes and configure balancing first"
            )
