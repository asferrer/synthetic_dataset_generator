"""
Synthetic Dataset Generator - Professional Interface
====================================================
Professional tool for photorealistic synthetic data generation
with advanced analysis, interactive visualizations, and full process control.

Author: Synthetic Data Team
Version: 2.0.0 (Remastered)
"""

import os
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data.coco_parser import load_coco_json, COCOValidationError
from src.analysis.class_analysis import analyze_coco_dataset
from src.augmentation.augmentor import SyntheticDataAugmentor

# Post-processing modules
from src.export import ExportManager, ExportConfig
from src.splits import DatasetSplitter, SplitConfig
from src.balancing import ClassBalancer, BalancingConfig
from src.pipeline import PostProcessingPipeline, PostProcessingConfig
from src.utils import LabelManager


# ========================= CONFIGURATION =========================

def load_config(config_path="configs/config.yaml") -> dict:
    """Loads configuration with portable cross-platform paths."""
    project_root = Path(__file__).parent.parent.resolve()
    config_file = project_root / config_path

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Convert relative paths to absolute
    if "paths" in config:
        for key, value in config["paths"].items():
            if isinstance(value, str):
                path = Path(value)
                if not path.is_absolute():
                    config["paths"][key] = str((project_root / path).resolve())
                else:
                    config["paths"][key] = str(path)

    return config


def setup_logging():
    """Configures logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )


# ========================= UTILITIES =========================

def get_available_classes_from_folder(path: str) -> List[str]:
    """Gets available classes from an objects folder."""
    if not path or not os.path.isdir(path):
        return []
    return sorted([d for d in os.listdir(path)
                   if os.path.isdir(os.path.join(path, d))])


def parse_class_mapping(mapping_text: str) -> Dict[str, List[str]]:
    """Parses class mapping from text."""
    mapping = {}
    if not mapping_text:
        return mapping

    for line in mapping_text.strip().split("\n"):
        if ":" in line:
            target, sources = line.split(":", 1)
            tgt = target.strip()
            srcs = [s.strip() for s in sources.split(",") if s.strip()]
            if tgt and srcs:
                mapping[tgt] = srcs
    return mapping


def calculate_dataset_stats(counts: Dict[str, int]) -> Dict:
    """Calculates dataset statistics."""
    if not counts:
        return {}

    values = list(counts.values())
    return {
        "total_samples": sum(values),
        "num_classes": len(counts),
        "mean": sum(values) / len(values),
        "median": sorted(values)[len(values) // 2],
        "min": min(values),
        "max": max(values),
        "std": (sum((x - sum(values)/len(values))**2 for x in values) / len(values)) ** 0.5
    }


# ========================= VISUALIZATIONS =========================

def create_distribution_chart(data: Dict[str, int], title: str, color_scheme: str = "viridis") -> go.Figure:
    """Creates interactive bar chart with Plotly."""
    if not data:
        return None

    df = pd.DataFrame(list(data.items()), columns=['Class', 'Count'])
    df = df.sort_values('Count', ascending=True)

    fig = px.bar(
        df,
        x='Count',
        y='Class',
        orientation='h',
        title=title,
        color='Count',
        color_continuous_scale=color_scheme,
        text='Count'
    )

    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(
        height=max(400, len(data) * 30),
        showlegend=False,
        xaxis_title="Number of Samples",
        yaxis_title="Class",
        font=dict(size=12),
        hovermode='closest',
        template="plotly",  # Adaptive theme
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        plot_bgcolor='rgba(0,0,0,0)'     # Transparent plot background
    )

    return fig


def create_comparison_chart(original: Dict, synthetic: Dict, final: Dict) -> go.Figure:
    """Creates comparison chart between datasets."""
    classes = sorted(set(original.keys()) | set(synthetic.keys()) | set(final.keys()))

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Original',
        x=classes,
        y=[original.get(c, 0) for c in classes],
        marker_color='#4A90E2'  # Saturated blue
    ))

    fig.add_trace(go.Bar(
        name='Synthetic',
        x=classes,
        y=[synthetic.get(c, 0) for c in classes],
        marker_color='#F5A623'  # Saturated orange
    ))

    fig.add_trace(go.Bar(
        name='Final',
        x=classes,
        y=[final.get(c, 0) for c in classes],
        marker_color='#7ED321'  # Saturated green
    ))

    fig.update_layout(
        title="Comparison: Original vs Synthetic vs Final",
        xaxis_title="Classes",
        yaxis_title="Number of Samples",
        barmode='group',
        height=500,
        hovermode='x unified',
        template="plotly",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig


def create_balance_heatmap(original: Dict, final: Dict) -> go.Figure:
    """Creates heatmap showing balance improvement."""
    classes = sorted(set(original.keys()) | set(final.keys()))

    original_vals = [original.get(c, 0) for c in classes]
    final_vals = [final.get(c, 0) for c in classes]
    improvement = [(f - o) / max(o, 1) * 100 for o, f in zip(original_vals, final_vals)]

    fig = go.Figure(data=go.Heatmap(
        z=[improvement],
        x=classes,
        y=['Improvement (%)'],
        colorscale='RdYlGn',
        text=[[f"{val:.1f}%" for val in improvement]],
        texttemplate='%{text}',
        textfont={"size": 10, "color": "black"},  # Black text for better contrast
        hoverongaps=False
    ))

    fig.update_layout(
        title="Balance Improvement per Class (%)",
        height=200,
        xaxis_title="Classes",
        template="plotly",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig


# ========================= UI COMPONENTS =========================

def render_header():
    """Renders professional application header."""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h1 style="color: white; margin: 0; text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🎨 Synthetic Dataset Generator Pro
        </h1>
        <p style="color: rgba(255,255,255,0.95); text-align: center; margin-top: 0.5rem; font-size: 1.1rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
            Photorealistic synthetic data generation with AI
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_metrics_dashboard(stats: Dict):
    """Renders key metrics dashboard."""
    st.markdown("### 📊 Dataset Metrics")

    cols = st.columns(5)

    metrics = [
        ("Total Samples", stats.get("total_samples", 0), "📦"),
        ("Classes", stats.get("num_classes", 0), "🏷️"),
        ("Mean", f"{stats.get('mean', 0):.1f}", "📈"),
        ("Std Dev", f"{stats.get('std', 0):.1f}", "📊"),
        ("Range", f"{stats.get('min', 0)}-{stats.get('max', 0)}", "📏")
    ]

    for col, (label, value, icon) in zip(cols, metrics):
        with col:
            st.metric(label=f"{icon} {label}", value=value)


def render_workflow_status():
    """Renders workflow status in the sidebar."""
    st.sidebar.markdown("## 📋 Session Status")

    # Check session state for workflow progress
    has_coco_data = 'coco_data' in st.session_state and st.session_state.coco_data is not None
    has_generated = 'last_generated_synthetic' in st.session_state and st.session_state.last_generated_synthetic is not None
    has_pp_data = 'pp_coco_data' in st.session_state and st.session_state.pp_coco_data is not None

    # Dataset loaded (either from analysis or post-processing)
    dataset_status = "✅" if (has_coco_data or has_pp_data) else "⬜"
    dataset_text = "Dataset loaded" if (has_coco_data or has_pp_data) else "No dataset loaded"

    # Synthetic generation (optional)
    gen_status = "✅" if has_generated else "➖"
    gen_text = "Synthetic data available" if has_generated else "Not generated (optional)"

    # Post-processing ready
    pp_status = "✅" if has_pp_data else "⬜"
    pp_text = "Ready for processing" if has_pp_data else "Load dataset in Post-Processing"

    st.sidebar.markdown(f"""
    {dataset_status} **Dataset**: {dataset_text}

    {gen_status} **Synthetic**: {gen_text}

    {pp_status} **Post-Process**: {pp_text}
    """)

    # Show quick stats if available
    if has_coco_data:
        with st.sidebar.expander("📊 Analyzed Dataset"):
            stats = st.session_state.get('stats', {})
            st.write(f"Images: {stats.get('total_samples', 'N/A')}")
            st.write(f"Classes: {stats.get('num_classes', 'N/A')}")

    if has_generated:
        with st.sidebar.expander("🎨 Generated Dataset"):
            gen_data = st.session_state.last_generated_synthetic
            st.write(f"Images: {len(gen_data.get('images', []))}")
            st.write(f"Annotations: {len(gen_data.get('annotations', []))}")

    st.sidebar.markdown("---")


def render_sidebar_config(config: dict) -> Tuple:
    """Renders configuration in sidebar and returns parameters."""
    st.sidebar.markdown("## ⚙️ Configuration")

    # Basic transformations
    with st.sidebar.expander("🔄 Geometric Transformations", expanded=True):
        rotate = st.checkbox("Random rotation", value=config["augmentation"]["rot"])
        scale = st.checkbox("Adaptive scaling", value=config["augmentation"]["scale"])
        translate = st.checkbox("Translation", value=config["augmentation"]["trans"])
        perspective = st.checkbox("Perspective transform", value=True)

    # Generation parameters
    with st.sidebar.expander("🎯 Generation Parameters", expanded=True):
        max_objects = st.slider(
            "Objects per image",
            min_value=1, max_value=15,
            value=config["augmentation"].get("max_objects_per_image", 5),
            help="Maximum number of synthetic objects per generated image"
        )

        overlap_threshold = st.slider(
            "Overlap threshold (%)",
            min_value=0, max_value=50,
            value=config["augmentation"]["overlap_threshold"],
            help="Maximum overlap percentage allowed between objects"
        )

        try_count = st.number_input(
            "Attempts per object",
            min_value=1, max_value=20,
            value=config["augmentation"]["try_count"],
            help="Number of attempts to place an object without excessive overlap"
        )

    # Object size control
    with st.sidebar.expander("📐 Size Control", expanded=True):
        min_area = st.slider(
            "Minimum area (%)",
            min_value=0.0, max_value=10.0,
            value=config["augmentation"]["min_area_ratio"] * 100,
            step=0.1,
            help="Minimum object size as % of total area"
        ) / 100

        max_area = st.slider(
            "Maximum area (%)",
            min_value=10.0, max_value=100.0,
            value=config["augmentation"]["max_area_ratio"] * 100,
            step=5.0,
            help="Maximum object size as % of total area"
        ) / 100

        max_upscale = st.slider(
            "Maximum upscale",
            min_value=1.0, max_value=10.0,
            value=3.0, step=0.5,
            help="Maximum scaling factor for small objects"
        )

    # Realism effects
    st.sidebar.markdown("## 🎨 Realism Effects")
    with st.sidebar.expander("✨ Effects Pipeline", expanded=True):
        realism_intensity = st.slider(
            "Global intensity",
            min_value=0.0, max_value=1.0,
            value=0.5, step=0.05,
            help="Overall intensity of all realism effects"
        )

        st.markdown("**Available effects:**")
        advanced_color = st.checkbox("🎨 Advanced color matching", value=True)
        blur_consist = st.checkbox("🌫️ Blur consistency", value=True)
        shadows = st.checkbox("☀️ Dynamic shadows", value=True)
        lighting = st.checkbox("💡 Lighting effects", value=False)
        underwater = st.checkbox("🌊 Underwater effect", value=True)
        motion_blur = st.checkbox("💨 Motion blur (20% of objects)", value=False)
        poisson = st.checkbox("🔬 Poisson blending", value=False)

    # Advanced options
    with st.sidebar.expander("🔧 Advanced Options"):
        save_intermediate = st.checkbox("Save intermediate steps", value=False)
        caustics_effect = st.checkbox("Caustics effects", value=True)

    return {
        "transformations": {
            "rotate": rotate,
            "scale": scale,
            "translate": translate,
            "perspective": perspective
        },
        "generation": {
            "max_objects": max_objects,
            "overlap_threshold": overlap_threshold / 100,
            "try_count": try_count
        },
        "size_control": {
            "min_area": min_area,
            "max_area": max_area,
            "max_upscale": max_upscale
        },
        "realism": {
            "intensity": realism_intensity,
            "advanced_color": advanced_color,
            "blur_consist": blur_consist,
            "shadows": shadows,
            "lighting": lighting,
            "underwater": underwater,
            "motion_blur": motion_blur,
            "poisson": poisson,
            "caustics": caustics_effect
        },
        "advanced": {
            "save_intermediate": save_intermediate
        }
    }


def render_dataset_analysis_tab(config: dict):
    """Tab for COCO dataset analysis."""
    st.markdown("## 📊 COCO Dataset Analysis")

    uploaded_file = st.file_uploader(
        "📁 Upload COCO JSON file",
        type=["json"],
        help="Upload your dataset in COCO format for analysis"
    )

    if uploaded_file is not None:
        try:
            with st.spinner("🔍 Validating COCO structure..."):
                coco_data = json.load(uploaded_file)

                # Robust validation
                from src.data.coco_parser import validate_coco_structure
                validate_coco_structure(coco_data, strict=False)

                st.success("✅ COCO dataset validated successfully")

            # Analysis
            analysis = analyze_coco_dataset(coco_data, {})
            class_counts = analysis.get("class_counts", {})
            stats = calculate_dataset_stats(class_counts)

            # Metrics dashboard
            render_metrics_dashboard(stats)

            # Visualizations
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📊 Class Distribution")
                fig1 = create_distribution_chart(
                    class_counts,
                    "Original Distribution",
                    "viridis"
                )
                if fig1:
                    st.plotly_chart(fig1, width='stretch')

            with col2:
                st.markdown("#### 📈 Detailed Statistics")
                df_stats = pd.DataFrame([
                    {"Class": k, "Samples": v, "% of Total": f"{v/stats['total_samples']*100:.1f}%"}
                    for k, v in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
                ])
                st.dataframe(df_stats, width='stretch', height=400)

            # Save to session_state for later use
            st.session_state['coco_data'] = coco_data
            st.session_state['class_counts'] = class_counts
            st.session_state['stats'] = stats

            # ===== QUICK ACTIONS FOR ANALYZED DATASET =====
            st.markdown("---")
            st.markdown("### 🚀 Quick Actions")

            action_cols = st.columns(4)

            with action_cols[0]:
                if st.button("📤 Export to YOLO", key="analysis_export_yolo", use_container_width=True):
                    output_dir = config["paths"]["images"].rsplit(os.sep, 1)[0]
                    images_dir = st.text_input("Images directory for export", value=config["paths"]["backgrounds_dataset"], key="yolo_img_dir_analysis")
                    if images_dir:
                        _quick_export(coco_data, images_dir, output_dir, 'yolo')

            with action_cols[1]:
                if st.button("📤 Export to COCO", key="analysis_export_coco", use_container_width=True):
                    output_dir = config["paths"]["images"].rsplit(os.sep, 1)[0]
                    images_dir = config["paths"]["backgrounds_dataset"]
                    _quick_export(coco_data, images_dir, output_dir, 'coco')

            with action_cols[2]:
                if st.button("✂️ Create Splits", key="analysis_splits", use_container_width=True):
                    output_dir = config["paths"]["images"].rsplit(os.sep, 1)[0]
                    images_dir = config["paths"]["backgrounds_dataset"]
                    _quick_split(coco_data, images_dir, output_dir)

            with action_cols[3]:
                st.markdown("**➡️ Go to Balancing tab to generate synthetic data**")

            # Tip for workflow
            st.info("""
            **💡 Workflow Tip:**
            1. Analyze your dataset here to understand class distribution
            2. Go to **Balancing** tab to generate synthetic samples for underrepresented classes
            3. Use **Post-Processing** tab to export, split, and combine datasets
            """)

        except COCOValidationError as e:
            st.error(f"❌ COCO validation error:\n\n{str(e)}")
        except Exception as e:
            st.error(f"❌ Error processing file:\n\n{str(e)}")
    else:
        st.info("👆 Upload a COCO JSON file to begin analysis")


def render_balancing_tab(config: dict, params: dict):
    """Tab for dataset balancing."""
    st.markdown("## ⚖️ Dataset Balancing")

    if 'coco_data' not in st.session_state:
        st.warning("⚠️ First analyze a COCO dataset in the 'Analysis' tab")
        return

    coco_data = st.session_state['coco_data']
    class_counts = st.session_state['class_counts']

    # Balancing configuration
    st.markdown("### 🎯 Balancing Configuration")

    col1, col2 = st.columns([2, 1])

    with col1:
        balance_strategy = st.radio(
            "Balancing strategy",
            ["Complete balancing (all to maximum)",
             "Partial balancing (custom target)",
             "Minority classes only"],
            help="Define how you want to balance the dataset"
        )

    with col2:
        if balance_strategy == "Minority classes only":
            minority_threshold = st.number_input(
                "Minority threshold",
                min_value=1,
                value=min(class_counts.values()) + (max(class_counts.values()) - min(class_counts.values())) // 3,
                help="Classes with fewer samples than this are considered minority"
            )

    # Class selection
    st.markdown("### 📋 Class Selection")

    if balance_strategy == "Minority classes only":
        minority_classes = [c for c, count in class_counts.items() if count < minority_threshold]
        selected_classes = st.multiselect(
            f"Detected minority classes (<{minority_threshold})",
            options=list(class_counts.keys()),
            default=minority_classes
        )
    else:
        selected_classes = st.multiselect(
            "Select classes to balance",
            options=list(class_counts.keys()),
            default=list(class_counts.keys())
        )

    if not selected_classes:
        st.info("👆 Select at least one class to balance")
        return

    # Define targets
    st.markdown("### 🎯 Generation Targets")

    max_count = max(class_counts.values())
    desired_synthetic = {}

    if balance_strategy == "Complete balancing (all to maximum)":
        st.info(f"ℹ️ All classes will be balanced to {max_count} samples")
        for cls in selected_classes:
            current = class_counts.get(cls, 0)
            desired_synthetic[cls] = max(0, max_count - current)
    else:
        # Custom input
        cols = st.columns(min(3, len(selected_classes)))
        for i, cls in enumerate(selected_classes):
            with cols[i % 3]:
                current = class_counts.get(cls, 0)
                target = st.number_input(
                    f"📌 {cls}",
                    min_value=current,
                    value=max_count if balance_strategy == "Complete balancing (all to maximum)" else current + 100,
                    help=f"Current: {current}. Define desired final total",
                    key=f"target_{cls}"
                )
                desired_synthetic[cls] = max(0, target - current)

    # Results preview
    st.markdown("### 🔮 Results Preview")

    original_counts = class_counts
    synthetic_counts = desired_synthetic
    final_counts = {cls: original_counts.get(cls, 0) + synthetic_counts.get(cls, 0)
                    for cls in set(original_counts.keys()) | set(synthetic_counts.keys())}

    fig_comparison = create_comparison_chart(original_counts, synthetic_counts, final_counts)
    st.plotly_chart(fig_comparison, width='stretch')

    fig_heatmap = create_balance_heatmap(original_counts, final_counts)
    st.plotly_chart(fig_heatmap, width='stretch')

    # Summary
    total_to_generate = sum(desired_synthetic.values())
    st.info(f"📦 Total synthetic samples to generate: **{total_to_generate}**")

    # Object source
    st.markdown("### 📂 Object Source")
    objects_source = st.radio(
        "Where do you want to extract objects from?",
        ["Input dataset (COCO segmentations)", "External objects folder"],
        help="Define the source of objects to paste"
    )

    if objects_source == "External objects folder":
        objects_path = st.text_input(
            "Path to objects folder",
            value=config["paths"]["objects_dataset"],
            help="Folder with subfolders per class containing images with alpha channel"
        )

        # Class mapping
        with st.expander("🔗 Class Mapping (Optional)"):
            available_folders = get_available_classes_from_folder(objects_path)
            st.info(f"Detected folders: {', '.join(available_folders)}")

            mapping_text = st.text_area(
                "Define mapping (target: source1, source2)",
                value="\n".join([f"{cls}: {cls}" for cls in selected_classes]),
                help="Example: bottle: bottle, plastic_bottle"
            )
            class_mapping = parse_class_mapping(mapping_text)
    else:
        objects_path = config["paths"]["backgrounds_dataset"]
        class_mapping = {}

    # Execution button
    st.markdown("---")

    if st.button("🚀 Start Balancing", type="primary", width='stretch'):
        execute_augmentation(
            config, params, coco_data, selected_classes,
            desired_synthetic, objects_source, objects_path, class_mapping,
            original_counts
        )


def render_generation_tab(config: dict, params: dict):
    """Tab for generation from scratch."""
    st.markdown("## 🎨 Generation from Scratch")

    objects_path = st.text_input(
        "📂 Path to objects folder",
        value=config["paths"]["objects_dataset"],
        help="Folder with subfolders per class"
    )

    available_classes = get_available_classes_from_folder(objects_path)

    if not available_classes:
        st.warning("⚠️ No classes found in the specified path")
        return

    st.success(f"✅ {len(available_classes)} classes detected: {', '.join(available_classes)}")

    # Optional mapping
    with st.expander("🔗 Class Grouping (Optional)"):
        mapping_text = st.text_area(
            "Define groupings",
            value="\n".join(available_classes),
            help="Format: target: source1, source2"
        )
        class_mapping = parse_class_mapping(mapping_text)

    # Selection and targets
    st.markdown("### 🎯 Generation Configuration")

    selected_classes = st.multiselect(
        "Select classes to generate",
        options=available_classes,
        default=available_classes
    )

    if not selected_classes:
        st.info("👆 Select at least one class")
        return

    desired_synthetic = {}
    cols = st.columns(min(4, len(selected_classes)))
    for i, cls in enumerate(selected_classes):
        with cols[i % 4]:
            count = st.number_input(
                f"📌 {cls}",
                min_value=0,
                value=100,
                key=f"gen_{cls}"
            )
            if count > 0:
                desired_synthetic[cls] = count

    total = sum(desired_synthetic.values())
    st.info(f"📦 Total to generate: **{total}** samples")

    if st.button("🚀 Generate Dataset", type="primary", width='stretch'):
        execute_augmentation(
            config, params, None, selected_classes,
            desired_synthetic, "External folder", objects_path, class_mapping, {}
        )


def _quick_export(coco_data: dict, images_dir: str, output_dir: str, format: str):
    """Quick export to a single format."""
    try:
        from src.export import ExportManager, ExportConfig

        export_path = os.path.join(output_dir, f"export_{format}")
        config = ExportConfig(output_dir=export_path, copy_images=True)
        manager = ExportManager(export_path)
        manager.configure_format(format, config)

        with st.spinner(f"Exporting to {format.upper()}..."):
            result = manager.export_format(format, coco_data, images_dir, "dataset")

        if result.success:
            st.success(f"✅ Exported to {format.upper()}: `{result.output_path}`")
        else:
            st.error(f"❌ Export failed: {result.errors}")
    except Exception as e:
        st.error(f"❌ Export error: {e}")


def _quick_split(coco_data: dict, images_dir: str, output_dir: str):
    """Quick split with default settings."""
    try:
        from src.splits import DatasetSplitter, SplitConfig

        split_path = os.path.join(output_dir, "splits")
        splitter = DatasetSplitter(split_path)
        config = SplitConfig(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)

        with st.spinner("Creating train/val/test splits..."):
            splits = splitter.split_dataset(
                coco_data, images_dir,
                strategy='stratified',
                config=config,
                copy_images=True
            )

        st.success(f"""✅ Splits created in `{split_path}`:
        - Train: {len(splits['train']['images'])} images
        - Val: {len(splits['val']['images'])} images
        - Test: {len(splits['test']['images'])} images
        """)
    except Exception as e:
        st.error(f"❌ Split error: {e}")


def execute_augmentation(config, params, coco_data, selected_classes,
                        desired_synthetic, objects_source, objects_path,
                        class_mapping, original_counts):
    """Executes the augmentation process."""

    # Output directory
    output_dir = config["paths"]["images"].rsplit(os.sep, 1)[0]

    # Create augmentor
    augmentor = SyntheticDataAugmentor(
        output_dir=output_dir,
        rot=params["transformations"]["rotate"],
        scale=params["transformations"]["scale"],
        trans=params["transformations"]["translate"],
        perspective_transform=params["transformations"]["perspective"],
        poisson_blending=params["realism"]["poisson"],
        advanced_color_correction=params["realism"]["advanced_color"],
        blur_consistency=params["realism"]["blur_consist"],
        add_shadows=params["realism"]["shadows"],
        lighting_effects=params["realism"]["lighting"],
        underwater_effect=params["realism"]["underwater"],
        motion_blur=params["realism"]["motion_blur"],
        try_count=params["generation"]["try_count"],
        overlap_threshold=params["generation"]["overlap_threshold"],
        realism_intensity=params["realism"]["intensity"],
        max_upscale_ratio=params["size_control"]["max_upscale"],
        min_area_ratio=params["size_control"]["min_area"],
        max_area_ratio=params["size_control"]["max_area"],
        save_intermediate_steps=params["advanced"]["save_intermediate"]
    )

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Prepare parameters
    augment_params = {
        "coco_data": coco_data,
        "images_path": config["paths"]["backgrounds_dataset"] if coco_data else None,
        "selected_classes": selected_classes,
        "objects_source": "folder" if "external" in objects_source.lower() else "input_dataset",
        "objects_dataset_path": objects_path,
        "backgrounds_dataset_path": config["paths"]["backgrounds_dataset"],
        "max_objects_per_image": params["generation"]["max_objects"],
        "desired_synthetic_per_class": desired_synthetic,
        "class_mapping": class_mapping,
        "progress_bar": progress_bar,
        "status_text": status_text
    }

    # Execute
    start_time = datetime.now()

    with st.spinner("🎨 Generating synthetic images..."):
        try:
            synthetic_counts, synthetic_total = augmentor.augment_dataset(**augment_params)

            elapsed = (datetime.now() - start_time).total_seconds()

            # ===== SAVE TO SESSION STATE FOR POST-PROCESSING =====
            synthetic_json_path = os.path.join(output_dir, "synthetic_dataset.json")
            synthetic_images_path = os.path.join(output_dir, "images")

            # Load the generated dataset and store in session state
            if os.path.exists(synthetic_json_path):
                with open(synthetic_json_path, 'r') as f:
                    generated_data = json.load(f)

                # Store in session state for use in Post-Processing
                st.session_state['last_generated_synthetic'] = generated_data
                st.session_state['last_generated_path'] = synthetic_json_path
                st.session_state['last_generated_images_path'] = synthetic_images_path
                st.session_state['last_generation_time'] = datetime.now().isoformat()

                # Also store the original dataset if provided
                if coco_data:
                    st.session_state['last_original_dataset'] = coco_data

            # Results
            st.success(f"✅ Generation completed in {elapsed:.1f}s")
            st.balloons()

            # Show results
            st.markdown("## 🎉 Results")

            final_counts = {
                cls: original_counts.get(cls, 0) + synthetic_counts.get(cls, 0)
                for cls in set(original_counts.keys()) | set(synthetic_counts.keys())
            }

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Generated images", synthetic_total)
            with col2:
                st.metric("Pasted objects", sum(synthetic_counts.values()))
            with col3:
                st.metric("Time (s)", f"{elapsed:.1f}")

            # Charts
            if original_counts:
                fig_comp = create_comparison_chart(original_counts, synthetic_counts, final_counts)
                st.plotly_chart(fig_comp, width='stretch')
            else:
                fig_synth = create_distribution_chart(synthetic_counts, "Generated Samples", "thermal")
                st.plotly_chart(fig_synth, width='stretch')

            st.info(f"📁 Results saved in: `{output_dir}`")

            # ===== QUICK ACTIONS =====
            st.markdown("---")
            st.markdown("### 🚀 Quick Actions")

            action_cols = st.columns(4)

            with action_cols[0]:
                if st.button("📤 Export to YOLO", key="quick_export_yolo", use_container_width=True):
                    _quick_export(generated_data, synthetic_images_path, output_dir, 'yolo')

            with action_cols[1]:
                if st.button("📤 Export to COCO", key="quick_export_coco", use_container_width=True):
                    _quick_export(generated_data, synthetic_images_path, output_dir, 'coco')

            with action_cols[2]:
                if st.button("✂️ Create Splits", key="quick_splits", use_container_width=True):
                    _quick_split(generated_data, synthetic_images_path, output_dir)

            with action_cols[3]:
                st.markdown("**Go to Post-Processing tab for more options**")

            # Show available data for post-processing
            st.markdown("---")
            st.info("""
            **💡 Tip:** Your generated dataset is now available in the **Post-Processing** tab.
            You can:
            - Export to multiple formats (COCO, YOLO, COCO Segmentation)
            - Create train/val/test splits with stratification
            - Combine with other datasets
            - Apply class balancing
            """)

        except Exception as e:
            st.error(f"❌ Error during generation:\n\n{str(e)}")
            logging.exception("Error in augmentation")


# ========================= POST-PROCESSING TAB =========================

def render_post_processing_tab(config: dict):
    """Renders the post-processing tab for export, splits, and balancing."""

    st.markdown("## Dataset Processing & Export")
    st.markdown("""
    **Process any COCO dataset** - no synthetic generation required!

    Features available:
    - 🏷️ **Label Management**: Rename, delete, merge labels
    - 📤 **Multi-format Export**: COCO, YOLO, Pascal VOC, COCO Segmentation
    - ✂️ **Dataset Splits**: Train/Val/Test with stratification
    - ⚖️ **Class Balancing**: Oversample, undersample, or compute weights
    """)

    # Initialize session state for all dataset sources
    if 'pp_original_data' not in st.session_state:
        st.session_state.pp_original_data = None
    if 'pp_synthetic_data' not in st.session_state:
        st.session_state.pp_synthetic_data = None
    if 'pp_external_data' not in st.session_state:
        st.session_state.pp_external_data = None
    if 'pp_coco_data' not in st.session_state:
        st.session_state.pp_coco_data = None

    # ===== Section 1: Load Datasets =====
    st.markdown("### 1. Load Dataset")

    # Check if there's data from other sources
    has_analyzed = 'coco_data' in st.session_state and st.session_state.coco_data is not None
    has_generated = 'last_generated_synthetic' in st.session_state and st.session_state.last_generated_synthetic is not None

    # Build options dynamically
    data_source_options = ["Upload COCO JSON file"]
    if has_analyzed:
        data_source_options.append("Use analyzed dataset (from Analysis tab)")
    if has_generated:
        data_source_options.append("Use generated synthetic dataset")
    data_source_options.append("Combine multiple datasets")

    # Dataset source selection
    data_source = st.radio(
        "Select data source",
        options=data_source_options,
        horizontal=True,
        key='pp_data_source'
    )

    if data_source == "Upload COCO JSON file":
        # Simple single dataset upload
        single_json = st.file_uploader(
            "Upload your COCO format annotation file",
            type=['json'],
            key='pp_single_json',
            help="Standard COCO JSON format with images, annotations, and categories"
        )
        if single_json:
            try:
                st.session_state.pp_coco_data = json.load(single_json)
                n_images = len(st.session_state.pp_coco_data.get('images', []))
                n_anns = len(st.session_state.pp_coco_data.get('annotations', []))
                n_cats = len(st.session_state.pp_coco_data.get('categories', []))
                st.success(f"✅ Loaded: {n_images} images, {n_anns} annotations, {n_cats} categories")
            except Exception as e:
                st.error(f"Error loading JSON: {e}")

    elif data_source == "Use analyzed dataset (from Analysis tab)" and has_analyzed:
        # Use dataset from Analysis tab
        st.session_state.pp_coco_data = st.session_state.coco_data
        n_images = len(st.session_state.pp_coco_data.get('images', []))
        n_anns = len(st.session_state.pp_coco_data.get('annotations', []))
        st.success(f"✅ Using analyzed dataset: {n_images} images, {n_anns} annotations")

    elif data_source == "Use generated synthetic dataset" and has_generated:
        # Use generated synthetic dataset
        st.session_state.pp_coco_data = st.session_state.last_generated_synthetic
        st.session_state.pp_images_dir = st.session_state.get('last_generated_images_path', config['paths']['images'])
        n_images = len(st.session_state.pp_coco_data.get('images', []))
        n_anns = len(st.session_state.pp_coco_data.get('annotations', []))
        gen_time = st.session_state.get('last_generation_time', 'Unknown')
        st.success(f"✅ Using generated dataset: {n_images} images, {n_anns} annotations (generated: {gen_time})")

    elif data_source == "Combine multiple datasets":
        # Multiple datasets to combine
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Original/Real Dataset**")
            original_json = st.file_uploader(
                "Upload Original COCO JSON",
                type=['json'],
                key='pp_original_json',
                help="Your base dataset with real images"
            )
            if original_json:
                try:
                    st.session_state.pp_original_data = json.load(original_json)
                    n_img = len(st.session_state.pp_original_data.get('images', []))
                    n_ann = len(st.session_state.pp_original_data.get('annotations', []))
                    st.success(f"{n_img} imgs, {n_ann} anns")
                except Exception as e:
                    st.error(f"Error: {e}")

        with col2:
            st.markdown("**Synthetic Dataset**")
            synthetic_json = st.file_uploader(
                "Upload Synthetic COCO JSON",
                type=['json'],
                key='pp_synthetic_json',
                help="Generated synthetic dataset"
            )
            if synthetic_json:
                try:
                    st.session_state.pp_synthetic_data = json.load(synthetic_json)
                    n_img = len(st.session_state.pp_synthetic_data.get('images', []))
                    n_ann = len(st.session_state.pp_synthetic_data.get('annotations', []))
                    st.success(f"{n_img} imgs, {n_ann} anns")
                except Exception as e:
                    st.error(f"Error: {e}")

        with col3:
            st.markdown("**External Dataset (Optional)**")
            external_json = st.file_uploader(
                "Upload External COCO JSON",
                type=['json'],
                key='pp_external_json',
                help="Any additional dataset to include"
            )
            if external_json:
                try:
                    st.session_state.pp_external_data = json.load(external_json)
                    n_img = len(st.session_state.pp_external_data.get('images', []))
                    n_ann = len(st.session_state.pp_external_data.get('annotations', []))
                    st.success(f"{n_img} imgs, {n_ann} anns")
                except Exception as e:
                    st.error(f"Error: {e}")

        # Combination options
        st.markdown("**Combination Options**")
        available_datasets = []
        if st.session_state.pp_original_data:
            available_datasets.append("Original")
        if st.session_state.pp_synthetic_data:
            available_datasets.append("Synthetic")
        if st.session_state.pp_external_data:
            available_datasets.append("External")

        if len(available_datasets) > 0:
            datasets_to_combine = st.multiselect(
                "Select datasets to combine",
                options=available_datasets,
                default=available_datasets,
                key='pp_datasets_to_combine'
            )

            if len(datasets_to_combine) > 0:
                # Merge selected datasets
                combined_data = None
                datasets_map = {
                    "Original": st.session_state.pp_original_data,
                    "Synthetic": st.session_state.pp_synthetic_data,
                    "External": st.session_state.pp_external_data
                }

                for ds_name in datasets_to_combine:
                    ds_data = datasets_map.get(ds_name)
                    if ds_data:
                        if combined_data is None:
                            combined_data = {
                                'info': ds_data.get('info', {}),
                                'licenses': ds_data.get('licenses', []),
                                'categories': ds_data['categories'],
                                'images': list(ds_data['images']),
                                'annotations': list(ds_data['annotations'])
                            }
                        else:
                            # Merge with ID offset
                            max_img_id = max((img['id'] for img in combined_data['images']), default=0)
                            max_ann_id = max((ann['id'] for ann in combined_data['annotations']), default=0)

                            img_id_map = {}
                            for img in ds_data['images']:
                                new_id = img['id'] + max_img_id + 1
                                img_id_map[img['id']] = new_id
                                new_img = img.copy()
                                new_img['id'] = new_id
                                new_img['source_dataset'] = ds_name.lower()
                                combined_data['images'].append(new_img)

                            for ann in ds_data['annotations']:
                                new_ann = ann.copy()
                                new_ann['id'] = ann['id'] + max_ann_id + 1
                                new_ann['image_id'] = img_id_map[ann['image_id']]
                                combined_data['annotations'].append(new_ann)

                st.session_state.pp_coco_data = combined_data

                if combined_data:
                    total_imgs = len(combined_data['images'])
                    total_anns = len(combined_data['annotations'])
                    st.info(f"**Combined Dataset:** {total_imgs} images, {total_anns} annotations from {len(datasets_to_combine)} source(s)")
        else:
            st.warning("Please upload at least one dataset")

    # Show current dataset status
    st.markdown("---")
    st.markdown("**Current Dataset Status**")
    if st.session_state.pp_coco_data:
        status_cols = st.columns(4)
        with status_cols[0]:
            st.metric("Images", len(st.session_state.pp_coco_data.get('images', [])))
        with status_cols[1]:
            st.metric("Annotations", len(st.session_state.pp_coco_data.get('annotations', [])))
        with status_cols[2]:
            st.metric("Categories", len(st.session_state.pp_coco_data.get('categories', [])))
        with status_cols[3]:
            # Check if has synthetic markers
            has_synthetic = any(
                img.get('is_synthetic') or img.get('source_dataset') == 'synthetic'
                for img in st.session_state.pp_coco_data.get('images', [])
            )
            st.metric("Has Synthetic", "Yes" if has_synthetic else "No")

        # Show class distribution preview
        with st.expander("Class Distribution Preview"):
            cat_id_to_name = {
                cat['id']: cat['name']
                for cat in st.session_state.pp_coco_data.get('categories', [])
            }
            class_counts = {}
            for ann in st.session_state.pp_coco_data.get('annotations', []):
                cat_name = cat_id_to_name.get(ann['category_id'], f"Unknown_{ann['category_id']}")
                class_counts[cat_name] = class_counts.get(cat_name, 0) + 1

            if class_counts:
                dist_df = pd.DataFrame([
                    {'Class': k, 'Count': v, 'Percentage': f"{v/sum(class_counts.values())*100:.1f}%"}
                    for k, v in sorted(class_counts.items(), key=lambda x: -x[1])
                ])
                st.dataframe(dist_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No dataset loaded yet. Please upload or select a dataset above.")

    st.markdown("---")

    # ===== Section 2: Label Management =====
    st.markdown("### 2. Label Management")
    st.markdown("Modify labels before exporting: rename, delete, merge, or add new classes.")

    if st.session_state.pp_coco_data:
        with st.expander("Label Operations", expanded=False):
            label_operation = st.radio(
                "Select operation",
                options=["Rename Labels", "Delete Labels", "Merge Labels", "Add New Labels", "Segmentation to BBox"],
                horizontal=True,
                key='pp_label_operation'
            )

            # Get current labels
            current_labels = [cat['name'] for cat in st.session_state.pp_coco_data.get('categories', [])]

            if label_operation == "Rename Labels":
                st.markdown("**Rename existing labels**")
                rename_col1, rename_col2 = st.columns(2)

                with rename_col1:
                    old_label = st.selectbox("Select label to rename", options=current_labels, key='pp_rename_old')
                with rename_col2:
                    new_label_name = st.text_input("New name", value=old_label if old_label else "", key='pp_rename_new')

                if st.button("Rename Label", key='pp_rename_btn'):
                    if old_label and new_label_name and old_label != new_label_name:
                        st.session_state.pp_coco_data = LabelManager.rename_label(
                            st.session_state.pp_coco_data, old_label, new_label_name
                        )
                        st.success(f"Renamed '{old_label}' to '{new_label_name}'")
                        st.rerun()
                    else:
                        st.warning("Please select a label and enter a different new name")

            elif label_operation == "Delete Labels":
                st.markdown("**Delete labels and their annotations**")
                labels_to_delete = st.multiselect(
                    "Select labels to delete",
                    options=current_labels,
                    key='pp_delete_labels'
                )

                delete_anns = st.checkbox("Also delete associated annotations", value=True, key='pp_delete_anns')

                if st.button("Delete Selected Labels", key='pp_delete_btn', type="primary"):
                    if labels_to_delete:
                        st.session_state.pp_coco_data = LabelManager.delete_labels_batch(
                            st.session_state.pp_coco_data, labels_to_delete, delete_anns
                        )
                        st.success(f"Deleted {len(labels_to_delete)} labels")
                        st.rerun()
                    else:
                        st.warning("Please select at least one label to delete")

            elif label_operation == "Merge Labels":
                st.markdown("**Merge multiple labels into one**")
                source_labels = st.multiselect(
                    "Select labels to merge (will be deleted)",
                    options=current_labels,
                    key='pp_merge_sources'
                )

                merge_target = st.text_input(
                    "Target label name (existing or new)",
                    key='pp_merge_target'
                )

                if st.button("Merge Labels", key='pp_merge_btn'):
                    if source_labels and merge_target:
                        st.session_state.pp_coco_data = LabelManager.merge_labels(
                            st.session_state.pp_coco_data, source_labels, merge_target
                        )
                        st.success(f"Merged {len(source_labels)} labels into '{merge_target}'")
                        st.rerun()
                    else:
                        st.warning("Please select source labels and enter a target name")

            elif label_operation == "Add New Labels":
                st.markdown("**Add new empty labels**")
                new_labels_text = st.text_area(
                    "New label names (one per line)",
                    height=100,
                    key='pp_new_labels'
                )

                supercategory = st.text_input("Supercategory (optional)", key='pp_supercategory')

                if st.button("Add Labels", key='pp_add_labels_btn'):
                    if new_labels_text.strip():
                        new_labels = [l.strip() for l in new_labels_text.strip().split('\n') if l.strip()]
                        st.session_state.pp_coco_data = LabelManager.add_labels_batch(
                            st.session_state.pp_coco_data, new_labels, supercategory or None
                        )
                        st.success(f"Added {len(new_labels)} new labels")
                        st.rerun()
                    else:
                        st.warning("Please enter at least one label name")

            elif label_operation == "Segmentation to BBox":
                st.markdown("**Convert segmentation masks to bounding boxes**")
                overwrite = st.checkbox("Overwrite existing bounding boxes", value=False, key='pp_overwrite_bbox')

                if st.button("Convert Segmentations", key='pp_seg_to_bbox_btn'):
                    st.session_state.pp_coco_data = LabelManager.segmentation_to_bbox(
                        st.session_state.pp_coco_data, overwrite
                    )
                    st.success("Converted segmentations to bounding boxes")
                    st.rerun()

            # Show label statistics
            st.markdown("---")
            st.markdown("**Current Label Statistics**")
            label_stats = LabelManager.get_label_statistics(st.session_state.pp_coco_data)
            if label_stats:
                stats_df = pd.DataFrame([
                    {
                        'Label': name,
                        'ID': stats['id'],
                        'Annotations': stats['annotation_count'],
                        'Images': stats['image_count'],
                        '%': f"{stats['percentage']:.1f}%"
                    }
                    for name, stats in sorted(label_stats.items(), key=lambda x: -x[1]['annotation_count'])
                ])
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
    else:
        st.info("Load a dataset to enable label management features")

    st.markdown("---")

    # ===== Section 3: Export Formats =====
    st.markdown("### 3. Export Formats")

    export_formats = st.multiselect(
        "Select annotation formats to export",
        options=['coco', 'yolo', 'coco_segmentation', 'pascal_voc'],
        default=['coco', 'yolo'],
        help="COCO: Standard JSON | YOLO: Per-image .txt | COCO Segmentation: With polygons | Pascal VOC: XML"
    )

    with st.expander("Export Format Details"):
        st.markdown("""
        | Format | Description | Output |
        |--------|-------------|--------|
        | **COCO** | Standard COCO JSON with enhanced metadata | `dataset.json` |
        | **YOLO** | Per-image labels with normalized coordinates | `images/`, `labels/`, `classes.txt`, `data.yaml` |
        | **COCO Segmentation** | COCO format with polygon segmentations | `dataset.json` with segmentation field |
        | **Pascal VOC** | XML annotations per image | `Annotations/`, `JPEGImages/`, `ImageSets/` |
        """)

    st.markdown("---")

    # ===== Section 4: Dataset Splits =====
    st.markdown("### 4. Dataset Splits")

    enable_splits = st.checkbox("Enable train/val/test splits", value=True)

    # Default values for splits
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1
    split_strategy = 'stratified'
    enable_kfolds = False
    n_folds = 5
    stratified_folds = True

    if enable_splits:
        col1, col2, col3 = st.columns(3)

        with col1:
            train_ratio = st.slider("Train %", 0, 100, 70, key='pp_train') / 100
        with col2:
            val_ratio = st.slider("Val %", 0, 100, 20, key='pp_val') / 100
        with col3:
            test_ratio = 1 - train_ratio - val_ratio
            st.metric("Test %", f"{test_ratio*100:.0f}%")

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
            st.warning("Ratios adjusted to sum to 100%")

        split_strategy = st.radio(
            "Split strategy",
            options=['stratified', 'random'],
            index=0,
            horizontal=True,
            help="Stratified maintains class distribution across splits"
        )

        # K-Folds option
        enable_kfolds = st.checkbox("Also generate K-Folds for cross-validation", value=False)
        if enable_kfolds:
            n_folds = st.number_input("Number of folds", min_value=2, max_value=10, value=5)
            stratified_folds = st.checkbox("Stratified K-Folds", value=True)

    st.markdown("---")

    # ===== Section 5: Class Balancing =====
    st.markdown("### 5. Class Balancing")

    enable_balancing = st.checkbox("Enable class balancing", value=False)

    if enable_balancing:
        balance_strategy = st.selectbox(
            "Balancing strategy",
            options=['oversample', 'undersample', 'hybrid'],
            index=0,
            help="Oversample: duplicate minorities | Undersample: reduce majorities | Hybrid: both to median"
        )

        target_type = st.radio(
            "Target count",
            options=['Auto (max/min/median)', 'Custom value'],
            horizontal=True
        )

        if target_type == 'Custom value':
            balance_target = st.number_input(
                "Target samples per class",
                min_value=10,
                max_value=10000,
                value=500
            )
        else:
            balance_target = None

        max_oversample = st.slider(
            "Max oversample ratio",
            min_value=1.0,
            max_value=10.0,
            value=5.0,
            help="Maximum times a sample can be duplicated"
        )
    else:
        balance_strategy = 'oversample'
        balance_target = None
        max_oversample = 5.0

    # Class weights
    compute_weights = st.checkbox("Compute class weights for training", value=True)
    if compute_weights:
        weights_method = st.selectbox(
            "Weights calculation method",
            options=['inverse_frequency', 'effective_samples', 'focal', 'sqrt_inverse'],
            index=0,
            help="Method to calculate class weights for loss functions"
        )
    else:
        weights_method = 'inverse_frequency'

    st.markdown("---")

    # ===== Section 6: Output Configuration =====
    st.markdown("### 6. Output Configuration")

    output_dir = st.text_input(
        "Output directory",
        value=config.get('pipeline', {}).get('output_dir', '/app/output/processed'),
        help="Directory where processed datasets will be saved"
    )

    images_dir = st.text_input(
        "Source images directory",
        value=config['paths']['images'],
        help="Directory containing the source images"
    )

    copy_images = st.checkbox("Copy images to output directories", value=True)
    random_seed = st.number_input("Random seed", value=42, help="For reproducibility")

    st.markdown("---")

    # ===== Section 7: Run Pipeline =====
    st.markdown("### 7. Run Pipeline")

    # Summary
    with st.expander("Pipeline Configuration Summary", expanded=True):
        summary_cols = st.columns(3)
        with summary_cols[0]:
            st.markdown("**Export**")
            st.write(f"Formats: {', '.join(export_formats)}")
        with summary_cols[1]:
            st.markdown("**Splits**")
            if enable_splits:
                st.write(f"Strategy: {split_strategy}")
                st.write(f"Ratios: {train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%}")
            else:
                st.write("Disabled")
        with summary_cols[2]:
            st.markdown("**Balancing**")
            if enable_balancing:
                st.write(f"Strategy: {balance_strategy}")
            else:
                st.write("Disabled")

    # Run button
    run_disabled = st.session_state.pp_coco_data is None

    if st.button("Run Post-Processing Pipeline", type="primary", disabled=run_disabled):
        if st.session_state.pp_coco_data is None:
            st.error("Please load a dataset first!")
            return

        try:
            with st.spinner("Running post-processing pipeline..."):
                # Create config
                # Note: Dataset combination is now handled in the UI loading section,
                # so pp_coco_data already contains the combined/selected dataset
                pp_config = PostProcessingConfig(
                    output_dir=output_dir,
                    export_formats=export_formats,
                    copy_images=copy_images,
                    enable_splits=enable_splits,
                    split_strategy=split_strategy,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                    enable_kfolds=enable_kfolds,
                    n_folds=n_folds,
                    stratified_folds=stratified_folds,
                    enable_balancing=enable_balancing,
                    balance_strategy=balance_strategy,
                    balance_target=balance_target,
                    max_oversample_ratio=max_oversample,
                    compute_weights=compute_weights,
                    weights_method=weights_method,
                    combine_synthetic=False,  # Combination already done in UI
                    random_seed=random_seed
                )

                # Run pipeline with the already combined/loaded dataset
                pipeline = PostProcessingPipeline(pp_config)
                results = pipeline.run(
                    coco_data=st.session_state.pp_coco_data,
                    images_dir=images_dir,
                    synthetic_data=None  # Not needed, combination done in UI
                )

                st.success("Pipeline completed successfully!")

                # Display results
                st.markdown("#### Results")

                # Splits results
                if 'splits' in results['stages'] and not results['stages']['splits'].get('skipped'):
                    split_info = results['stages']['splits']
                    st.markdown("**Dataset Splits:**")
                    split_df = pd.DataFrame({
                        'Split': ['Train', 'Val', 'Test'],
                        'Images': [
                            split_info.get('train_images', 0),
                            split_info.get('val_images', 0),
                            split_info.get('test_images', 0)
                        ],
                        'Annotations': [
                            split_info.get('train_annotations', 0),
                            split_info.get('val_annotations', 0),
                            split_info.get('test_annotations', 0)
                        ]
                    })
                    st.dataframe(split_df, use_container_width=True)

                # Balancing results
                if 'balancing' in results['stages'] and not results['stages']['balancing'].get('skipped'):
                    bal_info = results['stages']['balancing']
                    st.markdown("**Class Balancing:**")
                    st.write(f"Original annotations: {bal_info.get('total_original', 0)}")
                    st.write(f"Balanced annotations: {bal_info.get('total_balanced', 0)}")

                # Export results
                if 'exports' in results['stages']:
                    st.markdown("**Exports:**")
                    for split_name, formats in results['stages']['exports'].items():
                        if isinstance(formats, dict):
                            for fmt, info in formats.items():
                                if isinstance(info, dict):
                                    status = "OK" if info.get('success') else "FAILED"
                                    st.write(f"- {split_name}/{fmt}: {status}")

                st.info(f"Results saved to: `{output_dir}`")

        except Exception as e:
            st.error(f"Error running pipeline: {str(e)}")
            logging.exception("Error in post-processing pipeline")

    if run_disabled:
        st.info("Please load a dataset to enable the pipeline.")


# ========================= MAIN APPLICATION =========================

def main():
    """Main application."""

    # Page configuration
    st.set_page_config(
        page_title="Synthetic Dataset Generator Pro",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom adaptive CSS for light/dark mode
    st.markdown("""
    <style>
    /* Respect user theme (light/dark) */
    .stApp {
        /* No custom background - uses system theme */
    }

    /* Improve tabs style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: rgba(128, 128, 128, 0.1);
        border-radius: 8px 8px 0 0;
        padding: 12px 20px;
        transition: background-color 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(128, 128, 128, 0.2);
    }

    /* Ensure text is visible in both modes */
    .stMarkdown, .stText {
        color: inherit;
    }

    /* Improve metrics contrast */
    [data-testid="stMetricValue"] {
        font-weight: 600;
    }

    /* Improve info boxes visibility */
    .stAlert {
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Setup
    setup_logging()
    config = load_config()

    # Header
    render_header()

    # Sidebar - Workflow status first, then configuration
    render_workflow_status()
    params = render_sidebar_config(config)

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dataset Analysis",
        "⚖️ Balancing",
        "🎨 Generation from Scratch",
        "🔧 Post-Processing",
        "📚 Documentation"
    ])

    with tab1:
        render_dataset_analysis_tab(config)

    with tab2:
        render_balancing_tab(config, params)

    with tab3:
        render_generation_tab(config, params)

    with tab4:
        render_post_processing_tab(config)

    with tab5:
        st.markdown("""
        ## 📚 User Guide

        ### 🚀 Quick Start - Process Any Dataset

        **No synthetic generation required!** You can use Post-Processing directly:

        1. Go to **Post-Processing** tab
        2. Upload your COCO JSON file
        3. Use Label Management, Export, Splits, or Balancing features
        4. Run the pipeline

        ---

        ### 📊 Full Workflow (with Synthetic Generation)

        #### Step 1: Dataset Analysis (Optional)
        - Upload your COCO JSON in the **Analysis** tab
        - Review class distribution and statistics
        - Identify underrepresented classes

        #### Step 2: Synthetic Data Generation (Optional)
        - Go to **Balancing** tab to generate synthetic samples
        - Or use **Generation from Scratch** for new datasets

        #### Step 3: Post-Processing
        - Upload any dataset OR use analyzed/generated data
        - Apply label modifications
        - Export to multiple formats
        - Create train/val/test splits

        ### 📦 Export Formats

        | Format | Description | Output Files |
        |--------|-------------|--------------|
        | **COCO** | Standard JSON format | `dataset.json` |
        | **YOLO** | Per-image text files | `images/`, `labels/`, `classes.txt`, `data.yaml` |
        | **COCO Segmentation** | With polygon masks | `dataset.json` (with segmentation field) |
        | **Pascal VOC** | XML format per image | `Annotations/`, `JPEGImages/`, `ImageSets/` |

        ### 🏷️ Label Management

        - **Rename Labels**: Change label names without affecting annotations
        - **Delete Labels**: Remove labels and optionally their annotations
        - **Merge Labels**: Combine multiple labels into one
        - **Add New Labels**: Create empty class categories
        - **Segmentation to BBox**: Convert polygon masks to bounding boxes

        ### ✂️ Dataset Splitting

        - **Stratified Split**: Maintains class distribution across train/val/test
        - **Random Split**: Simple random assignment
        - **K-Folds**: Cross-validation support with configurable folds

        ### ⚖️ Class Balancing

        | Strategy | Description |
        |----------|-------------|
        | **Oversample** | Duplicate minority class samples |
        | **Undersample** | Reduce majority class samples |
        | **Hybrid** | Both strategies to reach median |

        #### Class Weights Methods
        - `inverse_frequency`: weight = total / (n_classes × count)
        - `effective_samples`: Based on "Class-Balanced Loss" paper
        - `focal`: weight = (1 - freq)^gamma
        - `sqrt_inverse`: weight = sqrt(total / count)

        ### ✨ Generation Features

        - ✅ **Dynamic shadows** with light direction analysis
        - ✅ **Z-ordering** for realistic depth
        - ✅ **Context-aware backgrounds** for semantic coherence
        - ✅ **Adaptive upscaling** (LANCZOS4 + sharpening)
        - ✅ **Optimized effects pipeline**
        - ✅ **Caustics cache** (500-1000x faster)
        - ✅ **Robust COCO validation**
        - ✅ **Cross-platform paths**

        ### 🎯 Expected Results

        - **+150% realism** with advanced effects
        - **100-200x faster** with optimizations
        - **No crashes** with robust validation
        - **Photorealistic images** indistinguishable from real ones

        ### 📖 More Information

        Check the [README.md](https://github.com/asferrer/synthetic-data-augmentation)
        for technical details and advanced examples.
        """)


if __name__ == "__main__":
    main()
