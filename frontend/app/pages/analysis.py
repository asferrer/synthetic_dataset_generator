"""
Analysis Page (Step 1)
======================
COCO dataset analysis with workflow stepper integration.
"""

import json
import streamlit as st
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from app.components.ui import (
    page_header, section_header, spacer, alert_box, empty_state,
    workflow_stepper, workflow_navigation
)
from app.config.theme import get_colors_dict


def analyze_coco_dataset(coco_data: Dict) -> Dict[str, Any]:
    """
    Analyze a COCO dataset and return statistics.
    """
    categories = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}

    class_counts = Counter()
    for ann in coco_data.get("annotations", []):
        cat_id = ann.get("category_id")
        if cat_id in categories:
            class_counts[categories[cat_id]] += 1

    num_images = len(coco_data.get("images", []))
    num_annotations = len(coco_data.get("annotations", []))

    anns_per_image = Counter()
    for ann in coco_data.get("annotations", []):
        anns_per_image[ann.get("image_id")] += 1

    ann_counts = list(anns_per_image.values()) if anns_per_image else [0]

    bbox_areas = []
    for ann in coco_data.get("annotations", []):
        bbox = ann.get("bbox", [0, 0, 0, 0])
        if len(bbox) >= 4:
            area = bbox[2] * bbox[3]
            bbox_areas.append(area)

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
    }


def calculate_balancing_targets(
    analysis: Dict,
    strategy: str,
    selected_classes: List[str],
) -> Dict[str, int]:
    """Calculate synthetic instances needed per class."""
    class_counts = analysis.get("class_counts", {})
    max_count = max(class_counts.values()) if class_counts else 0

    targets = {}

    for cls in selected_classes:
        current = class_counts.get(cls, 0)

        if strategy == "complete":
            targets[cls] = max(0, max_count - current)
        elif strategy == "partial":
            target = int(max_count * 0.75)
            targets[cls] = max(0, target - current)
        elif strategy == "minority":
            median = np.median(list(class_counts.values())) if class_counts else 0
            if current < median:
                targets[cls] = max(0, int(median) - current)
            else:
                targets[cls] = 0
        else:
            targets[cls] = 0

    return targets


def _render_metric_card(title: str, value: str, icon: str, color: str = "primary", subtitle: str = "") -> None:
    """Render a styled metric card."""
    c = get_colors_dict()
    color_map = {
        "primary": c['primary'],
        "success": c['success'],
        "warning": c['warning'],
        "error": c['error'],
        "info": "#3b82f6",
    }
    accent = color_map.get(color, color_map["primary"])
    subtitle_html = f'<div style="font-size: 0.7rem; color: {c["text_muted"]}; margin-top: 0.25rem;">{subtitle}</div>' if subtitle else ""

    st.markdown(f"""
    <div style="background: {c['bg_card']}; border: 1px solid {c['border']};
                border-radius: 0.75rem; padding: 1.25rem; text-align: center;
                border-top: 3px solid {accent};">
        <div style="font-size: 1.75rem; margin-bottom: 0.5rem;">{icon}</div>
        <div style="font-size: 2rem; font-weight: 700; color: {accent}; line-height: 1.2;">{value}</div>
        <div style="font-size: 0.8rem; color: {c['text_muted']}; margin-top: 0.5rem;
                    text-transform: uppercase; letter-spacing: 0.05em;">{title}</div>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)


def render_analysis_page():
    """Render the COCO analysis page (Step 1 of workflow)"""
    c = get_colors_dict()

    # Workflow stepper
    current_step = st.session_state.get("workflow_step", 1)
    completed = st.session_state.get("workflow_completed", [])
    workflow_stepper(current_step=1, completed_steps=completed)

    page_header(
        title="Análisis de Dataset",
        subtitle="Paso 1: Analiza tu dataset COCO y planifica la generación de datos sintéticos",
        icon="📊"
    )

    # Check if we already have data from Home
    has_source = st.session_state.get("source_dataset") is not None

    if not has_source:
        # File upload section
        section_header("Cargar Dataset", icon="📁")

        upload_method = st.radio(
            "Método de entrada",
            ["Subir archivo JSON", "Introducir ruta"],
            horizontal=True,
            key="analysis_input_method"
        )

        coco_data = None

        if upload_method == "Subir archivo JSON":
            uploaded = st.file_uploader(
                "Arrastra tu archivo COCO JSON aquí",
                type=["json"],
                key="coco_upload"
            )

            if uploaded:
                try:
                    coco_data = json.load(uploaded)
                    st.session_state.source_dataset = coco_data
                    st.session_state.source_filename = uploaded.name
                    st.success(f"Cargado: **{uploaded.name}**")
                except Exception as e:
                    alert_box(f"Error al parsear JSON: {e}", type="error")
        else:
            json_path = st.text_input(
                "Ruta del archivo COCO JSON",
                placeholder="/app/datasets/annotations.json",
                key="coco_path"
            )

            if json_path and Path(json_path).exists():
                try:
                    with open(json_path) as f:
                        coco_data = json.load(f)
                    st.session_state.source_dataset = coco_data
                    st.session_state.source_filename = Path(json_path).name
                    st.success(f"Cargado: **{json_path}**")
                except Exception as e:
                    alert_box(f"Error al cargar: {e}", type="error")
            elif json_path:
                alert_box(f"Archivo no encontrado: {json_path}", type="warning")

        if not st.session_state.get("source_dataset"):
            spacer(32)
            empty_state(
                title="Sin Dataset Cargado",
                message="Sube un archivo COCO JSON o introduce una ruta para comenzar el análisis.",
                icon="📁"
            )
            return

    # We have data - run analysis
    coco_data = st.session_state.source_dataset
    filename = st.session_state.get("source_filename", "dataset.json")

    with st.spinner("Analizando dataset..."):
        analysis = analyze_coco_dataset(coco_data)

    st.session_state.analysis_result = analysis

    spacer(16)

    # Dataset info banner
    st.markdown(f"""
    <div style="background: {c['bg_secondary']}; border: 1px solid {c['border']};
                border-radius: 0.5rem; padding: 0.75rem 1rem; margin-bottom: 1rem;
                display: flex; align-items: center; gap: 0.5rem;">
        <span style="font-size: 1.25rem;">📄</span>
        <span style="font-weight: 600; color: {c['text_primary']};">{filename}</span>
        <span style="color: {c['text_muted']};">|</span>
        <span style="color: {c['text_secondary']};">{analysis['num_images']:,} imágenes</span>
    </div>
    """, unsafe_allow_html=True)

    # Dataset Overview
    section_header("Vista General", icon="📈")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        _render_metric_card("Imágenes", f"{analysis['num_images']:,}", "🖼️", "primary")
    with col2:
        _render_metric_card("Anotaciones", f"{analysis['num_annotations']:,}", "🏷️", "info")
    with col3:
        _render_metric_card("Clases", str(analysis["num_classes"]), "📦", "success")
    with col4:
        _render_metric_card("Media/Imagen", f"{analysis['stats']['mean_anns_per_image']:.1f}", "📊", "warning")

    spacer(24)

    # Class Distribution
    section_header("Distribución de Clases", icon="📊")

    class_counts = analysis["class_counts"]

    if class_counts:
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        df = pd.DataFrame(sorted_classes, columns=["Clase", "Cantidad"])

        max_count = df["Cantidad"].max()
        min_count = df["Cantidad"].min()
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

        col1, col2 = st.columns([3, 1])

        with col1:
            st.bar_chart(df.set_index("Clase"), use_container_width=True, height=300)

        with col2:
            _render_metric_card("Máximo", f"{max_count:,}", "📈", "success")
            spacer(12)
            _render_metric_card("Mínimo", f"{min_count:,}", "📉", "error")
            spacer(12)

            if imbalance_ratio > 10:
                imbalance_color, imbalance_status = "error", "Crítico"
            elif imbalance_ratio > 3:
                imbalance_color, imbalance_status = "warning", "Alto"
            else:
                imbalance_color, imbalance_status = "success", "OK"

            _render_metric_card("Desbalance", f"{imbalance_ratio:.1f}x", "⚖️", imbalance_color, imbalance_status)

        # Detailed table
        with st.expander("📋 Estadísticas Detalladas por Clase"):
            df["Porcentaje"] = (df["Cantidad"] / df["Cantidad"].sum() * 100).round(1).astype(str) + "%"
            df["Gap al Máx"] = max_count - df["Cantidad"]
            st.dataframe(df, use_container_width=True, hide_index=True)

    spacer(24)

    # Balancing Configuration
    section_header("Configuración de Balanceo", icon="⚖️")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div style="background: {c['bg_secondary']}; padding: 1rem; border-radius: 0.5rem;
                    margin-bottom: 1rem; border-left: 3px solid {c['primary']};">
            <div style="font-size: 0.85rem; color: {c['text_secondary']};">
                Selecciona una estrategia para determinar cuántas imágenes sintéticas generar por clase.
            </div>
        </div>
        """, unsafe_allow_html=True)

        strategy = st.selectbox(
            "Estrategia de Balanceo",
            ["complete", "partial", "minority", "custom"],
            format_func=lambda x: {
                "complete": "🎯 Completo - Balancear todas al máximo",
                "partial": "📊 Parcial - Balancear al 75% del máximo",
                "minority": "📉 Minoritarias - Solo clases subrepresentadas",
                "custom": "✏️ Personalizado - Definir muestras por clase",
            }.get(x, x),
            key="balancing_strategy"
        )

        selected_classes = st.multiselect(
            "Clases a Balancear",
            options=analysis["categories"],
            default=analysis["categories"],
            key="selected_classes"
        )

        # Custom samples per class
        if strategy == "custom" and selected_classes:
            st.markdown(f"""
            <div style="background: {c['bg_tertiary']}; padding: 0.75rem; border-radius: 0.5rem;
                        margin-top: 1rem; margin-bottom: 0.5rem;">
                <div style="font-size: 0.8rem; font-weight: 600; color: {c['text_primary']};">
                    Muestras a generar por clase:
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Initialize custom targets in session state if needed
            if "custom_targets" not in st.session_state:
                st.session_state.custom_targets = {}

            custom_targets = {}
            for cls in selected_classes:
                current_count = class_counts.get(cls, 0)
                default_val = st.session_state.custom_targets.get(cls, 0)
                custom_targets[cls] = st.number_input(
                    f"{cls} (actual: {current_count})",
                    min_value=0,
                    max_value=10000,
                    value=default_val,
                    step=10,
                    key=f"custom_target_{cls}"
                )
            st.session_state.custom_targets = custom_targets

    with col2:
        if selected_classes:
            # Calculate targets based on strategy
            if strategy == "custom":
                targets = st.session_state.get("custom_targets", {})
            else:
                targets = calculate_balancing_targets(analysis, strategy, selected_classes)

            total_synthetic = sum(targets.values())
            classes_to_generate = len([cls for cls in selected_classes if targets.get(cls, 0) > 0])

            # Store targets
            st.session_state.balancing_targets = targets

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {c['primary']}, {c['primary_hover']});
                        border-radius: 0.75rem; padding: 1.5rem; text-align: center; color: white;">
                <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.9;">
                    Total Imágenes Sintéticas
                </div>
                <div style="font-size: 3rem; font-weight: 700; margin: 0.5rem 0;">
                    {total_synthetic:,}
                </div>
                <div style="font-size: 0.85rem; opacity: 0.9;">
                    en {classes_to_generate} clases
                </div>
            </div>
            """, unsafe_allow_html=True)

            spacer(16)

            target_df = pd.DataFrame([
                {"Clase": cls, "Actual": class_counts.get(cls, 0), "A Generar": targets.get(cls, 0)}
                for cls in selected_classes if targets.get(cls, 0) > 0
            ])

            if not target_df.empty:
                st.dataframe(target_df, use_container_width=True, hide_index=True)
        else:
            alert_box("Selecciona al menos una clase para configurar el balanceo.", type="info")

    # Mark step as ready and update workflow
    if selected_classes and sum(st.session_state.get("balancing_targets", {}).values()) > 0:
        st.session_state.workflow_step = max(st.session_state.get("workflow_step", 0), 1)

    spacer(16)

    # Workflow navigation
    can_proceed = bool(selected_classes and st.session_state.get("balancing_targets"))

    action = workflow_navigation(
        current_step=1,
        can_go_next=can_proceed,
        next_label="Configurar Generación",
        on_next="② Configurar"
    )

    if action == "next" and can_proceed:
        # Mark step 1 as completed
        if 1 not in st.session_state.get("workflow_completed", []):
            st.session_state.workflow_completed = st.session_state.get("workflow_completed", []) + [1]
        st.session_state.workflow_step = 2
        st.rerun()
