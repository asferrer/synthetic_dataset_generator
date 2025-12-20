"""
Export Page (Step 4)
====================
Export generated dataset to multiple formats.
"""

import os
import json
import streamlit as st
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from app.components.ui import (
    page_header, section_header, spacer, alert_box, empty_state,
    workflow_stepper, workflow_navigation
)
from app.utils import ExportManager
from app.config.theme import get_colors_dict


def render_export_page():
    """Render the export page (Step 4 of workflow)"""
    c = get_colors_dict()

    # Workflow stepper
    completed = st.session_state.get("workflow_completed", [])
    workflow_stepper(current_step=4, completed_steps=completed)

    page_header(
        title="Exportar Dataset",
        subtitle="Paso 4: Exporta el dataset generado a diferentes formatos",
        icon="📤"
    )

    # Check for generated dataset
    generated = st.session_state.get("generated_dataset")
    output_dir = st.session_state.get("generated_output_dir", "")

    if not generated:
        alert_box(
            "No hay dataset generado. Completa el paso anterior para generar imágenes.",
            type="warning",
            icon="⚠️"
        )

        action = workflow_navigation(
            current_step=4,
            can_go_next=False,
            on_prev="③ Generar"
        )

        if action == "prev":
            st.session_state.nav_menu = "③ Generar"
            st.rerun()
        return

    # Dataset summary
    _render_dataset_summary(generated, output_dir)

    spacer(16)

    # Export section
    section_header("Formatos de Exportación", icon="📦")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div style="background: {c['bg_secondary']}; padding: 1rem; border-radius: 0.5rem;
                    border-left: 3px solid {c['primary']}; margin-bottom: 1rem;">
            <div style="font-size: 0.85rem; color: {c['text_secondary']};">
                Selecciona los formatos en los que deseas exportar tu dataset.
                El formato COCO original siempre está disponible.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Format selection with descriptions
        st.markdown("**Formatos disponibles:**")

        export_coco = st.checkbox(
            "📋 COCO JSON",
            value=True,
            key="export_coco",
            help="Formato nativo COCO - compatible con la mayoría de frameworks"
        )

        export_yolo = st.checkbox(
            "🔲 YOLO (txt + yaml)",
            value=True,
            key="export_yolo",
            help="Formato YOLO v5/v8 - archivos txt por imagen + data.yaml"
        )

        export_voc = st.checkbox(
            "📄 Pascal VOC (xml)",
            value=False,
            key="export_voc",
            help="Formato Pascal VOC - archivos XML por imagen"
        )

    with col2:
        st.markdown("**Configuración de salida:**")

        export_output_dir = st.text_input(
            "Directorio de exportación",
            value=str(Path(output_dir) / "exported") if output_dir else "/app/output/exported",
            key="export_output_dir",
            help="Directorio donde se guardarán los archivos exportados"
        )

        images_source_dir = st.text_input(
            "Directorio de imágenes origen",
            value=output_dir if output_dir else "/app/output/synthetic",
            key="export_images_dir",
            help="Directorio donde están las imágenes generadas"
        )

        copy_images = st.checkbox(
            "📁 Copiar imágenes al directorio de salida",
            value=False,
            key="export_copy_images",
            help="Copiar las imágenes junto con las anotaciones exportadas"
        )

    spacer(16)

    # Build export formats list
    export_formats = []
    if export_coco:
        export_formats.append("coco")
    if export_yolo:
        export_formats.append("yolo")
    if export_voc:
        export_formats.append("pascal_voc")

    can_export = len(export_formats) > 0

    # Export button
    if st.button(
        "🚀 Exportar Dataset",
        type="primary",
        use_container_width=True,
        disabled=not can_export,
        key="export_btn"
    ):
        _perform_export(generated, export_output_dir, export_formats, copy_images, images_source_dir)

    spacer(16)

    # Show export results if available
    if st.session_state.get("export_results"):
        _render_export_results(st.session_state.export_results)

    spacer(24)

    # Quick download of COCO JSON
    with st.expander("⬇️ Descarga rápida COCO JSON", expanded=False):
        json_data = json.dumps(generated, indent=2)
        st.download_button(
            "📥 Descargar COCO JSON",
            data=json_data,
            file_name="synthetic_dataset.json",
            mime="application/json",
            key="quick_download_coco",
            use_container_width=True
        )

    spacer(24)

    # Workflow navigation
    has_exported = st.session_state.get("export_completed", False)

    action = workflow_navigation(
        current_step=4,
        can_go_next=True,  # Allow proceeding even without export
        next_label="Combinar Datasets",
        on_next="⑤ Combinar",
        on_prev="③ Generar"
    )

    if action == "next":
        if 4 not in st.session_state.get("workflow_completed", []):
            st.session_state.workflow_completed = st.session_state.get("workflow_completed", []) + [4]
        st.session_state.workflow_step = 5
        st.rerun()
    elif action == "prev":
        st.session_state.nav_menu = "③ Generar"
        st.rerun()


def _render_dataset_summary(dataset: Dict, output_dir: str) -> None:
    """Render summary of the generated dataset"""
    c = get_colors_dict()
    n_images = len(dataset.get("images", []))
    n_annotations = len(dataset.get("annotations", []))
    n_categories = len(dataset.get("categories", []))

    # Calculate class distribution
    cat_id_to_name = {cat["id"]: cat["name"] for cat in dataset.get("categories", [])}
    class_counts = {}
    for ann in dataset.get("annotations", []):
        cat_name = cat_id_to_name.get(ann.get("category_id"), "Unknown")
        class_counts[cat_name] = class_counts.get(cat_name, 0) + 1

    st.markdown(f"""
    <div style="background: {c['bg_card']}; border: 1px solid {c['border']};
                border-radius: 0.75rem; padding: 1.5rem; margin-bottom: 1rem;">
        <div style="font-size: 0.7rem; color: {c['text_muted']}; text-transform: uppercase;
                    letter-spacing: 0.05em; margin-bottom: 1rem;">
            Dataset Generado
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; text-align: center;">
            <div>
                <div style="font-size: 2.5rem; font-weight: 700; color: {c['primary']};">{n_images:,}</div>
                <div style="font-size: 0.85rem; color: {c['text_muted']};">Imágenes</div>
            </div>
            <div>
                <div style="font-size: 2.5rem; font-weight: 700; color: {c['text_primary']};">{n_annotations:,}</div>
                <div style="font-size: 0.85rem; color: {c['text_muted']};">Anotaciones</div>
            </div>
            <div>
                <div style="font-size: 2.5rem; font-weight: 700; color: {c['text_primary']};">{n_categories}</div>
                <div style="font-size: 0.85rem; color: {c['text_muted']};">Clases</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Class distribution in expander
    with st.expander("📊 Distribución por clase", expanded=False):
        if class_counts:
            df = pd.DataFrame([
                {"Clase": k, "Anotaciones": v}
                for k, v in sorted(class_counts.items(), key=lambda x: -x[1])
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)


def _perform_export(
    dataset: Dict,
    output_dir: str,
    formats: list,
    copy_images: bool,
    images_dir: str
) -> None:
    """Perform the actual export"""
    with st.spinner(f"Exportando a {', '.join(formats)}..."):
        try:
            results = ExportManager.export(
                dataset,
                output_dir,
                formats,
                copy_images,
                images_dir
            )

            st.session_state.export_results = results
            st.session_state.export_completed = True

            # Check if all succeeded
            all_success = all(r.get("success") for r in results.values())
            if all_success:
                st.success("✅ Exportación completada exitosamente")
            else:
                st.warning("⚠️ Algunos formatos tuvieron errores")

            st.rerun()

        except Exception as e:
            st.error(f"❌ Error durante la exportación: {e}")


def _render_export_results(results: Dict) -> None:
    """Render export results"""
    c = get_colors_dict()
    section_header("Resultados de Exportación", icon="✅")

    for fmt, result in results.items():
        if result.get("success"):
            output_path = result.get("output_path", "")
            st.markdown(f"""
            <div style="background: {c['success_bg']}; border: 1px solid {c['success']};
                        border-radius: 0.5rem; padding: 0.75rem; margin-bottom: 0.5rem;
                        display: flex; align-items: center; gap: 0.75rem;">
                <span style="color: {c['success']}; font-size: 1.25rem;">✓</span>
                <div>
                    <div style="font-weight: 600; color: {c['success']};">{fmt.upper()}</div>
                    <div style="font-size: 0.8rem; color: {c['text_muted']};">{output_path}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            error = result.get("error", "Unknown error")
            st.markdown(f"""
            <div style="background: {c['error_bg']}; border: 1px solid {c['error']};
                        border-radius: 0.5rem; padding: 0.75rem; margin-bottom: 0.5rem;
                        display: flex; align-items: center; gap: 0.75rem;">
                <span style="color: {c['error']}; font-size: 1.25rem;">✗</span>
                <div>
                    <div style="font-weight: 600; color: {c['error']};">{fmt.upper()}</div>
                    <div style="font-size: 0.8rem; color: {c['text_muted']};">{error}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
