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
from app.components.api_client import get_api_client
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

    # Check for active dataset - try multiple sources
    generated = st.session_state.get("generated_dataset")
    output_dir = st.session_state.get("generated_output_dir", "")

    # If no dataset in session, try to load from active dataset
    if not generated:
        active_dataset_id = st.session_state.get("active_dataset_id")

        # Try loading from last completed job if no active dataset set
        if not active_dataset_id:
            current_job = st.session_state.get("current_job_id")
            if current_job:
                active_dataset_id = current_job

        if active_dataset_id:
            client = get_api_client()

            with st.spinner(f"Cargando dataset {active_dataset_id[-8:]}..."):
                result = client.load_dataset_coco(active_dataset_id)

                if result.get("success"):
                    generated = result["data"]
                    st.session_state.generated_dataset = generated

                    # Get metadata for output_dir
                    metadata = client.get_dataset_metadata(active_dataset_id)
                    if not metadata.get("error"):
                        output_dir = metadata.get("images_dir", "").replace("/images", "")
                        st.session_state.generated_output_dir = output_dir

                    st.success("✅ Dataset cargado desde la base de datos")
                    st.rerun()

    # If still no dataset, offer to browse all available datasets
    if not generated:
        alert_box(
            "No hay dataset activo. Selecciona uno de los datasets disponibles abajo.",
            type="info",
            icon="ℹ️"
        )

        # Show all available datasets with better UI
        section_header("Datasets Disponibles", icon="📂")

        # Get API client
        client = get_api_client()

        # Fetch all datasets
        all_datasets_response = client.list_datasets()
        all_datasets = all_datasets_response.get("datasets", [])

        if not all_datasets:
            st.info("No hay datasets generados. Ve a ③ Generar para crear uno.")
        else:
            # Filter and categorize datasets
            # Note: database returns 'dataset_type', not 'type' or 'job_type'
            gen_datasets = [d for d in all_datasets if d.get("dataset_type") == "generation" or d.get("type") == "generation" or d.get("job_type") == "generation"]
            other_datasets = [d for d in all_datasets if d not in gen_datasets]

            st.markdown(f"**{len(all_datasets)} datasets encontrados** ({len(gen_datasets)} de generación)")

            # Tabs for different dataset types
            tab_gen, tab_all = st.tabs(["🎨 Generados", "📦 Todos"])

            with tab_gen:
                if gen_datasets:
                    for ds in sorted(gen_datasets, key=lambda x: x.get("created_at", ""), reverse=True):
                        _render_dataset_selection_card(ds, client, c)
                else:
                    st.info("No hay datasets de generación. Crea uno en ③ Generar.")

            with tab_all:
                if all_datasets:
                    for ds in sorted(all_datasets, key=lambda x: x.get("created_at", ""), reverse=True):
                        _render_dataset_selection_card(ds, client, c)
                else:
                    st.info("No hay datasets disponibles.")

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


def _render_dataset_selection_card(dataset: Dict, client, c: Dict) -> None:
    """Render a dataset card with selection button for export page."""
    from datetime import datetime

    job_id = dataset.get("job_id", "unknown")
    # Handle different field naming conventions
    job_type = dataset.get("dataset_type", dataset.get("type", dataset.get("job_type", "unknown")))
    num_images = dataset.get("num_images", dataset.get("images_count", 0))
    num_annotations = dataset.get("num_annotations", dataset.get("annotations_count", 0))
    created_at = dataset.get("created_at", "")
    status = dataset.get("status", "completed")

    # Type icon and label
    type_icons = {
        "generation": ("🎨", "Generación"),
        "extraction": ("🎯", "Extracción"),
        "sam3_conversion": ("🔬", "SAM3"),
    }
    type_icon, type_label = type_icons.get(job_type, ("📦", job_type.title()))

    # Format timestamp
    try:
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        time_str = dt.strftime("%Y-%m-%d %H:%M")
    except:
        time_str = created_at[:16] if created_at else "Desconocido"

    # Status color
    if status == "completed":
        status_color = c['success']
    elif status in ["processing", "queued"]:
        status_color = c['warning']
    else:
        status_color = c['error']

    st.markdown(f"""
    <div style="background: {c['bg_card']}; border: 1px solid {c['border']};
                border-radius: 0.5rem; padding: 1rem; margin-bottom: 0.75rem;
                border-left: 3px solid {status_color};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="font-size: 1.5rem;">{type_icon}</span>
                <div>
                    <div style="font-weight: 600; color: {c['text_primary']};">
                        {type_label}
                    </div>
                    <div style="font-size: 0.75rem; color: {c['text_muted']}; font-family: monospace;">
                        {job_id[:16]}...
                    </div>
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.9rem; font-weight: 600; color: {c['primary']};">
                    {num_images:,} imgs
                </div>
                <div style="font-size: 0.75rem; color: {c['text_muted']};">
                    📅 {time_str}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Action buttons
    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button(
            "✅ Usar este dataset",
            key=f"export_select_{job_id}",
            type="primary",
            use_container_width=True
        ):
            st.session_state.active_dataset_id = job_id
            st.rerun()

    with col2:
        if st.button(
            "👁️ Ver",
            key=f"export_view_{job_id}",
            use_container_width=True
        ):
            # Show dataset details in expander
            st.session_state[f"show_details_{job_id}"] = True
            st.rerun()

    # Show details if requested
    if st.session_state.get(f"show_details_{job_id}"):
        with st.expander("📋 Detalles del Dataset", expanded=True):
            metadata = client.get_dataset_metadata(job_id)
            if not metadata.get("error"):
                st.json(metadata)
            else:
                st.warning(f"No se pudieron cargar los detalles: {metadata.get('error')}")

            if st.button("Cerrar", key=f"close_details_{job_id}"):
                st.session_state.pop(f"show_details_{job_id}", None)
                st.rerun()
