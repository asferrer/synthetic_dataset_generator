"""
Combine Page (Step 5)
=====================
Merge generated dataset with original or other datasets.
"""

import json
import streamlit as st
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

from app.components.ui import (
    page_header, section_header, spacer, alert_box, empty_state,
    workflow_stepper, workflow_navigation
)
from app.utils import DatasetMerger
from app.config.theme import get_colors_dict


def render_combine_page():
    """Render the combine datasets page (Step 5 of workflow)"""
    c = get_colors_dict()

    # Workflow stepper
    completed = st.session_state.get("workflow_completed", [])
    workflow_stepper(current_step=5, completed_steps=completed)

    page_header(
        title="Combinar Datasets",
        subtitle="Paso 5: Combina el dataset sintético con el original u otros datasets",
        icon="🔗"
    )

    # Initialize datasets to combine list
    if "datasets_to_combine" not in st.session_state:
        st.session_state.datasets_to_combine = []

    # Auto-add available datasets from workflow
    _auto_add_workflow_datasets()

    spacer(16)

    # Datasets selection section
    section_header("Datasets a Combinar", icon="📦")

    # Current datasets in list
    if st.session_state.datasets_to_combine:
        _render_datasets_list()
    else:
        empty_state(
            title="No hay datasets seleccionados",
            message="Añade datasets para combinarlos.",
            icon="📭"
        )

    spacer(16)

    # Add more datasets section
    _render_add_dataset_section()

    spacer(24)

    # Merge configuration
    if len(st.session_state.datasets_to_combine) >= 2:
        _render_merge_config()

    spacer(24)

    # Workflow navigation
    has_combined = st.session_state.get("combined_dataset") is not None

    if has_combined:
        st.markdown(f"""
        <div style="background: {c['success_bg']}; border: 1px solid {c['success']};
                    border-radius: 0.5rem; padding: 1rem; text-align: center; margin-bottom: 1rem;">
            <span style="font-size: 1.5rem;">✅</span>
            <span style="font-weight: 600; margin-left: 0.5rem; color: {c['success']};">
                Datasets combinados exitosamente
            </span>
        </div>
        """, unsafe_allow_html=True)

    action = workflow_navigation(
        current_step=5,
        can_go_next=True,  # Allow skipping or proceeding with combined
        next_label="Crear Splits",
        on_next="⑥ Splits",
        on_prev="④ Exportar"
    )

    if action == "next":
        if 5 not in st.session_state.get("workflow_completed", []):
            st.session_state.workflow_completed = st.session_state.get("workflow_completed", []) + [5]
        st.session_state.workflow_step = 6
        st.rerun()
    elif action == "prev":
        st.session_state.nav_menu = "④ Exportar"
        st.rerun()


def _auto_add_workflow_datasets() -> None:
    """Auto-add datasets from the workflow if not already added"""
    # Check if we already auto-added
    if st.session_state.get("_combine_auto_added"):
        return

    datasets_to_add = []

    # Add generated dataset if available
    generated = st.session_state.get("generated_dataset")
    if generated:
        datasets_to_add.append({
            "name": "Dataset Sintético Generado",
            "data": generated,
            "source": "workflow_generated",
            "n_images": len(generated.get("images", [])),
            "n_annotations": len(generated.get("annotations", []))
        })

    # Add source dataset if available
    source = st.session_state.get("source_dataset")
    if source:
        datasets_to_add.append({
            "name": "Dataset Original",
            "data": source,
            "source": "workflow_source",
            "n_images": len(source.get("images", [])),
            "n_annotations": len(source.get("annotations", []))
        })

    if datasets_to_add:
        st.session_state.datasets_to_combine = datasets_to_add
        st.session_state._combine_auto_added = True


def _render_datasets_list() -> None:
    """Render the list of datasets to combine"""
    c = get_colors_dict()

    for i, ds in enumerate(st.session_state.datasets_to_combine):
        col1, col2, col3 = st.columns([3, 2, 1])

        with col1:
            st.markdown(f"""
            <div style="background: {c['bg_card']}; border: 1px solid {c['border']};
                        border-radius: 0.5rem; padding: 0.75rem;">
                <div style="font-weight: 600; color: {c['text_primary']};">
                    {ds['name']}
                </div>
                <div style="font-size: 0.8rem; color: {c['text_muted']}; margin-top: 0.25rem;">
                    {ds['n_images']:,} imágenes | {ds['n_annotations']:,} anotaciones
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            source_label = {
                "workflow_generated": "🏭 Generado",
                "workflow_source": "📊 Original",
                "uploaded": "📤 Subido"
            }.get(ds.get("source", ""), "📁 Externo")

            st.markdown(f"""
            <div style="padding: 0.75rem; text-align: center;">
                <span style="background: {c['bg_tertiary']}; padding: 0.25rem 0.5rem;
                            border-radius: 0.375rem; font-size: 0.8rem; color: {c['text_secondary']};">
                    {source_label}
                </span>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            if st.button("🗑️", key=f"remove_ds_{i}", help="Eliminar dataset"):
                st.session_state.datasets_to_combine.pop(i)
                st.rerun()

        spacer(8)


def _render_add_dataset_section() -> None:
    """Render section to add more datasets"""
    with st.expander("➕ Añadir otro dataset", expanded=False):
        add_method = st.radio(
            "Método de entrada",
            ["📤 Subir archivo JSON", "📂 Ruta de archivo"],
            horizontal=True,
            key="combine_add_method"
        )

        if add_method == "📤 Subir archivo JSON":
            uploaded = st.file_uploader(
                "Subir archivo COCO JSON",
                type=["json"],
                key="combine_upload"
            )

            if uploaded:
                try:
                    data = json.load(uploaded)
                    n_images = len(data.get("images", []))
                    n_anns = len(data.get("annotations", []))

                    st.info(f"📁 {uploaded.name}: {n_images} imágenes, {n_anns} anotaciones")

                    if st.button("Añadir Dataset", key="add_uploaded"):
                        st.session_state.datasets_to_combine.append({
                            "name": uploaded.name,
                            "data": data,
                            "source": "uploaded",
                            "n_images": n_images,
                            "n_annotations": n_anns
                        })
                        st.success(f"✅ Añadido: {uploaded.name}")
                        st.rerun()

                except Exception as e:
                    st.error(f"Error al parsear JSON: {e}")

        else:
            file_path = st.text_input(
                "Ruta del archivo COCO JSON",
                placeholder="/app/datasets/annotations.json",
                key="combine_path"
            )

            if file_path and st.button("Cargar y Añadir", key="add_from_path"):
                path = Path(file_path)
                if path.exists():
                    try:
                        with open(path) as f:
                            data = json.load(f)

                        n_images = len(data.get("images", []))
                        n_anns = len(data.get("annotations", []))

                        st.session_state.datasets_to_combine.append({
                            "name": path.name,
                            "data": data,
                            "source": "uploaded",
                            "n_images": n_images,
                            "n_annotations": n_anns
                        })
                        st.success(f"✅ Añadido: {path.name}")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error al cargar: {e}")
                else:
                    st.warning("Archivo no encontrado")


def _render_merge_config() -> None:
    """Render merge configuration and preview"""
    c = get_colors_dict()
    section_header("Configuración de Merge", icon="⚙️")

    col1, col2 = st.columns(2)

    with col1:
        id_strategy = st.radio(
            "Estrategia de IDs",
            ["offset", "reassign"],
            format_func=lambda x: {
                "offset": "📊 Offset - Preservar IDs con desplazamiento",
                "reassign": "🔄 Reasignar - Nuevos IDs secuenciales"
            }.get(x, x),
            key="merge_id_strategy",
            help="Offset: añade desplazamiento a los IDs para evitar colisiones. Reassign: reasigna todos los IDs desde 1."
        )

    with col2:
        category_strategy = st.radio(
            "Estrategia de Categorías",
            ["unify", "separate"],
            format_func=lambda x: {
                "unify": "🔗 Unificar - Fusionar categorías con mismo nombre",
                "separate": "📂 Separar - Mantener categorías distintas"
            }.get(x, x),
            key="merge_cat_strategy",
            help="Unify: categorías con el mismo nombre se fusionan. Separate: se añade sufijo para distinguir origen."
        )

    spacer(16)

    # Preview
    datasets = [ds["data"] for ds in st.session_state.datasets_to_combine]
    names = [ds["name"] for ds in st.session_state.datasets_to_combine]

    preview = DatasetMerger.preview_merge(datasets, names)

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {c['primary']}, {c['primary_hover']});
                border-radius: 0.75rem; padding: 1.5rem; color: white; text-align: center;">
        <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.9;">
            Resultado Combinado
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem;">
            <div>
                <div style="font-size: 2rem; font-weight: 700;">{preview['total_images']:,}</div>
                <div style="font-size: 0.85rem; opacity: 0.9;">Imágenes</div>
            </div>
            <div>
                <div style="font-size: 2rem; font-weight: 700;">{preview['total_annotations']:,}</div>
                <div style="font-size: 0.85rem; opacity: 0.9;">Anotaciones</div>
            </div>
            <div>
                <div style="font-size: 2rem; font-weight: 700;">{preview['unique_categories']}</div>
                <div style="font-size: 0.85rem; opacity: 0.9;">Clases Únicas</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    spacer(16)

    # Source breakdown
    with st.expander("📊 Desglose por Dataset", expanded=False):
        source_df = pd.DataFrame(preview["sources"])
        st.dataframe(source_df, use_container_width=True, hide_index=True)

        st.markdown("**Categorías únicas:**")
        st.write(", ".join(preview["all_categories"]))

    spacer(16)

    # Merge button
    if st.button("🔗 Combinar Datasets", type="primary", use_container_width=True, key="merge_btn"):
        _perform_merge(datasets, names, id_strategy, category_strategy)


def _perform_merge(
    datasets: List[Dict],
    names: List[str],
    id_strategy: str,
    category_strategy: str
) -> None:
    """Perform the actual merge"""
    with st.spinner("Combinando datasets..."):
        result = DatasetMerger.merge_datasets(
            datasets,
            names,
            id_strategy=id_strategy,
            category_strategy=category_strategy
        )

    if result.success:
        st.session_state.combined_dataset = result.merged_dataset

        st.success(f"""
        ✅ **Merge completado exitosamente**
        - {result.total_images:,} imágenes
        - {result.total_annotations:,} anotaciones
        - {result.total_categories} categorías
        """)

        if result.warnings:
            for warning in result.warnings:
                st.warning(warning)

        # Show download button
        json_data = json.dumps(result.merged_dataset, indent=2)
        st.download_button(
            "📥 Descargar Dataset Combinado (COCO JSON)",
            data=json_data,
            file_name="combined_dataset.json",
            mime="application/json",
            key="download_combined",
            use_container_width=True
        )

        st.rerun()

    else:
        st.error(f"❌ Error durante el merge: {result.error}")
