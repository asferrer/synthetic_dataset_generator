"""
Source Selection Page (Step 2.5)
================================
Select data source for balancing: generate new, use existing, or hybrid mode.
"""

import json
import streamlit as st
from typing import Dict, List, Optional, Any
from pathlib import Path

from app.components.ui import (
    page_header, section_header, spacer, alert_box,
    workflow_stepper, workflow_navigation
)
from app.components.dataset_matcher import DatasetMatcher, MatchResult
from app.components.api_client import get_api_client
from app.config.theme import get_colors_dict


def render_source_selection_page():
    """Render source selection page (Step 2.5)"""
    c = get_colors_dict()

    # Workflow stepper - use 2.5 visually as step 2
    completed = st.session_state.get("workflow_completed", [])
    workflow_stepper(current_step=2, completed_steps=completed)

    page_header(
        title="Seleccion de Fuente de Datos",
        subtitle="Paso 2.5: Elige como obtener los objetos sinteticos para balancear",
        icon="📂"
    )

    # Get targets from previous step
    targets = st.session_state.get("balancing_targets", {})
    config = st.session_state.get("generation_config", {})

    if not targets or not config:
        alert_box(
            "No hay configuracion de balanceo. Vuelve al paso anterior.",
            type="error",
            icon="⚠️"
        )
        if st.button("← Ir a Configuracion", type="primary"):
            st.session_state.nav_menu = "② Configurar"
            st.rerun()
        return

    # Filter to only classes with targets > 0
    targets = {k: v for k, v in targets.items() if v > 0}

    if not targets:
        alert_box(
            "No hay clases que necesiten balanceo. El dataset ya esta balanceado.",
            type="info",
            icon="ℹ️"
        )
        # Allow direct navigation to export
        if st.button("→ Continuar a Exportar", type="primary"):
            st.session_state.generation_source_mode = "skip"
            st.session_state.nav_menu = "④ Exportar"
            st.session_state.workflow_step = 4
            st.rerun()
        return

    # Requirements summary
    _render_requirements_summary(targets, c)

    spacer(16)

    # Tabs for three modes
    tab_new, tab_existing, tab_hybrid = st.tabs([
        "🏭 Generar Nuevas",
        "📂 Usar Existentes",
        "🔀 Modo Hibrido"
    ])

    with tab_new:
        _render_generate_new_tab(targets, c)

    with tab_existing:
        _render_use_existing_tab(targets, c)

    with tab_hybrid:
        _render_hybrid_tab(targets, c)


def _render_requirements_summary(targets: Dict[str, int], c: Dict) -> None:
    """Show summary of objects required per class"""
    total = sum(targets.values())
    num_classes = len(targets)

    st.markdown(f"""
    <div style="background: {c['bg_secondary']}; padding: 1rem; border-radius: 0.5rem;
                border-left: 3px solid {c['primary']}; margin-bottom: 1rem;">
        <div style="font-size: 0.9rem; color: {c['text_secondary']};">
            Se requieren <strong style="color: {c['primary']};">{total:,}</strong> objetos sinteticos
            en <strong>{num_classes}</strong> clases para balancear el dataset.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show per-class breakdown in expander
    with st.expander("📊 Ver detalle por clase", expanded=False):
        for cls, count in sorted(targets.items(), key=lambda x: -x[1]):
            st.markdown(f"• **{cls}**: {count:,} objetos")


def _render_generate_new_tab(targets: Dict[str, int], c: Dict) -> None:
    """Tab for generating all new objects"""
    section_header("Generar Objetos Nuevos", icon="🏭")

    st.markdown(f"""
    <div style="background: {c['info_bg']}; border: 1px solid {c['info']};
                border-radius: 0.5rem; padding: 1rem; margin-bottom: 1rem;">
        <span style="font-size: 0.9rem; color: {c['text_secondary']};">
            Se generaran <strong>todos</strong> los objetos sinteticos desde cero usando
            el servicio Augmentor con la configuracion definida en el paso anterior.
        </span>
    </div>
    """, unsafe_allow_html=True)

    total = sum(targets.values())

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {c['primary']}, {c['primary_hover']});
                border-radius: 0.75rem; padding: 1.5rem; text-align: center; color: white;
                margin: 1rem 0;">
        <div style="font-size: 1.5rem; margin-bottom: 0.25rem;">🚀</div>
        <div style="font-size: 2rem; font-weight: 700;">{total:,}</div>
        <div style="font-size: 0.85rem; opacity: 0.9;">objetos a generar</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("▶️ Continuar a Generacion", type="primary", use_container_width=True, key="btn_new"):
        st.session_state.generation_source_mode = "new"
        st.session_state.workflow_step = 3
        st.session_state.nav_menu = "③ Generar"
        if 2 not in st.session_state.get("workflow_completed", []):
            st.session_state.workflow_completed = st.session_state.get("workflow_completed", []) + [2]
        st.rerun()


def _render_use_existing_tab(targets: Dict[str, int], c: Dict) -> None:
    """Tab for using existing dataset"""
    section_header("Usar Dataset Existente", icon="📂")

    st.markdown(f"""
    <div style="background: {c['info_bg']}; border: 1px solid {c['info']};
                border-radius: 0.5rem; padding: 1rem; margin-bottom: 1rem;">
        <span style="font-size: 0.9rem; color: {c['text_secondary']};">
            Selecciona un dataset sintetico previamente generado.
            El sistema verificara que contenga las clases necesarias para el balanceo.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Initialize matcher
    matcher = DatasetMatcher()
    client = get_api_client()

    # Fetch available datasets
    datasets_response = client.list_datasets()
    datasets = datasets_response.get("datasets", [])

    if not datasets:
        alert_box(
            "No hay datasets sinteticos disponibles. Genera uno primero.",
            type="warning",
            icon="📭"
        )
        return

    # Filter to generation type datasets
    # Note: database returns 'dataset_type', not 'type' or 'job_type'
    gen_datasets = [d for d in datasets if d.get("dataset_type") == "generation" or d.get("type") == "generation" or d.get("job_type") == "generation"]

    if not gen_datasets:
        alert_box(
            "No hay datasets de generacion disponibles. Solo se pueden usar datasets generados previamente.",
            type="warning",
            icon="📭"
        )
        return

    # Dataset selector
    # Note: database returns 'num_images', not 'images_count'
    dataset_options = {
        f"{d.get('job_id', 'unknown')[:12]}... ({d.get('num_images', d.get('images_count', 'N/A'))} imgs)": d
        for d in gen_datasets
    }

    selected_label = st.selectbox(
        "Seleccionar dataset",
        options=list(dataset_options.keys()),
        key="existing_dataset_select"
    )

    if selected_label:
        selected_dataset = dataset_options[selected_label]
        job_id = selected_dataset.get("job_id")

        # Show dataset info
        with st.expander("📋 Informacion del dataset", expanded=False):
            st.json(selected_dataset)

        # Load and analyze button
        if st.button("🔍 Analizar Compatibilidad", key="analyze_existing"):
            with st.spinner("Cargando y analizando dataset..."):
                result = client.load_dataset_coco(job_id)

                if result.get("success"):
                    coco_data = result["data"]

                    # Analyze compatibility
                    match_result = matcher.analyze_compatibility(
                        coco_data, targets, list(targets.keys())
                    )

                    # Store in session
                    st.session_state.selected_existing_datasets = [selected_dataset]
                    st.session_state.existing_dataset_data = coco_data
                    st.session_state.dataset_match_result = match_result
                    st.rerun()
                else:
                    st.error(f"Error al cargar dataset: {result.get('error')}")

    # Show compatibility result if available
    if st.session_state.get("dataset_match_result"):
        _render_compatibility_result(
            st.session_state.dataset_match_result,
            targets,
            c,
            matcher
        )


def _render_compatibility_result(
    match: MatchResult,
    targets: Dict[str, int],
    c: Dict,
    matcher: DatasetMatcher
) -> None:
    """Show compatibility analysis result"""
    spacer(16)
    section_header("Analisis de Compatibilidad", icon="📊")

    # Coverage indicator
    if match.coverage_percentage >= 100:
        coverage_color = c['success']
        coverage_bg = c['success_bg']
        coverage_icon = "✅"
    elif match.coverage_percentage >= 50:
        coverage_color = c['warning']
        coverage_bg = c['warning_bg']
        coverage_icon = "⚠️"
    else:
        coverage_color = c['error']
        coverage_bg = c['error_bg']
        coverage_icon = "❌"

    st.markdown(f"""
    <div style="background: {coverage_bg}; border: 2px solid {coverage_color};
                border-radius: 0.75rem; padding: 1.5rem; text-align: center; margin-bottom: 1rem;">
        <div style="font-size: 3rem; font-weight: 700; color: {coverage_color};">
            {match.coverage_percentage:.0f}%
        </div>
        <div style="font-size: 0.9rem; color: {c['text_muted']};">
            Cobertura de Balanceo {coverage_icon}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Necesarias", f"{match.total_needed:,}")

    with col2:
        st.metric("Cubiertas", f"{match.total_covered:,}")

    with col3:
        total_missing = sum(match.missing_per_class.values())
        st.metric("Faltantes", f"{total_missing:,}")

    # Detail by class
    with st.expander("📋 Detalle por clase", expanded=True):
        for cls, needed in targets.items():
            available = len(match.matched_annotations.get(cls, []))
            missing = match.missing_per_class.get(cls, 0)

            if missing == 0:
                status = "✅"
                color = c['success']
            elif available > 0:
                status = "⚠️"
                color = c['warning']
            else:
                status = "❌"
                color = c['error']

            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center;
                        padding: 0.5rem; border-bottom: 1px solid {c['border']};">
                <span>{status} <strong>{cls}</strong></span>
                <span style="color: {color};">{available}/{needed}</span>
            </div>
            """, unsafe_allow_html=True)

            if missing > 0:
                st.caption(f"   Faltan {missing} objetos")

    # Warnings
    for warning in match.warnings:
        st.warning(warning)

    # Action buttons based on coverage
    spacer(16)

    if match.coverage_percentage >= 100:
        st.success("El dataset existente cubre todos los requerimientos de balanceo.")

        # Download options section
        _render_download_options(
            st.session_state.get("existing_dataset_data", {}),
            targets,
            matcher,
            c,
            "existing"
        )

        if st.button("✅ Usar Este Dataset", type="primary", use_container_width=True, key="btn_use_existing"):
            # Filter dataset to get only what's needed
            filtered = matcher.filter_for_balancing(
                st.session_state.existing_dataset_data,
                targets
            )
            st.session_state.generation_source_mode = "existing"
            st.session_state.generated_dataset = filtered
            st.session_state.workflow_step = 4  # Skip to Export
            st.session_state.nav_menu = "④ Exportar"

            # Mark steps as completed
            completed = st.session_state.get("workflow_completed", [])
            if 2 not in completed:
                completed.append(2)
            if 3 not in completed:
                completed.append(3)
            st.session_state.workflow_completed = completed

            st.rerun()
    else:
        st.warning(f"""
        El dataset no cubre todos los requerimientos.
        Considera usar el **Modo Hibrido** para generar los {sum(match.missing_per_class.values())} objetos faltantes.
        """)


def _render_download_options(
    dataset: Dict[str, Any],
    targets: Dict[str, int],
    matcher: DatasetMatcher,
    c: Dict,
    mode: str
) -> None:
    """Render download options for filtered/combined dataset in multiple formats."""
    section_header("Descargar Anotaciones", icon="⬇️")

    st.markdown(f"""
    <div style="background: {c['bg_secondary']}; padding: 0.75rem; border-radius: 0.5rem;
                margin-bottom: 1rem; font-size: 0.85rem; color: {c['text_secondary']};">
        Descarga el dataset filtrado/combinado en diferentes formatos antes de continuar.
    </div>
    """, unsafe_allow_html=True)

    # Filter dataset if needed
    if mode == "existing":
        filtered_dataset = matcher.filter_for_balancing(dataset, targets)
    else:
        filtered_dataset = dataset

    col1, col2, col3 = st.columns(3)

    # COCO JSON download
    with col1:
        coco_json = json.dumps(filtered_dataset, indent=2, ensure_ascii=False)
        st.download_button(
            "📋 COCO JSON",
            data=coco_json,
            file_name="balanced_dataset.json",
            mime="application/json",
            key=f"download_coco_{mode}",
            use_container_width=True
        )

    # YOLO format download
    with col2:
        yolo_content = _convert_to_yolo(filtered_dataset)
        if yolo_content:
            st.download_button(
                "🔲 YOLO (txt)",
                data=yolo_content,
                file_name="balanced_yolo.txt",
                mime="text/plain",
                key=f"download_yolo_{mode}",
                use_container_width=True
            )
        else:
            st.button("🔲 YOLO", disabled=True, use_container_width=True, key=f"yolo_disabled_{mode}")

    # Pascal VOC format download
    with col3:
        voc_content = _convert_to_voc_summary(filtered_dataset)
        if voc_content:
            st.download_button(
                "📄 VOC (xml)",
                data=voc_content,
                file_name="balanced_voc.xml",
                mime="application/xml",
                key=f"download_voc_{mode}",
                use_container_width=True
            )
        else:
            st.button("📄 VOC", disabled=True, use_container_width=True, key=f"voc_disabled_{mode}")


def _convert_to_yolo(dataset: Dict[str, Any]) -> Optional[str]:
    """Convert COCO dataset to YOLO format (combined txt file)."""
    try:
        # Build category mapping (name -> index)
        categories = dataset.get("categories", [])
        cat_id_to_idx = {cat["id"]: idx for idx, cat in enumerate(categories)}
        cat_id_to_name = {cat["id"]: cat["name"] for cat in categories}

        # Build image dimensions mapping
        images = {img["id"]: img for img in dataset.get("images", [])}

        lines = []
        lines.append("# YOLO Format Annotations")
        lines.append(f"# Classes: {', '.join([cat['name'] for cat in categories])}")
        lines.append("")

        # Group annotations by image
        anns_by_image: Dict[int, List] = {}
        for ann in dataset.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in anns_by_image:
                anns_by_image[img_id] = []
            anns_by_image[img_id].append(ann)

        for img_id, img_anns in anns_by_image.items():
            img = images.get(img_id)
            if not img:
                continue

            img_w = img.get("width", 1)
            img_h = img.get("height", 1)
            file_name = img.get("file_name", f"image_{img_id}")

            lines.append(f"# {file_name}")

            for ann in img_anns:
                bbox = ann.get("bbox", [0, 0, 0, 0])  # COCO format: [x, y, width, height]
                cat_idx = cat_id_to_idx.get(ann["category_id"], 0)

                # Convert to YOLO format: center_x, center_y, width, height (normalized)
                x, y, w, h = bbox
                center_x = (x + w / 2) / img_w
                center_y = (y + h / 2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h

                lines.append(f"{cat_idx} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")

            lines.append("")

        # Add data.yaml content
        lines.append("# --- data.yaml ---")
        lines.append(f"nc: {len(categories)}")
        lines.append(f"names: {[cat['name'] for cat in categories]}")

        return "\n".join(lines)

    except Exception as e:
        return None


def _convert_to_voc_summary(dataset: Dict[str, Any]) -> Optional[str]:
    """Convert COCO dataset to Pascal VOC XML summary format."""
    try:
        categories = dataset.get("categories", [])
        cat_id_to_name = {cat["id"]: cat["name"] for cat in categories}
        images = {img["id"]: img for img in dataset.get("images", [])}

        # Build XML structure
        xml_lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml_lines.append('<dataset>')
        xml_lines.append(f'  <name>Balanced Dataset</name>')
        xml_lines.append(f'  <images_count>{len(images)}</images_count>')
        xml_lines.append(f'  <annotations_count>{len(dataset.get("annotations", []))}</annotations_count>')

        # Categories
        xml_lines.append('  <categories>')
        for cat in categories:
            xml_lines.append(f'    <category id="{cat["id"]}">{cat["name"]}</category>')
        xml_lines.append('  </categories>')

        # Images with annotations
        xml_lines.append('  <annotations>')

        anns_by_image: Dict[int, List] = {}
        for ann in dataset.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in anns_by_image:
                anns_by_image[img_id] = []
            anns_by_image[img_id].append(ann)

        for img_id, img_anns in list(anns_by_image.items())[:50]:  # Limit for summary
            img = images.get(img_id)
            if not img:
                continue

            xml_lines.append(f'    <image id="{img_id}" file="{img.get("file_name", "")}">')

            for ann in img_anns:
                bbox = ann.get("bbox", [0, 0, 0, 0])
                cat_name = cat_id_to_name.get(ann["category_id"], "unknown")

                xml_lines.append(f'      <object>')
                xml_lines.append(f'        <name>{cat_name}</name>')
                xml_lines.append(f'        <bndbox>')
                xml_lines.append(f'          <xmin>{int(bbox[0])}</xmin>')
                xml_lines.append(f'          <ymin>{int(bbox[1])}</ymin>')
                xml_lines.append(f'          <xmax>{int(bbox[0] + bbox[2])}</xmax>')
                xml_lines.append(f'          <ymax>{int(bbox[1] + bbox[3])}</ymax>')
                xml_lines.append(f'        </bndbox>')
                xml_lines.append(f'      </object>')

            xml_lines.append('    </image>')

        if len(anns_by_image) > 50:
            xml_lines.append(f'    <!-- ... and {len(anns_by_image) - 50} more images -->')

        xml_lines.append('  </annotations>')
        xml_lines.append('</dataset>')

        return "\n".join(xml_lines)

    except Exception as e:
        return None


def _render_hybrid_tab(targets: Dict[str, int], c: Dict) -> None:
    """Tab for hybrid mode: use existing + generate missing"""
    section_header("Modo Hibrido", icon="🔀")

    st.markdown(f"""
    <div style="background: {c['info_bg']}; border: 1px solid {c['info']};
                border-radius: 0.5rem; padding: 1rem; margin-bottom: 1rem;">
        <span style="font-size: 0.9rem; color: {c['text_secondary']};">
            Combina objetos de un dataset existente con generacion de nuevos objetos
            <strong>solo para las clases que lo necesiten</strong>.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Check if we have existing analysis
    match = st.session_state.get("dataset_match_result")
    existing_data = st.session_state.get("existing_dataset_data")

    if not match or not existing_data:
        st.warning("Primero selecciona y analiza un dataset en la pestana 'Usar Existentes'.")
        return

    if match.coverage_percentage >= 100:
        st.info("El dataset existente ya cubre todos los requerimientos. Usa la pestana 'Usar Existentes'.")
        return

    # Show hybrid plan
    matcher = DatasetMatcher()
    hybrid_targets = matcher.calculate_hybrid_targets(targets, match.matched_annotations)

    st.markdown(f"""
    <div style="background: {c['bg_card']}; border: 1px solid {c['border']};
                border-radius: 0.75rem; padding: 1.5rem; margin: 1rem 0;">
        <div style="font-size: 0.7rem; color: {c['text_muted']}; text-transform: uppercase;
                    letter-spacing: 0.05em; margin-bottom: 1rem;">
            Plan de Generacion Hibrida
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📂 Del dataset existente:**")
        total_existing = 0
        for cls, ann_ids in match.matched_annotations.items():
            count = len(ann_ids)
            if count > 0:
                st.markdown(f"• {cls}: **{count}** objetos")
                total_existing += count

        st.markdown(f"**Total: {total_existing}**")

    with col2:
        st.markdown("**🏭 A generar:**")
        total_new = 0
        if hybrid_targets:
            for cls, count in hybrid_targets.items():
                if count > 0:
                    st.markdown(f"• {cls}: **{count}** objetos")
                    total_new += count
            st.markdown(f"**Total: {total_new}**")
        else:
            st.markdown("*Ninguno*")

    st.markdown("</div>", unsafe_allow_html=True)

    # Summary
    grand_total = total_existing + total_new
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {c['primary']}, {c['primary_hover']});
                border-radius: 0.5rem; padding: 1rem; text-align: center; color: white;
                margin: 1rem 0;">
        <div style="font-size: 0.85rem; opacity: 0.9;">Resumen</div>
        <div style="font-size: 1.5rem; font-weight: 700;">
            {total_existing} existentes + {total_new} nuevas = {grand_total} total
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Download options for combined dataset preview
    if total_existing > 0:
        with st.expander("⬇️ Descargar anotaciones existentes filtradas"):
            _render_download_options(
                existing_data,
                {cls: len(ann_ids) for cls, ann_ids in match.matched_annotations.items() if len(ann_ids) > 0},
                matcher,
                c,
                "hybrid_existing"
            )

    # Start hybrid generation
    if hybrid_targets and total_new > 0:
        if st.button("🚀 Iniciar Modo Hibrido", type="primary", use_container_width=True, key="btn_hybrid"):
            # Store hybrid configuration
            st.session_state.generation_source_mode = "hybrid"
            st.session_state.hybrid_targets = hybrid_targets
            st.session_state.existing_coverage = {
                cls: len(ann_ids) for cls, ann_ids in match.matched_annotations.items()
            }

            # Update generation config with reduced targets
            config = st.session_state.get("generation_config", {}).copy()
            config["targets_per_class"] = hybrid_targets
            config["num_images"] = sum(hybrid_targets.values())
            st.session_state.generation_config = config

            # Store filtered existing dataset for later merge
            filtered_existing = matcher.filter_for_balancing(
                existing_data,
                {cls: len(ann_ids) for cls, ann_ids in match.matched_annotations.items()}
            )
            st.session_state.hybrid_existing_dataset = filtered_existing

            # Navigate to generation
            st.session_state.workflow_step = 3
            st.session_state.nav_menu = "③ Generar"

            if 2 not in st.session_state.get("workflow_completed", []):
                st.session_state.workflow_completed = st.session_state.get("workflow_completed", []) + [2]

            st.rerun()
    else:
        st.info("No hay objetos nuevos que generar. Usa la pestana 'Usar Existentes'.")
