"""
Object Extraction Page
======================
Extract objects from COCO datasets as transparent PNG images.
"""

import os
import json
import base64
import hashlib
import streamlit as st
from pathlib import Path
from typing import Dict, Any, Optional, List
from io import BytesIO

from app.components.ui import (
    page_header, section_header, spacer, alert_box, empty_state,
    metric_card
)
from app.components.api_client import get_api_client
from app.config.theme import get_colors_dict


# Shared temp directory for large JSON files (accessible by both frontend and segmentation service)
TEMP_JSON_DIR = "/app/datasets/temp"


def _save_coco_to_shared_volume(coco_data: Dict, original_filename: str) -> str:
    """
    Save COCO JSON to shared volume for large file handling.
    Returns the path to the saved file.
    """
    # Create temp directory if it doesn't exist
    os.makedirs(TEMP_JSON_DIR, exist_ok=True)

    # Create unique filename based on content hash
    content_hash = hashlib.md5(json.dumps(coco_data, sort_keys=True).encode()).hexdigest()[:8]
    base_name = Path(original_filename).stem
    temp_filename = f"{base_name}_{content_hash}.json"
    temp_path = os.path.join(TEMP_JSON_DIR, temp_filename)

    # Save if not already exists
    if not os.path.exists(temp_path):
        with open(temp_path, 'w') as f:
            json.dump(coco_data, f)

    return temp_path


def _quick_health_check(client) -> Dict[str, Any]:
    """Quick health check with short timeout - just to get SAM3 availability info."""
    try:
        health = client.get_segmentation_health()
        return health
    except Exception as e:
        # Return degraded status but allow page to render
        return {"status": "unknown", "error": str(e), "sam3_available": False}


def render_object_extraction_page():
    """Render the object extraction tool page"""
    c = get_colors_dict()
    client = get_api_client()

    page_header(
        title="Extraer Objetos",
        subtitle="Extrae objetos recortados desde un dataset COCO usando mascaras o SAM3",
        icon="🎯"
    )

    # Check if there's a running job - redirect to monitor
    current_job_id = st.session_state.get("extract_current_job_id")
    if current_job_id:
        alert_box(
            f"Hay un trabajo de extraccion en progreso (Job: {current_job_id[:8]}...)",
            type="info",
            icon="⏳"
        )
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("📊 Ver en Monitor", type="primary", key="go_to_monitor"):
                st.session_state.nav_menu = "📊 Monitor"
                st.rerun()
        with col2:
            if st.button("🔄 Nueva Extraccion", key="new_extraction"):
                st.session_state.pop("extract_current_job_id", None)
                st.rerun()
        return

    # Check if we have pending extraction to start (from previous button click)
    # Handle this IMMEDIATELY without health check to avoid timeouts
    pending_settings = st.session_state.pop("extract_pending_settings", None)

    if pending_settings:
        # User clicked the button - start extraction immediately
        coco_json_path = pending_settings.get("coco_json_path")
        if not coco_json_path:
            alert_box("Error: no se encontro el archivo JSON preparado", type="error")
        else:
            with st.spinner("Iniciando extraccion en segundo plano..."):
                result = client.extract_objects(
                    coco_json_path=coco_json_path,
                    images_dir=pending_settings["images_dir"],
                    output_dir=pending_settings["output_dir"],
                    categories_to_extract=pending_settings["categories"],
                    use_sam3_for_bbox=pending_settings["use_sam3"],
                    padding=pending_settings["padding"],
                    min_object_area=pending_settings["min_area"],
                    save_individual_coco=pending_settings["save_json"]
                )

                if result.get("success"):
                    job_id = result.get("job_id")
                    st.session_state["extract_current_job_id"] = job_id
                    st.success(f"Job iniciado correctamente: {job_id[:8]}...")
                    # Redirect to monitor
                    st.session_state.nav_menu = "📊 Monitor"
                    st.rerun()
                else:
                    alert_box(f"Error al iniciar: {result.get('error', '?')}", type="error")
        return

    # Normal page load - quick health check (non-blocking for page render)
    health = _quick_health_check(client)
    sam3_available = health.get("sam3_available", False)
    service_status = health.get("status", "unknown")

    # Show service status but don't block page
    if service_status != "healthy":
        alert_box(
            f"Servicio de segmentacion: {service_status}. El job se iniciara cuando el servicio este disponible.",
            type="warning",
            icon="⚠️"
        )

    # Section 1: Load Dataset
    section_header("Cargar Dataset COCO", icon="📁")

    col1, col2 = st.columns(2)

    with col1:
        coco_file = st.file_uploader(
            "Archivo COCO JSON",
            type=["json"],
            key="extract_coco_upload",
            help="Sube el archivo JSON con las anotaciones COCO"
        )

    with col2:
        images_dir = st.text_input(
            "Directorio base de imagenes",
            value=st.session_state.get("extract_images_dir", "/app/datasets/images"),
            key="extract_images_dir_input",
            help="Directorio base donde estan las imagenes. Se combinara con el campo 'file_name' del JSON COCO para encontrar cada imagen (soporta subcarpetas)."
        )
        st.session_state["extract_images_dir"] = images_dir
        st.caption("📝 Ej: Si base=`/data` y file_name=`train/img.jpg` → `/data/train/img.jpg`")

    # Load and analyze dataset
    if coco_file:
        try:
            coco_data = json.load(coco_file)
            st.session_state["extract_coco_data"] = coco_data
            st.session_state["extract_coco_filename"] = coco_file.name
        except Exception as e:
            alert_box(f"Error al cargar el archivo: {str(e)}", type="error")
            return

    coco_data = st.session_state.get("extract_coco_data")

    if not coco_data:
        empty_state(
            title="No hay dataset cargado",
            message="Sube un archivo COCO JSON para comenzar",
            icon="📂"
        )
        return

    # Section 2: Dataset Analysis
    spacer(16)
    section_header("Analisis de Anotaciones", icon="🔍")

    # Save JSON to shared volume for analysis (avoids timeout with large files)
    coco_filename = st.session_state.get("extract_coco_filename", "dataset.json")
    try:
        coco_json_path = _save_coco_to_shared_volume(coco_data, coco_filename)
        st.session_state["extract_coco_json_path"] = coco_json_path
    except Exception as e:
        alert_box(f"Error al preparar datos: {str(e)}", type="error")
        return

    # Analyze dataset using file path
    with st.spinner("Analizando dataset..."):
        analysis = client.analyze_dataset_annotations(coco_json_path=coco_json_path)

    if not analysis.get("success", True):
        alert_box(f"Error al analizar: {analysis.get('error', 'Error desconocido')}", type="error")
        return

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Imagenes",
            value=analysis.get("total_images", 0)
        )

    with col2:
        st.metric(
            label="Total Anotaciones",
            value=analysis.get("total_annotations", 0)
        )

    with col3:
        st.metric(
            label="Con Mascara",
            value=analysis.get("annotations_with_segmentation", 0),
            help="Anotaciones con segmentacion a nivel de pixel"
        )

    with col4:
        st.metric(
            label="Solo BBox",
            value=analysis.get("annotations_bbox_only", 0),
            help="Anotaciones que solo tienen bounding box"
        )

    # Recommendation
    recommendation = analysis.get("recommendation", "")
    if recommendation == "use_masks":
        alert_box(
            "Todas las anotaciones tienen mascara. Se extraeran directamente.",
            type="success",
            icon="✅"
        )
    elif recommendation == "use_sam3":
        if sam3_available:
            alert_box(
                "Ninguna anotacion tiene mascara. Se usara SAM3 para segmentar automaticamente.",
                type="info",
                icon="🤖"
            )
        else:
            alert_box(
                "Ninguna anotacion tiene mascara y SAM3 no esta disponible. Solo se puede recortar por bounding box.",
                type="warning",
                icon="⚠️"
            )
    elif recommendation == "mixed":
        alert_box(
            f"Dataset mixto: {analysis.get('annotations_with_segmentation', 0)} con mascara, "
            f"{analysis.get('annotations_bbox_only', 0)} solo bbox. "
            f"{'SAM3 segmentara las que no tienen mascara.' if sam3_available else 'Las sin mascara se recortaran por bbox.'}",
            type="info",
            icon="📊"
        )

    # Section 3: Category Selection
    spacer(16)
    section_header("Clases a Extraer", icon="🏷️")

    categories = analysis.get("categories", [])
    category_names = [cat.get("name", f"ID:{cat.get('id')}") for cat in categories]

    # Show category stats
    if categories:
        cat_df_data = []
        for cat in categories:
            cat_df_data.append({
                "Clase": cat.get("name", ""),
                "Total": cat.get("count", 0),
                "Con Mascara": cat.get("with_segmentation", 0),
                "Solo BBox": cat.get("bbox_only", 0)
            })

        st.dataframe(
            cat_df_data,
            use_container_width=True,
            hide_index=True
        )

    selected_categories = st.multiselect(
        "Selecciona las clases a extraer (vacio = todas)",
        options=category_names,
        default=[],
        key="extract_selected_categories",
        help="Deja vacio para extraer todas las clases"
    )

    # Section 4: Extraction Options
    spacer(16)
    section_header("Opciones de Extraccion", icon="⚙️")

    col1, col2 = st.columns(2)

    with col1:
        output_dir = st.text_input(
            "Directorio de salida",
            value=st.session_state.get("extract_output_dir", "/app/datasets/Extracted_objects"),
            key="extract_output_dir_input",
            help="Directorio donde se guardaran los objetos extraidos"
        )
        st.session_state["extract_output_dir"] = output_dir

        use_sam3 = st.checkbox(
            "Usar SAM3 para anotaciones sin mascara",
            value=sam3_available,
            disabled=not sam3_available,
            key="extract_use_sam3",
            help="Segmenta automaticamente los objetos que solo tienen bounding box"
        )

    with col2:
        padding = st.slider(
            "Padding (pixeles)",
            min_value=0,
            max_value=50,
            value=5,
            key="extract_padding",
            help="Pixeles adicionales alrededor del objeto recortado"
        )

        min_area = st.number_input(
            "Area minima (pixeles)",
            min_value=0,
            max_value=10000,
            value=100,
            key="extract_min_area",
            help="Area minima del bounding box para extraer un objeto"
        )

    save_individual_json = st.checkbox(
        "Guardar JSON COCO individual por objeto",
        value=True,
        key="extract_save_json",
        help="Genera un archivo JSON con anotaciones COCO para cada objeto extraido"
    )

    # Section 5: Preview
    spacer(16)

    with st.expander("👁️ Preview de Extraccion", expanded=False):
        if st.button("Generar Preview de un Objeto", key="extract_preview_btn"):
            annotations = coco_data.get("annotations", [])
            images = {img["id"]: img for img in coco_data.get("images", [])}
            cats = {cat["id"]: cat for cat in coco_data.get("categories", [])}

            if annotations:
                # Get a sample annotation
                import random
                sample_ann = random.choice(annotations)
                sample_img = images.get(sample_ann.get("image_id"))
                sample_cat = cats.get(sample_ann.get("category_id"))

                if sample_img:
                    img_path = str(Path(images_dir) / sample_img.get("file_name", ""))

                    with st.spinner("Extrayendo objeto de preview..."):
                        preview = client.extract_single_object(
                            image_path=img_path,
                            annotation=sample_ann,
                            category_name=sample_cat.get("name", "unknown") if sample_cat else "unknown",
                            use_sam3=use_sam3,
                            padding=padding
                        )

                    if preview.get("success"):
                        col1, col2 = st.columns(2)

                        with col1:
                            # Decode and display image
                            img_data = base64.b64decode(preview["cropped_image_base64"])
                            st.image(img_data, caption=f"Objeto: {sample_cat.get('name', '?')}", use_container_width=True)

                        with col2:
                            st.markdown(f"""
                            **Detalles:**
                            - Tipo de anotacion: `{preview.get('annotation_type', '?')}`
                            - Metodo usado: `{preview.get('method_used', '?')}`
                            - Tamano extraido: {preview.get('extracted_size', [0,0])}
                            - Cobertura mascara: {preview.get('mask_coverage', 0):.1%}
                            - Tiempo: {preview.get('processing_time_ms', 0):.0f}ms
                            """)
                    else:
                        alert_box(f"Error en preview: {preview.get('error', '?')}", type="error")
            else:
                alert_box("No hay anotaciones en el dataset", type="warning")

    # Section 6: Extract Button
    spacer(24)

    total_to_extract = analysis.get("total_annotations", 0)
    if selected_categories:
        total_to_extract = sum(
            cat.get("count", 0) for cat in categories
            if cat.get("name") in selected_categories
        )

    # Define callback to set pending settings before rerun
    def on_extract_click():
        st.session_state["extract_pending_settings"] = {
            "coco_json_path": st.session_state.get("extract_coco_json_path"),
            "images_dir": images_dir,
            "output_dir": output_dir,
            "categories": selected_categories if selected_categories else None,
            "use_sam3": use_sam3,
            "padding": padding,
            "min_area": min_area,
            "save_json": save_individual_json
        }

    # Show button for starting extraction
    st.button(
        f"🚀 Iniciar Extraccion ({total_to_extract} objetos)",
        type="primary",
        use_container_width=True,
        disabled=total_to_extract == 0,
        on_click=on_extract_click
    )


