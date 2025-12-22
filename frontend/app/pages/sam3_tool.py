"""
SAM3 Tool Page
==============
Standalone SAM3 segmentation tool for individual images and batch dataset conversion.
"""

import os
import json
import time
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


def render_sam3_tool_page():
    """Render the SAM3 segmentation tool page"""
    c = get_colors_dict()
    client = get_api_client()

    page_header(
        title="SAM3 Segmentation",
        subtitle="Segmenta objetos usando SAM3 con prompts de bounding box o punto",
        icon="🔬"
    )

    # Check if there's a running conversion job - redirect to monitor
    current_job_id = st.session_state.get("sam3_convert_current_job_id")
    if current_job_id:
        alert_box(
            f"Hay un trabajo de conversion SAM3 en progreso (Job: {current_job_id[:8]}...)",
            type="info",
            icon="⏳"
        )
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("📊 Ver en Monitor", type="primary", key="go_to_monitor_sam3"):
                st.session_state.nav_menu = "📊 Monitor"
                st.rerun()
        with col2:
            if st.button("🔄 Nueva Conversion", key="new_sam3_conversion"):
                st.session_state.pop("sam3_convert_current_job_id", None)
                st.rerun()
        return

    # Quick health check (non-blocking for page render)
    health = client.get_segmentation_health()
    sam3_available = health.get("sam3_available", False)
    service_status = health.get("status", "unknown")

    if service_status != "healthy":
        alert_box(
            f"Servicio de segmentacion: {service_status}. El job se iniciara cuando el servicio este disponible.",
            type="warning",
            icon="⚠️"
        )

    if not sam3_available:
        alert_box(
            "SAM3 no esta disponible en el servicio de segmentacion.",
            type="error",
            icon="❌"
        )
        return

    # Tabs for different modes
    tab1, tab2 = st.tabs(["🖼️ Imagen Individual", "📦 Conversion de Dataset"])

    with tab1:
        _render_individual_segmentation(client, c)

    with tab2:
        _render_batch_conversion(client, c)


def _render_individual_segmentation(client, c: Dict):
    """Render individual image segmentation tab"""

    section_header("Segmentar Imagen", icon="🎯")

    # Image input
    col1, col2 = st.columns(2)

    with col1:
        input_method = st.radio(
            "Metodo de entrada",
            options=["Subir imagen", "Ruta del servidor"],
            key="sam3_input_method",
            horizontal=True
        )

    image_data = None
    image_path = None

    if input_method == "Subir imagen":
        uploaded_file = st.file_uploader(
            "Selecciona una imagen",
            type=["jpg", "jpeg", "png", "webp"],
            key="sam3_image_upload",
            help="Sube la imagen que quieres segmentar"
        )

        if uploaded_file:
            image_data = base64.b64encode(uploaded_file.read()).decode("utf-8")
            uploaded_file.seek(0)
            st.image(uploaded_file, caption="Imagen cargada", use_container_width=True)
    else:
        image_path = st.text_input(
            "Ruta de la imagen",
            value=st.session_state.get("sam3_image_path", ""),
            key="sam3_image_path_input",
            help="Ruta completa a la imagen en el servidor"
        )
        st.session_state["sam3_image_path"] = image_path

    spacer(16)

    # Prompt type selection
    section_header("Prompt de Segmentacion", icon="📍")

    prompt_type = st.radio(
        "Tipo de prompt",
        options=["Bounding Box", "Punto", "Texto"],
        key="sam3_prompt_type",
        horizontal=True,
        help="Selecciona como quieres indicar el objeto a segmentar"
    )

    bbox = None
    point = None
    text_prompt = None

    if prompt_type == "Bounding Box":
        st.markdown("**Introduce las coordenadas del bounding box:**")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            bbox_x = st.number_input("X", min_value=0, value=0, key="sam3_bbox_x")
        with col2:
            bbox_y = st.number_input("Y", min_value=0, value=0, key="sam3_bbox_y")
        with col3:
            bbox_w = st.number_input("Ancho", min_value=1, value=100, key="sam3_bbox_w")
        with col4:
            bbox_h = st.number_input("Alto", min_value=1, value=100, key="sam3_bbox_h")

        bbox = [float(bbox_x), float(bbox_y), float(bbox_w), float(bbox_h)]

    elif prompt_type == "Punto":
        st.markdown("**Introduce las coordenadas del punto:**")
        col1, col2 = st.columns(2)

        with col1:
            point_x = st.number_input("X", min_value=0, value=0, key="sam3_point_x")
        with col2:
            point_y = st.number_input("Y", min_value=0, value=0, key="sam3_point_y")

        point = [int(point_x), int(point_y)]

    else:  # Text prompt
        text_prompt = st.text_input(
            "Descripcion del objeto",
            value="",
            key="sam3_text_prompt",
            placeholder="Ej: a red car, a person wearing a hat",
            help="Describe el objeto que quieres segmentar"
        )

    spacer(16)

    # Options
    section_header("Opciones", icon="⚙️")

    col1, col2 = st.columns(2)

    with col1:
        return_polygon = st.checkbox(
            "Retornar poligono",
            value=True,
            key="sam3_return_polygon",
            help="Obtener la segmentacion como coordenadas de poligono"
        )

        simplify_polygon = st.checkbox(
            "Simplificar poligono",
            value=True,
            key="sam3_simplify_polygon",
            help="Reducir el numero de puntos del poligono"
        )

    with col2:
        return_mask = st.checkbox(
            "Retornar mascara",
            value=True,
            key="sam3_return_mask",
            help="Obtener la mascara de segmentacion como imagen"
        )

        simplify_tolerance = st.slider(
            "Tolerancia de simplificacion",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.5,
            key="sam3_simplify_tolerance",
            help="Mayor tolerancia = menos puntos en el poligono"
        )

    spacer(24)

    # Segment button
    can_segment = (image_data or image_path) and (bbox or point or text_prompt)

    if st.button(
        "🔬 Segmentar",
        type="primary",
        use_container_width=True,
        disabled=not can_segment
    ):
        with st.spinner("Segmentando con SAM3..."):
            result = client.sam3_segment_image(
                image_path=image_path,
                image_base64=image_data,
                bbox=bbox,
                point=point,
                text_prompt=text_prompt,
                return_polygon=return_polygon,
                return_mask=return_mask,
                simplify_polygon=simplify_polygon,
                simplify_tolerance=simplify_tolerance
            )

        if result.get("success"):
            st.session_state["sam3_last_result"] = result
            alert_box(
                f"Segmentacion completada en {result.get('processing_time_ms', 0):.0f}ms",
                type="success",
                icon="✅"
            )
        else:
            alert_box(
                f"Error en segmentacion: {result.get('error', 'Error desconocido')}",
                type="error",
                icon="❌"
            )

    # Display results
    result = st.session_state.get("sam3_last_result")

    if result and result.get("success"):
        spacer(16)
        section_header("Resultado", icon="📊")

        col1, col2 = st.columns(2)

        with col1:
            # Show mask if available
            if result.get("mask_base64"):
                mask_data = base64.b64decode(result["mask_base64"])
                st.image(mask_data, caption="Mascara de segmentacion", use_container_width=True)

                # Download button for mask
                st.download_button(
                    "📥 Descargar Mascara PNG",
                    data=mask_data,
                    file_name="segmentation_mask.png",
                    mime="image/png"
                )

        with col2:
            # Show metrics
            st.metric("Area", f"{result.get('area', 0):,} px")
            st.metric("Confianza", f"{result.get('confidence', 0):.1%}")

            if result.get("bbox"):
                bbox_result = result["bbox"]
                st.markdown(f"**BBox:** `[{bbox_result[0]:.0f}, {bbox_result[1]:.0f}, {bbox_result[2]:.0f}, {bbox_result[3]:.0f}]`")

            # Show polygon info
            if result.get("segmentation_polygon"):
                polygon = result["segmentation_polygon"]
                st.markdown(f"**Puntos del poligono:** {len(polygon)}")

                # Download button for polygon JSON
                polygon_json = json.dumps({
                    "segmentation": result.get("segmentation_coco", []),
                    "bbox": result.get("bbox", []),
                    "area": result.get("area", 0)
                }, indent=2)

                st.download_button(
                    "📥 Descargar Poligono JSON",
                    data=polygon_json,
                    file_name="segmentation_polygon.json",
                    mime="application/json"
                )


def _render_batch_conversion(client, c: Dict):
    """Render batch dataset conversion tab"""

    section_header("Convertir Dataset COCO", icon="📦")

    st.markdown("""
    Convierte anotaciones de bounding box a segmentaciones usando SAM3.
    El resultado es un nuevo archivo COCO JSON con el campo `segmentation` poblado.
    """)

    spacer(16)

    # Input section
    col1, col2 = st.columns(2)

    with col1:
        coco_file = st.file_uploader(
            "Archivo COCO JSON",
            type=["json"],
            key="sam3_convert_coco_upload",
            help="Sube el archivo JSON con las anotaciones COCO (solo bbox)"
        )

    with col2:
        images_dir = st.text_input(
            "Directorio base de imagenes",
            value=st.session_state.get("sam3_convert_images_dir", "/app/datasets/images"),
            key="sam3_convert_images_dir_input",
            help="Directorio base donde estan las imagenes. Se combinara con el campo 'file_name' del JSON COCO (soporta subcarpetas)."
        )
        st.session_state["sam3_convert_images_dir"] = images_dir
        st.caption("📝 Ej: Si base=`/data` y file_name=`train/img.jpg` → `/data/train/img.jpg`")

    # Load and analyze dataset
    if coco_file:
        try:
            coco_data = json.load(coco_file)
            st.session_state["sam3_convert_coco_data"] = coco_data
            st.session_state["sam3_convert_coco_filename"] = coco_file.name
        except Exception as e:
            alert_box(f"Error al cargar el archivo: {str(e)}", type="error")
            return

    coco_data = st.session_state.get("sam3_convert_coco_data")

    if not coco_data:
        empty_state(
            title="No hay dataset cargado",
            message="Sube un archivo COCO JSON para analizar y convertir",
            icon="📂"
        )
        return

    # Analyze dataset
    spacer(16)
    section_header("Analisis del Dataset", icon="🔍")

    # Save JSON to shared volume for analysis (avoids timeout with large files)
    coco_filename = st.session_state.get("sam3_convert_coco_filename", "dataset.json")
    try:
        coco_json_path = _save_coco_to_shared_volume(coco_data, coco_filename)
        st.session_state["sam3_convert_coco_json_path"] = coco_json_path
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
        st.metric("Total Imagenes", analysis.get("total_images", 0))

    with col2:
        st.metric("Total Anotaciones", analysis.get("total_annotations", 0))

    with col3:
        st.metric(
            "Ya con Mascara",
            analysis.get("annotations_with_segmentation", 0),
            help="Anotaciones que ya tienen segmentacion"
        )

    with col4:
        st.metric(
            "A Convertir",
            analysis.get("annotations_bbox_only", 0),
            help="Anotaciones que solo tienen bounding box"
        )

    # Recommendation
    bbox_only = analysis.get("annotations_bbox_only", 0)
    with_seg = analysis.get("annotations_with_segmentation", 0)

    if bbox_only == 0:
        alert_box(
            "Todas las anotaciones ya tienen segmentacion. No hay nada que convertir.",
            type="info",
            icon="ℹ️"
        )
        return
    elif with_seg == 0:
        alert_box(
            f"Todas las {bbox_only} anotaciones seran convertidas usando SAM3.",
            type="info",
            icon="🤖"
        )
    else:
        alert_box(
            f"{bbox_only} anotaciones seran convertidas. {with_seg} ya tienen segmentacion.",
            type="info",
            icon="📊"
        )

    # Category selection
    spacer(16)
    section_header("Clases a Convertir", icon="🏷️")

    categories = analysis.get("categories", [])
    category_names = [cat.get("name", f"ID:{cat.get('id')}") for cat in categories]

    # Filter to show only categories with bbox_only
    categories_with_bbox = [
        cat for cat in categories if cat.get("bbox_only", 0) > 0
    ]

    if categories_with_bbox:
        cat_df_data = []
        for cat in categories_with_bbox:
            cat_df_data.append({
                "Clase": cat.get("name", ""),
                "Total": cat.get("count", 0),
                "A Convertir": cat.get("bbox_only", 0)
            })

        st.dataframe(
            cat_df_data,
            use_container_width=True,
            hide_index=True
        )

    category_names_bbox = [cat.get("name") for cat in categories_with_bbox]

    selected_categories = st.multiselect(
        "Selecciona las clases a convertir (vacio = todas)",
        options=category_names_bbox,
        default=[],
        key="sam3_convert_selected_categories",
        help="Deja vacio para convertir todas las clases con bbox"
    )

    # Conversion options
    spacer(16)
    section_header("Opciones de Conversion", icon="⚙️")

    col1, col2 = st.columns(2)

    with col1:
        output_path = st.text_input(
            "Ruta de salida",
            value=st.session_state.get("sam3_convert_output_path", "/app/datasets/converted_annotations.json"),
            key="sam3_convert_output_path_input",
            help="Ruta para el archivo COCO JSON de salida con segmentaciones"
        )
        st.session_state["sam3_convert_output_path"] = output_path

        overwrite_existing = st.checkbox(
            "Sobrescribir segmentaciones existentes",
            value=False,
            key="sam3_convert_overwrite",
            help="Reemplazar segmentaciones existentes con las generadas por SAM3"
        )

    with col2:
        simplify_polygons = st.checkbox(
            "Simplificar poligonos",
            value=True,
            key="sam3_convert_simplify",
            help="Reducir el numero de puntos en los poligonos generados"
        )

        simplify_tolerance = st.slider(
            "Tolerancia de simplificacion",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.5,
            key="sam3_convert_tolerance",
            help="Mayor tolerancia = menos puntos en los poligonos"
        )

    # Start Conversion Button
    spacer(24)

    total_to_convert = bbox_only
    if selected_categories:
        total_to_convert = sum(
            cat.get("bbox_only", 0) for cat in categories_with_bbox
            if cat.get("name") in selected_categories
        )

    if st.button(
        f"🚀 Iniciar Conversion ({total_to_convert} anotaciones)",
        type="primary",
        use_container_width=True,
        disabled=total_to_convert == 0
    ):
        # Get the already-saved JSON path from analysis step
        coco_json_path = st.session_state.get("sam3_convert_coco_json_path")
        if not coco_json_path:
            alert_box("Error: no se encontro el archivo JSON preparado", type="error")
            return

        with st.spinner("Iniciando conversion en segundo plano..."):
            # Use file path instead of sending data in request body
            result = client.sam3_convert_dataset(
                coco_json_path=coco_json_path,
                images_dir=images_dir,
                output_path=output_path,
                categories_to_convert=selected_categories if selected_categories else None,
                overwrite_existing=overwrite_existing,
                simplify_polygons=simplify_polygons,
                simplify_tolerance=simplify_tolerance
            )

            if result.get("success"):
                job_id = result.get("job_id")
                st.session_state["sam3_convert_current_job_id"] = job_id
                st.success(f"Job iniciado correctamente: {job_id[:8]}...")
                # Redirect to monitor
                st.session_state.nav_menu = "📊 Monitor"
                st.rerun()
            else:
                alert_box(f"Error al iniciar: {result.get('error', '?')}", type="error")


def _render_conversion_job_progress(client, job_id: str, c: Dict):
    """Render SAM3 conversion job progress"""

    status = client.get_sam3_job_status(job_id)

    if "error" in status and not status.get("status"):
        alert_box(f"Error al obtener estado: {status.get('error')}", type="error")
        if st.button("Reintentar"):
            st.rerun()
        return

    job_status = status.get("status", "unknown")
    converted = status.get("converted_annotations", 0)
    skipped = status.get("skipped_annotations", 0)
    failed = status.get("failed_annotations", 0)
    total = status.get("total_annotations", 1)
    current_image = status.get("current_image", "")

    # Progress
    progress = (converted + skipped + failed) / max(total, 1)

    st.markdown(f"""
    <div style="background: {c['bg_secondary']}; padding: 1.5rem; border-radius: 0.5rem;
                border: 1px solid {c['border']}; margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="font-weight: 600;">Estado: {job_status.upper()}</span>
            <span>{converted + skipped + failed} / {total}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(progress)

    if current_image:
        st.caption(f"Procesando: {current_image}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Convertidas", converted)
    with col2:
        st.metric("Omitidas", skipped, help="Ya tenian segmentacion")
    with col3:
        st.metric("Fallidas", failed)
    with col4:
        st.metric("Tiempo", f"{status.get('processing_time_ms', 0)/1000:.1f}s")

    # Auto-refresh while processing
    if job_status in ["queued", "processing"]:
        time.sleep(1)
        st.rerun()
    elif job_status == "completed":
        alert_box(
            f"Conversion completada: {converted} anotaciones convertidas. Archivo guardado en: {status.get('output_path', '')}",
            type="success",
            icon="✅"
        )

        # Show category breakdown
        cat_progress = status.get("categories_progress", {})
        if cat_progress:
            st.markdown("**Anotaciones por clase:**")
            for cat_name, count in cat_progress.items():
                st.write(f"- {cat_name}: {count}")

        if st.button("Nueva Conversion", type="primary"):
            st.session_state.pop("sam3_convert_current_job_id", None)
            st.rerun()

    elif job_status == "failed":
        errors = status.get("errors", [])
        alert_box(
            f"Conversion fallida. {len(errors)} errores.",
            type="error",
            icon="❌"
        )

        if errors:
            with st.expander("Ver errores"):
                for err in errors[:20]:
                    st.text(err)

        if st.button("Reintentar"):
            st.session_state.pop("sam3_convert_current_job_id", None)
            st.rerun()
