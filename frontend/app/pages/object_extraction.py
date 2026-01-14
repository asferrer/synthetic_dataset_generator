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
                    force_bbox_only=pending_settings.get("force_bbox_only", False),
                    force_sam3_resegmentation=pending_settings.get("force_sam3_resegmentation", False),
                    force_sam3_text_prompt=pending_settings.get("force_sam3_text_prompt", False),
                    padding=pending_settings["padding"],
                    min_object_area=pending_settings["min_area"],
                    save_individual_coco=pending_settings["save_json"],
                    deduplication=pending_settings.get("deduplication")
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

    # Create tabs for different extraction modes
    spacer(8)
    tab1, tab2, tab3 = st.tabs(["📁 Dataset COCO", "🗂️ Estilo ImageNet", "✏️ Nombres Custom"])

    # =============================================================================
    # TAB 1: COCO DATASET EXTRACTION
    # =============================================================================
    with tab1:
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
                "✨ Usar SAM3 para anotaciones sin mascara",
                value=sam3_available,
                disabled=not sam3_available,
                key="extract_use_sam3",
                help="Cuando una anotación solo tiene bbox (sin polygon/RLE), SAM3 genera la máscara automáticamente"
            )

            force_bbox_only = st.checkbox(
                "⚠️ Ignorar mascaras existentes (usar solo bbox)",
                value=False,
                key="extract_force_bbox_only",
                help="Extraer usando solo bounding boxes, ignorando polygon/RLE masks"
            )

            force_sam3_resegmentation = st.checkbox(
                "🔄 Regenerar mascaras con SAM3 (usa bbox como guía)",
                value=False,
                disabled=not sam3_available,
                key="extract_force_sam3_reseg",
                help="Toma los bounding boxes del dataset y usa SAM3 para generar máscaras más precisas, ignorando las máscaras existentes de baja calidad"
            )

            force_sam3_text_prompt = st.checkbox(
                "🎯 Regenerar TODO con SAM3 (solo confío en la etiqueta)",
                value=False,
                disabled=not sam3_available,
                key="extract_force_sam3_text",
                help="No confías ni en bbox ni en máscaras. SAM3 usa solo el nombre de la clase como text prompt para regenerar máscara y bbox desde cero"
            )

        # Validación de opciones conflictivas
        spacer(4)
        conflicts_detected = False

        if force_bbox_only and (force_sam3_resegmentation or force_sam3_text_prompt):
            conflicts_detected = True
            st.warning("""
            ⚠️ **Opciones conflictivas detectadas:**

            No puedes activar **"Ignorar máscaras existentes"** junto con opciones de SAM3.

            - **Ignorar máscaras** → Crop rectangular simple (sin segmentación)
            - **Opciones SAM3** → Segmentación con SAM3

            **Recomendación:** Desactiva "Ignorar máscaras existentes".
            """)

        elif force_sam3_resegmentation and force_sam3_text_prompt:
            conflicts_detected = True
            st.warning("""
            ⚠️ **Opciones conflictivas detectadas:**

            No puedes activar **"Regenerar con SAM3 (bbox)"** y **"Regenerar TODO (text prompt)"** al mismo tiempo.

            - **Regenerar con bbox** → Usa bbox existente como guía para SAM3
            - **Regenerar TODO** → Ignora bbox y máscaras, usa solo etiqueta de clase

            **Recomendación:** Elige una de las dos según tu nivel de confianza en los bbox.
            """)
        elif force_sam3_resegmentation and not sam3_available:
            st.error("""
            🚫 **SAM3 no está disponible**

            La opción "Regenerar máscaras con SAM3" requiere que el servicio SAM3 esté activo.

            Verifica que el servicio de segmentación esté corriendo correctamente.
            """)
        elif force_sam3_text_prompt and not sam3_available:
            st.error("""
            🚫 **SAM3 no está disponible**

            La opción "Regenerar TODO con SAM3" requiere que el servicio SAM3 esté activo.

            Verifica que el servicio de segmentación esté corriendo correctamente.
            """)
        elif force_sam3_text_prompt:
            st.info("""
            🎯 **Modo: Regeneración total con text prompt**

            - SAM3 usará **solo el nombre de la clase** como prompt
            - Bounding boxes **ignorados** (no confiables)
            - Máscaras polygon/RLE **ignoradas** (no confiables)
            - Se generarán **máscara y bbox nuevos** desde cero basándose en el contenido visual

            💡 Ideal cuando bbox y máscaras son incorrectos, pero la etiqueta de clase es correcta.

            ⚠️ **Nota:** Requiere que SAM3 soporte text prompts (Grounded-SAM o similar).
            """, icon="🎯")
        elif force_sam3_resegmentation:
            st.info("""
            ✅ **Modo: Regeneración precisa con SAM3**

            - SAM3 usará los **bounding boxes** como guía
            - Máscaras polygon/RLE existentes serán **ignoradas**
            - Se generarán **máscaras nuevas** basadas en el contenido visual real

            💡 Ideal para mejorar datasets con máscaras de baja calidad pero bbox precisos.
            """, icon="🔄")
        elif force_bbox_only:
            st.info("""
            ⚠️ **Modo: Crops rectangulares sin segmentación**

            - Se usarán solo los **bounding boxes**
            - Las máscaras serán **ignoradas**
            - Resultado: **recortes rectangulares** sin transparencia

            💡 Ideal para entrenar modelos que no requieren segmentación precisa.
            """, icon="📦")

        # Ayuda contextual - Guía de opciones
        spacer(8)
        with st.expander("💡 Guía: ¿Qué opciones activar según tu caso?", expanded=False):
            st.markdown("""
            ### 📋 Escenarios Comunes

            #### 1️⃣ **Dataset solo tiene bounding boxes (sin máscaras)**
            ```
            ✅ Usar SAM3 para anotaciones sin mascara: ON
            ⬜ Ignorar mascaras existentes: OFF
            ⬜ Regenerar mascaras con SAM3: OFF
            ```
            **Resultado:** SAM3 genera máscaras automáticamente desde los bbox

            ---

            #### 2️⃣ **Dataset tiene máscaras, pero son de baja calidad**
            ```
            ✅ Usar SAM3 para anotaciones sin mascara: ON (recomendado)
            ⬜ Ignorar mascaras existentes: OFF
            ✅ Regenerar mascaras con SAM3: ON ← OPCIÓN CLAVE
            ```
            **Resultado:** SAM3 usa los bbox como guía para regenerar máscaras precisas

            **💡 Caso de uso:** Cuando tus máscaras polygon/RLE son imprecisas pero los bbox están bien posicionados

            ---

            #### 3️⃣ **Bbox y máscaras incorrectos (solo confío en la etiqueta)**
            ```
            ✅ Usar SAM3 para anotaciones sin mascara: ON (recomendado)
            ⬜ Ignorar mascaras existentes: OFF
            ⬜ Regenerar mascaras con SAM3: OFF
            ✅ Regenerar TODO con SAM3: ON ← OPCIÓN CLAVE
            ```
            **Resultado:** SAM3 usa solo el nombre de clase para regenerar máscara y bbox desde cero

            **💡 Caso de uso:** Dataset con anotaciones muy malas (bbox y máscaras incorrectos) pero etiquetas de clase correctas

            ---

            #### 4️⃣ **Solo quiero crops rectangulares (sin segmentación)**
            ```
            ⬜ Usar SAM3 para anotaciones sin mascara: OFF
            ✅ Ignorar mascaras existentes: ON ← OPCIÓN CLAVE
            ⬜ Regenerar mascaras con SAM3: OFF
            ⬜ Regenerar TODO con SAM3: OFF
            ```
            **Resultado:** Crop rectangular del bbox sin aplicar máscara

            ---

            #### 5️⃣ **Dataset tiene máscaras buenas, usarlas tal cual**
            ```
            ✅ Usar SAM3 para anotaciones sin mascara: ON (para bbox-only)
            ⬜ Ignorar mascaras existentes: OFF
            ⬜ Regenerar mascaras con SAM3: OFF
            ```
            **Resultado:** Usa polygon/RLE existentes + SAM3 solo para objetos sin máscara

            ---

            ### 🎯 Resumen de Opciones

            | Opción | Cuándo activar |
            |--------|----------------|
            | **Usar SAM3 para anotaciones sin mascara** | Siempre (si SAM3 disponible), procesa objetos que solo tienen bbox |
            | **Ignorar mascaras existentes** | Cuando quieres crops rectangulares sin segmentación |
            | **Regenerar mascaras con SAM3** | Cuando las máscaras son malas pero los bbox son buenos ✨ |
            | **Regenerar TODO con SAM3** | Cuando bbox Y máscaras son malas, solo confías en la etiqueta 🎯 |

            ---

            ### ⚙️ Flujo Técnico de "Regenerar con SAM3 (bbox)"

            Cuando activas **"🔄 Regenerar mascaras con SAM3"**:

            1. **Lee el bbox** de la anotación existente `[x, y, width, height]`
            2. **Ignora** las máscaras polygon/RLE que ya existen
            3. **Llama a SAM3** usando el bbox como prompt rectangular
            4. **Genera máscara precisa** basada en el contenido visual real
            5. **Extrae el objeto** con la nueva máscara de SAM3

            **Ventaja:** SAM3 es muy bueno segmentando cuando tiene un bbox de guía, produciendo máscaras mucho más precisas que las anotaciones manuales rápidas.

            ---

            ### ⚙️ Flujo Técnico de "Regenerar TODO (text prompt)"

            Cuando activas **"🎯 Regenerar TODO con SAM3"**:

            1. **Lee el nombre de la clase** de la anotación (ej: "fish", "coral")
            2. **Ignora** tanto el bbox como las máscaras existentes
            3. **Llama a SAM3** con el nombre de clase como text prompt
            4. SAM3 **busca el objeto en toda la imagen** basándose en la descripción
            5. **Genera máscara Y bbox nuevos** desde cero
            6. **Extrae el objeto** con la máscara generada

            **Ventaja:** Ideal cuando las anotaciones originales (bbox + máscaras) son completamente incorrectas pero sabes que la etiqueta de clase es correcta.

            ---

            ### 🔀 Diagrama de Flujo de Decisión

            ```
            ¿Tienes máscaras en tu dataset?
                │
                ├─ NO (solo bbox)
                │   └─> ✅ Usar SAM3 para anotaciones sin máscara: ON
                │       └─> Resultado: SAM3 genera máscaras desde bbox
                │
                └─ SÍ (polygon/RLE)
                    │
                    ├─ ¿Las máscaras son de BUENA calidad?
                    │   │
                    │   ├─ SÍ → Usar máscaras existentes
                    │   │   └─> ⬜ Regenerar con SAM3: OFF
                    │   │       └─> Resultado: Usa polygon/RLE tal cual
                    │   │
                    │   └─ NO → ¿Qué quieres hacer?
                    │       │
                    │       ├─ Mejorar con SAM3
                    │       │   └─> ✅ Regenerar con SAM3: ON
                    │       │       └─> Resultado: SAM3 crea máscaras desde bbox
                    │       │
                    │       └─ Crops rectangulares
                    │           └─> ✅ Ignorar máscaras: ON
                    │               └─> Resultado: Crop bbox sin segmentación
            ```

            ---

            ### 🎓 Casos de Uso Reales

            **Ejemplo 1: Dataset COCO con segmentaciones imperfectas**
            > Tienes un dataset con máscaras polygon pero fueron anotadas rápidamente y tienen errores.
            >
            > ✅ Solución: `Regenerar con SAM3: ON`
            >
            > SAM3 tomará los bbox y generará máscaras precisas ignorando las imperfectas.

            **Ejemplo 2: Dataset Open Images (solo bbox)**
            > Open Images tiene millones de bbox pero sin segmentaciones.
            >
            > ✅ Solución: `Usar SAM3 para anotaciones sin máscara: ON`
            >
            > SAM3 generará máscaras automáticamente para cada bbox.

            **Ejemplo 3: Preentrenamiento de clasificación (no necesitas máscaras)**
            > Solo quieres los objetos recortados para entrenar un clasificador.
            >
            > ✅ Solución: `Ignorar máscaras existentes: ON`
            >
            > Obtienes crops rectangulares rápidos sin procesamiento extra.

            ---

            ### 📊 Tabla Comparativa Rápida

            | Situación | Usar SAM3 sin máscara | Ignorar máscaras | Regenerar con SAM3 (bbox) | Regenerar TODO (text) | Resultado |
            |-----------|:---------------------:|:----------------:|:-------------------------:|:---------------------:|-----------|
            | **Dataset solo bbox** | ✅ ON | ⬜ OFF | ⬜ OFF | ⬜ OFF | SAM3 genera máscaras |
            | **Máscaras de baja calidad (bbox buenos)** | ✅ ON | ⬜ OFF | ✅ ON | ⬜ OFF | SAM3 regenera con bbox como guía ⭐ |
            | **Bbox Y máscaras incorrectos** | ✅ ON | ⬜ OFF | ⬜ OFF | ✅ ON | SAM3 regenera TODO desde etiqueta 🎯 |
            | **Máscaras de buena calidad** | ✅ ON | ⬜ OFF | ⬜ OFF | ⬜ OFF | Usa máscaras existentes |
            | **Solo crops rectangulares** | ⬜ OFF | ✅ ON | ⬜ OFF | ⬜ OFF | Bbox crop sin segmentación |

            **⭐ = Caso: Mejorar máscaras usando bbox como guía**
            **🎯 = Caso: Regenerar TODO cuando bbox y máscaras son malos**
            """)

        # =====================================================================
        # DEDUPLICATION CONFIGURATION
        # =====================================================================
        spacer(16)
        st.markdown("### 🔍 Configuración de Deduplicación")
        st.markdown("Previene extracciones duplicadas cuando múltiples anotaciones apuntan al mismo objeto")

        col_dedup1, col_dedup2 = st.columns(2)

        with col_dedup1:
            enable_dedup = st.checkbox(
                "✅ Prevenir duplicados",
                value=True,
                key="extract_enable_dedup",
                help="Evita extraer el mismo objeto múltiples veces usando detección de solapamiento (IoU)"
            )

            if enable_dedup:
                iou_threshold = st.slider(
                    "Umbral de IoU para duplicados",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.05,
                    key="extract_iou_threshold",
                    help="Mayor valor = más estricto (solo marca duplicados muy obvios). 0.7 = 70% de solapamiento."
                )

                st.caption(f"📊 Umbral actual: {iou_threshold:.0%} de solapamiento")

        with col_dedup2:
            if enable_dedup:
                matching_strategy = st.selectbox(
                    "Estrategia de emparejamiento (modo text prompt)",
                    options=["bbox_iou", "mask_iou", "center_distance"],
                    index=0,
                    key="extract_matching_strategy",
                    help="Cómo emparejar instancias de SAM3 con anotaciones en modo text prompt"
                )

                cross_category_dedup = st.checkbox(
                    "Deduplicación entre categorías",
                    value=False,
                    key="extract_cross_category_dedup",
                    help="Marcar como duplicados objetos de diferentes clases si se solapan (normalmente desactivado)"
                )

        # Info box explaining deduplication
        spacer(8)
        if enable_dedup:
            st.info(f"""
✅ **Deduplicación habilitada**

- **Umbral IoU**: {iou_threshold:.0%} - Objetos con solapamiento ≥ {iou_threshold:.0%} se consideran duplicados
- **Estrategia**: {matching_strategy} - Método para emparejar instancias SAM3 con anotaciones
- **Entre categorías**: {'Sí' if cross_category_dedup else 'No'} - {'Detecta duplicados entre clases diferentes' if cross_category_dedup else 'Solo dentro de la misma clase'}

💡 **Beneficios**:
- Previene extraer el mismo pez 10 veces si hay 10 anotaciones en la imagen
- Asigna cada instancia SAM3 a la anotación más cercana (matching uno-a-uno)
- Reduce tamaño del dataset y evita sesgo en entrenamiento
            """)
        else:
            st.warning("""
⚠️ **Deduplicación deshabilitada**

En modo text prompt con múltiples anotaciones de la misma clase, **todas las anotaciones pueden extraer el mismo objeto**.

Ejemplo: 5 anotaciones "fish" → 5 archivos PNG idénticos del mismo pez.
            """)

        # Deduplication guide
        spacer(8)
        with st.expander("💡 Guía: ¿Cuándo usar deduplicación?", expanded=False):
            st.markdown("""
            ### 📋 Casos de Uso

            #### 1️⃣ **Modo Text Prompt con múltiples instancias**
            ```
            Situación: Imagen con 10 peces, 10 anotaciones "fish"
            Sin dedup: SAM3 detecta 5 peces reales → 10 anotaciones extraen los MISMOS 5 peces → duplicados
            ✅ Con dedup: Matching 1-a-1 → 5 anotaciones obtienen peces únicos → 5 extracciones correctas
            ```

            **Recomendación:** ✅ Deduplicación **ACTIVADA** (default)

            ---

            #### 2️⃣ **Modo Bbox con bboxes solapados**
            ```
            Situación: 2 bbox de "coral" en misma área (anotador duplicó por error)
            Sin dedup: Extrae el mismo coral 2 veces
            ✅ Con dedup: Detecta IoU > 70% → Skip segundo → 1 extracción correcta
            ```

            **Recomendación:** ✅ Deduplicación **ACTIVADA**

            ---

            #### 3️⃣ **Dataset limpio sin duplicados**
            ```
            Situación: Dataset bien curado, cada anotación es única
            Dedup activada: Pequeño overhead (~20%) pero previene edge cases
            ```

            **Recomendación:** ✅ Deduplicación **ACTIVADA** (es seguro, mínimo impacto)

            ---

            ### ⚙️ Cómo Funciona

            **Text Prompt Mode:**
            1. SAM3 detecta N instancias de "fish" en imagen
            2. Hay M anotaciones "fish" en el dataset
            3. Matching greedy por bbox IoU: asigna cada anotación a mejor instancia
            4. Registry verifica duplicados antes de extraer
            5. Solo extrae instancias únicas

            **Bbox/Mask Mode:**
            1. Para cada anotación, extrae máscara
            2. Registry compara con máscaras ya extraídas de la imagen
            3. Si IoU ≥ threshold → Skip (duplicado)
            4. Si único → Extrae y registra

            ---

            ### 📊 Configuración Recomendada

            | Parámetro | Valor Default | Recomendación |
            |-----------|:-------------:|---------------|
            | **Enabled** | ✅ True | Siempre activado |
            | **IoU Threshold** | 0.7 (70%) | 0.7 para duplicados obvios, 0.5 para más agresivo |
            | **Matching Strategy** | bbox_iou | bbox_iou (rápido), mask_iou (preciso pero lento) |
            | **Cross-category** | ❌ False | False (pez y coral pueden superponerse) |

            ---

            ### ⚠️ Escenarios M > N (Más anotaciones que instancias)

            ```
            Situación: 10 anotaciones "fish", SAM3 solo encuentra 5 peces
            Comportamiento:
            - 5 anotaciones obtienen peces únicos (matching)
            - 5 anotaciones FALLAN con mensaje claro: "No SAM3 instance matched"
            ```

            **Esto es INTENCIONAL** (preferencia del usuario):
            - ✅ Mejor fallar explícitamente que crear duplicados
            - ✅ Logs claros: "Found 5 instances for 10 annotations - 5 will fail"
            - ✅ Usuario puede revisar anotaciones fallidas

            ---

            ### 💡 Caso Real: CleanSea Dataset

            **Antes (sin dedup):**
            - 1000 imágenes con 10,000 anotaciones "fish"
            - Modo text prompt → 10,000 extracciones
            - **Problema:** Muchas imágenes tenían 10 anotaciones pero solo 3-4 peces reales
            - **Resultado:** 10,000 archivos PNG con ~60% duplicados

            **Después (con dedup):**
            - SAM3 detecta ~4,200 peces únicos
            - Matching + dedup → 4,200 extracciones únicas
            - **Beneficio:** Dataset 58% más pequeño, sin duplicados, mejor para entrenamiento
            """)

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
            # Show which extraction mode is active
            active_modes = []
            if force_bbox_only:
                active_modes.append("🔲 **Solo bbox** (ignorando máscaras)")
            if force_sam3_resegmentation:
                active_modes.append("🔄 **SAM3 re-segmentación** (bbox como guía)")
            if force_sam3_text_prompt:
                active_modes.append("🎯 **SAM3 text prompt** (solo etiqueta)")

            if active_modes:
                st.info("**Modo activo:** " + " | ".join(active_modes))
            else:
                st.info("**Modo automático:** Usa máscaras existentes o bbox según lo disponible")

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
                                padding=padding,
                                force_bbox_only=force_bbox_only,
                                force_sam3_resegmentation=force_sam3_resegmentation,
                                force_sam3_text_prompt=force_sam3_text_prompt
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
            # Build deduplication config
            dedup_config = None
            if enable_dedup:
                dedup_config = {
                    "enabled": True,
                    "iou_threshold": iou_threshold,
                    "matching_strategy": matching_strategy,
                    "cross_category_dedup": cross_category_dedup
                }

            st.session_state["extract_pending_settings"] = {
                "coco_json_path": st.session_state.get("extract_coco_json_path"),
                "images_dir": images_dir,
                "output_dir": output_dir,
                "categories": selected_categories if selected_categories else None,
                "use_sam3": use_sam3,
                "force_bbox_only": force_bbox_only,
                "force_sam3_resegmentation": force_sam3_resegmentation,
                "force_sam3_text_prompt": force_sam3_text_prompt,
                "padding": padding,
                "min_area": min_area,
                "save_json": save_individual_json,
                "deduplication": dedup_config
            }

        # Check for conflicting options
        has_conflicts = (
            (force_bbox_only and (force_sam3_resegmentation or force_sam3_text_prompt)) or
            (force_sam3_resegmentation and force_sam3_text_prompt) or
            conflicts_detected
        )
        button_disabled = total_to_extract == 0 or has_conflicts

        # Show button for starting extraction
        if has_conflicts:
            st.error("❌ No se puede iniciar: opciones conflictivas seleccionadas (ver advertencia arriba)")

        st.button(
            f"🚀 Iniciar Extraccion ({total_to_extract} objetos)",
            type="primary",
            use_container_width=True,
            disabled=button_disabled,
            on_click=on_extract_click
        )

    # =============================================================================
    # TAB 2: IMAGENET-STYLE EXTRACTION
    # =============================================================================
    with tab2:
        section_header("Extraer desde Estructura ImageNet", icon="🗂️")

        st.info("""
        **Estructura esperada:**
        ```
        root_dir/
        ├── clase1/
        │   ├── img001.jpg
        │   └── img002.jpg
        ├── clase2/
        │   └── ...
        ```
        SAM3 segmentará cada imagen usando el nombre de la carpeta (clase) como prompt.
        """)

        if not sam3_available:
            alert_box(
                "⚠️ SAM3 no está disponible. Esta funcionalidad requiere SAM3 para segmentar objetos por clase.",
                type="error",
                icon="🚫"
            )
            return

        # Input fields
        col1, col2 = st.columns(2)

        with col1:
            imagenet_root_dir = st.text_input(
                "Directorio raíz",
                value=st.session_state.get("imagenet_root_dir", "/app/datasets/imagenet_style"),
                key="imagenet_root_input",
                help="Directorio con subdirectorios por clase"
            )
            st.session_state["imagenet_root_dir"] = imagenet_root_dir

        with col2:
            imagenet_output_dir = st.text_input(
                "Directorio de salida",
                value=st.session_state.get("imagenet_output_dir", "/app/datasets/extracted_imagenet"),
                key="imagenet_output_input",
                help="Directorio donde se guardarán los objetos extraídos"
            )
            st.session_state["imagenet_output_dir"] = imagenet_output_dir

        # Options
        col1, col2 = st.columns(2)

        with col1:
            imagenet_padding = st.slider(
                "Padding (píxeles)",
                min_value=0,
                max_value=50,
                value=5,
                key="imagenet_padding"
            )

        with col2:
            imagenet_min_area = st.number_input(
                "Área mínima (píxeles)",
                min_value=0,
                max_value=10000,
                value=100,
                key="imagenet_min_area"
            )

        imagenet_max_per_class = st.number_input(
            "Máximo de objetos por clase (0 = todos)",
            min_value=0,
            max_value=10000,
            value=0,
            key="imagenet_max_per_class",
            help="Limita el número de imágenes procesadas por clase"
        )

        spacer(24)

        # Extract button
        if st.button(
            "🚀 Iniciar Extracción ImageNet",
            type="primary",
            use_container_width=True,
            key="imagenet_extract_btn"
        ):
            with st.spinner("Iniciando extracción desde estructura ImageNet..."):
                result = client.extract_from_imagenet(
                    root_dir=imagenet_root_dir,
                    output_dir=imagenet_output_dir,
                    padding=imagenet_padding,
                    min_object_area=imagenet_min_area,
                    max_objects_per_class=imagenet_max_per_class if imagenet_max_per_class > 0 else None
                )

                if result.get("success"):
                    job_id = result.get("job_id")
                    st.session_state["extract_current_job_id"] = job_id
                    st.success(f"✅ Job iniciado correctamente: {job_id[:8]}...")
                    st.info("Redirigiendo al monitor...")
                    st.session_state.nav_menu = "📊 Monitor"
                    st.rerun()
                else:
                    alert_box(f"Error al iniciar extracción: {result.get('error', '?')}", type="error")

    # =============================================================================
    # TAB 3: CUSTOM OBJECT NAMES EXTRACTION
    # =============================================================================
    with tab3:
        st.markdown("""
        ### 🎯 Extracción por Nombres Personalizados

        Segmenta objetos especificando nombres directamente, **sin necesidad de JSON COCO**.

        SAM3 buscará todas las instancias de cada objeto en tus imágenes usando
        reconocimiento visual basado en los nombres que proporciones.
        """)

        spacer(16)

        # Section 1: Input Configuration
        section_header("Configuración de Entrada", icon="📝")

        col1, col2 = st.columns(2)

        with col1:
            custom_images_dir = st.text_input(
                "Directorio de imágenes",
                value=st.session_state.get("custom_images_dir", "/app/datasets/images"),
                key="custom_images_dir_input",
                help="Directorio que contiene las imágenes a procesar"
            )
            st.session_state["custom_images_dir"] = custom_images_dir

        with col2:
            custom_output_dir = st.text_input(
                "Directorio de salida",
                value=st.session_state.get("custom_output_dir", "/app/datasets/custom_extracted"),
                key="custom_output_dir_input",
                help="Directorio donde se guardarán los objetos extraídos (organizados por tipo)"
            )
            st.session_state["custom_output_dir"] = custom_output_dir

        spacer(8)

        # Object names input
        object_names_input = st.text_area(
            "Nombres de objetos a buscar (uno por línea o separados por coma)",
            value=st.session_state.get("custom_object_names", ""),
            height=120,
            key="custom_object_names_input",
            placeholder="Ejemplos:\nfish\ncoral\nplastic debris\nseaweed\n\nO separados por coma: fish, coral, plastic debris",
            help="Escribe los nombres de los objetos que quieres segmentar. SAM3 buscará estos objetos en todas las imágenes."
        )
        st.session_state["custom_object_names"] = object_names_input

        # Parse object names
        if object_names_input.strip():
            # Support both newlines and commas
            object_names = [
                name.strip()
                for name in object_names_input.replace(',', '\n').split('\n')
                if name.strip()
            ]
        else:
            object_names = []

        # Show preview of parsed names
        if object_names:
            st.info(f"🔍 Se buscarán **{len(object_names)}** tipos de objetos: {', '.join(f'`{name}`' for name in object_names)}")
        else:
            st.warning("⚠️ Debes especificar al menos un nombre de objeto")

        # Section 2: Extraction Options
        spacer(16)
        section_header("Opciones de Extracción", icon="⚙️")

        col1, col2 = st.columns(2)

        with col1:
            custom_padding = st.slider(
                "Padding (px)",
                min_value=0,
                max_value=50,
                value=5,
                key="custom_padding",
                help="Píxeles de relleno alrededor de cada objeto extraído"
            )

            custom_min_area = st.number_input(
                "Área mínima (px²)",
                min_value=0,
                max_value=10000,
                value=100,
                key="custom_min_area",
                help="Área mínima en píxeles para considerar un objeto válido"
            )

        with col2:
            custom_save_coco = st.checkbox(
                "Guardar COCO JSON individual",
                value=True,
                key="custom_save_coco",
                help="Guardar un archivo JSON COCO por cada objeto extraído"
            )

            if not sam3_available:
                alert_box(
                    "SAM3 no está disponible. Esta funcionalidad requiere SAM3 para segmentación por texto.",
                    type="error",
                    icon="❌"
                )

        # Deduplication settings
        with st.expander("🔍 Configuración de Deduplicación", expanded=False):
            st.markdown("""
            La deduplicación previene extraer el mismo objeto múltiples veces cuando SAM3
            detecta instancias superpuestas.
            """)

            custom_dedup_enabled = st.checkbox(
                "Activar deduplicación",
                value=True,
                key="custom_dedup_enabled",
                help="Previene duplicados usando IoU (Intersection over Union)"
            )

            if custom_dedup_enabled:
                custom_iou_threshold = st.slider(
                    "Umbral de IoU para duplicados",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.05,
                    key="custom_iou_threshold",
                    help="Threshold de superposición. Más alto = solo marca duplicados obvios (0.7 recomendado)"
                )
                st.caption(f"Objetos con ≥ {custom_iou_threshold:.0%} de superposición se consideran duplicados")

        # Section 3: Start Extraction
        spacer(16)
        section_header("Iniciar Extracción", icon="🚀")

        # Validation
        can_extract = True
        validation_messages = []

        if not custom_images_dir or not os.path.exists(custom_images_dir):
            can_extract = False
            validation_messages.append("❌ Directorio de imágenes no existe")

        if not custom_output_dir:
            can_extract = False
            validation_messages.append("❌ Debes especificar un directorio de salida")

        if not object_names:
            can_extract = False
            validation_messages.append("❌ Debes especificar al menos un nombre de objeto")

        if not sam3_available:
            can_extract = False
            validation_messages.append("❌ SAM3 no está disponible")

        if validation_messages:
            for msg in validation_messages:
                st.warning(msg)

        # Extraction button
        if st.button(
            "🚀 Extraer Objetos Custom",
            disabled=not can_extract,
            key="custom_extract_button",
            type="primary",
            use_container_width=True
        ):
            with st.spinner("Iniciando extracción personalizada..."):
                # Prepare deduplication config
                deduplication_config = None
                if custom_dedup_enabled:
                    deduplication_config = {
                        "enabled": True,
                        "iou_threshold": custom_iou_threshold,
                        "matching_strategy": "bbox_iou",
                        "cross_category_dedup": False
                    }

                # Call API
                result = client.extract_custom_objects(
                    images_dir=custom_images_dir,
                    output_dir=custom_output_dir,
                    object_names=object_names,
                    padding=custom_padding,
                    min_object_area=custom_min_area,
                    save_individual_coco=custom_save_coco,
                    deduplication=deduplication_config
                )

                if result.get("success"):
                    job_id = result.get("job_id")
                    st.session_state["extract_current_job_id"] = job_id
                    st.success(f"✅ Job personalizado iniciado: {job_id[:8]}...")
                    st.info(f"📊 {result.get('message', 'Procesando...')}")
                    st.info("Redirigiendo al monitor...")
                    st.session_state.nav_menu = "📊 Monitor"
                    st.rerun()
                else:
                    alert_box(f"Error al iniciar extracción custom: {result.get('error', '?')}", type="error")

        # Help section
        with st.expander("❓ Ayuda y Ejemplos"):
            st.markdown("""
            ### Cómo usar esta funcionalidad

            1. **Especifica el directorio de imágenes**: Carpeta con las imágenes a procesar
            2. **Escribe los nombres de objetos**: Uno por línea o separados por comas
            3. **Ajusta opciones**: Padding, área mínima, deduplicación
            4. **Inicia extracción**: SAM3 buscará y segmentará los objetos automáticamente

            ### Ejemplos de nombres válidos

            - `fish` - Peces en general
            - `plastic bottle` - Botellas de plástico
            - `coral reef` - Arrecifes de coral
            - `plastic debris` - Desechos plásticos
            - `seaweed` - Algas marinas

            ### Estructura de salida

            Los objetos extraídos se organizarán en carpetas por tipo:
            ```
            output_dir/
            ├── fish/
            │   ├── image001_fish_instance000.png
            │   ├── image001_fish_instance001.png
            │   └── ...
            ├── coral/
            │   ├── image001_coral_instance000.png
            │   └── ...
            └── extraction_summary.json
            ```

            ### Notas importantes

            - SAM3 detectará **todas las instancias** de cada objeto en cada imagen
            - La deduplicación previene extraer el mismo objeto múltiples veces
            - Si SAM3 no encuentra un objeto, simplemente no extraerá nada (no es un error)
            - Los nombres genéricos funcionan mejor (ej: "fish" en lugar de "red snapper")
            """)


