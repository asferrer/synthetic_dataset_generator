"""
Labeling Tool Page
==================
Tool for labeling/relabeling datasets using SAM3 text-based segmentation.
Supports detection, segmentation, and classification tasks.
"""

import json
import time
import streamlit as st
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from app.components.ui import (
    page_header, section_header, spacer, alert_box, empty_state
)
from app.components.api_client import get_api_client
from app.config.theme import get_colors_dict


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s" if secs > 0 else f"{minutes}m"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m" if minutes > 0 else f"{hours}h"


# =============================================================================
# PREDEFINED TEMPLATES FOR COMMON LABELING SCENARIOS
# =============================================================================

LABELING_TEMPLATES = {
    "🐟 Vida Marina": {
        "description": "Objetos comunes en ambientes marinos y acuáticos",
        "classes": [
            "fish", "tropical fish", "reef fish", "school of fish",
            "coral", "coral reef", "sea anemone",
            "jellyfish", "octopus", "squid", "crab", "lobster", "shrimp",
            "sea turtle", "shark", "ray", "manta ray", "stingray",
            "seahorse", "starfish", "sea urchin", "seaweed", "kelp",
            "clownfish", "angelfish", "parrotfish", "grouper", "barracuda"
        ]
    },
    "🚗 Vehículos": {
        "description": "Vehículos terrestres de todo tipo",
        "classes": [
            "car", "sedan", "SUV", "truck", "pickup truck", "van", "minivan",
            "motorcycle", "scooter", "bicycle", "bus", "school bus",
            "taxi", "police car", "ambulance", "fire truck",
            "sports car", "convertible", "limousine", "vintage car",
            "damaged car", "parked car", "moving car"
        ]
    },
    "👤 Personas": {
        "description": "Personas en diferentes poses y actividades",
        "classes": [
            "person", "man", "woman", "child", "baby", "elderly person",
            "walking person", "running person", "sitting person", "standing person",
            "person with backpack", "person with umbrella", "cyclist",
            "pedestrian", "crowd of people", "group of people"
        ]
    },
    "🏠 Objetos Domesticos": {
        "description": "Objetos comunes en interiores",
        "classes": [
            "chair", "table", "sofa", "couch", "bed", "desk", "lamp",
            "television", "computer monitor", "laptop", "keyboard", "mouse",
            "bottle", "cup", "mug", "plate", "bowl", "fork", "knife", "spoon",
            "book", "plant", "potted plant", "vase", "clock", "picture frame"
        ]
    },
    "🏗️ Infraestructura Urbana": {
        "description": "Elementos de entornos urbanos",
        "classes": [
            "building", "skyscraper", "house", "apartment building",
            "traffic light", "traffic sign", "stop sign", "street sign",
            "street lamp", "fire hydrant", "parking meter", "bench",
            "sidewalk", "crosswalk", "road", "bridge", "tunnel",
            "construction site", "crane", "scaffolding"
        ]
    },
    "🐕 Animales": {
        "description": "Animales domésticos y salvajes",
        "classes": [
            "dog", "cat", "bird", "horse", "cow", "sheep", "pig", "goat",
            "chicken", "duck", "rabbit", "hamster", "guinea pig",
            "elephant", "lion", "tiger", "bear", "deer", "fox", "wolf",
            "monkey", "gorilla", "giraffe", "zebra", "hippo", "rhino"
        ]
    },
    "🗑️ Residuos y Basura": {
        "description": "Objetos de basura para detección de contaminación",
        "classes": [
            "plastic bottle", "plastic bag", "plastic wrapper", "plastic container",
            "cardboard box", "paper", "newspaper", "cardboard",
            "glass bottle", "broken glass", "metal can", "aluminum can",
            "food waste", "organic waste", "cigarette butt", "cigarette",
            "tire", "rubber", "electronic waste", "battery",
            "fishing net", "rope", "styrofoam", "foam"
        ]
    },
    "🏭 Industrial": {
        "description": "Equipamiento y maquinaria industrial",
        "classes": [
            "machine", "robot", "conveyor belt", "forklift", "pallet",
            "container", "shipping container", "tank", "pipe", "valve",
            "gauge", "control panel", "generator", "motor", "pump",
            "safety equipment", "hard hat", "safety vest", "fire extinguisher"
        ]
    }
}


def _render_template_selector(c: Dict) -> None:
    """Render template selector for predefined class sets."""
    st.markdown("**Selecciona una plantilla predefinida:**")

    # Show templates as cards
    cols = st.columns(2)

    for idx, (name, template) in enumerate(LABELING_TEMPLATES.items()):
        with cols[idx % 2]:
            with st.container():
                st.markdown(f"""
                <div style="background: {c['bg_card']}; border: 1px solid {c['border']};
                            border-radius: 0.5rem; padding: 0.75rem; margin-bottom: 0.5rem;">
                    <div style="font-weight: 600; font-size: 1rem;">{name}</div>
                    <div style="font-size: 0.75rem; color: {c['text_muted']};">{template['description']}</div>
                    <div style="font-size: 0.7rem; color: {c['text_secondary']}; margin-top: 0.25rem;">
                        {len(template['classes'])} clases
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if st.button(f"Usar {name.split()[0]}", key=f"template_{idx}", use_container_width=True):
                    st.session_state.labeling_classes = template["classes"].copy()
                    st.success(f"✓ Cargadas {len(template['classes'])} clases de '{name}'")
                    st.rerun()

    spacer(8)

    # Option to combine templates
    with st.expander("Combinar plantillas"):
        selected_templates = st.multiselect(
            "Selecciona varias plantillas",
            options=list(LABELING_TEMPLATES.keys()),
            key="combine_templates"
        )

        if selected_templates:
            combined_classes = []
            for tpl_name in selected_templates:
                combined_classes.extend(LABELING_TEMPLATES[tpl_name]["classes"])
            combined_classes = list(set(combined_classes))  # Remove duplicates

            st.info(f"Total: {len(combined_classes)} clases unicas")

            if st.button("Cargar clases combinadas", type="primary"):
                st.session_state.labeling_classes = combined_classes
                st.rerun()


def _render_detailed_class_input(c: Dict) -> None:
    """Render detailed class input with variations and synonyms."""
    st.markdown(f"""
    <div style="background: {c['bg_secondary']}; padding: 0.75rem; border-radius: 0.5rem;
                margin-bottom: 1rem; font-size: 0.85rem; color: {c['text_muted']};">
        Añade clases con sus variaciones para una detección más exhaustiva.
        Cada variación se busca por separado pero se etiqueta con la clase principal.
    </div>
    """, unsafe_allow_html=True)

    # Initialize detailed classes structure
    if "detailed_classes" not in st.session_state:
        st.session_state.detailed_classes = {}

    # Display current detailed classes
    if st.session_state.detailed_classes:
        st.markdown("**Clases configuradas:**")

        for main_class, variations in st.session_state.detailed_classes.items():
            with st.expander(f"🏷️ **{main_class}** ({len(variations)} variaciones)", expanded=False):
                st.markdown(f"Variaciones: {', '.join(variations)}")

                if st.button(f"Eliminar '{main_class}'", key=f"del_detailed_{main_class}"):
                    del st.session_state.detailed_classes[main_class]
                    # Update flat list
                    _update_flat_classes_from_detailed()
                    st.rerun()

    spacer(8)

    # Add new class with variations
    st.markdown("**Añadir nueva clase:**")

    col1, col2 = st.columns([1, 2])

    with col1:
        new_main_class = st.text_input(
            "Nombre de la clase",
            placeholder="ej: fish",
            key="new_main_class",
            help="Este será el nombre de la etiqueta final"
        )

    with col2:
        new_variations = st.text_input(
            "Variaciones (separadas por comas)",
            placeholder="ej: fish, tropical fish, reef fish, school of fish",
            key="new_variations",
            help="SAM3 buscará cada variación pero etiquetará con el nombre principal"
        )

    if st.button("➕ Añadir clase con variaciones", type="primary"):
        if new_main_class:
            variations = [v.strip() for v in new_variations.split(",") if v.strip()]
            if not variations:
                variations = [new_main_class]
            elif new_main_class not in variations:
                variations.insert(0, new_main_class)

            st.session_state.detailed_classes[new_main_class] = variations
            _update_flat_classes_from_detailed()
            st.rerun()

    spacer(8)

    # Quick add common variations
    with st.expander("Generador de variaciones"):
        base_object = st.text_input(
            "Objeto base",
            placeholder="ej: bottle",
            key="variation_base"
        )

        if base_object:
            st.markdown("**Variaciones sugeridas:**")

            # Generate common variations
            prefixes = ["", "small ", "large ", "big ", "damaged ", "broken "]
            materials = ["plastic ", "glass ", "metal ", ""]
            colors = ["red ", "blue ", "green ", "white ", "black ", ""]

            suggested = set()
            suggested.add(base_object)
            for prefix in prefixes[:3]:
                suggested.add(f"{prefix}{base_object}".strip())
            for material in materials:
                suggested.add(f"{material}{base_object}".strip())

            suggested = list(suggested)[:10]  # Limit to 10

            st.code(", ".join(suggested))

            if st.button("Usar estas variaciones"):
                st.session_state.detailed_classes[base_object] = suggested
                _update_flat_classes_from_detailed()
                st.rerun()


def _update_flat_classes_from_detailed():
    """Update the flat labeling_classes list from detailed_classes."""
    all_variations = []
    for variations in st.session_state.detailed_classes.values():
        all_variations.extend(variations)
    st.session_state.labeling_classes = list(set(all_variations))


def _render_simple_class_input(c: Dict, client) -> None:
    """Render simple class input (original functionality)."""
    # Class input methods
    class_input_method = st.radio(
        "Metodo de entrada",
        ["Manual", "Desde archivo", "Desde dataset existente"],
        horizontal=True,
        key="class_input_method"
    )

    if class_input_method == "Manual":
        # Display current classes as tags
        if st.session_state.labeling_classes:
            st.markdown("**Clases actuales:**")

            # Show in a more compact way
            classes_display = ", ".join([f"`{cls}`" for cls in st.session_state.labeling_classes[:20]])
            if len(st.session_state.labeling_classes) > 20:
                classes_display += f" ... y {len(st.session_state.labeling_classes) - 20} más"
            st.markdown(classes_display)

            if st.button("🗑️ Limpiar todas", key="clear_all_classes"):
                st.session_state.labeling_classes = []
                st.rerun()

        # Add new class
        col_cls, col_add_cls = st.columns([4, 1])
        with col_cls:
            new_class = st.text_input(
                "Nueva clase",
                placeholder="nombre de objeto (ej: fish, bottle, person)",
                key="new_class_name"
            )
        with col_add_cls:
            if st.button("Añadir", key="add_class", type="primary"):
                if new_class and new_class.strip() not in st.session_state.labeling_classes:
                    st.session_state.labeling_classes.append(new_class.strip())
                    st.rerun()

        # Bulk add
        with st.expander("📝 Añadir múltiples clases"):
            st.markdown(f"""
            <div style="font-size: 0.8rem; color: {c['text_muted']}; margin-bottom: 0.5rem;">
                Introduce una clase por línea o separadas por comas.
                <strong>Tip:</strong> Añade variaciones para mejor detección.
            </div>
            """, unsafe_allow_html=True)

            bulk_classes = st.text_area(
                "Clases",
                placeholder="fish\ntropical fish\nreef fish\ncoral\njellyfish\nplastic bottle\nplastic bag",
                height=150,
                key="bulk_classes_input"
            )
            if st.button("Añadir todas", key="bulk_add"):
                if bulk_classes:
                    new_classes = []
                    for line in bulk_classes.split("\n"):
                        for cls in line.split(","):
                            cls = cls.strip()
                            if cls and cls not in st.session_state.labeling_classes:
                                new_classes.append(cls)
                    st.session_state.labeling_classes.extend(new_classes)
                    st.success(f"✓ Añadidas {len(new_classes)} clases")
                    st.rerun()

    elif class_input_method == "Desde archivo":
        uploaded = st.file_uploader(
            "Subir archivo de clases (JSON o TXT)",
            type=["json", "txt"],
            key="class_file_upload"
        )
        if uploaded:
            try:
                if uploaded.name.endswith(".json"):
                    data = json.load(uploaded)
                    if isinstance(data, list):
                        classes = data
                    elif "categories" in data:
                        classes = [c.get("name", c) for c in data["categories"]]
                    elif "names" in data:
                        classes = data["names"]
                    else:
                        classes = list(data.keys()) if isinstance(data, dict) else []
                else:
                    content = uploaded.read().decode("utf-8")
                    classes = [c.strip() for c in content.split("\n") if c.strip()]

                if classes:
                    st.session_state.labeling_classes = classes
                    st.success(f"✓ Cargadas {len(classes)} clases")
            except Exception as e:
                st.error(f"Error al cargar archivo: {e}")

    else:  # Desde dataset existente
        datasets_response = client.list_datasets()
        datasets = datasets_response.get("datasets", [])

        if datasets:
            dataset_options = {
                f"{d.get('job_id', 'unknown')[:12]}... ({d.get('num_images', 0)} imgs)": d
                for d in datasets
            }
            selected = st.selectbox(
                "Seleccionar dataset",
                options=list(dataset_options.keys()),
                key="class_source_dataset"
            )

            if st.button("Cargar clases del dataset"):
                ds = dataset_options[selected]
                metadata = client.get_dataset_metadata(ds.get("job_id"))
                if metadata and not metadata.get("error"):
                    categories = metadata.get("categories", [])
                    if categories:
                        st.session_state.labeling_classes = [c.get("name", c) for c in categories]
                        st.success(f"✓ Cargadas {len(categories)} clases")
                        st.rerun()
        else:
            st.info("No hay datasets disponibles")


def render_labeling_tool_page():
    """Render the labeling tool page."""
    c = get_colors_dict()

    page_header(
        title="Herramienta de Etiquetado",
        subtitle="Etiqueta o re-etiqueta datasets usando SAM3 con prompts de texto",
        icon="🏷️"
    )

    # Initialize session state
    if "labeling_classes" not in st.session_state:
        st.session_state.labeling_classes = []
    if "labeling_image_dirs" not in st.session_state:
        st.session_state.labeling_image_dirs = []

    # Check SAM3 availability
    client = get_api_client()
    health = client.get_segmentation_health()

    sam3_available = health.get("sam3_available", False)

    if not sam3_available:
        alert_box(
            "SAM3 no esta disponible. Esta herramienta requiere SAM3 para segmentacion basada en texto.",
            type="error",
            icon="⚠️"
        )
        st.info("SAM3 se carga automaticamente si esta configurado correctamente en el servicio de segmentacion.")
        return

    # Main tabs
    tab_new, tab_relabel, tab_jobs = st.tabs([
        "🆕 Etiquetar Nuevo",
        "🔄 Re-etiquetar Dataset",
        "📊 Monitor de Jobs"
    ])

    with tab_new:
        _render_new_labeling_tab(c, client)

    with tab_relabel:
        _render_relabel_tab(c, client)

    with tab_jobs:
        _render_labeling_jobs_tab(c, client)


def _render_new_labeling_tab(c: Dict, client) -> None:
    """Tab for labeling images from scratch."""
    section_header("Etiquetar Imagenes Sin Anotaciones", icon="🆕")

    st.markdown(f"""
    <div style="background: {c['info_bg']}; border: 1px solid {c['info']};
                border-radius: 0.5rem; padding: 1rem; margin-bottom: 1rem;">
        <span style="font-size: 0.9rem; color: {c['text_secondary']};">
            Selecciona directorios de imagenes y define las clases a detectar.
            SAM3 segmentara automaticamente todos los objetos que coincidan con los nombres de clase.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Image directories section
    section_header("Directorios de Imagenes", icon="📁")

    st.markdown(f"""
    <div style="background: {c['bg_secondary']}; padding: 0.75rem; border-radius: 0.5rem;
                margin-bottom: 1rem; font-size: 0.85rem; color: {c['text_muted']};">
        Puedes especificar multiples directorios. Las imagenes se buscaran en todos ellos.
    </div>
    """, unsafe_allow_html=True)

    # Display current directories
    if st.session_state.labeling_image_dirs:
        for i, dir_path in enumerate(st.session_state.labeling_image_dirs):
            col_dir, col_del = st.columns([5, 1])
            with col_dir:
                st.text_input(
                    f"Directorio {i+1}",
                    value=dir_path,
                    key=f"dir_{i}",
                    disabled=True
                )
            with col_del:
                if st.button("🗑️", key=f"del_dir_{i}", help="Eliminar"):
                    st.session_state.labeling_image_dirs.pop(i)
                    st.rerun()

    # Add new directory
    col_new_dir, col_add = st.columns([5, 1])
    with col_new_dir:
        new_dir = st.text_input(
            "Añadir directorio",
            placeholder="/app/data/images",
            key="new_image_dir"
        )
    with col_add:
        if st.button("➕", key="add_dir", help="Añadir directorio"):
            if new_dir and new_dir not in st.session_state.labeling_image_dirs:
                st.session_state.labeling_image_dirs.append(new_dir)
                st.rerun()

    spacer(16)

    # Classes section
    section_header("Objetos a Detectar y Etiquetar", icon="🏷️")

    st.markdown(f"""
    <div style="background: {c['info_bg']}; border: 1px solid {c['info']};
                border-radius: 0.5rem; padding: 1rem; margin-bottom: 1rem;">
        <div style="font-weight: 600; color: {c['text_primary']}; margin-bottom: 0.5rem;">
            💡 Consejos para un etiquetado detallado:
        </div>
        <ul style="font-size: 0.85rem; color: {c['text_secondary']}; margin: 0; padding-left: 1.25rem;">
            <li><strong>Se especifico:</strong> "tropical fish" detecta mejor que solo "fish"</li>
            <li><strong>Usa variaciones:</strong> añade "bottle", "plastic bottle", "water bottle" para detectar mas instancias</li>
            <li><strong>Incluye subtipos:</strong> "clownfish", "angelfish", "grouper" en lugar de solo "fish"</li>
            <li><strong>Describe atributos:</strong> "red car", "blue car", "damaged car" para mayor precision</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Labeling mode
    labeling_mode = st.radio(
        "Modo de etiquetado",
        ["Listado simple", "Detallado con variaciones", "Plantillas predefinidas"],
        horizontal=True,
        key="labeling_mode",
        help="El modo detallado permite añadir variaciones y sinonimos para cada clase"
    )

    if labeling_mode == "Plantillas predefinidas":
        _render_template_selector(c)

    elif labeling_mode == "Detallado con variaciones":
        _render_detailed_class_input(c)

    else:  # Listado simple
        _render_simple_class_input(c, client)

    spacer(8)

    # Show current classes summary
    if st.session_state.labeling_classes:
        total_prompts = len(st.session_state.labeling_classes)
        st.markdown(f"""
        <div style="background: {c['success_bg']}; border: 1px solid {c['success']};
                    border-radius: 0.5rem; padding: 0.75rem; margin-top: 0.5rem;">
            <span style="color: {c['success']}; font-weight: 600;">
                ✓ {total_prompts} objetos/prompts configurados para buscar
            </span>
        </div>
        """, unsafe_allow_html=True)

    # Output configuration
    section_header("Configuracion de Salida", icon="⚙️")

    col1, col2 = st.columns(2)

    with col1:
        output_dir = st.text_input(
            "Directorio de salida",
            value="/app/output/labeled",
            key="labeling_output_dir"
        )

        output_format = st.selectbox(
            "Formato de salida",
            ["COCO JSON", "YOLO", "Pascal VOC", "Todos"],
            key="labeling_output_format"
        )

    with col2:
        task_type = st.selectbox(
            "Tipo de tarea",
            ["Segmentacion (instancias)", "Deteccion (bboxes)", "Ambos"],
            key="labeling_task_type"
        )

        min_confidence = st.slider(
            "Confianza minima",
            0.1, 1.0, 0.5, 0.05,
            help="Umbral de confianza para aceptar detecciones",
            key="labeling_min_confidence"
        )

    # Advanced options
    with st.expander("Opciones avanzadas"):
        col_adv1, col_adv2 = st.columns(2)

        with col_adv1:
            min_area = st.number_input(
                "Area minima (px²)",
                min_value=10,
                max_value=100000,
                value=100,
                key="labeling_min_area"
            )

            simplify_polygons = st.checkbox(
                "Simplificar poligonos",
                value=True,
                help="Reduce el numero de puntos en las segmentaciones",
                key="labeling_simplify"
            )

        with col_adv2:
            max_instances = st.number_input(
                "Max instancias por imagen",
                min_value=1,
                max_value=1000,
                value=100,
                key="labeling_max_instances"
            )

            save_visualizations = st.checkbox(
                "Guardar visualizaciones",
                value=False,
                help="Guarda imagenes con las anotaciones superpuestas",
                key="labeling_save_viz"
            )

        # Padding row
        col_pad1, col_pad2 = st.columns(2)
        with col_pad1:
            padding = st.slider(
                "Padding (pixeles)",
                min_value=0,
                max_value=50,
                value=0,
                key="labeling_padding",
                help="Pixeles adicionales alrededor del bounding box detectado"
            )

    spacer(24)

    # Validation and start
    can_start = (
        len(st.session_state.labeling_image_dirs) > 0 and
        len(st.session_state.labeling_classes) > 0 and
        output_dir
    )

    if not can_start:
        missing = []
        if not st.session_state.labeling_image_dirs:
            missing.append("directorio de imagenes")
        if not st.session_state.labeling_classes:
            missing.append("clases a detectar")
        if not output_dir:
            missing.append("directorio de salida")

        alert_box(
            f"Faltan campos requeridos: {', '.join(missing)}",
            type="warning",
            icon="⚠️"
        )

    # Summary
    if can_start:
        st.markdown(f"""
        <div style="background: {c['bg_card']}; border: 1px solid {c['border']};
                    border-radius: 0.75rem; padding: 1rem; margin-bottom: 1rem;">
            <div style="font-size: 0.75rem; color: {c['text_muted']}; text-transform: uppercase;
                        letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                Resumen del Job
            </div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;">
                <div>
                    <div style="color: {c['text_muted']}; font-size: 0.8rem;">Directorios</div>
                    <div style="font-weight: 600; color: {c['primary']};">{len(st.session_state.labeling_image_dirs)}</div>
                </div>
                <div>
                    <div style="color: {c['text_muted']}; font-size: 0.8rem;">Clases</div>
                    <div style="font-weight: 600; color: {c['primary']};">{len(st.session_state.labeling_classes)}</div>
                </div>
                <div>
                    <div style="color: {c['text_muted']}; font-size: 0.8rem;">Formato</div>
                    <div style="font-weight: 600; color: {c['text_primary']};">{output_format}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if st.button(
        "🚀 Iniciar Etiquetado",
        type="primary",
        use_container_width=True,
        disabled=not can_start,
        key="start_labeling"
    ):
        # Prepare request
        formats_map = {
            "COCO JSON": ["coco"],
            "YOLO": ["yolo"],
            "Pascal VOC": ["voc"],
            "Todos": ["coco", "yolo", "voc"]
        }

        task_map = {
            "Segmentacion (instancias)": "segmentation",
            "Deteccion (bboxes)": "detection",
            "Ambos": "both"
        }

        # Build class mapping for variations (if using detailed mode)
        class_mapping = None
        if st.session_state.get("detailed_classes"):
            # Create mapping: variation -> main_class
            class_mapping = {}
            for main_class, variations in st.session_state.detailed_classes.items():
                for var in variations:
                    class_mapping[var] = main_class

        request = {
            "image_directories": st.session_state.labeling_image_dirs,
            "classes": st.session_state.labeling_classes,
            "class_mapping": class_mapping,  # Maps variations to main class names
            "output_dir": output_dir,
            "output_formats": formats_map.get(output_format, ["coco"]),
            "task_type": task_map.get(task_type, "segmentation"),
            "min_confidence": min_confidence,
            "min_area": min_area,
            "max_instances_per_image": max_instances,
            "simplify_polygons": simplify_polygons,
            "save_visualizations": save_visualizations,
            "padding": padding
        }

        with st.spinner("Iniciando job de etiquetado..."):
            result = client.start_labeling_job(request)

        if result.get("success"):
            job_id = result.get("job_id")
            st.success(f"✅ Job iniciado: `{job_id}`")
            st.session_state.current_labeling_job = job_id
            st.rerun()
        else:
            st.error(f"Error: {result.get('error', 'Error desconocido')}")


def _render_relabel_tab(c: Dict, client) -> None:
    """Tab for relabeling existing datasets."""
    section_header("Re-etiquetar Dataset Existente", icon="🔄")

    st.markdown(f"""
    <div style="background: {c['info_bg']}; border: 1px solid {c['info']};
                border-radius: 0.5rem; padding: 1rem; margin-bottom: 1rem;">
        <span style="font-size: 0.9rem; color: {c['text_secondary']};">
            Carga un dataset existente y re-etiquetalo con nuevas clases o mejora las anotaciones existentes.
            Util para corregir datasets con etiquetado de baja calidad.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Dataset source selection
    section_header("Dataset de Origen", icon="📂")

    source_method = st.radio(
        "Origen del dataset",
        ["Subir archivo COCO JSON", "Desde datasets generados", "Solo imagenes (sin anotaciones)"],
        horizontal=True,
        key="relabel_source_method"
    )

    coco_data = None
    images_dir = None
    additional_dirs = []

    if source_method == "Subir archivo COCO JSON":
        col1, col2 = st.columns(2)

        with col1:
            uploaded = st.file_uploader(
                "Archivo COCO JSON",
                type=["json"],
                key="relabel_coco_upload"
            )
            if uploaded:
                try:
                    coco_data = json.load(uploaded)
                    st.success(f"✓ Cargado: {len(coco_data.get('images', []))} imagenes, {len(coco_data.get('annotations', []))} anotaciones")
                except Exception as e:
                    st.error(f"Error: {e}")

        with col2:
            images_dir = st.text_input(
                "Directorio de imagenes principal",
                placeholder="/app/data/images",
                key="relabel_images_dir"
            )

    elif source_method == "Desde datasets generados":
        datasets_response = client.list_datasets()
        datasets = datasets_response.get("datasets", [])

        if datasets:
            dataset_options = {
                f"{d.get('job_id', 'unknown')[:12]}... ({d.get('num_images', 0)} imgs)": d
                for d in datasets
            }
            selected = st.selectbox(
                "Seleccionar dataset",
                options=list(dataset_options.keys()),
                key="relabel_source_dataset"
            )

            if selected:
                ds = dataset_options[selected]

                if st.button("Cargar dataset", key="load_relabel_dataset"):
                    result = client.load_dataset_coco(ds.get("job_id"))
                    if result.get("success"):
                        coco_data = result["data"]
                        st.session_state.relabel_coco_data = coco_data

                        # Get images directory from metadata
                        metadata = client.get_dataset_metadata(ds.get("job_id"))
                        if metadata and not metadata.get("error"):
                            images_dir = metadata.get("images_dir", "")
                            st.session_state.relabel_images_dir = images_dir

                        st.success(f"✓ Cargado dataset")
                        st.rerun()

                # Use cached data if available
                if st.session_state.get("relabel_coco_data"):
                    coco_data = st.session_state.relabel_coco_data
                    cached_images_dir = st.session_state.get("relabel_images_dir", "")
                    st.info(f"Dataset activo: {len(coco_data.get('images', []))} imagenes")

                    # Allow user to edit/set images directory
                    images_dir = st.text_input(
                        "Directorio de imagenes",
                        value=cached_images_dir,
                        placeholder="/app/data/images",
                        key="relabel_images_dir_edit",
                        help="Directorio donde se encuentran las imagenes del dataset"
                    )

                    if not images_dir:
                        st.warning("⚠️ Especifica el directorio donde estan las imagenes")
        else:
            st.info("No hay datasets disponibles")

    else:  # Solo imagenes
        images_dir = st.text_input(
            "Directorio de imagenes",
            placeholder="/app/data/images",
            key="relabel_only_images_dir"
        )

    # Additional directories
    spacer(8)
    st.markdown("**Directorios adicionales de busqueda de imagenes:**")
    st.caption("Si las imagenes estan distribuidas en multiples directorios, añadelos aqui.")

    num_additional = st.number_input(
        "Numero de directorios adicionales",
        min_value=0,
        max_value=10,
        value=0,
        key="num_additional_dirs"
    )

    for i in range(num_additional):
        additional_dir = st.text_input(
            f"Directorio adicional {i+1}",
            placeholder=f"/app/data/images_{i+1}",
            key=f"additional_dir_{i}"
        )
        if additional_dir:
            additional_dirs.append(additional_dir)

    spacer(16)

    # Relabeling options
    section_header("Opciones de Re-etiquetado", icon="⚙️")

    relabel_mode = st.radio(
        "Modo de re-etiquetado",
        [
            "Añadir nuevas clases (mantener existentes)",
            "Reemplazar todas las anotaciones",
            "Solo mejorar segmentaciones (bbox → segmentation)"
        ],
        key="relabel_mode"
    )

    # New classes (for add/replace modes)
    if relabel_mode != "Solo mejorar segmentaciones (bbox → segmentation)":
        st.markdown("**Nuevas clases a detectar:**")

        new_classes_text = st.text_area(
            "Clases (una por linea)",
            placeholder="fish\nbottle\nplastic",
            height=100,
            key="relabel_new_classes"
        )

        new_classes = [c.strip() for c in new_classes_text.split("\n") if c.strip()]

        if new_classes:
            st.markdown(f"Clases a detectar: **{', '.join(new_classes)}**")
    else:
        new_classes = []

    # Output configuration
    col1, col2 = st.columns(2)

    with col1:
        output_dir = st.text_input(
            "Directorio de salida",
            value="/app/output/relabeled",
            key="relabel_output_dir"
        )

    with col2:
        output_format = st.selectbox(
            "Formato de salida",
            ["COCO JSON", "YOLO", "Pascal VOC", "Todos"],
            key="relabel_output_format"
        )

    spacer(24)

    # Combine all image directories for validation
    all_dirs = []
    if images_dir:
        all_dirs.append(images_dir)
    all_dirs.extend(additional_dirs)

    # Validation - must have at least one directory
    needs_classes = relabel_mode != "Solo mejorar segmentaciones (bbox → segmentation)"
    can_start = (
        len(all_dirs) > 0 and
        output_dir and
        (new_classes or not needs_classes)
    )

    # Show helpful validation messages
    if not can_start:
        missing = []
        if len(all_dirs) == 0:
            missing.append("directorio de imagenes")
        if not output_dir:
            missing.append("directorio de salida")
        if needs_classes and not new_classes:
            missing.append("clases a detectar")

        if missing:
            st.warning(f"⚠️ Falta: {', '.join(missing)}")

    if st.button(
        "🔄 Iniciar Re-etiquetado",
        type="primary",
        use_container_width=True,
        disabled=not can_start,
        key="start_relabeling"
    ):
        formats_map = {
            "COCO JSON": ["coco"],
            "YOLO": ["yolo"],
            "Pascal VOC": ["voc"],
            "Todos": ["coco", "yolo", "voc"]
        }

        mode_map = {
            "Añadir nuevas clases (mantener existentes)": "add",
            "Reemplazar todas las anotaciones": "replace",
            "Solo mejorar segmentaciones (bbox → segmentation)": "improve_segmentation"
        }

        # all_dirs already computed above for validation

        request = {
            "coco_data": coco_data,
            "image_directories": all_dirs,
            "new_classes": new_classes,
            "relabel_mode": mode_map.get(relabel_mode, "add"),
            "output_dir": output_dir,
            "output_formats": formats_map.get(output_format, ["coco"]),
        }

        with st.spinner("Iniciando job de re-etiquetado..."):
            result = client.start_relabeling_job(request)

        if result.get("success"):
            job_id = result.get("job_id")
            st.success(f"✅ Job iniciado: `{job_id}`")
            st.session_state.current_labeling_job = job_id
            st.rerun()
        else:
            st.error(f"Error: {result.get('error', 'Error desconocido')}")


def _render_labeling_jobs_tab(c: Dict, client) -> None:
    """Tab for monitoring labeling jobs."""
    section_header("Jobs de Etiquetado", icon="📊")

    if st.button("🔄 Actualizar", key="refresh_labeling_jobs"):
        st.rerun()

    # Fetch labeling jobs
    jobs_response = client.list_labeling_jobs()
    jobs = jobs_response.get("jobs", [])

    if not jobs:
        empty_state(
            title="No hay jobs de etiquetado",
            message="Inicia un nuevo job desde las pestañas anteriores.",
            icon="📭"
        )
        return

    # Categorize jobs
    active_jobs = [j for j in jobs if j.get("status") in ["processing", "queued"]]
    completed_jobs = [j for j in jobs if j.get("status") == "completed"]
    failed_jobs = [j for j in jobs if j.get("status") == "failed"]

    # Active jobs
    if active_jobs:
        st.markdown("### ⏳ Jobs Activos")

        for job in active_jobs:
            _render_labeling_job_card(job, c, client, active=True)

        # Auto-refresh
        time.sleep(2)
        st.rerun()

    # Completed jobs
    if completed_jobs:
        st.markdown("### ✅ Jobs Completados")

        for job in completed_jobs[:5]:
            _render_labeling_job_card(job, c, client, active=False)

        if len(completed_jobs) > 5:
            st.caption(f"Mostrando 5 de {len(completed_jobs)} jobs completados")

    # Failed jobs
    if failed_jobs:
        with st.expander(f"❌ Jobs Fallidos ({len(failed_jobs)})"):
            for job in failed_jobs[:5]:
                _render_labeling_job_card(job, c, client, active=False)


def _render_labeling_job_card(job: Dict, c: Dict, client, active: bool = False) -> None:
    """Render a labeling job card."""
    job_id = job.get("job_id", "unknown")
    status = job.get("status", "unknown")
    job_type = job.get("job_type", "labeling")

    total_images = job.get("total_images", 0)
    processed_images = job.get("processed_images", 0)
    total_objects = job.get("total_objects_found", 0)
    objects_by_class = job.get("objects_by_class", {})

    progress = processed_images / total_images if total_images > 0 else 0

    # Calculate elapsed time and ETA
    started_at = job.get("started_at")
    elapsed_str = ""
    eta_str = ""
    if started_at and status == "processing":
        try:
            # Parse start time
            start_str = started_at.replace("Z", "+00:00")
            # Handle both ISO format with and without timezone
            if "+" not in start_str and "-" not in start_str[-6:]:
                start_dt = datetime.fromisoformat(start_str)
                now = datetime.now()
            else:
                start_dt = datetime.fromisoformat(start_str)
                now = datetime.now(start_dt.tzinfo) if start_dt.tzinfo else datetime.now()

            elapsed = (now - start_dt).total_seconds()
            if elapsed > 0:
                elapsed_str = _format_duration(elapsed)

                if processed_images > 0:
                    rate = processed_images / elapsed
                    remaining = total_images - processed_images
                    if rate > 0:
                        eta_seconds = remaining / rate
                        eta_str = _format_duration(eta_seconds)
        except Exception:
            pass

    # Status styling
    if status == "completed":
        status_color = c['success']
        status_icon = "✅"
    elif status == "processing":
        status_color = c['warning']
        status_icon = "⏳"
    elif status == "queued":
        status_color = c['info']
        status_icon = "📋"
    else:
        status_color = c['error']
        status_icon = "❌"

    st.markdown(f"""
    <div style="background: {c['bg_card']}; border: 1px solid {c['border']};
                border-radius: 0.5rem; padding: 1rem; margin-bottom: 0.75rem;
                border-left: 4px solid {status_color};">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.25rem;">{status_icon}</span>
                <span style="font-family: monospace; font-weight: 600; color: {c['text_primary']}; font-size: 0.85rem;">
                    {job_id[:16]}...
                </span>
            </div>
            <span style="font-size: 0.75rem; color: {status_color}; font-weight: 600;">
                {status.upper()}
            </span>
        </div>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; font-size: 0.8rem;">
            <div>
                <span style="color: {c['text_muted']};">Imagenes:</span>
                <span style="font-weight: 600;"> {processed_images}/{total_images}</span>
            </div>
            <div>
                <span style="color: {c['text_muted']};">Objetos:</span>
                <span style="color: {c['success']}; font-weight: 600;"> {total_objects}</span>
            </div>
            <div>
                <span style="color: {c['text_muted']};">Progreso:</span>
                <span style="color: {c['primary']}; font-weight: 600;"> {progress*100:.0f}%</span>
            </div>
            <div>
                <span style="color: {c['text_muted']};">Tipo:</span>
                <span style="font-weight: 600;"> {job_type}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if active:
        st.progress(progress)

        # Current processing info and timing
        col_info, col_time = st.columns([2, 1])

        with col_info:
            current_image = job.get("current_image", "")
            if current_image:
                st.caption(f"📍 Procesando: **{Path(current_image).name}**")

        with col_time:
            if elapsed_str:
                time_text = f"⏱️ {elapsed_str}"
                if eta_str:
                    time_text += f" | ETA: ~{eta_str}"
                st.caption(time_text)

        # Show objects by class for active jobs
        if objects_by_class and any(v > 0 for v in objects_by_class.values()):
            with st.expander("📊 Objetos por clase", expanded=False):
                # Sort by count descending
                sorted_classes = sorted(objects_by_class.items(), key=lambda x: -x[1])
                for cls_name, count in sorted_classes:
                    if count > 0:
                        st.markdown(f"• **{cls_name}**: {count} objetos")

    # Download results for completed jobs
    if status == "completed":
        output_dir = job.get("output_dir", "")
        processing_time_ms = job.get("processing_time_ms", 0)

        # Show processing time
        if processing_time_ms > 0:
            processing_time_str = _format_duration(processing_time_ms / 1000)
            st.caption(f"⏱️ Tiempo total: {processing_time_str}")

        # Show objects by class summary for completed jobs
        if objects_by_class and any(v > 0 for v in objects_by_class.values()):
            with st.expander("📊 Objetos por clase", expanded=False):
                sorted_classes = sorted(objects_by_class.items(), key=lambda x: -x[1])
                for cls_name, count in sorted_classes:
                    if count > 0:
                        st.markdown(f"• **{cls_name}**: {count} objetos")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📥 Descargar COCO", key=f"dl_coco_{job_id}"):
                # Load and offer download
                coco_path = f"{output_dir}/annotations.json"
                st.info(f"Dataset disponible en: {coco_path}")

        with col2:
            if st.button("👁️ Ver detalles", key=f"view_{job_id}"):
                st.session_state[f"show_details_{job_id}"] = True
                st.rerun()

        with col3:
            if st.button("📊 Cargar en workflow", key=f"load_{job_id}"):
                result = client.load_labeling_result(job_id)
                if result.get("success"):
                    st.session_state.generated_dataset = result["data"]
                    st.success("✓ Dataset cargado")

    # Retry option for failed jobs
    if status == "failed":
        can_resume = job.get("can_resume", False)

        col1, col2 = st.columns(2)

        with col1:
            if can_resume:
                if st.button("🔄 Reintentar desde checkpoint", key=f"retry_{job_id}",
                            help="Continuar desde la ultima imagen procesada"):
                    with st.spinner("Reanudando job..."):
                        result = client.resume_labeling_job(job_id)
                    if result.get("success"):
                        st.success(f"✓ Job reanudado: {result.get('message', '')}")
                        st.rerun()
                    else:
                        st.error(f"Error: {result.get('error', 'Error desconocido')}")
            else:
                st.caption("⚠️ Sin checkpoint disponible - debe reiniciarse desde el principio")

        with col2:
            if st.button("👁️ Ver errores", key=f"view_errors_{job_id}"):
                st.session_state[f"show_details_{job_id}"] = True
                st.rerun()

        # Show error count
        errors = job.get("errors", [])
        if errors:
            st.markdown(f"""
            <div style="background: {c['error_bg']}; border: 1px solid {c['error']};
                        border-radius: 0.25rem; padding: 0.5rem; margin-top: 0.5rem;">
                <span style="color: {c['error']}; font-size: 0.85rem;">
                    {len(errors)} error(es) registrado(s) - Progreso guardado en imagen {processed_images}
                </span>
            </div>
            """, unsafe_allow_html=True)

    # Show details
    if st.session_state.get(f"show_details_{job_id}"):
        with st.expander("📋 Detalles del Job", expanded=True):
            st.json(job)

            if st.button("Cerrar", key=f"close_{job_id}"):
                st.session_state.pop(f"show_details_{job_id}", None)
                st.rerun()
