"""
Configure Page (Step 2)
=======================
Generation configuration with effects, directories, and options.
"""

import os
import json
from datetime import datetime
import streamlit as st
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

from app.components.ui import (
    page_header, section_header, spacer, alert_box,
    workflow_stepper, workflow_navigation
)
from app.config.theme import get_colors_dict


# =============================================================================
# Configuration Import/Export Functions
# =============================================================================

def _get_config_version() -> str:
    """Return current config schema version."""
    return "1.0"


def _build_effects_preset(
    effects: List[str],
    effects_config: Dict,
    max_objects: int,
    overlap_threshold: float,
    depth_aware: bool,
    validate_quality: bool,
    validate_physics: bool,
    save_debug: bool,
) -> Dict:
    """Build an effects preset dictionary suitable for export.

    Creates a reusable preset with ONLY effects and advanced options.
    Does not include dataset-specific info (directories, targets, etc.).
    """
    return {
        "preset_version": _get_config_version(),
        "preset_type": "effects_config",
        "created_at": datetime.now().isoformat(),
        "description": "Preset de efectos y opciones avanzadas reutilizable",

        # Effects configuration
        "effects": {
            "enabled": effects,
            "intensities": {
                "color_intensity": effects_config.get("color_intensity", 0.7),
                "blur_strength": effects_config.get("blur_strength", 1.0),
                "shadow_opacity": effects_config.get("shadow_opacity", 0.4),
                "underwater_intensity": effects_config.get("underwater_intensity", 0.25),
                "caustics_intensity": effects_config.get("caustics_intensity", 0.15),
                "edge_feather": effects_config.get("edge_feather", 5),
                "lighting_intensity": effects_config.get("lighting_intensity", 0.5),
                "motion_blur_probability": effects_config.get("motion_blur_probability", 0.2),
            },
            "parameters": {
                "lighting_type": effects_config.get("lighting_type", "ambient"),
                "water_color": list(effects_config.get("water_color", [20, 80, 120])),
                "water_clarity": effects_config.get("water_clarity", "clear"),
            },
        },

        # Advanced generation options
        "advanced_options": {
            "max_objects_per_image": max_objects,
            "overlap_threshold": overlap_threshold,
            "depth_aware": depth_aware,
        },

        # Validation settings
        "validation": {
            "validate_quality": validate_quality,
            "validate_physics": validate_physics,
        },

        # Debug settings
        "debug": {
            "save_pipeline_debug": save_debug,
        },
    }


def _apply_loaded_preset(preset: Dict) -> bool:
    """Apply a loaded effects preset to session state. Returns True on success.

    Supports both the new preset format (preset_version) and legacy config format
    (config_version) for backward compatibility.

    Applies:
    - Enabled effects (checkboxes)
    - Effect intensity values and parameters (sliders)
    - Advanced generation options
    - Validation and debug settings

    Does NOT apply (dataset-specific, must be set manually):
    - Directory paths
    - Targets per class
    """
    try:
        # Detect format: new preset format vs legacy config format
        is_new_format = "preset_version" in preset or "preset_type" in preset

        # Effects enabled (checkboxes)
        effects_data = preset.get("effects", {})
        enabled_effects = effects_data.get("enabled", [])

        st.session_state.fx_color = "color_correction" in enabled_effects
        st.session_state.fx_blur = "blur_matching" in enabled_effects
        st.session_state.fx_shadows = "shadows" in enabled_effects
        st.session_state.fx_caustics = "caustics" in enabled_effects
        st.session_state.fx_underwater = "underwater" in enabled_effects
        st.session_state.fx_edge = "edge_smoothing" in enabled_effects
        st.session_state.fx_motion = "motion_blur" in enabled_effects
        st.session_state.fx_lighting = "lighting" in enabled_effects

        # Effects intensities - handle both formats
        if is_new_format:
            # New format: effects.intensities and effects.parameters
            intensities = effects_data.get("intensities", {})
            parameters = effects_data.get("parameters", {})
        else:
            # Legacy format: effects.config contains everything
            intensities = effects_data.get("config", {})
            parameters = effects_data.get("config", {})

        # Core intensity values
        if "color_intensity" in intensities:
            st.session_state.int_color = float(intensities["color_intensity"])
        if "blur_strength" in intensities:
            st.session_state.int_blur = float(intensities["blur_strength"])
        if "shadow_opacity" in intensities:
            st.session_state.int_shadow = float(intensities["shadow_opacity"])
        if "underwater_intensity" in intensities:
            st.session_state.int_underwater = float(intensities["underwater_intensity"])
        if "caustics_intensity" in intensities:
            st.session_state.int_caustics = float(intensities["caustics_intensity"])
        if "edge_feather" in intensities:
            st.session_state.int_edge = int(intensities["edge_feather"])

        # Generation options (advanced) - handle both formats
        if is_new_format:
            gen = preset.get("advanced_options", {})
        else:
            gen = preset.get("generation", {})

        if "max_objects_per_image" in gen:
            st.session_state.config_max_objects = int(gen["max_objects_per_image"])
        if "overlap_threshold" in gen:
            st.session_state.config_overlap = float(gen["overlap_threshold"])
        if "depth_aware" in gen:
            st.session_state.config_depth_aware = bool(gen["depth_aware"])

        # Validation options
        val = preset.get("validation", {})
        if "validate_quality" in val:
            st.session_state.config_val_quality = bool(val["validate_quality"])
        if "validate_physics" in val:
            st.session_state.config_val_physics = bool(val["validate_physics"])

        # Debug options
        debug = preset.get("debug", {})
        if "save_pipeline_debug" in debug:
            st.session_state.config_save_debug = bool(debug["save_pipeline_debug"])

        return True
    except Exception as e:
        st.error(f"Error al aplicar preset: {e}")
        return False


def _render_config_management_section() -> None:
    """Render the effects preset import/export section."""
    c = get_colors_dict()

    with st.expander("💾 Presets de Efectos", expanded=False):
        st.markdown(f"""
        <div style="background: {c['info_bg']}; border: 1px solid {c['info']};
                    border-radius: 0.5rem; padding: 0.75rem; margin-bottom: 1rem;">
            <span style="font-size: 0.85rem; color: {c['text_secondary']};">
                Guarda tus efectos y opciones avanzadas como preset reutilizable.
                Ideal para replicar configuraciones que funcionan bien (ej: efectos submarinos).
            </span>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**📥 Cargar Preset**")

            uploaded_preset = st.file_uploader(
                "Seleccionar archivo de preset",
                type=["json"],
                key="preset_upload",
                label_visibility="collapsed"
            )

            if uploaded_preset:
                try:
                    preset_data = json.load(uploaded_preset)

                    # Validate it's a valid preset/config file (support both formats)
                    has_version = "preset_version" in preset_data or "config_version" in preset_data
                    if not has_version:
                        st.error("❌ Archivo no válido: falta versión de preset")
                    else:
                        # Determine format and get version
                        is_new_format = "preset_version" in preset_data
                        version = preset_data.get("preset_version") or preset_data.get("config_version", "N/A")
                        effects_count = len(preset_data.get("effects", {}).get("enabled", []))

                        # Show preset preview
                        st.markdown(f"""
                        <div style="background: {c['bg_secondary']}; border: 1px solid {c['border']};
                                    border-radius: 0.5rem; padding: 0.75rem; margin: 0.5rem 0;">
                            <div style="font-size: 0.75rem; color: {c['text_muted']};">Preset detectado:</div>
                            <div style="font-size: 0.85rem; color: {c['text_primary']}; margin-top: 0.25rem;">
                                • Versión: {version}<br>
                                • Creado: {preset_data.get('created_at', 'N/A')[:10] if preset_data.get('created_at') else 'N/A'}<br>
                                • Efectos activos: {effects_count}<br>
                                • Formato: {"Nuevo (effects preset)" if is_new_format else "Legacy (config)"}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        if st.button("✅ Aplicar Preset", key="apply_preset", use_container_width=True):
                            if _apply_loaded_preset(preset_data):
                                st.success("✅ Preset aplicado correctamente")
                                st.rerun()

                except json.JSONDecodeError:
                    st.error("❌ Error al leer el archivo JSON")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

        with col2:
            st.markdown(f"**📤 Exportar Preset Actual**")

            preset_name = st.text_input(
                "Nombre del preset",
                value=f"effects_preset_{datetime.now().strftime('%Y%m%d_%H%M')}",
                key="preset_export_name",
                label_visibility="collapsed",
                placeholder="Nombre para el archivo de preset"
            )

            # Get current values from session state or defaults
            current_preset = _build_effects_preset(
                effects=[
                    eff for eff, key in [
                        ("color_correction", "fx_color"),
                        ("blur_matching", "fx_blur"),
                        ("shadows", "fx_shadows"),
                        ("caustics", "fx_caustics"),
                        ("underwater", "fx_underwater"),
                        ("edge_smoothing", "fx_edge"),
                        ("motion_blur", "fx_motion"),
                        ("lighting", "fx_lighting"),
                    ] if st.session_state.get(key, key in ["fx_color", "fx_blur", "fx_shadows", "fx_caustics", "fx_underwater", "fx_edge"])
                ],
                effects_config={
                    "color_intensity": st.session_state.get("int_color", 0.7),
                    "blur_strength": st.session_state.get("int_blur", 1.0),
                    "shadow_opacity": st.session_state.get("int_shadow", 0.4),
                    "underwater_intensity": st.session_state.get("int_underwater", 0.25),
                    "caustics_intensity": st.session_state.get("int_caustics", 0.15),
                    "edge_feather": st.session_state.get("int_edge", 5),
                    "lighting_type": "ambient",
                    "lighting_intensity": 0.5,
                    "water_color": [20, 80, 120],
                    "water_clarity": "clear",
                    "motion_blur_probability": 0.2,
                },
                max_objects=st.session_state.get("config_max_objects", 5),
                overlap_threshold=st.session_state.get("config_overlap", 0.1),
                depth_aware=st.session_state.get("config_depth_aware", True),
                validate_quality=st.session_state.get("config_val_quality", False),
                validate_physics=st.session_state.get("config_val_physics", False),
                save_debug=st.session_state.get("config_save_debug", False),
            )

            preset_json = json.dumps(current_preset, indent=2, ensure_ascii=False)

            st.download_button(
                "📥 Descargar Preset",
                data=preset_json,
                file_name=f"{preset_name}.json",
                mime="application/json",
                key="download_preset",
                use_container_width=True
            )

            # Show current preset summary
            st.markdown(f"""
            <div style="background: {c['bg_tertiary']}; border-radius: 0.5rem;
                        padding: 0.5rem; margin-top: 0.5rem; font-size: 0.75rem;">
                <span style="color: {c['text_muted']};">
                    Efectos: {len(current_preset['effects']['enabled'])} |
                    Max obj: {current_preset['advanced_options']['max_objects_per_image']} |
                    Depth-aware: {"Sí" if current_preset['advanced_options']['depth_aware'] else "No"}
                </span>
            </div>
            """, unsafe_allow_html=True)


def render_configure_page():
    """Render the generation configuration page (Step 2 of workflow)"""
    c = get_colors_dict()

    # Workflow stepper
    completed = st.session_state.get("workflow_completed", [])
    workflow_stepper(current_step=2, completed_steps=completed)

    page_header(
        title="Configuración de Generación",
        subtitle="Paso 2: Configura los efectos, directorios y opciones de generación",
        icon="⚙️"
    )

    # Configuration management section (import/export)
    _render_config_management_section()

    spacer(8)

    # Get data from previous step
    targets = st.session_state.get("balancing_targets", {})
    analysis = st.session_state.get("analysis_result", {})

    total_images = sum(targets.values())

    # Summary from analysis
    st.markdown(f"""
    <div style="background: {c['bg_secondary']}; border: 1px solid {c['border']};
                border-radius: 0.5rem; padding: 1rem; margin-bottom: 1.5rem;
                display: flex; justify-content: space-between; align-items: center;">
        <div>
            <span style="color: {c['text_muted']}; font-size: 0.85rem;">Imágenes a generar:</span>
            <span style="font-weight: 700; font-size: 1.25rem; margin-left: 0.5rem; color: {c['primary']};">
                {total_images:,}
            </span>
        </div>
        <div>
            <span style="color: {c['text_muted']}; font-size: 0.85rem;">Clases:</span>
            <span style="font-weight: 600; margin-left: 0.5rem; color: {c['text_primary']};">
                {len([cls for cls in targets if targets[cls] > 0])}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Two columns layout
    col1, col2 = st.columns(2)

    with col1:
        # Directories section
        section_header("Directorios", icon="📁")

        backgrounds_dir = st.text_input(
            "Directorio de fondos",
            value=os.environ.get("BACKGROUNDS_PATH", "/app/datasets/Backgrounds_filtered"),
            key="config_bg_dir",
            help="Directorio con imágenes de fondo para composición"
        )

        objects_dir = st.text_input(
            "Directorio de objetos",
            value=os.environ.get("OBJECTS_PATH", "/app/datasets/Objects"),
            key="config_obj_dir",
            help="Directorio con objetos recortados organizados por clase"
        )

        output_dir = st.text_input(
            "Directorio de salida",
            value="/app/output/synthetic",
            key="config_output_dir",
            help="Directorio donde se guardarán las imágenes generadas"
        )

        # Validate directories
        bg_exists = Path(backgrounds_dir).exists() if backgrounds_dir else False
        obj_exists = Path(objects_dir).exists() if objects_dir else False

        if not bg_exists:
            st.warning("⚠️ Directorio de fondos no encontrado")
        else:
            bg_count = len(list(Path(backgrounds_dir).glob("*.jpg")) + list(Path(backgrounds_dir).glob("*.png")))
            st.success(f"✓ {bg_count} fondos disponibles")

        if not obj_exists:
            st.warning("⚠️ Directorio de objetos no encontrado")
        else:
            obj_classes = [d.name for d in Path(objects_dir).iterdir() if d.is_dir()]
            st.success(f"✓ {len(obj_classes)} clases de objetos disponibles")

    with col2:
        # Effects section
        section_header("Efectos de Realismo", icon="✨")

        st.markdown(f"""
        <div style="font-size: 0.85rem; color: {c['text_secondary']}; margin-bottom: 1rem;">
            Selecciona los efectos a aplicar para mejorar el realismo de las imágenes generadas.
        </div>
        """, unsafe_allow_html=True)

        # Effects checkboxes in two columns
        fx_col1, fx_col2 = st.columns(2)

        with fx_col1:
            fx_color = st.checkbox("🎨 Color Correction", value=True, key="fx_color")
            fx_blur = st.checkbox("🔍 Blur Matching", value=True, key="fx_blur")
            fx_shadows = st.checkbox("🌑 Shadows", value=True, key="fx_shadows")
            fx_caustics = st.checkbox("💧 Caustics", value=True, key="fx_caustics")

        with fx_col2:
            fx_underwater = st.checkbox("🌊 Underwater", value=True, key="fx_underwater")
            fx_edge = st.checkbox("✂️ Edge Smoothing", value=True, key="fx_edge")
            fx_motion = st.checkbox("💨 Motion Blur", value=False, key="fx_motion")
            fx_lighting = st.checkbox("💡 Lighting", value=False, key="fx_lighting")

    spacer(24)

    # Advanced options
    section_header("Opciones Avanzadas", icon="🔧")

    col1, col2, col3 = st.columns(3)

    with col1:
        depth_aware = st.checkbox(
            "Depth-aware placement",
            value=True,
            key="config_depth_aware",
            help="Posicionar objetos según la profundidad de la escena"
        )

        save_debug = st.checkbox(
            "Guardar debug pipeline",
            value=False,
            key="config_save_debug",
            help="Guardar imágenes intermedias del pipeline para documentación"
        )

    with col2:
        max_objects = st.slider(
            "Máx objetos por imagen",
            min_value=1,
            max_value=10,
            value=5,
            key="config_max_objects",
            help="Número máximo de objetos a colocar en cada imagen"
        )

        overlap_threshold = st.slider(
            "Overlap threshold",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.05,
            key="config_overlap",
            help="Máximo solapamiento permitido entre objetos"
        )

    with col3:
        validate_quality = st.checkbox(
            "Validar calidad (LPIPS)",
            value=False,
            key="config_val_quality",
            help="Validar calidad perceptual de las imágenes generadas"
        )

        validate_physics = st.checkbox(
            "Validar física",
            value=False,
            key="config_val_physics",
            help="Validar posicionamiento físicamente plausible"
        )

    spacer(16)

    # Object sizes configuration
    section_header("Tamaños de Objetos", icon="📏")

    st.markdown(f"""
    <div style="font-size: 0.85rem; color: {c['text_secondary']}; margin-bottom: 1rem;">
        Configura los tamaños reales (en metros) de los objetos para un escalado depth-aware más realista.
    </div>
    """, unsafe_allow_html=True)

    from app.components.api_client import get_api_client

    try:
        client = get_api_client()
        config_data = client.get_object_sizes()
        sizes = config_data.get("sizes", {})
        reference_distance = config_data.get("reference_capture_distance", 2.0)

        # Get object classes from targets
        object_classes = [cls for cls, count in targets.items() if count > 0]

        if object_classes:
            # Show sizes for classes in this generation
            size_col1, size_col2, size_col3 = st.columns([2, 1, 1])

            with size_col1:
                st.markdown("**Clase de Objeto**")
            with size_col2:
                st.markdown("**Tamaño Actual (m)**")
            with size_col3:
                st.markdown("**Editar**")

            st.markdown(f"<hr style='margin: 0.5rem 0; border: none; border-top: 1px solid {c['border']};'>", unsafe_allow_html=True)

            # Track if any sizes were modified
            sizes_modified = False

            # Show up to 5 most common classes, rest in expander
            display_classes = object_classes[:5]

            for obj_class in display_classes:
                col1, col2, col3 = st.columns([2, 1, 1])

                # Get current size for this class
                current_size = sizes.get(obj_class.lower(), sizes.get("default", 0.25))

                with col1:
                    st.markdown(f"**{obj_class}**")

                with col2:
                    st.markdown(f"<div style='padding-top: 0.3rem;'>{current_size:.2f} m</div>", unsafe_allow_html=True)

                with col3:
                    new_size = st.number_input(
                        f"size_{obj_class}",
                        min_value=0.01,
                        max_value=100.0,
                        value=float(current_size),
                        step=0.01,
                        key=f"obj_size_{obj_class}",
                        label_visibility="collapsed"
                    )

                    # Check if modified
                    if abs(new_size - current_size) > 0.001:
                        sizes_modified = True

            # Show remaining classes in expander if there are more
            if len(object_classes) > 5:
                with st.expander(f"Ver {len(object_classes) - 5} clases adicionales"):
                    for obj_class in object_classes[5:]:
                        col1, col2, col3 = st.columns([2, 1, 1])

                        current_size = sizes.get(obj_class.lower(), sizes.get("default", 0.25))

                        with col1:
                            st.markdown(f"**{obj_class}**")

                        with col2:
                            st.markdown(f"{current_size:.2f} m")

                        with col3:
                            new_size = st.number_input(
                                f"size_{obj_class}",
                                min_value=0.01,
                                max_value=100.0,
                                value=float(current_size),
                                step=0.01,
                                key=f"obj_size_{obj_class}",
                                label_visibility="collapsed"
                            )

                            if abs(new_size - current_size) > 0.001:
                                sizes_modified = True

            # Save button if sizes were modified
            if sizes_modified:
                if st.button("💾 Guardar Cambios en Tamaños", type="primary", use_container_width=True):
                    try:
                        # Collect all modified sizes
                        updated_sizes = {}
                        for obj_class in object_classes:
                            new_size = st.session_state.get(f"obj_size_{obj_class}")
                            if new_size is not None:
                                updated_sizes[obj_class] = new_size

                        # Update via API
                        client.update_multiple_object_sizes(updated_sizes)
                        st.success(f"✅ Se actualizaron {len(updated_sizes)} tamaños correctamente")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error al guardar: {str(e)}")

            # Link to full configuration page
            st.markdown(f"""
            <div style="margin-top: 1rem; padding: 0.75rem; background: {c['bg_secondary']};
                        border-radius: 0.5rem; font-size: 0.85rem;">
                <span style="color: {c['text_muted']};">
                    💡 Para configuración avanzada de todos los objetos, visita
                    <strong>Herramientas → 📏 Tamaños</strong>
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No hay clases de objetos seleccionadas para esta generación")

    except Exception as e:
        st.warning(f"⚠️ No se pudo cargar la configuración de tamaños: {str(e)}")

    spacer(16)

    # Intensity sliders (collapsed by default)
    with st.expander("🎚️ Ajustar Intensidades de Efectos"):
        int_col1, int_col2 = st.columns(2)

        with int_col1:
            color_intensity = st.slider("Color Correction", 0.0, 1.0, 0.7, key="int_color")
            blur_strength = st.slider("Blur Strength", 0.0, 3.0, 1.0, key="int_blur")
            shadow_opacity = st.slider("Shadow Opacity", 0.0, 1.0, 0.4, key="int_shadow")

        with int_col2:
            underwater_intensity = st.slider("Underwater", 0.0, 1.0, 0.25, key="int_underwater")
            caustics_intensity = st.slider("Caustics", 0.0, 0.5, 0.15, key="int_caustics")
            edge_feather = st.slider("Edge Feather", 1, 20, 5, key="int_edge")

    spacer(16)

    # Targets review (collapsed)
    with st.expander("📊 Ver Targets por Clase"):
        if targets:
            target_df = pd.DataFrame([
                {"Clase": cls, "Imágenes a generar": count}
                for cls, count in targets.items() if count > 0
            ])
            st.dataframe(target_df, use_container_width=True, hide_index=True)

    spacer(24)

    # Build configuration object
    effects = []
    if fx_color: effects.append("color_correction")
    if fx_blur: effects.append("blur_matching")
    if fx_shadows: effects.append("shadows")
    if fx_caustics: effects.append("caustics")
    if fx_underwater: effects.append("underwater")
    if fx_edge: effects.append("edge_smoothing")
    if fx_motion: effects.append("motion_blur")
    if fx_lighting: effects.append("lighting")

    # Get intensity values from session state (more reliable than local vars)
    effects_config = {
        "color_intensity": st.session_state.get("int_color", 0.7),
        "blur_strength": st.session_state.get("int_blur", 1.0),
        "underwater_intensity": st.session_state.get("int_underwater", 0.25),
        "caustics_intensity": st.session_state.get("int_caustics", 0.15),
        "shadow_opacity": st.session_state.get("int_shadow", 0.4),
        "edge_feather": st.session_state.get("int_edge", 5),
        "lighting_type": "ambient",
        "lighting_intensity": 0.5,
        "water_color": [20, 80, 120],  # Use list for JSON serialization
        "water_clarity": "clear",
        "motion_blur_probability": 0.2,
    }

    generation_config = {
        "backgrounds_dir": backgrounds_dir,
        "objects_dir": objects_dir,
        "output_dir": output_dir,
        "num_images": total_images,
        "targets_per_class": targets,
        "max_objects_per_image": max_objects,
        "overlap_threshold": overlap_threshold,
        "depth_aware": depth_aware,
        "effects": effects,
        "effects_config": effects_config,
        "validate_quality": validate_quality,
        "validate_physics": validate_physics,
        "save_pipeline_debug": save_debug,
    }

    # Configuration summary
    st.markdown(f"""
    <div style="background: {c['bg_card']}; border: 1px solid {c['border']};
                border-radius: 0.75rem; padding: 1rem;">
        <div style="font-size: 0.75rem; color: {c['text_muted']}; text-transform: uppercase;
                    letter-spacing: 0.05em; margin-bottom: 0.75rem;">
            Resumen de Configuración
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;">
            <div>
                <div style="font-weight: 600; color: {c['text_primary']};">{total_images:,}</div>
                <div style="font-size: 0.8rem; color: {c['text_muted']};">Imágenes</div>
            </div>
            <div>
                <div style="font-weight: 600; color: {c['text_primary']};">{len(effects)}</div>
                <div style="font-size: 0.8rem; color: {c['text_muted']};">Efectos</div>
            </div>
            <div>
                <div style="font-weight: 600; color: {c['text_primary']};">{max_objects}</div>
                <div style="font-size: 0.8rem; color: {c['text_muted']};">Max obj/img</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Store configuration
    st.session_state.generation_config = generation_config

    # Workflow navigation
    can_proceed = bg_exists and obj_exists and total_images > 0

    if not can_proceed:
        alert_box(
            "Verifica que los directorios de fondos y objetos existan antes de continuar.",
            type="warning",
            icon="⚠️"
        )

    action = workflow_navigation(
        current_step=2,
        can_go_next=can_proceed,
        next_label="Iniciar Generación",
        on_next="③ Generar",
        on_prev="① Análisis"
    )

    if action == "next" and can_proceed:
        if 2 not in st.session_state.get("workflow_completed", []):
            st.session_state.workflow_completed = st.session_state.get("workflow_completed", []) + [2]
        st.session_state.workflow_step = 3
        st.rerun()
    elif action == "prev":
        st.session_state.nav_menu = "① Análisis"
        st.rerun()
