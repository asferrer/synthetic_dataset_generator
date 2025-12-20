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
from typing import Dict, List, Any, Optional
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


def _build_exportable_config(
    backgrounds_dir: str,
    objects_dir: str,
    output_dir: str,
    effects: List[str],
    effects_config: Dict,
    max_objects: int,
    overlap_threshold: float,
    depth_aware: bool,
    validate_quality: bool,
    validate_physics: bool,
    save_debug: bool,
    targets_per_class: Optional[Dict] = None,
) -> Dict:
    """Build a configuration dictionary suitable for export."""
    return {
        "config_version": _get_config_version(),
        "created_at": datetime.now().isoformat(),
        "directories": {
            "backgrounds_dir": backgrounds_dir,
            "objects_dir": objects_dir,
            "output_dir": output_dir,
        },
        "effects": {
            "enabled": effects,
            "config": effects_config,
        },
        "generation": {
            "max_objects_per_image": max_objects,
            "overlap_threshold": overlap_threshold,
            "depth_aware": depth_aware,
        },
        "validation": {
            "validate_quality": validate_quality,
            "validate_physics": validate_physics,
        },
        "debug": {
            "save_pipeline_debug": save_debug,
        },
        "targets_per_class": targets_per_class or {},
    }


def _apply_loaded_config(config: Dict) -> bool:
    """Apply a loaded configuration to session state. Returns True on success."""
    try:
        # Directories
        dirs = config.get("directories", {})
        if dirs.get("backgrounds_dir"):
            st.session_state.config_bg_dir = dirs["backgrounds_dir"]
        if dirs.get("objects_dir"):
            st.session_state.config_obj_dir = dirs["objects_dir"]
        if dirs.get("output_dir"):
            st.session_state.config_output_dir = dirs["output_dir"]

        # Effects enabled
        effects_data = config.get("effects", {})
        enabled_effects = effects_data.get("enabled", [])

        st.session_state.fx_color = "color_correction" in enabled_effects
        st.session_state.fx_blur = "blur_matching" in enabled_effects
        st.session_state.fx_shadows = "shadows" in enabled_effects
        st.session_state.fx_caustics = "caustics" in enabled_effects
        st.session_state.fx_underwater = "underwater" in enabled_effects
        st.session_state.fx_edge = "edge_smoothing" in enabled_effects
        st.session_state.fx_motion = "motion_blur" in enabled_effects
        st.session_state.fx_lighting = "lighting" in enabled_effects

        # Effects config (intensities)
        fx_config = effects_data.get("config", {})
        if "color_intensity" in fx_config:
            st.session_state.int_color = fx_config["color_intensity"]
        if "blur_strength" in fx_config:
            st.session_state.int_blur = fx_config["blur_strength"]
        if "shadow_opacity" in fx_config:
            st.session_state.int_shadow = fx_config["shadow_opacity"]
        if "underwater_intensity" in fx_config:
            st.session_state.int_underwater = fx_config["underwater_intensity"]
        if "caustics_intensity" in fx_config:
            st.session_state.int_caustics = fx_config["caustics_intensity"]
        if "edge_feather" in fx_config:
            st.session_state.int_edge = fx_config["edge_feather"]

        # Generation options
        gen = config.get("generation", {})
        if "max_objects_per_image" in gen:
            st.session_state.config_max_objects = gen["max_objects_per_image"]
        if "overlap_threshold" in gen:
            st.session_state.config_overlap = gen["overlap_threshold"]
        if "depth_aware" in gen:
            st.session_state.config_depth_aware = gen["depth_aware"]

        # Validation options
        val = config.get("validation", {})
        if "validate_quality" in val:
            st.session_state.config_val_quality = val["validate_quality"]
        if "validate_physics" in val:
            st.session_state.config_val_physics = val["validate_physics"]

        # Debug options
        debug = config.get("debug", {})
        if "save_pipeline_debug" in debug:
            st.session_state.config_save_debug = debug["save_pipeline_debug"]

        # Targets per class (optional - only if present and user wants to override)
        if config.get("targets_per_class"):
            st.session_state.loaded_targets = config["targets_per_class"]

        return True
    except Exception as e:
        st.error(f"Error al aplicar configuración: {e}")
        return False


def _render_config_management_section() -> None:
    """Render the configuration import/export section."""
    c = get_colors_dict()

    with st.expander("💾 Gestión de Configuración", expanded=False):
        st.markdown(f"""
        <div style="background: {c['info_bg']}; border: 1px solid {c['info']};
                    border-radius: 0.5rem; padding: 0.75rem; margin-bottom: 1rem;">
            <span style="font-size: 0.85rem; color: {c['text_secondary']};">
                Guarda tu configuración actual para reutilizarla en futuras generaciones,
                o carga una configuración previamente guardada.
            </span>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**📥 Cargar Configuración**")

            uploaded_config = st.file_uploader(
                "Seleccionar archivo de configuración",
                type=["json"],
                key="config_upload",
                label_visibility="collapsed"
            )

            if uploaded_config:
                try:
                    config_data = json.load(uploaded_config)

                    # Validate it's a valid config file
                    if "config_version" not in config_data:
                        st.error("❌ Archivo no válido: falta versión de configuración")
                    else:
                        # Show config preview
                        st.markdown(f"""
                        <div style="background: {c['bg_secondary']}; border: 1px solid {c['border']};
                                    border-radius: 0.5rem; padding: 0.75rem; margin: 0.5rem 0;">
                            <div style="font-size: 0.75rem; color: {c['text_muted']};">Configuración detectada:</div>
                            <div style="font-size: 0.85rem; color: {c['text_primary']}; margin-top: 0.25rem;">
                                • Versión: {config_data.get('config_version', 'N/A')}<br>
                                • Creada: {config_data.get('created_at', 'N/A')[:10] if config_data.get('created_at') else 'N/A'}<br>
                                • Efectos: {len(config_data.get('effects', {}).get('enabled', []))}<br>
                                • Targets: {len(config_data.get('targets_per_class', {}))} clases
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        if st.button("✅ Aplicar Configuración", key="apply_config", use_container_width=True):
                            if _apply_loaded_config(config_data):
                                st.success("✅ Configuración aplicada correctamente")
                                st.rerun()

                except json.JSONDecodeError:
                    st.error("❌ Error al leer el archivo JSON")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

        with col2:
            st.markdown(f"**📤 Exportar Configuración Actual**")

            config_name = st.text_input(
                "Nombre de la configuración",
                value=f"config_{datetime.now().strftime('%Y%m%d_%H%M')}",
                key="config_export_name",
                label_visibility="collapsed",
                placeholder="Nombre para el archivo de configuración"
            )

            # Get current values from session state or defaults
            current_config = _build_exportable_config(
                backgrounds_dir=st.session_state.get("config_bg_dir", "/app/datasets/Backgrounds_filtered"),
                objects_dir=st.session_state.get("config_obj_dir", "/app/datasets/Objects"),
                output_dir=st.session_state.get("config_output_dir", "/app/output/synthetic"),
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
                targets_per_class=st.session_state.get("balancing_targets", {}),
            )

            config_json = json.dumps(current_config, indent=2, ensure_ascii=False)

            st.download_button(
                "📥 Descargar Configuración",
                data=config_json,
                file_name=f"{config_name}.json",
                mime="application/json",
                key="download_config",
                use_container_width=True
            )

            # Show current config summary
            st.markdown(f"""
            <div style="background: {c['bg_tertiary']}; border-radius: 0.5rem;
                        padding: 0.5rem; margin-top: 0.5rem; font-size: 0.75rem;">
                <span style="color: {c['text_muted']};">
                    Efectos: {len(current_config['effects']['enabled'])} |
                    Max obj: {current_config['generation']['max_objects_per_image']} |
                    Targets: {len(current_config['targets_per_class'])} clases
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

    # Check if there are loaded targets from a config file
    loaded_targets = st.session_state.get("loaded_targets")
    if loaded_targets:
        st.markdown(f"""
        <div style="background: {c['warning_bg']}; border: 1px solid {c['warning']};
                    border-radius: 0.5rem; padding: 0.75rem; margin-bottom: 1rem;
                    display: flex; align-items: center; gap: 0.75rem;">
            <span style="font-size: 1.25rem;">📋</span>
            <div style="flex: 1;">
                <div style="font-weight: 600; color: {c['text_primary']};">
                    Targets cargados desde configuración
                </div>
                <div style="font-size: 0.8rem; color: {c['text_muted']};">
                    {len(loaded_targets)} clases | {sum(loaded_targets.values()):,} imágenes totales
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_apply, col_dismiss = st.columns(2)
        with col_apply:
            if st.button("✅ Usar targets cargados", key="use_loaded_targets", use_container_width=True):
                st.session_state.balancing_targets = loaded_targets
                targets = loaded_targets
                del st.session_state.loaded_targets
                st.rerun()
        with col_dismiss:
            if st.button("❌ Mantener actuales", key="dismiss_loaded_targets", use_container_width=True):
                del st.session_state.loaded_targets
                st.rerun()

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

    effects_config = {
        "color_intensity": color_intensity if 'color_intensity' in dir() else 0.7,
        "blur_strength": blur_strength if 'blur_strength' in dir() else 1.0,
        "underwater_intensity": underwater_intensity if 'underwater_intensity' in dir() else 0.25,
        "caustics_intensity": caustics_intensity if 'caustics_intensity' in dir() else 0.15,
        "shadow_opacity": shadow_opacity if 'shadow_opacity' in dir() else 0.4,
        "edge_feather": edge_feather if 'edge_feather' in dir() else 5,
        "lighting_type": "ambient",
        "lighting_intensity": 0.5,
        "water_color": (20, 80, 120),
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
