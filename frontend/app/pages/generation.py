"""
Generation Page (Step 3)
========================
Batch generation monitoring with progress tracking and ETA.
"""

import os
import time
import json
from datetime import datetime
import streamlit as st
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional

from app.components.api_client import get_api_client
from app.components.ui import (
    page_header, section_header, spacer, alert_box, empty_state,
    workflow_stepper, workflow_navigation
)
from app.config.theme import get_colors_dict


def _make_json_serializable(obj):
    """Convert non-JSON-serializable objects to serializable format."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def _save_generation_config(output_dir: str, job_id: str, config: Dict) -> None:
    """Save the generation configuration alongside the generated dataset.

    This allows users to retrieve and reuse the exact configuration
    used for a specific generation job.

    Saves:
    - Directories configuration
    - Effects enabled and their intensity values
    - Advanced generation options
    - Validation settings
    - Targets per class
    """
    try:
        # Build job folder path
        job_folder = f"job_{job_id}" if not job_id.startswith("job_") else job_id
        base_output = _normalize_path(output_dir)

        # Try to find the actual job folder
        job_path = base_output / job_folder
        if not job_path.exists():
            job_path = base_output  # Fallback

        # Extract effects config with all intensity values
        effects_config = config.get("effects_config", {})

        # Build comprehensive config with metadata
        saved_config = {
            "config_version": "1.0",
            "job_id": job_id,
            "created_at": datetime.now().isoformat(),
            "directories": {
                "backgrounds_dir": config.get("backgrounds_dir", ""),
                "objects_dir": config.get("objects_dir", ""),
                "output_dir": config.get("output_dir", ""),
            },
            "effects": {
                "enabled": config.get("effects", []),
                "config": {
                    # Core intensity values
                    "color_intensity": effects_config.get("color_intensity", 0.7),
                    "blur_strength": effects_config.get("blur_strength", 1.0),
                    "shadow_opacity": effects_config.get("shadow_opacity", 0.4),
                    "underwater_intensity": effects_config.get("underwater_intensity", 0.25),
                    "caustics_intensity": effects_config.get("caustics_intensity", 0.15),
                    "edge_feather": effects_config.get("edge_feather", 5),
                    # Additional effect settings
                    "lighting_type": effects_config.get("lighting_type", "ambient"),
                    "lighting_intensity": effects_config.get("lighting_intensity", 0.5),
                    "water_color": list(effects_config.get("water_color", [20, 80, 120])),
                    "water_clarity": effects_config.get("water_clarity", "clear"),
                    "motion_blur_probability": effects_config.get("motion_blur_probability", 0.2),
                },
            },
            "generation": {
                "num_images": config.get("num_images", 0),
                "max_objects_per_image": config.get("max_objects_per_image", 5),
                "overlap_threshold": config.get("overlap_threshold", 0.1),
                "depth_aware": config.get("depth_aware", True),
            },
            "validation": {
                "validate_quality": config.get("validate_quality", False),
                "validate_physics": config.get("validate_physics", False),
            },
            "debug": {
                "save_pipeline_debug": config.get("save_pipeline_debug", False),
            },
            "targets_per_class": config.get("targets_per_class", {}),
        }

        # Make sure everything is JSON serializable
        saved_config = _make_json_serializable(saved_config)

        # Save to job folder
        config_path = job_path / "generation_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(saved_config, f, indent=2, ensure_ascii=False)

    except Exception:
        # Silently fail - this is a convenience feature, not critical
        pass


def render_generation_page():
    """Render the generation monitoring page (Step 3 of workflow)"""

    # Workflow stepper
    completed = st.session_state.get("workflow_completed", [])
    workflow_stepper(current_step=3, completed_steps=completed)

    page_header(
        title="Generación en Progreso",
        subtitle="Paso 3: Monitorea el proceso de generación de imágenes sintéticas",
        icon="🏭"
    )

    # Check if we have config from previous step
    config = st.session_state.get("generation_config")

    if not config:
        alert_box(
            "No hay configuración de generación. Vuelve al paso anterior para configurar.",
            type="warning",
            icon="⚠️"
        )

        action = workflow_navigation(
            current_step=3,
            can_go_next=False,
            on_prev="② Configurar"
        )

        if action == "prev":
            st.session_state.nav_menu = "② Configurar"
            st.rerun()
        return

    # Configuration summary
    _render_config_summary(config)

    spacer(16)

    # Check if there's an active job
    current_job = st.session_state.get("current_job_id")

    if current_job:
        _render_job_monitor(current_job, config)
    else:
        _render_start_generation(config)


def _render_config_summary(config: Dict) -> None:
    """Show configuration summary from Step 2"""
    c = get_colors_dict()
    num_images = config.get("num_images", 0)
    effects = config.get("effects", [])
    max_objects = config.get("max_objects_per_image", 5)
    output_dir = config.get("output_dir", "")

    st.markdown(f"""
    <div style="background: {c['bg_secondary']}; border: 1px solid {c['border']};
                border-radius: 0.5rem; padding: 1rem; margin-bottom: 1rem;">
        <div style="font-size: 0.7rem; color: {c['text_muted']}; text-transform: uppercase;
                    letter-spacing: 0.05em; margin-bottom: 0.75rem;">
            Configuración de Generación
        </div>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;">
            <div>
                <div style="font-weight: 700; font-size: 1.5rem; color: {c['primary']};">
                    {num_images:,}
                </div>
                <div style="font-size: 0.8rem; color: {c['text_muted']};">Imágenes</div>
            </div>
            <div>
                <div style="font-weight: 700; font-size: 1.5rem; color: {c['text_primary']};">
                    {len(effects)}
                </div>
                <div style="font-size: 0.8rem; color: {c['text_muted']};">Efectos</div>
            </div>
            <div>
                <div style="font-weight: 700; font-size: 1.5rem; color: {c['text_primary']};">
                    {max_objects}
                </div>
                <div style="font-size: 0.8rem; color: {c['text_muted']};">Max obj/img</div>
            </div>
            <div>
                <div style="font-weight: 600; font-size: 0.9rem; color: {c['text_primary']};
                            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                    {Path(output_dir).name if output_dir else 'N/A'}
                </div>
                <div style="font-size: 0.8rem; color: {c['text_muted']};">Salida</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_start_generation(config: Dict) -> None:
    """Render the start generation button and confirmation"""
    c = get_colors_dict()

    section_header("Iniciar Generación", icon="🚀")

    # Pre-flight checks
    backgrounds_dir = config.get("backgrounds_dir", "")
    objects_dir = config.get("objects_dir", "")
    num_images = config.get("num_images", 0)

    bg_exists = Path(backgrounds_dir).exists() if backgrounds_dir else False
    obj_exists = Path(objects_dir).exists() if objects_dir else False

    checks_passed = bg_exists and obj_exists and num_images > 0

    if not checks_passed:
        col1, col2 = st.columns(2)
        with col1:
            if bg_exists:
                st.success("✓ Directorio de fondos")
            else:
                st.error("✗ Directorio de fondos no encontrado")
        with col2:
            if obj_exists:
                st.success("✓ Directorio de objetos")
            else:
                st.error("✗ Directorio de objetos no encontrado")

        alert_box(
            "Verifica la configuración en el paso anterior antes de continuar.",
            type="error"
        )
        return

    # Ready to start
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {c['primary']}, {c['primary_hover']});
                border-radius: 0.75rem; padding: 1.5rem; text-align: center; color: white;
                margin-bottom: 1rem;">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🚀</div>
        <div style="font-size: 1rem; margin-bottom: 0.5rem;">
            Todo listo para generar
        </div>
        <div style="font-size: 2.5rem; font-weight: 700;">
            {num_images:,} imágenes
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Start button
    if st.button("▶️ Iniciar Generación", type="primary", use_container_width=True, key="start_batch"):
        _start_batch_job(config)


def _start_batch_job(config: Dict) -> None:
    """Start the batch generation job"""
    client = get_api_client()

    with st.spinner("Iniciando generación..."):
        result = client.compose_batch(
            backgrounds_dir=config.get("backgrounds_dir"),
            objects_dir=config.get("objects_dir"),
            output_dir=config.get("output_dir"),
            num_images=config.get("num_images", 0),
            targets_per_class=config.get("targets_per_class"),
            max_objects_per_image=config.get("max_objects_per_image", 5),
            overlap_threshold=config.get("overlap_threshold", 0.1),
            effects=config.get("effects", []),
            effects_config=config.get("effects_config", {}),
            depth_aware=config.get("depth_aware", True),
            validate_quality=config.get("validate_quality", False),
            validate_physics=config.get("validate_physics", False),
            save_pipeline_debug=config.get("save_pipeline_debug", False),
        )

    if result.get("success"):
        job_id = result.get("job_id")
        st.session_state.current_job_id = job_id
        st.session_state.job_start_time = time.time()
        st.success(f"✅ Job iniciado: `{job_id}`")
        st.rerun()
    else:
        st.error(f"❌ Error: {result.get('error')}")


def _render_job_monitor(job_id: str, config: Dict) -> None:
    """Render the job monitoring interface"""
    c = get_colors_dict()

    section_header("Monitor de Generación", icon="📊")

    # Manual refresh button
    if st.button("🔄 Actualizar Estado", key="refresh_job", use_container_width=False):
        st.rerun()

    # Fetch job status
    client = get_api_client()
    job = client.get_job_status(job_id)

    if job.get("error"):
        alert_box(f"Error al obtener estado: {job.get('error')}", type="error")

        # Allow retry or go back
        if st.button("🔄 Reintentar", key="retry_job"):
            st.session_state.current_job_id = None
            st.rerun()
        return

    status = job.get("status", "unknown")
    generated = job.get("images_generated", 0)
    rejected = job.get("images_rejected", 0)
    pending = job.get("images_pending", 0)
    total = generated + rejected + pending
    error = job.get("error")
    output_dir = job.get("output_dir", config.get("output_dir", ""))

    # Status banner
    if status == "completed":
        status_color = c['success']
        status_bg = c['success_bg']
        status_icon = "✅"
        status_text = "Completado"
    elif status == "processing":
        status_color = c['warning']
        status_bg = c['warning_bg']
        status_icon = "⏳"
        status_text = "En progreso"
    elif status == "failed":
        status_color = c['error']
        status_bg = c['error_bg']
        status_icon = "❌"
        status_text = "Error"
    elif status == "cancelled":
        status_color = c['text_muted']
        status_bg = c['bg_secondary']
        status_icon = "⏹️"
        status_text = "Cancelado"
    else:
        status_color = c['info']
        status_bg = c['bg_secondary']
        status_icon = "⏸️"
        status_text = status.capitalize()

    st.markdown(f"""
    <div style="background: {status_bg}; border: 1px solid {status_color};
                border-radius: 0.5rem; padding: 1rem; margin-bottom: 1rem;
                display: flex; align-items: center; gap: 1rem;">
        <span style="font-size: 1.5rem;">{status_icon}</span>
        <div>
            <div style="font-weight: 600; color: {status_color};">{status_text}</div>
            <div style="font-size: 0.8rem; color: {c['text_muted']};">Job: {job_id}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Progress bar
    if total > 0:
        progress = generated / total
        st.progress(progress, text=f"{generated:,} / {total:,} imágenes ({progress*100:.1f}%)")

    spacer(16)

    # Metrics grid
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div style="background: {c['bg_card']}; border: 1px solid {c['border']};
                    border-radius: 0.5rem; padding: 1rem; text-align: center;">
            <div style="font-size: 2rem; font-weight: 700; color: {c['success']};">{generated:,}</div>
            <div style="font-size: 0.8rem; color: {c['text_muted']};">Generadas</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background: {c['bg_card']}; border: 1px solid {c['border']};
                    border-radius: 0.5rem; padding: 1rem; text-align: center;">
            <div style="font-size: 2rem; font-weight: 700; color: {c['error']};">{rejected}</div>
            <div style="font-size: 0.8rem; color: {c['text_muted']};">Rechazadas</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="background: {c['bg_card']}; border: 1px solid {c['border']};
                    border-radius: 0.5rem; padding: 1rem; text-align: center;">
            <div style="font-size: 2rem; font-weight: 700; color: {c['text_secondary']};">{pending}</div>
            <div style="font-size: 0.8rem; color: {c['text_muted']};">Pendientes</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        # Calculate ETA
        eta = _calculate_eta(generated, pending, status)
        st.markdown(f"""
        <div style="background: {c['bg_card']}; border: 1px solid {c['border']};
                    border-radius: 0.5rem; padding: 1rem; text-align: center;">
            <div style="font-size: 1.5rem; font-weight: 700; color: {c['primary']};">{eta}</div>
            <div style="font-size: 0.8rem; color: {c['text_muted']};">ETA</div>
        </div>
        """, unsafe_allow_html=True)

    spacer(16)

    # Error display
    if error:
        alert_box(f"Error: {error}", type="error", icon="❌")

    # Cancel button for active jobs
    if status in ["processing", "queued"]:
        if st.button("⏹️ Cancelar Job", type="secondary", key="cancel_job"):
            result = client.cancel_job(job_id)
            if result.get("success"):
                st.toast("Job cancelado")
                st.rerun()
            else:
                st.error(f"Error al cancelar: {result.get('error')}")

    # Preview of generated images (for processing or completed)
    if generated > 0 and output_dir:
        _render_preview_gallery(output_dir, job_id, generated)

    # Pipeline debug preview (if enabled in config)
    if config.get("save_pipeline_debug", False) and output_dir:
        spacer(16)
        _render_pipeline_debug_preview(output_dir, job_id)

    spacer(24)

    # Workflow navigation
    job_completed = status in ["completed", "cancelled"]

    if job_completed:
        # Store generated dataset info and save config
        if status == "completed":
            coco_path = Path(output_dir) / "synthetic_dataset.json"
            if coco_path.exists():
                try:
                    with open(coco_path) as f:
                        st.session_state.generated_dataset = json.load(f)
                    st.session_state.generated_output_dir = output_dir

                    # Save configuration used for this generation
                    _save_generation_config(output_dir, job_id, config)
                except Exception as e:
                    st.warning(f"No se pudo cargar el dataset generado: {e}")

        st.markdown(f"""
        <div style="background: {c['success_bg']}; border: 1px solid {c['success']};
                    border-radius: 0.5rem; padding: 1rem; text-align: center; margin-bottom: 1rem;">
            <span style="font-size: 1.5rem;">🎉</span>
            <span style="font-weight: 600; margin-left: 0.5rem; color: {c['success']};">
                Generación completada
            </span>
        </div>
        """, unsafe_allow_html=True)

        # New generation button
        if st.button("🔄 Nueva Generación", key="new_generation"):
            st.session_state.current_job_id = None
            st.rerun()

    action = workflow_navigation(
        current_step=3,
        can_go_next=job_completed,
        next_label="Exportar Dataset",
        on_next="④ Exportar",
        on_prev="② Configurar"
    )

    if action == "next" and job_completed:
        if 3 not in st.session_state.get("workflow_completed", []):
            st.session_state.workflow_completed = st.session_state.get("workflow_completed", []) + [3]
        st.session_state.workflow_step = 4
        st.session_state.current_job_id = None  # Clear job for next time
        st.rerun()
    elif action == "prev":
        st.session_state.nav_menu = "② Configurar"
        st.rerun()


def _calculate_eta(generated: int, pending: int, status: str) -> str:
    """Calculate estimated time remaining"""
    if status == "completed":
        return "✓"
    if status != "processing":
        return "—"
    if generated == 0:
        return "Calculando..."

    start_time = st.session_state.get("job_start_time")
    if not start_time:
        return "—"

    elapsed = time.time() - start_time
    rate = generated / elapsed  # images per second

    if rate > 0 and pending > 0:
        eta_seconds = pending / rate
        if eta_seconds < 60:
            return f"~{int(eta_seconds)}s"
        elif eta_seconds < 3600:
            return f"~{int(eta_seconds / 60)}min"
        else:
            return f"~{eta_seconds / 3600:.1f}h"
    return "—"


def _normalize_path(path_str: str) -> Path:
    """Normalize a path string to handle Docker/Windows path translations.

    Converts Docker paths like /app/output/synthetic to local paths.
    """
    import os

    # Normalize slashes
    normalized = path_str.replace("\\", "/")

    # Remove /app prefix for local development
    if normalized.startswith("/app/"):
        normalized = normalized[5:]  # Remove "/app/"
    elif normalized.startswith("/app"):
        normalized = normalized[4:]  # Remove "/app"

    # Convert to Path and resolve
    try:
        p = Path(normalized)
        # If path is not absolute, try to resolve it
        if not p.is_absolute():
            # Try relative to current working directory
            cwd = Path.cwd()
            resolved = cwd / p
            if resolved.exists():
                return resolved
            # Try relative to project root (go up from frontend/app/pages)
            project_root = Path(__file__).parent.parent.parent.parent
            resolved = project_root / p
            if resolved.exists():
                return resolved
        return p
    except Exception:
        return Path(path_str)


def _render_preview_gallery(output_dir: str, job_id: str, count: int) -> None:
    """Render a preview gallery of generated images.

    Images are expected at: {output_dir}/job_{job_id}/images/
    The output_dir is the user's configured base path (e.g., /app/output/synthetic)
    """
    c = get_colors_dict()

    # Show preview section with count badge
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
        <span style="font-size: 1rem;">🖼️</span>
        <span style="font-weight: 600; color: {c['text_primary']};">Preview de Imágenes Generadas</span>
        <span style="background: {c['primary']}; color: white; font-size: 0.7rem;
                    padding: 0.15rem 0.5rem; border-radius: 1rem; font-weight: 600;">
            {count}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Build the job folder name
    job_folder = f"job_{job_id}" if not job_id.startswith("job_") else job_id

    # Normalize the base output directory path
    base_output = _normalize_path(output_dir)
    found_images_dir = None

    # Get project root for path resolution
    project_root = Path(__file__).parent.parent.parent.parent

    # Images are at: {base_output}/job_{id}/images/
    # Build list of image directory paths to try
    paths_to_try = [
        # Primary path: base_output/job_{id}/images/
        base_output / job_folder / "images",
        # Project root relative paths
        project_root / "output" / "synthetic" / job_folder / "images",
        project_root / output_dir.replace("/app/", "").lstrip("/").replace("\\", "/") / job_folder / "images" if output_dir else None,
        # Without /app/ prefix
        Path(output_dir.replace("/app/", "").replace("\\", "/")) / job_folder / "images" if output_dir else None,
        # Fallback: maybe output_dir already includes job folder (check for /images directly)
        base_output / "images",
    ]

    # Filter out None values
    paths_to_try = [p for p in paths_to_try if p is not None]

    # Try to find an existing path with images
    for p in paths_to_try:
        try:
            if p.exists() and p.is_dir():
                # Check if there are images in this directory
                has_images = any(p.glob("*.jpg")) or any(p.glob("*.png")) or any(p.glob("*.jpeg"))
                if has_images:
                    found_images_dir = p
                    break
        except Exception:
            continue

    if not found_images_dir:
        with st.expander("📁 Directorio de imágenes no encontrado", expanded=False):
            st.warning(f"Ruta base: `{output_dir}`")
            st.caption(f"Ruta normalizada: `{base_output}`")
            st.caption(f"Job folder: `{job_folder}`")
            st.caption(f"Project root: `{project_root}`")
            st.caption("Rutas buscadas:")
            for p in paths_to_try:
                try:
                    exists = "✓" if p.exists() else "✗"
                except:
                    exists = "?"
                st.caption(f"  {exists} `{p}`")
        return

    # Find images in the directory
    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        try:
            images.extend(found_images_dir.glob(ext))
        except Exception:
            pass

    # Remove duplicates
    images = list(set(images))

    if not images:
        st.info(f"📂 Directorio encontrado pero sin imágenes: `{found_images_dir}`")
        return

    # Sort by modification time (newest first)
    try:
        images = sorted(images, key=lambda x: x.stat().st_mtime, reverse=True)[:8]
    except Exception:
        images = images[:8]

    # Render gallery
    cols = st.columns(4)
    for i, img_path in enumerate(images):
        with cols[i % 4]:
            try:
                img = Image.open(img_path)
                st.image(img, caption=img_path.name, use_container_width=True)
            except Exception as e:
                st.markdown(f"""
                <div style="background: {c['error_bg']}; border: 1px solid {c['error']};
                            border-radius: 0.25rem; padding: 0.5rem; font-size: 0.75rem;">
                    ❌ {img_path.name}
                </div>
                """, unsafe_allow_html=True)

    if len(images) < count:
        st.caption(f"Mostrando {len(images)} de {count} imágenes (desde `{found_images_dir}`)")


def _render_pipeline_debug_preview(output_dir: str, job_id: str) -> None:
    """Render a preview of pipeline debug images showing the evolution of the first iteration.

    Debug images are expected at: {output_dir}/job_{job_id}/pipeline_debug/
    The output_dir is the user's configured base path (e.g., /app/output/synthetic)

    Images are displayed in order by filename to show pipeline evolution.
    The pipeline_debug folder contains both root-level images and subfolders for each object.
    """
    c = get_colors_dict()

    # Build the job folder name
    job_folder = f"job_{job_id}" if not job_id.startswith("job_") else job_id

    # Normalize the base output directory path
    base_output = _normalize_path(output_dir)
    found_debug_dir = None

    # Get project root for path resolution
    project_root = Path(__file__).parent.parent.parent.parent

    # Debug images are at: {base_output}/job_{id}/pipeline_debug/
    # Build list of debug directory paths to try
    paths_to_try = [
        # Primary path: base_output/job_{id}/pipeline_debug/
        base_output / job_folder / "pipeline_debug",
        # Project root relative paths
        project_root / "output" / "synthetic" / job_folder / "pipeline_debug",
        project_root / output_dir.replace("/app/", "").lstrip("/").replace("\\", "/") / job_folder / "pipeline_debug" if output_dir else None,
        # Without /app/ prefix
        Path(output_dir.replace("/app/", "").replace("\\", "/")) / job_folder / "pipeline_debug" if output_dir else None,
        # Fallback: maybe output_dir already includes job folder (check for /pipeline_debug directly)
        base_output / "pipeline_debug",
    ]

    # Filter out None values
    paths_to_try = [p for p in paths_to_try if p is not None]

    # Try to find an existing path with debug images (check recursively)
    for p in paths_to_try:
        try:
            if p.exists() and p.is_dir():
                # Check if there are images in this directory or subdirectories
                has_images = any(p.glob("*.jpg")) or any(p.glob("*.png")) or any(p.glob("**/*.jpg")) or any(p.glob("**/*.png"))
                if has_images:
                    found_debug_dir = p
                    break
        except Exception:
            continue

    if not found_debug_dir:
        # Show informative message when debug is enabled but no images found yet
        with st.expander("🔬 Pipeline Debug - Esperando imágenes...", expanded=False):
            st.info("Las imágenes de debug se generan con la primera iteración. Espera a que comience la generación.")
            st.caption(f"Ruta base: `{output_dir}`")
            st.caption(f"Ruta normalizada: `{base_output}`")
            st.caption(f"Job folder: `{job_folder}`")
            st.caption("Rutas buscadas:")
            for p in paths_to_try:
                try:
                    exists = "✓" if p.exists() else "✗"
                except:
                    exists = "?"
                st.caption(f"  {exists} `{p}`")
        return

    # Find all debug images (including in subdirectories for per-object debug)
    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        try:
            # Get root-level images first
            images.extend(found_debug_dir.glob(ext))
        except Exception:
            pass

    # Remove duplicates
    images = list(set(images))

    if not images:
        return

    # Sort by filename to show pipeline evolution in order
    # Typically named like: 01_background.png, 02_depth.png, 03_placement.png, etc.
    try:
        images = sorted(images, key=lambda x: x.name)
    except Exception:
        pass

    # Render debug section with expander
    with st.expander("🔬 Pipeline Debug - Evolución del Proceso", expanded=False):
        st.markdown(f"""
        <div style="background: {c['info_bg']}; border: 1px solid {c['info']};
                    border-radius: 0.5rem; padding: 0.75rem; margin-bottom: 1rem;">
            <span style="font-size: 0.85rem; color: {c['text_secondary']};">
                Vista paso a paso del pipeline de generación (primera iteración).
                Las imágenes muestran la evolución desde el fondo original hasta la imagen final.
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Display images in a grid - show all debug images in order
        num_cols = 3
        for row_start in range(0, len(images), num_cols):
            cols = st.columns(num_cols)
            for col_idx, img_path in enumerate(images[row_start:row_start + num_cols]):
                with cols[col_idx]:
                    try:
                        img = Image.open(img_path)
                        # Extract step name from filename (e.g., "01_background" -> "Background")
                        step_name = img_path.stem
                        # Clean up the step name for display
                        if "_" in step_name:
                            parts = step_name.split("_", 1)
                            # Remove leading number if present
                            if parts[0].isdigit() and len(parts) > 1:
                                step_name = parts[1].replace("_", " ").title()
                            else:
                                step_name = step_name.replace("_", " ").title()
                        else:
                            step_name = step_name.title()

                        st.image(img, caption=step_name, use_container_width=True)
                    except Exception as e:
                        st.markdown(f"""
                        <div style="background: {c['error_bg']}; border: 1px solid {c['error']};
                                    border-radius: 0.25rem; padding: 0.5rem; font-size: 0.75rem;">
                            ❌ {img_path.name}
                        </div>
                        """, unsafe_allow_html=True)

        st.caption(f"📁 {len(images)} imágenes de debug desde: `{found_debug_dir}`")


# =============================================================================
# STUDIO: Single Image Generation (kept as standalone tool)
# =============================================================================

def render_studio_page():
    """Render studio page for single image generation (standalone tool)"""

    page_header(
        title="Studio",
        subtitle="Generación de imágenes sintéticas individuales",
        icon="🎨"
    )

    # 3-Column Layout
    col_controls, col_canvas, col_info = st.columns([1, 2, 1])

    # ===== COLUMN 1: Controls =====
    with col_controls:
        effects, effects_config, gen_config = _render_studio_controls()

    # ===== COLUMN 2: Canvas =====
    with col_canvas:
        _render_studio_canvas(effects, effects_config, gen_config)

    # ===== COLUMN 3: Objects & Metrics =====
    with col_info:
        _render_studio_objects()


def _render_studio_controls() -> tuple:
    """Render studio control panel"""
    c = get_colors_dict()
    st.markdown(f"""
    <div style="font-weight: 600; font-size: 1rem; margin-bottom: 1rem; color: {c['text_primary']};">
        ⚙️ Configuración
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🎯 Generación", expanded=True):
        depth_aware = st.checkbox("Depth-aware", value=True, key="studio_depth_aware")
        max_objects = st.slider("Máx objetos", 1, 10, 5, key="studio_max_objects")
        overlap = st.slider("Overlap", 0.0, 0.5, 0.1, key="studio_overlap")

    with st.expander("✨ Efectos", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            fx_color = st.checkbox("Color", value=True, key="studio_fx_color")
            fx_blur = st.checkbox("Blur", value=True, key="studio_fx_blur")
            fx_shadows = st.checkbox("Sombras", value=True, key="studio_fx_shadows")
            fx_caustics = st.checkbox("Caustics", value=True, key="studio_fx_caustics")
        with col2:
            fx_underwater = st.checkbox("Underwater", value=True, key="studio_fx_underwater")
            fx_lighting = st.checkbox("Lighting", value=False, key="studio_fx_lighting")
            fx_motion = st.checkbox("Motion", value=False, key="studio_fx_motion")
            fx_edge = st.checkbox("Edge", value=True, key="studio_fx_edge")

    with st.expander("🎚️ Intensidades", expanded=False):
        color_int = st.slider("Color", 0.0, 1.0, 0.7, key="studio_int_color")
        blur_str = st.slider("Blur", 0.0, 3.0, 1.0, key="studio_int_blur")
        underwater_int = st.slider("Underwater", 0.0, 1.0, 0.25, key="studio_int_underwater")
        caustics_int = st.slider("Caustics", 0.0, 0.5, 0.15, key="studio_int_caustics")
        shadow_opacity = st.slider("Sombra", 0.0, 1.0, 0.4, key="studio_int_shadow")

    with st.expander("✅ Validación", expanded=False):
        val_quality = st.checkbox("Calidad (LPIPS)", value=False, key="studio_val_quality")
        val_physics = st.checkbox("Física", value=False, key="studio_val_physics")

    # Build effects list
    effects = []
    if fx_color: effects.append("color_correction")
    if fx_blur: effects.append("blur_matching")
    if fx_shadows: effects.append("shadows")
    if fx_caustics: effects.append("caustics")
    if fx_underwater: effects.append("underwater")
    if fx_lighting: effects.append("lighting")
    if fx_motion: effects.append("motion_blur")
    if fx_edge: effects.append("edge_smoothing")

    effects_config = {
        "color_intensity": color_int if 'color_int' in dir() else 0.7,
        "blur_strength": blur_str if 'blur_str' in dir() else 1.0,
        "underwater_intensity": underwater_int if 'underwater_int' in dir() else 0.25,
        "caustics_intensity": caustics_int if 'caustics_int' in dir() else 0.15,
        "shadow_opacity": shadow_opacity if 'shadow_opacity' in dir() else 0.4,
        "lighting_type": "ambient",
        "lighting_intensity": 0.5,
        "water_color": (20, 80, 120),
        "water_clarity": "clear",
        "motion_blur_probability": 0.2,
        "edge_feather": 5,
    }

    gen_config = {
        "depth_aware": depth_aware,
        "max_objects": max_objects,
        "overlap_threshold": overlap,
        "validate_quality": val_quality if 'val_quality' in dir() else False,
        "validate_physics": val_physics if 'val_physics' in dir() else False,
    }

    return effects, effects_config, gen_config


def _render_studio_canvas(effects: List[str], effects_config: Dict, gen_config: Dict) -> None:
    """Render the studio canvas area"""
    import random
    c = get_colors_dict()

    st.markdown(f"""
    <div style="font-weight: 600; font-size: 1rem; margin-bottom: 1rem; color: {c['text_primary']};">
        🖼️ Canvas
    </div>
    """, unsafe_allow_html=True)

    input_method = st.radio(
        "Fuente del fondo",
        ["📤 Subir archivo", "📂 Ruta", "🗂️ Explorar datasets"],
        horizontal=True,
        key="studio_input_method"
    )

    background_path = None

    if input_method == "📤 Subir archivo":
        uploaded = st.file_uploader(
            "Subir fondo", type=["jpg", "jpeg", "png"],
            key="studio_bg_upload", label_visibility="collapsed"
        )
        if uploaded:
            shared_dir = Path("/shared/images/input")
            shared_dir.mkdir(parents=True, exist_ok=True)
            save_path = shared_dir / uploaded.name
            with open(save_path, "wb") as f:
                f.write(uploaded.getvalue())
            background_path = str(save_path)

    elif input_method == "📂 Ruta":
        background_path = st.text_input(
            "Ruta del fondo",
            placeholder="/app/datasets/Backgrounds_filtered/image.jpg",
            key="studio_bg_path", label_visibility="collapsed"
        )
    else:
        datasets_dir = Path(os.environ.get("BACKGROUNDS_PATH", "/app/datasets/Backgrounds_filtered"))
        if datasets_dir.exists():
            bg_files = list(datasets_dir.glob("*.jpg")) + list(datasets_dir.glob("*.png"))
            bg_names = [f.name for f in bg_files[:50]]
            if bg_names:
                selected = st.selectbox("Seleccionar fondo", options=bg_names, key="studio_bg_select")
                if selected:
                    background_path = str(datasets_dir / selected)
            else:
                st.warning("No hay fondos en el directorio")
        else:
            st.warning(f"Directorio no encontrado: {datasets_dir}")

    # Show background preview
    if background_path and Path(background_path).exists():
        try:
            img = Image.open(background_path)
            st.image(img, caption="Fondo seleccionado", use_container_width=True)
        except Exception as e:
            st.error(f"Error al cargar imagen: {e}")

    st.markdown(f"<hr style='border: none; border-top: 1px solid {c['border']}; margin: 1rem 0;'>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        output_dir = st.text_input("Directorio salida", value="/app/output/single", key="studio_output_dir")
    with col_b:
        output_filename = st.text_input("Nombre archivo", value="synthetic_001.jpg", key="studio_output_name")

    output_path = str(Path(output_dir) / output_filename)
    generate_disabled = not background_path or not Path(background_path).exists() if background_path else True

    if st.button("🚀 Generar Imagen", type="primary", disabled=generate_disabled, use_container_width=True, key="studio_generate_btn"):
        objects_to_place = st.session_state.get("studio_objects_to_place", [])
        client = get_api_client()

        with st.spinner("Generando..."):
            start = time.time()
            result = client.compose_image(
                background_path=background_path,
                objects=objects_to_place,
                effects=effects,
                effects_config=effects_config,
                output_path=output_path,
                validate_quality=gen_config.get("validate_quality", False),
                validate_physics=gen_config.get("validate_physics", False),
            )
            elapsed = time.time() - start

        st.session_state["studio_result"] = result
        st.session_state["studio_elapsed"] = elapsed

    if "studio_result" in st.session_state:
        result = st.session_state["studio_result"]
        elapsed = st.session_state.get("studio_elapsed", 0)

        if result.get("success"):
            st.success(f"✅ Generado en {elapsed:.1f}s")
            output = result.get("output_path", "")
            if output and Path(output).exists():
                try:
                    img = Image.open(output)
                    st.image(img, caption="Imagen generada", use_container_width=True)
                except:
                    pass

                with open(output, "rb") as f:
                    st.download_button(
                        "📥 Descargar Imagen", f,
                        file_name=Path(output).name, mime="image/jpeg",
                        key="studio_download", use_container_width=True
                    )
        else:
            st.error(f"❌ Error: {result.get('error', 'Unknown')}")


def _render_studio_objects() -> None:
    """Render the studio objects panel"""
    import random
    c = get_colors_dict()

    st.markdown(f"""
    <div style="font-weight: 600; font-size: 1rem; margin-bottom: 1rem; color: {c['text_primary']};">
        📦 Objetos
    </div>
    """, unsafe_allow_html=True)

    objects_dir = Path(os.environ.get("OBJECTS_PATH", "/app/datasets/Objects"))
    objects_to_place = []

    if objects_dir.exists():
        class_dirs = [d for d in objects_dir.iterdir() if d.is_dir()]

        if class_dirs:
            selected_classes = st.multiselect(
                "Clases a colocar",
                options=[d.name for d in class_dirs],
                key="studio_obj_classes"
            )

            for cls in selected_classes:
                cls_dir = objects_dir / cls
                obj_files = list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg"))
                if obj_files:
                    obj_file = random.choice(obj_files)
                    objects_to_place.append({
                        "image_path": str(obj_file),
                        "class_name": cls,
                        "position": None,
                        "scale": None,
                        "rotation": None,
                    })

            if selected_classes:
                st.info(f"📦 {len(objects_to_place)} objetos seleccionados")

                for obj in objects_to_place[:3]:
                    try:
                        thumb = Image.open(obj["image_path"])
                        thumb.thumbnail((80, 80))
                        st.image(thumb, caption=obj["class_name"], width=80)
                    except:
                        pass

            st.session_state["studio_objects_to_place"] = objects_to_place
        else:
            st.warning("No hay clases de objetos")
    else:
        st.warning(f"Directorio no encontrado: {objects_dir}")

    st.markdown(f"<hr style='border: none; border-top: 1px solid {c['border']}; margin: 1rem 0;'>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="font-weight: 600; font-size: 1rem; margin-bottom: 1rem; color: {c['text_primary']};">
        📊 Métricas
    </div>
    """, unsafe_allow_html=True)

    if "studio_result" in st.session_state:
        result = st.session_state["studio_result"]
        if result.get("success"):
            st.metric("Tiempo", f"{result.get('processing_time_ms', 0):.0f}ms")
            st.metric("Objetos", result.get("objects_placed", 0))
            st.metric("Depth", "Sí" if result.get("depth_used") else "No")
            st.metric("Válido", "Sí" if result.get("is_valid", True) else "No")

            fx = result.get("effects_applied", [])
            if fx:
                st.caption(f"Efectos: {', '.join(fx)}")

            if result.get("quality_score"):
                with st.expander("Quality Score"):
                    st.json(result["quality_score"])

            if result.get("annotations"):
                with st.expander("Annotations"):
                    st.json(result["annotations"])
    else:
        st.caption("Genera una imagen para ver métricas")
