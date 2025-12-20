"""
Home Page
=========
Dashboard de inicio con quick actions y estado del sistema.
"""

import streamlit as st
from typing import Dict, Any, Optional

from app.components.ui import page_header, section_header, spacer, alert_box
from app.components.api_client import get_api_client
from app.config.theme import get_colors_dict


def render_home_page():
    """Render the home dashboard page"""
    c = get_colors_dict()

    # Header
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem 0;">
        <div style="font-size: 4rem; margin-bottom: 0.5rem;">🔬</div>
        <h1 style="font-size: 2rem; font-weight: 700; margin: 0; color: {c['text_primary']};">
            Synthetic Dataset Generator
        </h1>
        <p style="color: {c['text_muted']}; margin-top: 0.5rem;">
            Genera datasets sintéticos para entrenar modelos de detección de objetos
        </p>
    </div>
    """, unsafe_allow_html=True)

    spacer(16)

    # Main content: Two columns
    col_main, col_quick = st.columns([2, 1])

    with col_main:
        _render_workflow_start()

    with col_quick:
        _render_quick_access()

    spacer(24)

    # System status
    _render_system_status()

    spacer(24)

    # Session info
    _render_session_info()


def _render_workflow_start():
    """Render the main workflow start card"""
    c = get_colors_dict()

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {c['primary']}, {c['primary_hover']});
                border-radius: 0.75rem; padding: 2rem; color: white;">
        <div style="font-size: 2.5rem; margin-bottom: 1rem;">📊</div>
        <h2 style="font-size: 1.5rem; font-weight: 700; margin: 0 0 0.5rem 0; color: white;">
            Iniciar Workflow
        </h2>
        <p style="opacity: 0.9; margin-bottom: 1.5rem; font-size: 0.95rem;">
            Sube tu dataset COCO para analizar el desbalance de clases y generar datos sintéticos automáticamente.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # File uploader for starting workflow
    uploaded = st.file_uploader(
        "Subir dataset COCO JSON",
        type=["json"],
        key="home_upload",
        help="Arrastra tu archivo de anotaciones COCO aquí para comenzar"
    )

    if uploaded:
        import json
        try:
            coco_data = json.load(uploaded)
            n_images = len(coco_data.get('images', []))
            n_anns = len(coco_data.get('annotations', []))
            n_cats = len(coco_data.get('categories', []))

            # Store in session state
            st.session_state.source_dataset = coco_data
            st.session_state.source_filename = uploaded.name
            st.session_state.workflow_step = 1

            st.success(f"Dataset cargado: **{n_images}** imágenes, **{n_anns}** anotaciones, **{n_cats}** clases")

            if st.button("Continuar al Análisis →", type="primary", use_container_width=True):
                st.session_state.nav_menu = "① Análisis"
                st.rerun()

        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")

    spacer(16)

    # Workflow steps preview
    st.markdown(f"""
    <div style="background: {c['bg_secondary']}; border-radius: 0.5rem; padding: 1rem;
                border: 1px solid {c['border']};">
        <div style="font-size: 0.75rem; color: {c['text_muted']}; text-transform: uppercase;
                    letter-spacing: 0.05em; margin-bottom: 0.75rem;">
            Pasos del Workflow
        </div>
        <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
            <span style="background: {c['primary_light']}; color: {c['primary']};
                        padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.8rem;">
                ① Análisis
            </span>
            <span style="color: {c['text_muted']};">→</span>
            <span style="background: {c['bg_tertiary']}; color: {c['text_secondary']};
                        padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.8rem;">
                ② Configurar
            </span>
            <span style="color: {c['text_muted']};">→</span>
            <span style="background: {c['bg_tertiary']}; color: {c['text_secondary']};
                        padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.8rem;">
                ③ Generar
            </span>
            <span style="color: {c['text_muted']};">→</span>
            <span style="background: {c['bg_tertiary']}; color: {c['text_secondary']};
                        padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.8rem;">
                ④ Exportar
            </span>
            <span style="color: {c['text_muted']};">→</span>
            <span style="background: {c['bg_tertiary']}; color: {c['text_secondary']};
                        padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.8rem;">
                ⑤ Combinar
            </span>
            <span style="color: {c['text_muted']};">→</span>
            <span style="background: {c['bg_tertiary']}; color: {c['text_secondary']};
                        padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.8rem;">
                ⑥ Splits
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_quick_access():
    """Render quick access buttons"""
    c = get_colors_dict()

    st.markdown(f"""
    <div style="font-size: 0.75rem; color: {c['text_muted']}; text-transform: uppercase;
                letter-spacing: 0.05em; margin-bottom: 0.75rem;">
        Acceso Rápido
    </div>
    """, unsafe_allow_html=True)

    # Quick access cards as buttons
    quick_actions = [
        ("📊", "Análisis", "Analizar cualquier dataset", "① Análisis"),
        ("🏷️", "Etiquetas", "Gestionar labels", "🏷️ Etiquetas"),
        ("✂️", "Splits", "Crear train/val/test", "⑥ Splits"),
        ("📤", "Exportar", "Convertir formatos", "④ Exportar"),
    ]

    for icon, title, desc, nav_key in quick_actions:
        st.markdown(f"""
        <div style="background: {c['bg_card']}; border: 1px solid {c['border']};
                    border-radius: 0.5rem; padding: 0.75rem; margin-bottom: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="font-size: 1.5rem;">{icon}</span>
                <div>
                    <div style="font-weight: 600; font-size: 0.9rem; color: {c['text_primary']};">
                        {title}
                    </div>
                    <div style="font-size: 0.75rem; color: {c['text_muted']};">
                        {desc}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(f"Ir a {title}", key=f"quick_{title}", use_container_width=True):
            st.session_state.nav_menu = nav_key
            st.rerun()


def _render_system_status():
    """Render system/services status"""
    c = get_colors_dict()
    section_header("Estado del Sistema", icon="🔧")

    client = get_api_client()
    health = client.get_health()

    if health.get("error"):
        alert_box(
            "Gateway no disponible. Verifica que los servicios estén ejecutándose.",
            type="error",
            icon="🔌"
        )
        return

    services = health.get("services", [])
    overall_status = health.get("status", "unknown")

    # Overall status indicator
    if overall_status == "healthy":
        status_color = c['success']
        status_text = "Todos los servicios operativos"
        status_icon = "🟢"
    elif overall_status == "degraded":
        status_color = c['warning']
        status_text = "Algunos servicios con problemas"
        status_icon = "🟡"
    else:
        status_color = c['error']
        status_text = "Servicios no disponibles"
        status_icon = "🔴"

    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
        <span style="font-size: 1.25rem;">{status_icon}</span>
        <span style="font-weight: 600; color: {status_color};">{status_text}</span>
    </div>
    """, unsafe_allow_html=True)

    # Service cards in a row
    if services:
        cols = st.columns(min(len(services), 5))

        for i, service in enumerate(services):
            with cols[i % 5]:
                name = service.get("name", "unknown")
                status = service.get("status", "unknown")
                latency = service.get("latency_ms", 0)

                if status == "healthy":
                    bg_color = c['success_bg']
                    border_color = c['success']
                    icon = "✓"
                elif status == "degraded":
                    bg_color = c['warning_bg']
                    border_color = c['warning']
                    icon = "!"
                else:
                    bg_color = c['error_bg']
                    border_color = c['error']
                    icon = "✗"

                st.markdown(f"""
                <div style="background: {bg_color}; border: 1px solid {border_color};
                            border-radius: 0.5rem; padding: 0.75rem; text-align: center;">
                    <div style="font-weight: 600; font-size: 0.85rem; color: {c['text_primary']};">
                        {icon} {name.capitalize()}
                    </div>
                    <div style="font-size: 0.7rem; color: {c['text_muted']}; margin-top: 0.25rem;">
                        {latency:.0f}ms
                    </div>
                </div>
                """, unsafe_allow_html=True)


def _render_session_info():
    """Render current session information"""
    c = get_colors_dict()
    section_header("Sesión Actual", icon="📋")

    # Check for active workflow data
    has_source = st.session_state.get("source_dataset") is not None
    has_generated = st.session_state.get("generated_dataset") is not None
    has_combined = st.session_state.get("combined_dataset") is not None
    current_step = st.session_state.get("workflow_step", 0)
    current_job = st.session_state.get("current_job_id")

    if not has_source and not has_generated:
        st.markdown(f"""
        <div style="background: {c['bg_secondary']}; border-radius: 0.5rem;
                    padding: 1.5rem; text-align: center; border: 1px dashed {c['border']};">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📭</div>
            <div style="color: {c['text_muted']};">
                No hay workflow activo. Sube un dataset para comenzar.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Session stats
    col1, col2, col3 = st.columns(3)

    with col1:
        if has_source:
            source = st.session_state.source_dataset
            filename = st.session_state.get("source_filename", "dataset.json")
            n_imgs = len(source.get('images', []))

            st.markdown(f"""
            <div style="background: {c['bg_card']}; border: 1px solid {c['border']};
                        border-radius: 0.5rem; padding: 1rem;">
                <div style="font-size: 0.7rem; color: {c['text_muted']}; text-transform: uppercase;">
                    Dataset Origen
                </div>
                <div style="font-weight: 600; font-size: 1rem; margin-top: 0.25rem;">
                    {filename}
                </div>
                <div style="font-size: 0.85rem; color: {c['text_secondary']};">
                    {n_imgs:,} imágenes
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        if current_job:
            st.markdown(f"""
            <div style="background: {c['bg_card']}; border: 1px solid {c['border']};
                        border-radius: 0.5rem; padding: 1rem;">
                <div style="font-size: 0.7rem; color: {c['text_muted']}; text-transform: uppercase;">
                    Job Actual
                </div>
                <div style="font-weight: 600; font-size: 1rem; margin-top: 0.25rem;">
                    {current_job[:12]}...
                </div>
                <div style="font-size: 0.85rem; color: {c['warning']};">
                    En progreso
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif has_generated:
            gen = st.session_state.generated_dataset
            n_gen = len(gen.get('images', []))

            st.markdown(f"""
            <div style="background: {c['bg_card']}; border: 1px solid {c['border']};
                        border-radius: 0.5rem; padding: 1rem;">
                <div style="font-size: 0.7rem; color: {c['text_muted']}; text-transform: uppercase;">
                    Dataset Generado
                </div>
                <div style="font-weight: 600; font-size: 1rem; margin-top: 0.25rem; color: {c['success']};">
                    ✓ Completado
                </div>
                <div style="font-size: 0.85rem; color: {c['text_secondary']};">
                    {n_gen:,} imágenes sintéticas
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col3:
        if current_step > 0:
            st.markdown(f"""
            <div style="background: {c['bg_card']}; border: 1px solid {c['border']};
                        border-radius: 0.5rem; padding: 1rem;">
                <div style="font-size: 0.7rem; color: {c['text_muted']}; text-transform: uppercase;">
                    Progreso Workflow
                </div>
                <div style="font-weight: 600; font-size: 1.5rem; margin-top: 0.25rem; color: {c['primary']};">
                    {current_step}/6
                </div>
                <div style="font-size: 0.85rem; color: {c['text_secondary']};">
                    pasos completados
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Reset workflow button
    spacer(16)
    if st.button("🔄 Reiniciar Workflow", use_container_width=True):
        # Clear workflow-related session state
        keys_to_clear = [
            'source_dataset', 'source_filename', 'workflow_step', 'workflow_completed',
            'analysis_result', 'generation_config', 'current_job_id',
            'generated_dataset', 'combined_dataset', 'final_splits'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
