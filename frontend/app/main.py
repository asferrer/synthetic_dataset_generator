"""
Synthetic Data Generator - Unified Frontend
============================================
Professional UI for synthetic dataset generation with microservices.
"""

import os
import streamlit as st
from pathlib import Path

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Synthetic Data Generator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/asferrer/synthetic_dataset_generator/issues",
        "About": "Synthetic Data Generator v2.0 - Professional Edition"
    }
)

# Import styles
from app.config.styles import inject_styles
from app.config.theme import render_theme_toggle
from app.components.ui import page_header, section_header, spacer, service_card, alert_box, empty_state

# Inject custom styles
inject_styles()


# =============================================================================
# NAVIGATION CONFIGURATION
# =============================================================================

# Workflow steps (main flow)
WORKFLOW_NAV = {
    "① Análisis": "analysis",
    "② Configurar": "configure",
    "③ Generar": "generate",
    "④ Exportar": "export",
    "⑤ Combinar": "combine",
    "⑥ Splits": "splits",
}

# Tools (independent access)
TOOLS_NAV = {
    "🏷️ Etiquetas": "labels",
    "📤 Exportar": "export_tool",
    "🔗 Combinar": "combine_tool",
    "✂️ Splits": "splits_tool",
    "📊 Monitor": "monitor_tool",
    "🔧 Servicios": "services",
    "📚 Docs": "docs",
}


# =============================================================================
# SIDEBAR: Navigation + Service Status
# =============================================================================

def render_sidebar():
    """Render sidebar with navigation and service status"""
    from app.config.theme import get_colors_dict
    c = get_colors_dict()

    with st.sidebar:
        # App branding
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid {c['border']}; margin-bottom: 1rem;">
            <div style="font-size: 2.5rem;">🔬</div>
            <div style="font-size: 1.1rem; font-weight: 700; margin-top: 0.25rem; color: {c['text_primary']};">SDG</div>
            <div style="font-size: 0.7rem; color: {c['text_muted']};">Synthetic Data Generator</div>
        </div>
        """, unsafe_allow_html=True)

        # Home button
        if st.button("🏠 Home", key="nav_home", use_container_width=True):
            st.session_state.nav_menu = "🏠 Home"

        st.markdown(f"<hr style='border: none; border-top: 1px solid {c['border']}; margin: 0.75rem 0;'>",
                    unsafe_allow_html=True)

        # Workflow section
        st.markdown(f"""
        <div style="font-size: 0.7rem; color: {c['text_muted']}; text-transform: uppercase;
                    letter-spacing: 0.05em; margin-bottom: 0.5rem;">
            Workflow Principal
        </div>
        """, unsafe_allow_html=True)

        # Get current workflow step for highlighting
        current_step = st.session_state.get("workflow_step", 0)
        completed_steps = st.session_state.get("workflow_completed", [])

        # Workflow navigation with styled step indicators
        step_names = ["Análisis", "Configurar", "Generar", "Exportar", "Combinar", "Splits"]

        for i, (label, _) in enumerate(WORKFLOW_NAV.items(), 1):
            is_current = st.session_state.get("nav_menu") == label
            is_completed = i in completed_steps
            is_accessible = i <= current_step + 1 or current_step == 0

            # Build display label with step indicator
            if is_completed:
                # Green checkmark for completed steps
                display_label = f"✅ {step_names[i-1]}"
            else:
                # Number circle for non-completed steps
                number_icons = ["①", "②", "③", "④", "⑤", "⑥"]
                display_label = f"{number_icons[i-1]} {step_names[i-1]}"

            button_type = "primary" if is_current else "secondary"

            if st.button(
                display_label,
                key=f"nav_{label}",
                use_container_width=True,
                type=button_type,
                disabled=not is_accessible and current_step > 0
            ):
                st.session_state.nav_menu = label

        st.markdown(f"<hr style='border: none; border-top: 1px solid {c['border']}; margin: 0.75rem 0;'>",
                    unsafe_allow_html=True)

        # Tools section
        st.markdown(f"""
        <div style="font-size: 0.7rem; color: {c['text_muted']}; text-transform: uppercase;
                    letter-spacing: 0.05em; margin-bottom: 0.5rem;">
            Herramientas
        </div>
        """, unsafe_allow_html=True)

        for label, _ in TOOLS_NAV.items():
            is_current = st.session_state.get("nav_menu") == label
            button_type = "primary" if is_current else "secondary"

            if st.button(label, key=f"nav_{label}", use_container_width=True, type=button_type):
                st.session_state.nav_menu = label

        st.markdown(f"<hr style='border: none; border-top: 1px solid {c['border']}; margin: 0.75rem 0;'>",
                    unsafe_allow_html=True)

        # Theme toggle
        st.markdown(f"""
        <div style="font-size: 0.7rem; color: {c['text_muted']}; text-transform: uppercase;
                    letter-spacing: 0.05em; margin-bottom: 0.5rem;">
            Tema
        </div>
        """, unsafe_allow_html=True)
        render_theme_toggle()

        st.markdown(f"<hr style='border: none; border-top: 1px solid {c['border']}; margin: 0.75rem 0;'>",
                    unsafe_allow_html=True)

        # Service Status (always visible)
        render_service_status_widget()

        # Return current page
        return st.session_state.get("nav_menu", "🏠 Home")


def render_service_status_widget():
    """Render compact service status in sidebar"""
    from app.config.theme import get_colors_dict
    c = get_colors_dict()

    st.markdown(f"""
    <div style="font-size: 0.7rem; color: {c['text_muted']}; text-transform: uppercase;
                letter-spacing: 0.05em; margin-bottom: 0.5rem;">
        Estado de Servicios
    </div>
    """, unsafe_allow_html=True)

    from app.components.api_client import get_api_client
    client = get_api_client()
    health = client.get_health()

    if health and "services" in health:
        services = health.get("services", [])

        # Overall status
        overall_status = health.get("status", "unknown")
        if overall_status == "healthy":
            st.success("Sistema Operativo", icon="🟢")
        elif overall_status == "degraded":
            st.warning("Rendimiento Reducido", icon="🟡")
        else:
            st.error("Servicios No Disponibles", icon="🔴")

        # Compact service indicators
        if services:
            status_html = '<div style="display: flex; flex-wrap: wrap; gap: 4px; margin-top: 0.5rem;">'
            for service in services:
                status = service.get("status", "unknown")
                name = service.get("name", "unknown")[:4].capitalize()

                if status == "healthy":
                    icon = "🟢"
                    bg_color = "#ECFDF5"
                    border_color = "#10B981"
                elif status == "degraded":
                    icon = "🟡"
                    bg_color = "#FFFBEB"
                    border_color = "#F59E0B"
                else:
                    icon = "🔴"
                    bg_color = "#FEF2F2"
                    border_color = "#EF4444"

                status_html += f'''<div style="display: inline-flex; align-items: center; gap: 3px;
                    padding: 3px 8px; background: {bg_color}; border-radius: 4px; font-size: 11px;
                    border: 1px solid {border_color}; color: #1E293B;">{icon} {name}</div>'''
            status_html += '</div>'
            st.markdown(status_html, unsafe_allow_html=True)
    else:
        st.error("Gateway no disponible", icon="🔴")


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point"""

    # Initialize session state
    if "nav_menu" not in st.session_state:
        st.session_state.nav_menu = "🏠 Home"
    if "workflow_step" not in st.session_state:
        st.session_state.workflow_step = 0
    if "workflow_completed" not in st.session_state:
        st.session_state.workflow_completed = []

    # Render sidebar and get current page
    current_page = render_sidebar()

    # Route to appropriate page
    if current_page == "🏠 Home":
        render_home_page()

    # Workflow pages
    elif current_page == "① Análisis":
        render_analysis_page()
    elif current_page == "② Configurar":
        render_configure_page()
    elif current_page == "③ Generar":
        render_generation_page()
    elif current_page == "④ Exportar":
        render_export_page()
    elif current_page == "⑤ Combinar":
        render_combine_page()
    elif current_page == "⑥ Splits":
        render_splits_page()

    # Tools pages
    elif current_page == "🏷️ Etiquetas":
        render_labels_page()
    elif current_page == "📤 Exportar":
        render_export_tool_page()
    elif current_page == "🔗 Combinar":
        render_combine_tool_page()
    elif current_page == "✂️ Splits":
        render_splits_tool_page()
    elif current_page == "📊 Monitor":
        render_monitor_tool_page()
    elif current_page == "🔧 Servicios":
        render_services_page()
    elif current_page == "📚 Docs":
        render_documentation_page()


# =============================================================================
# PAGE: Home
# =============================================================================

def render_home_page():
    """Render home dashboard page"""
    from app.pages.home import render_home_page as home
    home()


# =============================================================================
# PAGE: Analysis (Step 1)
# =============================================================================

def render_analysis_page():
    """Render analysis page (Step 1 of workflow)"""
    from app.pages.analysis import render_analysis_page as analysis
    analysis()


# =============================================================================
# PAGE: Configure (Step 2)
# =============================================================================

def render_configure_page():
    """Render configuration page (Step 2 of workflow)"""
    # Check if we have data from analysis
    if not st.session_state.get("source_dataset"):
        alert_box(
            "Primero debes cargar y analizar un dataset en el paso anterior.",
            type="warning",
            icon="⚠️"
        )
        if st.button("← Ir a Análisis", type="primary"):
            st.session_state.nav_menu = "① Análisis"
            st.rerun()
        return

    from app.pages.configure import render_configure_page as configure
    configure()


# =============================================================================
# PAGE: Generation (Step 3)
# =============================================================================

def render_generation_page():
    """Render generation page (Step 3 of workflow)"""
    # Check if we have configuration
    if not st.session_state.get("generation_config"):
        alert_box(
            "Primero debes configurar la generación en el paso anterior.",
            type="warning",
            icon="⚠️"
        )
        if st.button("← Ir a Configuración", type="primary"):
            st.session_state.nav_menu = "② Configurar"
            st.rerun()
        return

    from app.pages.generation import render_generation_page as generation
    generation()


# =============================================================================
# PAGE: Export (Step 4)
# =============================================================================

def render_export_page():
    """Render export page (Step 4 of workflow)"""
    from app.pages.export import render_export_page as export
    export()


# =============================================================================
# PAGE: Combine (Step 5)
# =============================================================================

def render_combine_page():
    """Render combine datasets page (Step 5 of workflow)"""
    from app.pages.combine import render_combine_page as combine
    combine()


# =============================================================================
# PAGE: Splits (Step 6)
# =============================================================================

def render_splits_page():
    """Render splits page (Step 6 of workflow)"""
    from app.pages.splits import render_splits_page as splits
    splits()


# =============================================================================
# PAGE: Labels (Tool)
# =============================================================================

def render_labels_page():
    """Render labels management page"""
    from app.pages.post_processing import render_labels_section
    page_header(
        title="Gestión de Etiquetas",
        subtitle="Renombrar, eliminar, fusionar y añadir etiquetas a tu dataset",
        icon="🏷️"
    )
    render_labels_section()


# =============================================================================
# PAGE: Export Tool (Standalone)
# =============================================================================

def render_export_tool_page():
    """Render export page as standalone tool (not part of workflow)"""

    page_header(
        title="Exportar Dataset",
        subtitle="Exporta cualquier dataset COCO a diferentes formatos (YOLO, Pascal VOC, etc.)",
        icon="📤"
    )

    # Initialize session state for tool
    if "export_tool_data" not in st.session_state:
        st.session_state.export_tool_data = None

    # Dataset loader section
    section_header("Cargar Dataset", icon="📁")

    upload_method = st.radio(
        "Método de entrada",
        ["Subir archivo JSON", "Usar dataset del workflow"],
        horizontal=True,
        key="export_tool_input_method"
    )

    if upload_method == "Subir archivo JSON":
        uploaded = st.file_uploader(
            "Arrastra tu archivo COCO JSON aquí",
            type=["json"],
            key="export_tool_upload"
        )
        if uploaded:
            import json
            try:
                st.session_state.export_tool_data = json.load(uploaded)
                st.success(f"✓ Cargado: {uploaded.name}")
            except Exception as e:
                st.error(f"Error al cargar: {e}")
    else:
        # Use data from workflow
        if st.session_state.get("generated_dataset"):
            st.session_state.export_tool_data = st.session_state.generated_dataset
            st.success("✓ Usando dataset generado del workflow")
        elif st.session_state.get("source_dataset"):
            st.session_state.export_tool_data = st.session_state.source_dataset
            st.success("✓ Usando dataset original del workflow")
        else:
            st.warning("No hay dataset en el workflow. Sube un archivo JSON.")

    spacer(16)

    if st.session_state.export_tool_data:
        from app.pages.export import _render_dataset_summary, _perform_export, _render_export_results

        dataset = st.session_state.export_tool_data
        _render_dataset_summary(dataset, "")

        spacer(16)
        section_header("Formatos de Exportación", icon="📦")

        col1, col2 = st.columns(2)
        with col1:
            export_coco = st.checkbox("📋 COCO JSON", value=True, key="et_coco")
            export_yolo = st.checkbox("🔲 YOLO (txt + yaml)", value=True, key="et_yolo")
            export_voc = st.checkbox("📄 Pascal VOC (xml)", value=False, key="et_voc")

        with col2:
            export_output_dir = st.text_input(
                "Directorio de exportación",
                value="/app/output/exported",
                key="et_output_dir"
            )

        formats = []
        if export_coco: formats.append("coco")
        if export_yolo: formats.append("yolo")
        if export_voc: formats.append("pascal_voc")

        if st.button("🚀 Exportar", type="primary", use_container_width=True, key="et_export_btn"):
            _perform_export(dataset, export_output_dir, formats, False, "")

        if st.session_state.get("export_results"):
            _render_export_results(st.session_state.export_results)
    else:
        empty_state(
            title="Sin dataset cargado",
            message="Carga un archivo COCO JSON para exportar.",
            icon="📤"
        )


# =============================================================================
# PAGE: Combine Tool (Standalone)
# =============================================================================

def render_combine_tool_page():
    """Render combine datasets page as standalone tool"""
    from app.pages.combine import render_combine_page as combine
    combine()


# =============================================================================
# PAGE: Splits Tool (Standalone)
# =============================================================================

def render_splits_tool_page():
    """Render splits page as standalone tool"""
    from app.pages.post_processing import _render_splits_section

    page_header(
        title="Crear Splits",
        subtitle="Divide cualquier dataset COCO en Train/Val/Test o K-Fold",
        icon="✂️"
    )

    # Initialize session state
    if "splits_tool_data" not in st.session_state:
        st.session_state.splits_tool_data = None

    section_header("Cargar Dataset", icon="📁")

    upload_method = st.radio(
        "Método de entrada",
        ["Subir archivo JSON", "Usar dataset del workflow"],
        horizontal=True,
        key="splits_tool_input_method"
    )

    if upload_method == "Subir archivo JSON":
        uploaded = st.file_uploader(
            "Arrastra tu archivo COCO JSON aquí",
            type=["json"],
            key="splits_tool_upload"
        )
        if uploaded:
            import json
            try:
                st.session_state.pp_coco_data = json.load(uploaded)
                st.success(f"✓ Cargado: {uploaded.name}")
            except Exception as e:
                st.error(f"Error al cargar: {e}")
    else:
        if st.session_state.get("generated_dataset"):
            st.session_state.pp_coco_data = st.session_state.generated_dataset
            st.success("✓ Usando dataset generado del workflow")
        elif st.session_state.get("combined_dataset"):
            st.session_state.pp_coco_data = st.session_state.combined_dataset
            st.success("✓ Usando dataset combinado del workflow")
        elif st.session_state.get("source_dataset"):
            st.session_state.pp_coco_data = st.session_state.source_dataset
            st.success("✓ Usando dataset original del workflow")
        else:
            st.warning("No hay dataset en el workflow. Sube un archivo JSON.")

    spacer(16)

    if st.session_state.get("pp_coco_data"):
        _render_splits_section()
    else:
        empty_state(
            title="Sin dataset cargado",
            message="Carga un archivo COCO JSON para crear splits.",
            icon="✂️"
        )


# =============================================================================
# PAGE: Monitor Tool (Standalone)
# =============================================================================

def render_monitor_tool_page():
    """Render the job monitoring tool page - shows all background generation jobs"""
    from app.config.theme import get_colors_dict
    c = get_colors_dict()

    page_header(
        title="Monitor de Jobs",
        subtitle="Monitoriza todos los trabajos de generación sintética en ejecución o completados",
        icon="📊"
    )

    # Manual refresh button
    if st.button("🔄 Actualizar Estado", key="monitor_refresh_btn"):
        st.rerun()

    spacer(16)

    # Fetch all jobs
    from app.components.api_client import get_api_client
    client = get_api_client()
    jobs_response = client.list_jobs()

    if jobs_response.get("error"):
        alert_box(f"Error al obtener jobs: {jobs_response.get('error')}", type="error")
        return

    jobs = jobs_response.get("jobs", [])

    if not jobs:
        empty_state(
            title="No hay trabajos",
            message="No hay trabajos de generación en el sistema. Inicia una generación desde el workflow.",
            icon="📭"
        )
        return

    # Categorize jobs
    active_jobs = [j for j in jobs if j.get("status") in ["processing", "queued", "pending"]]
    completed_jobs = [j for j in jobs if j.get("status") == "completed"]
    failed_jobs = [j for j in jobs if j.get("status") in ["failed", "cancelled", "error"]]

    # Summary cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div style="background: {c['bg_card']}; border: 1px solid {c['border']};
                    border-radius: 0.5rem; padding: 1rem; text-align: center;
                    border-top: 3px solid {c['primary']};">
            <div style="font-size: 2rem; font-weight: 700; color: {c['primary']};">{len(jobs)}</div>
            <div style="font-size: 0.8rem; color: {c['text_muted']};">Total Jobs</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background: {c['bg_card']}; border: 1px solid {c['border']};
                    border-radius: 0.5rem; padding: 1rem; text-align: center;
                    border-top: 3px solid {c['warning']};">
            <div style="font-size: 2rem; font-weight: 700; color: {c['warning']};">{len(active_jobs)}</div>
            <div style="font-size: 0.8rem; color: {c['text_muted']};">En Progreso</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="background: {c['bg_card']}; border: 1px solid {c['border']};
                    border-radius: 0.5rem; padding: 1rem; text-align: center;
                    border-top: 3px solid {c['success']};">
            <div style="font-size: 2rem; font-weight: 700; color: {c['success']};">{len(completed_jobs)}</div>
            <div style="font-size: 0.8rem; color: {c['text_muted']};">Completados</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div style="background: {c['bg_card']}; border: 1px solid {c['border']};
                    border-radius: 0.5rem; padding: 1rem; text-align: center;
                    border-top: 3px solid {c['error']};">
            <div style="font-size: 2rem; font-weight: 700; color: {c['error']};">{len(failed_jobs)}</div>
            <div style="font-size: 0.8rem; color: {c['text_muted']};">Fallidos</div>
        </div>
        """, unsafe_allow_html=True)

    spacer(24)

    # Active jobs section
    if active_jobs:
        section_header("Jobs Activos", icon="⏳")

        for job in active_jobs:
            _render_job_card(job, c, client, is_active=True)

        spacer(16)

    # Completed jobs section
    if completed_jobs:
        section_header("Jobs Completados", icon="✅")

        for job in completed_jobs[:10]:  # Limit to 10 most recent
            _render_job_card(job, c, client, is_active=False)

        if len(completed_jobs) > 10:
            st.caption(f"Mostrando 10 de {len(completed_jobs)} jobs completados")

        spacer(16)

    # Failed jobs section
    if failed_jobs:
        with st.expander(f"❌ Jobs Fallidos ({len(failed_jobs)})", expanded=False):
            for job in failed_jobs[:5]:
                _render_job_card(job, c, client, is_active=False)


def _render_job_card(job: dict, c: dict, client, is_active: bool = False) -> None:
    """Render a single job card"""
    job_id = job.get("job_id", "unknown")
    status = job.get("status", "unknown")
    generated = job.get("images_generated", 0)
    rejected = job.get("images_rejected", 0)
    pending = job.get("images_pending", 0)
    total = generated + rejected + pending if pending > 0 else generated + rejected
    output_dir = job.get("output_dir", "")
    created_at = job.get("created_at", "")
    error = job.get("error", "")

    # Status styling
    if status == "completed":
        status_color, status_icon = c['success'], "✅"
    elif status == "processing":
        status_color, status_icon = c['warning'], "⏳"
    elif status == "queued":
        status_color, status_icon = c['info'], "📋"
    elif status == "failed":
        status_color, status_icon = c['error'], "❌"
    elif status == "cancelled":
        status_color, status_icon = c['text_muted'], "⏹️"
    else:
        status_color, status_icon = c['text_muted'], "❔"

    progress = generated / total if total > 0 else 0

    st.markdown(f"""
    <div style="background: {c['bg_card']}; border: 1px solid {c['border']};
                border-radius: 0.5rem; padding: 1rem; margin-bottom: 0.75rem;
                border-left: 4px solid {status_color};">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.25rem;">{status_icon}</span>
                <span style="font-family: monospace; font-weight: 600; color: {c['text_primary']};">
                    {job_id[:20]}...
                </span>
            </div>
            <span style="font-size: 0.75rem; color: {status_color}; font-weight: 600;">
                {status.upper()}
            </span>
        </div>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; font-size: 0.8rem;">
            <div>
                <span style="color: {c['text_muted']};">Generadas:</span>
                <span style="color: {c['success']}; font-weight: 600;"> {generated}</span>
            </div>
            <div>
                <span style="color: {c['text_muted']};">Rechazadas:</span>
                <span style="color: {c['error']}; font-weight: 600;"> {rejected}</span>
            </div>
            <div>
                <span style="color: {c['text_muted']};">Pendientes:</span>
                <span style="font-weight: 600;"> {pending}</span>
            </div>
            <div>
                <span style="color: {c['text_muted']};">Progreso:</span>
                <span style="color: {c['primary']}; font-weight: 600;"> {progress*100:.0f}%</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show progress bar for active jobs
    if is_active and total > 0:
        st.progress(progress)

    # Action buttons for active jobs
    if is_active:
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("👁️ Ver Detalles", key=f"view_{job_id}", use_container_width=True):
                st.session_state.current_job_id = job_id
                st.session_state.nav_menu = "③ Generar"
                st.rerun()
        with col2:
            if st.button("⏹️ Cancelar", key=f"cancel_{job_id}", use_container_width=True):
                result = client.cancel_job(job_id)
                if result.get("success"):
                    st.toast(f"Job {job_id[:8]}... cancelado")
                    st.rerun()
                else:
                    st.error(f"Error: {result.get('error')}")

    # Show error if present
    if error:
        st.error(f"Error: {error}")


# =============================================================================
# PAGE: Services
# =============================================================================

def render_services_page():
    """Render services information page"""
    from app.config.theme import get_colors_dict
    c = get_colors_dict()

    page_header(
        title="Estado de Servicios",
        subtitle="Monitorea el estado y rendimiento de los microservicios",
        icon="🔧"
    )

    from app.components.api_client import get_api_client
    client = get_api_client()
    health = client.get_health()

    if health.get("error"):
        alert_box(
            f"Gateway no disponible: {health.get('error')}",
            type="error",
            icon="🔌"
        )
        empty_state(
            title="No se puede conectar al Gateway",
            message="Asegúrate de que el API Gateway esté ejecutándose y accesible.",
            icon="🔌"
        )
        return

    # Overall status
    status = health.get("status", "unknown")
    if status == "healthy":
        st.success("🟢 **Estado del Sistema: SALUDABLE** - Todos los servicios operativos")
    elif status == "degraded":
        st.warning("🟡 **Estado del Sistema: DEGRADADO** - Algunos servicios con problemas")
    else:
        st.error("🔴 **Estado del Sistema: NO SALUDABLE** - Servicios críticos no disponibles")

    spacer(24)
    section_header("Microservicios", icon="🐳")

    services = health.get("services", [])

    if services:
        cols = st.columns(min(len(services), 4))

        for i, service in enumerate(services):
            with cols[i % 4]:
                name = service.get("name", "unknown")
                svc_status = service.get("status", "unknown")
                latency = service.get("latency_ms", 0)

                ports = {
                    "depth": 8001,
                    "segmentation": 8002,
                    "effects": 8003,
                    "augmentor": 8004,
                }
                port = ports.get(name.lower())

                service_card(
                    name=name,
                    status=svc_status,
                    latency=latency,
                    port=port
                )

                if st.button(f"Detalles", key=f"svc_details_{name}", use_container_width=True):
                    info = client.get_service_info(name)
                    if info:
                        st.json(info)
    else:
        empty_state(
            title="No hay servicios",
            message="No hay microservicios registrados en el gateway.",
            icon="📭"
        )

    spacer(32)
    section_header("Documentación API", icon="📖")

    gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:8000")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📘</div>
            <div style="font-weight: 600; color: {c['text_primary']};">Swagger UI</div>
            <div style="font-size: 0.875rem; color: {c['text_muted']};">Explorador interactivo</div>
            <a href="{gateway_url}/docs" target="_blank" style="display: inline-block; margin-top: 0.75rem;
                color: {c['primary']};">Abrir Swagger →</a>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📗</div>
            <div style="font-weight: 600; color: {c['text_primary']};">ReDoc</div>
            <div style="font-size: 0.875rem; color: {c['text_muted']};">Documentación limpia</div>
            <a href="{gateway_url}/redoc" target="_blank" style="display: inline-block; margin-top: 0.75rem;
                color: {c['primary']};">Abrir ReDoc →</a>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📙</div>
            <div style="font-weight: 600; color: {c['text_primary']};">OpenAPI Spec</div>
            <div style="font-size: 0.875rem; color: {c['text_muted']};">Especificación JSON</div>
            <a href="{gateway_url}/openapi.json" target="_blank" style="display: inline-block; margin-top: 0.75rem;
                color: {c['primary']};">Descargar JSON →</a>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE: Documentation
# =============================================================================

def render_documentation_page():
    """Render documentation page"""
    page_header(
        title="Documentación",
        subtitle="Guías de arquitectura, referencia de efectos y tutoriales",
        icon="📚"
    )

    # Sub-navigation for docs
    doc_section = st.radio(
        "Sección",
        ["🏗️ Arquitectura", "✨ Efectos", "📖 Guía de Uso"],
        horizontal=True,
        key="docs_section"
    )

    spacer(16)

    if doc_section == "🏗️ Arquitectura":
        _render_architecture_docs()
    elif doc_section == "✨ Efectos":
        _render_effects_docs()
    else:
        _render_usage_guide()


def _render_architecture_docs():
    """Render architecture documentation"""
    st.markdown("""
    ## Arquitectura de Microservicios

    El Generador de Datos Sintéticos utiliza una arquitectura distribuida de microservicios
    para escalabilidad y modularidad.
    """)

    st.code("""
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                        FRONTEND (Streamlit - 8501)                        │
    └────────────────────────────────────┬─────────────────────────────────────┘
                                         │ HTTP/REST
                                         ▼
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                      API GATEWAY (FastAPI - 8000)                         │
    └──────┬─────────────┬─────────────┬─────────────┬─────────────────────────┘
           │             │             │             │
           ▼             ▼             ▼             ▼
    ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
    │   DEPTH    │ │  EFFECTS   │ │ AUGMENTOR  │ │SEGMENTATION│
    │   (8001)   │ │   (8003)   │ │   (8004)   │ │   (8002)   │
    │  GPU: Yes  │ │  GPU: No   │ │  GPU: Yes  │ │  GPU: Yes  │
    └────────────┘ └────────────┘ └────────────┘ └────────────┘
    """, language="text")

    spacer(16)

    import pandas as pd
    services_data = [
        {"Servicio": "Gateway", "Puerto": "8000", "GPU": "❌", "Descripción": "Orquestación y routing de APIs"},
        {"Servicio": "Depth", "Puerto": "8001", "GPU": "✅", "Descripción": "Estimación de profundidad"},
        {"Servicio": "Segmentation", "Puerto": "8002", "GPU": "✅", "Descripción": "Segmentación de objetos"},
        {"Servicio": "Effects", "Puerto": "8003", "GPU": "❌", "Descripción": "Pipeline de efectos de realismo"},
        {"Servicio": "Augmentor", "Puerto": "8004", "GPU": "✅", "Descripción": "Composición y validación"},
        {"Servicio": "Frontend", "Puerto": "8501", "GPU": "❌", "Descripción": "Interfaz de usuario Streamlit"},
    ]
    st.dataframe(pd.DataFrame(services_data), use_container_width=True, hide_index=True)


def _render_effects_docs():
    """Render effects documentation"""
    st.markdown("""
    ## Efectos Disponibles

    El pipeline ofrece múltiples efectos de mejora fotorealista.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Efectos Principales
        | Efecto | Rango | Descripción |
        |--------|-------|-------------|
        | **Color Correction** | 0.0 - 1.0 | Transferencia de color LAB |
        | **Blur Matching** | 0.0 - 3.0 | Igualar niveles de enfoque |
        | **Shadows** | 0.0 - 1.0 | Generación de sombras |
        | **Edge Smoothing** | 1 - 20px | Anti-aliasing de bordes |
        """)

    with col2:
        st.markdown("""
        ### Efectos Avanzados
        | Efecto | Rango | Descripción |
        |--------|-------|-------------|
        | **Caustics** | 0.0 - 0.5 | Patrones de luz subacuática |
        | **Underwater** | 0.0 - 1.0 | Tinte de color de agua |
        | **Motion Blur** | 0.0 - 1.0 | Simulación de movimiento |
        | **Poisson Blend** | On/Off | Clonación seamless |
        """)


def _render_usage_guide():
    """Render usage guide"""
    st.markdown("""
    ## Guía Rápida - Workflow de 6 Pasos

    ### ① Análisis
    1. Sube tu archivo JSON COCO desde **Home** o **Análisis**
    2. Revisa la distribución de clases y el ratio de desbalance
    3. El sistema calcula automáticamente cuántas imágenes sintéticas necesitas

    ### ② Configuración
    1. Selecciona qué clases balancear
    2. Configura los efectos de realismo
    3. Especifica los directorios de fondos y objetos

    ### ③ Generación
    1. Revisa el resumen de configuración
    2. Inicia el batch de generación
    3. Monitorea el progreso en tiempo real

    ### ④ Exportar
    1. Elige los formatos de exportación (COCO, YOLO, Pascal VOC)
    2. Configura el directorio de salida
    3. Opcionalmente copia las imágenes

    ### ⑤ Combinar
    1. Selecciona datasets a combinar (original + sintético)
    2. Configura la estrategia de merge
    3. Genera el dataset combinado

    ### ⑥ Splits
    1. Configura los ratios de train/val/test
    2. Selecciona estratificación
    3. Genera los splits para entrenamiento
    """)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
