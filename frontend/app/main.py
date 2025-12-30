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
    "🎯 Extraer Objetos": "extract_objects",
    "🔬 SAM3": "sam3_tool",
    "🏷️ Etiquetas": "labels",
    "📤 Exportar": "export_tool",
    "🔗 Combinar": "combine_tool",
    "✂️ Splits": "splits_tool",
    "📊 Monitor": "monitor_tool",
    "📏 Tamaños": "object_sizes",
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
    elif current_page == "🎯 Extraer Objetos":
        render_object_extraction_page()
    elif current_page == "🔬 SAM3":
        render_sam3_tool_page()
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
    elif current_page == "📏 Tamaños":
        render_object_sizes_page()
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
# PAGE: Object Extraction (Tool)
# =============================================================================

def render_object_extraction_page():
    """Render object extraction tool page"""
    from app.pages.object_extraction import render_object_extraction_page as extraction
    extraction()


# =============================================================================
# PAGE: SAM3 Tool
# =============================================================================

def render_sam3_tool_page():
    """Render SAM3 segmentation tool page"""
    from app.pages.sam3_tool import render_sam3_tool_page as sam3
    sam3()


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
    """Render the job monitoring tool page - shows all background jobs"""
    import time
    from app.config.theme import get_colors_dict
    c = get_colors_dict()

    page_header(
        title="Monitor de Jobs",
        subtitle="Monitoriza todos los trabajos en ejecución o completados",
        icon="📊"
    )

    # Fetch all jobs from different sources
    from app.components.api_client import get_api_client
    client = get_api_client()

    # Generation jobs (from augmentor)
    gen_jobs_response = client.list_jobs()
    gen_jobs = gen_jobs_response.get("jobs", [])
    for job in gen_jobs:
        job["job_type"] = "generation"

    # Extraction jobs (from segmentation)
    extract_jobs_response = client.list_extraction_jobs()
    extract_jobs = extract_jobs_response.get("jobs", [])

    # SAM3 conversion jobs (from segmentation)
    sam3_jobs_response = client.list_sam3_jobs()
    sam3_jobs = sam3_jobs_response.get("jobs", [])

    # Combine all jobs
    jobs = gen_jobs + extract_jobs + sam3_jobs

    if not jobs:
        empty_state(
            title="No hay trabajos",
            message="No hay trabajos en el sistema. Inicia una generacion, extraccion o conversion SAM3.",
            icon="📭"
        )
        return

    # Categorize jobs by status
    active_jobs = [j for j in jobs if j.get("status") in ["processing", "queued", "pending"]]
    completed_jobs = [j for j in jobs if j.get("status") == "completed"]
    interrupted_jobs = [j for j in jobs if j.get("status") == "interrupted"]
    failed_jobs = [j for j in jobs if j.get("status") in ["failed", "cancelled", "error"]]

    # Count by type
    gen_count = len([j for j in jobs if j.get("job_type") == "generation"])
    extract_count = len([j for j in jobs if j.get("job_type") == "extraction"])
    sam3_count = len([j for j in jobs if j.get("job_type") == "sam3_conversion"])

    # Auto-refresh indicator for active jobs
    if active_jobs:
        col_refresh, col_status = st.columns([3, 1])
        with col_refresh:
            st.caption("🔄 Auto-actualizando cada 2 segundos...")
        with col_status:
            if st.button("⏹️ Pausar", key="pause_refresh"):
                st.session_state["monitor_paused"] = True
                st.rerun()
    else:
        if st.button("🔄 Actualizar Estado", key="monitor_refresh_btn"):
            st.rerun()

    spacer(8)

    # Summary cards - row 1: status
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
        # Show interrupted count if any, otherwise failed
        if interrupted_jobs:
            st.markdown(f"""
            <div style="background: {c['bg_card']}; border: 1px solid {c['border']};
                        border-radius: 0.5rem; padding: 1rem; text-align: center;
                        border-top: 3px solid {c['info']};">
                <div style="font-size: 2rem; font-weight: 700; color: {c['info']};">{len(interrupted_jobs)}</div>
                <div style="font-size: 0.8rem; color: {c['text_muted']};">Interrumpidos</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: {c['bg_card']}; border: 1px solid {c['border']};
                        border-radius: 0.5rem; padding: 1rem; text-align: center;
                        border-top: 3px solid {c['error']};">
                <div style="font-size: 2rem; font-weight: 700; color: {c['error']};">{len(failed_jobs)}</div>
                <div style="font-size: 0.8rem; color: {c['text_muted']};">Fallidos</div>
            </div>
            """, unsafe_allow_html=True)

    # Summary cards - row 2: by type
    spacer(8)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div style="background: {c['bg_secondary']}; border: 1px solid {c['border']};
                    border-radius: 0.5rem; padding: 0.75rem; text-align: center;">
            <div style="font-size: 1.25rem; font-weight: 600;">🎨 {gen_count}</div>
            <div style="font-size: 0.75rem; color: {c['text_muted']};">Generacion</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background: {c['bg_secondary']}; border: 1px solid {c['border']};
                    border-radius: 0.5rem; padding: 0.75rem; text-align: center;">
            <div style="font-size: 1.25rem; font-weight: 600;">🎯 {extract_count}</div>
            <div style="font-size: 0.75rem; color: {c['text_muted']};">Extraccion</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="background: {c['bg_secondary']}; border: 1px solid {c['border']};
                    border-radius: 0.5rem; padding: 0.75rem; text-align: center;">
            <div style="font-size: 1.25rem; font-weight: 600;">🔬 {sam3_count}</div>
            <div style="font-size: 0.75rem; color: {c['text_muted']};">SAM3 Conversion</div>
        </div>
        """, unsafe_allow_html=True)

    spacer(24)

    # Active jobs section - DETAILED VIEW
    if active_jobs:
        section_header("Jobs Activos", icon="⏳")

        for job in active_jobs:
            _render_detailed_job_card(job, c, client)

        spacer(16)

    # Completed jobs section
    if completed_jobs:
        section_header("Jobs Completados", icon="✅")

        for job in completed_jobs[:10]:  # Limit to 10 most recent
            _render_job_card(job, c, client, is_active=False)

        if len(completed_jobs) > 10:
            st.caption(f"Mostrando 10 de {len(completed_jobs)} jobs completados")

        spacer(16)

    # Interrupted jobs section (can be resumed)
    if interrupted_jobs:
        section_header("Jobs Interrumpidos (Reanudables)", icon="⏸️")
        st.info("Estos jobs fueron interrumpidos y pueden reanudarse desde donde se quedaron.")

        for job in interrupted_jobs:
            _render_job_card_with_actions(job, c, client)

        spacer(16)

    # Failed jobs section
    if failed_jobs:
        with st.expander(f"❌ Jobs Fallidos ({len(failed_jobs)})", expanded=False):
            for job in failed_jobs[:5]:
                _render_job_card_with_actions(job, c, client)

    # Auto-refresh for active jobs (at the end to avoid blocking UI)
    if active_jobs and not st.session_state.get("monitor_paused"):
        time.sleep(2)
        st.rerun()
    elif st.session_state.get("monitor_paused"):
        st.session_state.pop("monitor_paused", None)


def _render_detailed_job_card(job: dict, c: dict, client) -> None:
    """Render a detailed job card with full progress visualization for active jobs"""
    from datetime import datetime

    job_id = job.get("job_id", "unknown")
    status = job.get("status", "unknown")
    job_type = job.get("job_type", "generation")

    # Job type styling and data extraction
    if job_type == "generation":
        type_icon, type_label = "🎨", "Generacion"
        done = job.get("images_generated", 0)
        failed = job.get("images_rejected", 0)
        pending = job.get("images_pending", 0)
        total = job.get("total_items", 0)  # Use total_items from backend
        current_info = job.get("current_category", "")
    elif job_type == "extraction":
        type_icon, type_label = "🎯", "Extraccion de Objetos"
        done = job.get("extracted_objects", 0)
        failed = job.get("failed_objects", 0)
        total = job.get("total_objects", 0)
        pending = max(0, total - done - failed)
        current_info = job.get("current_category", "")
    elif job_type == "sam3_conversion":
        type_icon, type_label = "🔬", "Conversion SAM3"
        done = job.get("converted_annotations", 0)
        skipped = job.get("skipped_annotations", 0)
        failed = job.get("failed_annotations", 0)
        total = job.get("total_annotations", 0)
        pending = max(0, total - done - skipped - failed)
        current_info = job.get("current_image", "")
    else:
        type_icon, type_label = "❔", "Desconocido"
        done, failed, pending, total = 0, 0, 0, 0
        current_info = ""

    progress = done / total if total > 0 else 0

    # Calculate elapsed time from started_at (for active jobs) or processing_time_ms (for completed)
    elapsed_seconds = 0
    started_at = job.get("started_at")
    if started_at and status in ["processing", "queued"]:
        try:
            start_time = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            elapsed_seconds = (datetime.now() - start_time.replace(tzinfo=None)).total_seconds()
        except (ValueError, TypeError):
            elapsed_seconds = job.get("processing_time_ms", 0) / 1000
    else:
        elapsed_seconds = job.get("processing_time_ms", 0) / 1000

    # Estimate remaining time
    if done > 0 and elapsed_seconds > 0:
        rate = done / elapsed_seconds  # items per second
        remaining_items = pending
        eta_seconds = remaining_items / rate if rate > 0 else 0
        eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s" if eta_seconds > 60 else f"{int(eta_seconds)}s"
    else:
        eta_str = "Calculando..."

    elapsed_str = f"{int(elapsed_seconds // 60)}m {int(elapsed_seconds % 60)}s" if elapsed_seconds > 60 else f"{int(elapsed_seconds)}s"

    # Main container
    st.markdown(f"""
    <div style="background: {c['bg_card']}; border: 1px solid {c['border']};
                border-radius: 0.75rem; padding: 1.5rem; margin-bottom: 1rem;
                border-left: 4px solid {c['warning']};">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="font-size: 2rem;">{type_icon}</span>
                <div>
                    <div style="font-weight: 700; font-size: 1.1rem; color: {c['text_primary']};">{type_label}</div>
                    <div style="font-family: monospace; font-size: 0.75rem; color: {c['text_muted']};">ID: {job_id[:20]}...</div>
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 2rem; font-weight: 700; color: {c['primary']};">{progress*100:.1f}%</div>
                <div style="font-size: 0.75rem; color: {c['text_muted']};">completado</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Progress bar
    st.progress(progress)

    # Stats row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Completados", f"{done:,}", help=f"De {total:,} total")
    with col2:
        if job_type == "sam3_conversion":
            st.metric("Omitidos", f"{skipped:,}")
        else:
            st.metric("Pendientes", f"{pending:,}")
    with col3:
        st.metric("Fallidos", f"{failed:,}")
    with col4:
        st.metric("Tiempo", elapsed_str)

    # Current activity and ETA
    col_left, col_right = st.columns(2)
    with col_left:
        if current_info:
            st.caption(f"📍 Procesando: **{current_info}**")
        else:
            st.caption("📍 Procesando...")
    with col_right:
        st.caption(f"⏱️ Tiempo restante estimado: **{eta_str}**")

    # Category breakdown for extraction jobs
    if job_type == "extraction":
        categories_progress = job.get("categories_progress", {})
        if categories_progress:
            with st.expander("📊 Progreso por Categoria", expanded=False):
                for cat_name, cat_count in categories_progress.items():
                    st.write(f"• {cat_name}: {cat_count} objetos")

    # Separator
    st.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #333;'>", unsafe_allow_html=True)


def _render_job_card(job: dict, c: dict, client, is_active: bool = False) -> None:
    """Render a single job card for any job type"""
    job_id = job.get("job_id", "unknown")
    status = job.get("status", "unknown")
    job_type = job.get("job_type", "generation")

    # Job type styling
    if job_type == "generation":
        type_icon, type_label = "🎨", "Generacion"
        # Generation job fields
        done = job.get("images_generated", 0)
        failed = job.get("images_rejected", 0)
        pending = job.get("images_pending", 0)
        total = job.get("total_items", 0)  # Use total_items from backend
        label1, val1 = "Generadas", done
        label2, val2 = "Rechazadas", failed
        label3, val3 = "Pendientes", pending
    elif job_type == "extraction":
        type_icon, type_label = "🎯", "Extraccion"
        # Extraction job fields
        done = job.get("extracted_objects", 0)
        failed = job.get("failed_objects", 0)
        total = job.get("total_objects", 0)
        pending = max(0, total - done - failed)
        label1, val1 = "Extraidos", done
        label2, val2 = "Fallidos", failed
        label3, val3 = "Pendientes", pending
    elif job_type == "sam3_conversion":
        type_icon, type_label = "🔬", "SAM3"
        # SAM3 conversion job fields
        done = job.get("converted_annotations", 0)
        skipped = job.get("skipped_annotations", 0)
        failed = job.get("failed_annotations", 0)
        total = job.get("total_annotations", 0)
        pending = max(0, total - done - skipped - failed)
        label1, val1 = "Convertidos", done
        label2, val2 = "Omitidos", skipped
        label3, val3 = "Fallidos", failed
    else:
        type_icon, type_label = "❔", "Desconocido"
        done, failed, pending, total = 0, 0, 0, 0
        label1, val1 = "Completados", 0
        label2, val2 = "Fallidos", 0
        label3, val3 = "Pendientes", 0

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
    elif status == "interrupted":
        status_color, status_icon = c['info'], "⏸️"
    else:
        status_color, status_icon = c['text_muted'], "❔"

    progress = done / total if total > 0 else 0

    st.markdown(f"""
    <div style="background: {c['bg_card']}; border: 1px solid {c['border']};
                border-radius: 0.5rem; padding: 1rem; margin-bottom: 0.75rem;
                border-left: 4px solid {status_color};">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.25rem;">{status_icon}</span>
                <span style="background: {c['bg_secondary']}; padding: 0.15rem 0.4rem; border-radius: 0.25rem;
                            font-size: 0.7rem; color: {c['text_secondary']};">
                    {type_icon} {type_label}
                </span>
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
                <span style="color: {c['text_muted']};">{label1}:</span>
                <span style="color: {c['success']}; font-weight: 600;"> {val1}</span>
            </div>
            <div>
                <span style="color: {c['text_muted']};">{label2}:</span>
                <span style="color: {c['error'] if 'Fallid' in label2 or 'Rechaz' in label2 else c['text_secondary']}; font-weight: 600;"> {val2}</span>
            </div>
            <div>
                <span style="color: {c['text_muted']};">{label3}:</span>
                <span style="font-weight: 600;"> {val3}</span>
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

    # Action buttons for active jobs (only for generation jobs that support cancellation)
    if is_active and job_type == "generation":
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


def _render_job_card_with_actions(job: dict, c: dict, client) -> None:
    """Render a job card with resume/retry actions for interrupted/failed jobs"""
    from app.config.theme import get_colors_dict

    job_id = job.get("job_id", "unknown")
    status = job.get("status", "unknown")
    job_type = job.get("job_type", "generation")

    # First render the base card
    _render_job_card(job, c, client, is_active=False)

    # Only show actions for generation jobs (others don't support resume yet)
    if job_type != "generation":
        return

    # Action buttons based on status
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        # Resume button - only for interrupted jobs or failed jobs with checkpoint
        if status == "interrupted":
            if st.button("▶️ Reanudar", key=f"resume_{job_id}", use_container_width=True):
                result = client.resume_job(job_id)
                if result.get("success"):
                    st.toast(f"Job {job_id[:8]}... reanudado desde checkpoint")
                    st.rerun()
                else:
                    st.error(f"Error: {result.get('error', 'No se pudo reanudar')}")

    with col2:
        # Retry button - for failed/cancelled jobs
        if status in ["failed", "cancelled"]:
            if st.button("🔄 Reintentar", key=f"retry_{job_id}", use_container_width=True):
                result = client.retry_job(job_id)
                if result.get("success"):
                    new_job_id = result.get("job_id", "")
                    st.toast(f"Nuevo job creado: {new_job_id[:8]}...")
                    st.rerun()
                else:
                    st.error(f"Error: {result.get('error', 'No se pudo reintentar')}")
        elif status == "interrupted":
            # For interrupted, also offer retry (start fresh)
            if st.button("🔄 Reiniciar", key=f"retry_{job_id}", use_container_width=True,
                        help="Iniciar desde cero (ignora progreso)"):
                result = client.retry_job(job_id)
                if result.get("success"):
                    new_job_id = result.get("job_id", "")
                    st.toast(f"Nuevo job creado: {new_job_id[:8]}...")
                    st.rerun()
                else:
                    st.error(f"Error: {result.get('error')}")

    with col3:
        # View logs button
        if st.button("📋 Ver Logs", key=f"logs_{job_id}", use_container_width=True):
            st.session_state[f"show_logs_{job_id}"] = True
            st.rerun()

    # Show logs if requested
    if st.session_state.get(f"show_logs_{job_id}"):
        with st.expander(f"📋 Logs del Job {job_id[:12]}...", expanded=True):
            logs_response = client.get_job_logs(job_id, limit=50)
            logs = logs_response.get("logs", [])

            if logs:
                for log in logs:
                    level = log.get("level", "INFO")
                    timestamp = log.get("timestamp", "")[:19]  # Trim to seconds
                    message = log.get("message", "")

                    if level == "ERROR":
                        st.error(f"`{timestamp}` {message}")
                    elif level == "WARNING":
                        st.warning(f"`{timestamp}` {message}")
                    else:
                        st.info(f"`{timestamp}` {message}")
            else:
                st.caption("No hay logs disponibles")

            if st.button("Cerrar", key=f"close_logs_{job_id}"):
                st.session_state.pop(f"show_logs_{job_id}", None)
                st.rerun()

    # Add visual separator
    st.markdown("<hr style='margin: 0.5rem 0; border: none; border-top: 1px solid #333;'>", unsafe_allow_html=True)


# =============================================================================
# PAGE: Object Sizes Configuration
# =============================================================================

def render_object_sizes_page():
    """Render object sizes configuration page"""
    from app.config.theme import get_colors_dict
    from app.components.api_client import get_api_client
    import json
    c = get_colors_dict()

    page_header(
        title="Configuración de Tamaños de Objetos",
        subtitle="Configura los tamaños reales de los objetos para mejorar el realismo del escalado",
        icon="📏"
    )

    client = get_api_client()

    # Fetch current configuration
    try:
        config_data = client.get_object_sizes()
        sizes = config_data.get("sizes", {})
        reference_distance = config_data.get("reference_capture_distance", 2.0)
        keyword_mappings = config_data.get("keyword_mappings", {})
    except Exception as e:
        alert_box(
            f"Error al cargar la configuración: {str(e)}",
            type="error",
            icon="❌"
        )
        return

    # Layout with two columns
    col1, col2 = st.columns([2, 1])

    with col1:
        section_header("Tamaños Configurados", icon="📊")

        st.markdown(f"""
        <div style='background: {c['bg_secondary']}; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
            <div style='font-size: 0.9rem; color: {c['text_muted']}; margin-bottom: 0.5rem;'>
                Los tamaños están en <strong>metros</strong> y representan el tamaño real aproximado de los objetos.
                Estos valores se utilizan para escalar correctamente los objetos según la profundidad estimada del fondo.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Filter and search
        search_term = st.text_input("🔍 Buscar clase de objeto", placeholder="Ej: fish, bottle, tire...")

        # Filter sizes based on search
        filtered_sizes = {k: v for k, v in sizes.items() if search_term.lower() in k.lower()} if search_term else sizes

        # Display sizes in a table-like format
        if filtered_sizes:
            st.markdown(f"""
            <div style='background: {c['bg_secondary']}; padding: 0.5rem; border-radius: 8px; margin-bottom: 0.5rem;'>
                <div style='display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 1rem; font-weight: 600; padding: 0.5rem;'>
                    <div>Clase de Objeto</div>
                    <div style='text-align: center;'>Tamaño (m)</div>
                    <div style='text-align: center;'>Acciones</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Show sorted sizes
            for class_name, size in sorted(filtered_sizes.items()):
                col_name, col_size, col_actions = st.columns([2, 1, 1])

                with col_name:
                    # Show keywords if available
                    keywords = []
                    for key, kw_list in keyword_mappings.items():
                        if key == class_name:
                            keywords = kw_list
                            break

                    if keywords:
                        keywords_str = ", ".join(keywords[:3])
                        st.markdown(f"**{class_name}**")
                        st.caption(f"Keywords: {keywords_str}")
                    else:
                        st.markdown(f"**{class_name}**")

                with col_size:
                    st.markdown(f"<div style='text-align: center; padding-top: 0.5rem;'>{size:.2f} m</div>", unsafe_allow_html=True)

                with col_actions:
                    if st.button("🗑️", key=f"delete_{class_name}", help="Eliminar", use_container_width=True):
                        try:
                            client.delete_object_size(class_name)
                            alert_box("Tamaño eliminado correctamente", type="success", icon="✅")
                            st.rerun()
                        except Exception as e:
                            alert_box(f"Error al eliminar: {str(e)}", type="error", icon="❌")

                st.markdown(f"<hr style='margin: 0.25rem 0; border: none; border-top: 1px solid {c['border']};'>", unsafe_allow_html=True)
        else:
            empty_state(
                title="No se encontraron resultados",
                message="No hay objetos que coincidan con tu búsqueda",
                icon="🔍"
            )

    with col2:
        section_header("Configuración", icon="⚙️")

        # Reference distance configuration
        st.markdown("**Distancia de Captura de Referencia**")
        new_ref_distance = st.number_input(
            "Distancia (m)",
            min_value=0.1,
            max_value=100.0,
            value=reference_distance,
            step=0.1,
            help="Distancia a la que se capturaron las imágenes de referencia"
        )

        if new_ref_distance != reference_distance:
            if st.button("Actualizar Distancia", type="primary", use_container_width=True):
                try:
                    client.set_reference_distance(new_ref_distance)
                    alert_box("Distancia actualizada correctamente", type="success", icon="✅")
                    st.rerun()
                except Exception as e:
                    alert_box(f"Error: {str(e)}", type="error", icon="❌")

        spacer(2)

        # Add/Update single object
        with st.expander("➕ Agregar/Actualizar Objeto", expanded=True):
            new_class = st.text_input("Clase de objeto", placeholder="Ej: fish, bottle")
            new_size = st.number_input("Tamaño (metros)", min_value=0.01, max_value=100.0, value=0.25, step=0.01)

            if st.button("Guardar", type="primary", use_container_width=True, disabled=not new_class):
                try:
                    client.update_object_size(new_class.strip(), new_size)
                    alert_box(f"Tamaño de '{new_class}' guardado correctamente", type="success", icon="✅")
                    st.rerun()
                except Exception as e:
                    alert_box(f"Error: {str(e)}", type="error", icon="❌")

        spacer(1)

        # Batch update
        with st.expander("📦 Actualización por Lotes"):
            st.markdown("""
            <div style='font-size: 0.85rem; margin-bottom: 1rem;'>
                Formato JSON:
                <pre style='background: #1a1a1a; padding: 0.5rem; border-radius: 4px; overflow-x: auto;'>
{
  "fish": 0.25,
  "shark": 2.5,
  "bottle": 0.25
}</pre>
            </div>
            """, unsafe_allow_html=True)

            batch_json = st.text_area("JSON de tamaños", height=150, placeholder='{"fish": 0.25, "shark": 2.5}')

            if st.button("Actualizar por Lotes", type="primary", use_container_width=True, disabled=not batch_json):
                try:
                    batch_sizes = json.loads(batch_json)
                    client.update_multiple_object_sizes(batch_sizes)
                    alert_box(f"Se actualizaron {len(batch_sizes)} objetos correctamente", type="success", icon="✅")
                    st.rerun()
                except json.JSONDecodeError:
                    alert_box("JSON inválido. Verifica el formato.", type="error", icon="❌")
                except Exception as e:
                    alert_box(f"Error: {str(e)}", type="error", icon="❌")

        spacer(1)

        # Export configuration
        with st.expander("💾 Exportar Configuración"):
            st.markdown("Descarga la configuración actual en formato JSON")

            config_json = json.dumps(config_data, indent=2, ensure_ascii=False)

            st.download_button(
                label="⬇️ Descargar JSON",
                data=config_json,
                file_name="object_sizes_config.json",
                mime="application/json",
                use_container_width=True
            )

    # Statistics section
    spacer(2)
    section_header("Estadísticas", icon="📈")

    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

    with stat_col1:
        st.metric("Total de Objetos", len(sizes))

    with stat_col2:
        avg_size = sum(sizes.values()) / len(sizes) if sizes else 0
        st.metric("Tamaño Promedio", f"{avg_size:.2f} m")

    with stat_col3:
        max_size = max(sizes.values()) if sizes else 0
        max_class = max(sizes, key=sizes.get) if sizes else "N/A"
        st.metric("Objeto Más Grande", f"{max_size:.2f} m", delta=max_class)

    with stat_col4:
        min_size = min(sizes.values()) if sizes else 0
        min_class = min(sizes, key=sizes.get) if sizes else "N/A"
        st.metric("Objeto Más Pequeño", f"{min_size:.2f} m", delta=min_class)


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
