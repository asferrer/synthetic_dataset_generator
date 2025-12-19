"""
Synthetic Data Generator - Unified Frontend
============================================
Professional UI for synthetic dataset generation with microservices.

Features:
- COCO dataset analysis
- Class balancing
- Single and batch generation
- 11+ realism effects
- Quality and physics validation
- Dark/Light/Ocean themes
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

# Import theme and styles
from app.config.theme import ThemeManager
from app.config.styles import inject_styles
from app.components.ui import (
    page_header, section_header, metric_card, metric_row,
    service_card, alert_box, empty_state, spacer, divider_with_text
)

# Inject custom styles
inject_styles()


def render_app_header():
    """Render the application header with branding"""
    col1, col2, col3 = st.columns([2, 6, 2])

    with col1:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 2rem;">🔬</span>
            <div>
                <div style="font-size: 1.25rem; font-weight: 700; color: var(--color-text-primary);">SDG</div>
                <div style="font-size: 0.65rem; color: var(--color-text-muted);">v2.0</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <h1 style="margin: 0; font-size: 1.75rem; font-weight: 700; color: var(--color-text-primary);">
                Synthetic Data Generator
            </h1>
            <p style="margin: 0.25rem 0 0 0; font-size: 0.875rem; color: var(--color-text-muted);">
                Microservices-based photorealistic dataset generation
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        # Quick status indicator
        from app.components.api_client import get_api_client
        client = get_api_client()
        health = client.get_health()

        status = health.get("status", "unknown")
        if status == "healthy":
            status_icon = "🟢"
            status_text = "All Systems Operational"
        elif status == "degraded":
            status_icon = "🟡"
            status_text = "Degraded Performance"
        else:
            status_icon = "🔴"
            status_text = "Services Unavailable"

        st.markdown(f"""
        <div style="text-align: right;">
            <div style="font-size: 0.75rem; color: var(--color-text-muted);">System Status</div>
            <div style="font-size: 0.875rem; font-weight: 500;">
                {status_icon} {status_text}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px solid var(--color-border);'>", unsafe_allow_html=True)


def render_sidebar():
    """Render the enhanced sidebar"""
    from app.components.effects_sidebar import render_effects_sidebar

    with st.sidebar:
        # Theme selector
        st.markdown("""
        <div style="padding: 0.5rem 0; margin-bottom: 1rem; border-bottom: 1px solid var(--color-border);">
            <div style="font-size: 0.75rem; color: var(--color-text-muted); margin-bottom: 0.5rem;">THEME</div>
        </div>
        """, unsafe_allow_html=True)

        ThemeManager.render_theme_selector()

        spacer(16)
        st.markdown("<hr style='border: none; border-top: 1px solid var(--color-border);'>", unsafe_allow_html=True)
        spacer(8)

        # Effects configuration
        return render_effects_sidebar()


def main():
    """Main application entry point"""

    # Render app header
    render_app_header()

    # Render sidebar and get configuration
    generation_config, effects, effects_config = render_sidebar()

    # Import page components
    from app.pages.analysis import render_analysis_page
    from app.pages.generation import render_generation_page
    from app.pages.post_processing import render_post_processing_page

    # Main content tabs with icons
    tab_analysis, tab_generation, tab_post_processing, tab_services, tab_docs = st.tabs([
        "📊 Dataset Analysis",
        "🎨 Generation",
        "⚙️ Post-Processing",
        "🔧 Services",
        "📚 Documentation"
    ])

    # Tab 1: Analysis
    with tab_analysis:
        render_analysis_page()

    # Tab 2: Generation
    with tab_generation:
        render_generation_page(effects, effects_config, generation_config)

    # Tab 3: Post-Processing
    with tab_post_processing:
        render_post_processing_page()

    # Tab 4: Services
    with tab_services:
        render_services_page()

    # Tab 5: Documentation
    with tab_docs:
        render_documentation_page()


def render_services_page():
    """Render services information page with enhanced UI"""
    page_header(
        title="Service Status",
        subtitle="Monitor the health and performance of all microservices",
        icon="🔧"
    )

    from app.components.api_client import get_api_client
    client = get_api_client()

    # Get health
    health = client.get_health()

    if health.get("error"):
        alert_box(
            f"Gateway unavailable: {health.get('error')}",
            type="error",
            icon="🔌"
        )
        empty_state(
            title="Cannot Connect to Gateway",
            message="Please ensure the API Gateway is running and accessible at the configured URL.",
            icon="🔌"
        )
        return

    # Overall system status
    status = health.get("status", "unknown")
    if status == "healthy":
        st.success("🟢 **System Status: HEALTHY** - All services are operational")
    elif status == "degraded":
        st.warning("🟡 **System Status: DEGRADED** - Some services may be experiencing issues")
    else:
        st.error("🔴 **System Status: UNHEALTHY** - Critical services are unavailable")

    spacer(24)

    # Service cards grid
    section_header("Microservices", icon="🐳")

    services = health.get("services", [])

    if services:
        cols = st.columns(min(len(services), 4))

        for i, service in enumerate(services):
            with cols[i % 4]:
                name = service.get("name", "unknown")
                svc_status = service.get("status", "unknown")
                latency = service.get("latency_ms", 0)

                # Determine port from service name
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

                # Get service info button
                if st.button(f"Details", key=f"svc_details_{name}", use_container_width=True):
                    info = client.get_service_info(name)
                    if info:
                        st.json(info)
    else:
        empty_state(
            title="No Services Found",
            message="No microservices are currently registered with the gateway.",
            icon="📭"
        )

    spacer(32)

    # API Documentation
    section_header("API Documentation", icon="📖")

    gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:8000")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📘</div>
            <div style="font-weight: 600; margin-bottom: 0.25rem;">Swagger UI</div>
            <div style="font-size: 0.875rem; color: var(--color-text-muted);">Interactive API explorer</div>
            <a href="{gateway_url}/docs" target="_blank" style="display: inline-block; margin-top: 0.75rem; color: var(--color-primary); text-decoration: none;">
                Open Swagger →
            </a>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📗</div>
            <div style="font-weight: 600; margin-bottom: 0.25rem;">ReDoc</div>
            <div style="font-size: 0.875rem; color: var(--color-text-muted);">Clean API documentation</div>
            <a href="{gateway_url}/redoc" target="_blank" style="display: inline-block; margin-top: 0.75rem; color: var(--color-primary); text-decoration: none;">
                Open ReDoc →
            </a>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📙</div>
            <div style="font-weight: 600; margin-bottom: 0.25rem;">OpenAPI Spec</div>
            <div style="font-size: 0.875rem; color: var(--color-text-muted);">Raw JSON specification</div>
            <a href="{gateway_url}/openapi.json" target="_blank" style="display: inline-block; margin-top: 0.75rem; color: var(--color-primary); text-decoration: none;">
                Download JSON →
            </a>
        </div>
        """, unsafe_allow_html=True)

    spacer(24)

    # Endpoints reference
    with st.expander("📋 Available Endpoints Reference", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### Health & Info
            | Endpoint | Method | Description |
            |----------|--------|-------------|
            | `/health` | GET | Service health status |
            | `/info` | GET | Service information |

            #### Generation
            | Endpoint | Method | Description |
            |----------|--------|-------------|
            | `/augment/compose` | POST | Compose single image |
            | `/augment/compose-batch` | POST | Batch composition |
            | `/augment/jobs/{job_id}` | GET | Job status |
            | `/augment/jobs` | GET | List all jobs |
            """)

        with col2:
            st.markdown("""
            #### Validation & Effects
            | Endpoint | Method | Description |
            |----------|--------|-------------|
            | `/augment/validate` | POST | Quality validation |
            | `/augment/lighting` | POST | Lighting estimation |
            | `/augment/backgrounds` | GET | List backgrounds |
            | `/augment/objects` | GET | List objects |

            #### Services
            | Endpoint | Method | Description |
            |----------|--------|-------------|
            | `/services/depth` | GET | Depth service info |
            | `/services/effects` | GET | Effects service info |
            | `/services/augmentor` | GET | Augmentor service info |
            """)


def render_documentation_page():
    """Render documentation page with enhanced UI"""
    page_header(
        title="Documentation",
        subtitle="Architecture guides, effect references, and usage tutorials",
        icon="📚"
    )

    tab1, tab2, tab3 = st.tabs(["🏗️ Architecture", "✨ Effects", "📖 Usage Guide"])

    with tab1:
        render_architecture_docs()

    with tab2:
        render_effects_docs()

    with tab3:
        render_usage_guide()


def render_architecture_docs():
    """Render architecture documentation"""

    st.markdown("""
    ## Microservices Architecture

    The Synthetic Data Generator uses a distributed microservices architecture for scalability
    and modularity. Each service handles a specific domain of the generation pipeline.
    """)

    # Architecture diagram
    st.markdown("""
    ```
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                        FRONTEND (Streamlit - 8501)                        │
    │                     Professional UI / Dashboard / Monitor                  │
    └────────────────────────────────────┬─────────────────────────────────────┘
                                         │ HTTP/REST
                                         ▼
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                      API GATEWAY (FastAPI - 8000)                         │
    │                 Orchestration / Load Balancing / Auth                     │
    └──────┬─────────────┬─────────────┬─────────────┬─────────────────────────┘
           │             │             │             │
           ▼             ▼             ▼             ▼
    ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
    │   DEPTH    │ │  EFFECTS   │ │ AUGMENTOR  │ │SEGMENTATION│
    │   (8001)   │ │   (8003)   │ │   (8004)   │ │   (8002)   │
    ├────────────┤ ├────────────┤ ├────────────┤ ├────────────┤
    │ DA-V3      │ │ 11 Effects │ │ Composer   │ │ SAM3       │
    │ +44% acc.  │ │ Caustics   │ │ Validator  │ │ (optional) │
    │ Zone class │ │ Poisson    │ │ Physics    │ │            │
    │ GPU: Yes   │ │ GPU: No    │ │ GPU: Yes   │ │ GPU: Yes   │
    └────────────┘ └────────────┘ └────────────┘ └────────────┘
    ```
    """)

    spacer(24)

    # Service overview
    section_header("Service Overview", icon="📋")

    services_data = [
        {"Service": "**Gateway**", "Port": "8000", "GPU": "❌", "Description": "API orchestration, routing, and load balancing"},
        {"Service": "**Depth**", "Port": "8001", "GPU": "✅", "Description": "Depth Anything V3 estimation with zone classification"},
        {"Service": "**Segmentation**", "Port": "8002", "GPU": "✅", "Description": "SAM3 segmentation for object extraction"},
        {"Service": "**Effects**", "Port": "8003", "GPU": "❌", "Description": "11+ realism effects pipeline (caustics, color, blur)"},
        {"Service": "**Augmentor**", "Port": "8004", "GPU": "✅", "Description": "Composition, validation, physics simulation"},
        {"Service": "**Frontend**", "Port": "8501", "GPU": "❌", "Description": "Streamlit UI with monitoring and controls"},
    ]

    import pandas as pd
    st.dataframe(
        pd.DataFrame(services_data),
        use_container_width=True,
        hide_index=True
    )

    spacer(24)

    # Data flow
    section_header("Data Flow Pipeline", icon="🔄")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Generation Pipeline

        1. **📥 Input** - Background image + object assets
        2. **🔍 Analysis** - Depth estimation + zone classification
        3. **📍 Placement** - Depth-aware object positioning
        4. **✨ Effects** - Apply realism effects
        5. **✅ Validation** - Quality + physics checks
        6. **📤 Output** - Image + COCO annotations
        """)

    with col2:
        st.markdown("""
        ### Processing Flow

        ```
        Background → Depth Service
                          ↓
        Objects → Augmentor → Effects
                          ↓
        Validation ← Physics Engine
                          ↓
        Output → COCO JSON + Images
        ```
        """)


def render_effects_docs():
    """Render effects documentation"""

    st.markdown("""
    ## Available Effects

    The effects pipeline provides 11+ photorealistic enhancement options for
    synthetic image generation. Each effect can be individually configured.
    """)

    # Core effects
    section_header("Core Effects", icon="🎨")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        | Effect | Range | Description |
        |--------|-------|-------------|
        | **Color Correction** | 0.0 - 1.0 | LAB color space transfer |
        | **Blur Matching** | 0.0 - 3.0 | Match focus/blur levels |
        | **Shadows** | 0.0 - 1.0 | Multi-source shadow generation |
        | **Edge Smoothing** | 1 - 20px | Anti-aliasing for edges |
        """)

    with col2:
        st.markdown("""
        | Effect | Range | Description |
        |--------|-------|-------------|
        | **Caustics** | 0.0 - 0.5 | Underwater light patterns |
        | **Underwater** | 0.0 - 1.0 | Water color tinting |
        | **Motion Blur** | 0.0 - 1.0 | Movement simulation |
        | **Poisson Blend** | On/Off | Seamless cloning |
        """)

    spacer(24)

    # Advanced effects
    section_header("Advanced Lighting", icon="💡")

    st.markdown("""
    | Lighting Type | Description | Best For |
    |---------------|-------------|----------|
    | **Ambient** | Uniform soft lighting | General scenes |
    | **Spotlight** | Directional focused light | Dramatic effects |
    | **Gradient** | Smooth light transition | Underwater depth |
    """)

    spacer(24)

    # Depth-aware features
    section_header("Depth-Aware Placement", icon="📐")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Zone Classification

        Objects are automatically placed based on depth zones:

        | Zone | Depth Range | Scale Range |
        |------|-------------|-------------|
        | **NEAR** | 0 - 33% | 0.8x - 3.0x |
        | **MID** | 33 - 66% | 0.4x - 1.5x |
        | **FAR** | 66 - 100% | 0.05x - 0.4x |
        """)

    with col2:
        st.markdown("""
        ### Automatic Features

        - **Scale Mapping** - Objects sized based on depth
        - **Blur Matching** - Focus matches depth zone
        - **Occlusion** - Proper object layering
        - **Shadow Direction** - Light-consistent shadows
        """)


def render_usage_guide():
    """Render usage guide documentation"""

    st.markdown("""
    ## Quick Start Guide

    Follow these steps to generate your first synthetic dataset.
    """)

    # Step 1
    section_header("Step 1: Dataset Analysis", icon="1️⃣")

    st.markdown("""
    1. Navigate to **📊 Dataset Analysis** tab
    2. Upload your COCO JSON annotation file
    3. Review class distribution and imbalance metrics
    4. Select a balancing strategy:
       - **Complete** - Balance all classes to maximum count
       - **Partial** - Balance to 75% of maximum
       - **Minority** - Only balance underrepresented classes
    5. Click **"Proceed to Generation"** to save targets
    """)

    spacer(16)

    # Step 2
    section_header("Step 2: Single Image Generation", icon="2️⃣")

    st.markdown("""
    1. Navigate to **🎨 Generation** → **Single Image** tab
    2. Select a background image (upload, path, or browse)
    3. Choose object classes to place
    4. Configure effects in the sidebar
    5. Click **"Generate Image"**
    6. Download the result and annotations
    """)

    spacer(16)

    # Step 3
    section_header("Step 3: Batch Generation", icon="3️⃣")

    st.markdown("""
    1. Navigate to **🎨 Generation** → **Batch Generation** tab
    2. Configure directories:
       - **Backgrounds** - Folder with background images
       - **Objects** - Folder with class subfolders
       - **Output** - Destination folder
    3. Set the number of images to generate
    4. Enable depth-aware placement (recommended)
    5. Click **"Start Batch Generation"**
    6. Monitor progress in the Jobs tab
    7. Download COCO JSON when complete
    """)

    spacer(24)

    # Directory structure
    section_header("Directory Structure", icon="📁")

    st.code("""
    /app/datasets/
    ├── Backgrounds_filtered/
    │   ├── background_001.jpg
    │   ├── background_002.jpg
    │   └── ...
    └── Objects/
        ├── fish/
        │   ├── fish_001.png (with alpha)
        │   └── ...
        ├── coral/
        │   └── ...
        └── debris/
            └── ...

    /app/output/
    └── batch_20240101_120000/
        ├── images/
        │   ├── synthetic_00000.jpg
        │   └── ...
        └── synthetic_dataset.json (COCO format)
    """, language="text")

    spacer(24)

    # Tips
    section_header("Tips & Best Practices", icon="💡")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Image Preparation

        - ✅ Use **PNG with alpha** for objects
        - ✅ High-resolution backgrounds (1920x1080+)
        - ✅ Diverse lighting conditions
        - ✅ Clean object cutouts
        """)

    with col2:
        st.markdown("""
        ### Generation Settings

        - ✅ Enable **depth-aware placement**
        - ✅ Start with **color_correction + blur_matching**
        - ✅ Add **caustics** for underwater scenes
        - ✅ Enable **validation** for quality assurance
        """)


if __name__ == "__main__":
    main()
