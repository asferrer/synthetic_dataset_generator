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
"""

import os
import streamlit as st
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Synthetic Data Generator",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/asferrer/synthetic_dataset_generator/issues",
        "About": "Synthetic Data Generator v2.0 - Microservices Edition"
    }
)

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #ff4b4b;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application entry point"""

    # Title
    st.title("Synthetic Data Generator")
    st.markdown("*Microservices-based photorealistic synthetic dataset generation*")

    # Import components
    from app.components.effects_sidebar import render_effects_sidebar
    from app.components.api_client import get_api_client
    from app.pages.analysis import render_analysis_page
    from app.pages.generation import render_generation_page
    from app.pages.post_processing import render_post_processing_page

    # Render sidebar and get configuration
    generation_config, effects, effects_config = render_effects_sidebar()

    # Main content tabs
    tab_analysis, tab_generation, tab_post_processing, tab_services, tab_docs = st.tabs([
        "Dataset Analysis",
        "Generation",
        "Post-Processing",
        "Services",
        "Documentation"
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
    """Render services information page"""
    st.header("Service Status")

    from app.components.api_client import get_api_client
    client = get_api_client()

    # Get health
    health = client.get_health()

    if health.get("error"):
        st.error(f"Gateway unavailable: {health.get('error')}")
        return

    # Overall status
    status = health.get("status", "unknown")
    if status == "healthy":
        st.success(f"System Status: {status.upper()}")
    elif status == "degraded":
        st.warning(f"System Status: {status.upper()}")
    else:
        st.error(f"System Status: {status.upper()}")

    st.divider()

    # Service cards
    services = health.get("services", [])

    cols = st.columns(len(services) if services else 1)

    for i, service in enumerate(services):
        with cols[i % len(cols)]:
            name = service.get("name", "unknown").capitalize()
            svc_status = service.get("status", "unknown")
            latency = service.get("latency_ms", 0)

            if svc_status == "healthy":
                st.success(f"**{name}**")
            elif svc_status == "degraded":
                st.warning(f"**{name}**")
            else:
                st.error(f"**{name}**")

            st.caption(f"Latency: {latency:.0f}ms" if latency else "N/A")

            # Get service info
            if st.button(f"Details", key=f"svc_details_{name}"):
                info = client.get_service_info(service.get("name", ""))
                if info:
                    st.json(info)

    st.divider()

    # API Documentation link
    st.subheader("API Documentation")

    gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:8000")

    st.markdown(f"""
    - **Gateway Swagger UI**: [{gateway_url}/docs]({gateway_url}/docs)
    - **Gateway ReDoc**: [{gateway_url}/redoc]({gateway_url}/redoc)
    """)

    # Endpoints reference
    with st.expander("Available Endpoints"):
        st.markdown("""
        **Health & Info:**
        - `GET /health` - Service health status
        - `GET /info` - Service information

        **Generation (Legacy):**
        - `POST /generate/image` - Generate single image
        - `POST /generate/batch` - Batch generation

        **Augmentation (New):**
        - `POST /augment/compose` - Compose single image
        - `POST /augment/compose-batch` - Batch composition
        - `GET /augment/jobs/{job_id}` - Job status
        - `POST /augment/validate` - Quality validation
        - `POST /augment/lighting` - Lighting estimation

        **Services:**
        - `GET /services/depth` - Depth service info
        - `GET /services/effects` - Effects service info
        - `GET /services/augmentor` - Augmentor service info
        """)


def render_documentation_page():
    """Render documentation page"""
    st.header("Documentation")

    tab1, tab2, tab3 = st.tabs(["Architecture", "Effects", "Usage Guide"])

    with tab1:
        st.markdown("""
        ## Microservices Architecture

        ```
        +------------------------------------------------------------------+
        |                    FRONTEND (Streamlit - 8501)                   |
        +------------------------------------------------------------------+
                                        |
                                   HTTP/REST
                                        v
        +------------------------------------------------------------------+
        |                    API GATEWAY (FastAPI - 8000)                  |
        +------------------------------------------------------------------+
                |              |              |              |
                v              v              v              v
        +------------+  +------------+  +------------+  +------------+
        |   DEPTH    |  |  EFFECTS   |  | AUGMENTOR  |  |SEGMENTATION|
        |   (8001)   |  |   (8003)   |  |   (8004)   |  |   (8002)   |
        +------------+  +------------+  +------------+  +------------+
        | DA-V3      |  | 11 Effects |  | Composer   |  | SAM3       |
        | +44% acc.  |  | Caustics   |  | Validator  |  | (optional) |
        | Zone class |  | Poisson    |  | Physics    |  |            |
        +------------+  +------------+  +------------+  +------------+
        ```

        ### Services Overview

        | Service | Port | Description | GPU Required |
        |---------|------|-------------|--------------|
        | **Gateway** | 8000 | API orchestrator | No |
        | **Depth** | 8001 | Depth Anything V3 estimation | Yes |
        | **Effects** | 8003 | Realism effects pipeline | No |
        | **Augmentor** | 8004 | Composition + Validation | Yes |
        | **Segmentation** | 8002 | SAM3 segmentation | Yes |
        | **Frontend** | 8501 | This Streamlit UI | No |

        ### Data Flow

        1. **Analysis**: Upload COCO JSON, analyze class distribution
        2. **Configuration**: Select effects, intensities, validation
        3. **Generation**: Single or batch with depth-aware placement
        4. **Validation**: Quality (LPIPS) and physics checks
        5. **Output**: Images + COCO annotations
        """)

    with tab2:
        st.markdown("""
        ## Available Effects

        ### Core Effects

        | Effect | Description | Intensity Range |
        |--------|-------------|-----------------|
        | **Color Correction** | LAB color space transfer | 0.0 - 1.0 |
        | **Blur Matching** | Match focus/blur levels | 0.0 - 3.0 |
        | **Shadows** | Multi-source shadow generation | 0.0 - 1.0 |
        | **Caustics** | Underwater light patterns | 0.0 - 0.5 |
        | **Underwater** | Water color tinting | 0.0 - 1.0 |

        ### Advanced Effects

        | Effect | Description | Notes |
        |--------|-------------|-------|
        | **Lighting** | Directional/ambient lighting | Types: ambient, spotlight, gradient |
        | **Motion Blur** | Simulate movement | Probability-based |
        | **Edge Smoothing** | Anti-aliasing for edges | Feather pixels: 1-20 |
        | **Poisson Blend** | Seamless cloning | High quality, slower |

        ### Depth-Aware Features

        - **Zone Classification**: NEAR (0-33%), MID (33-66%), FAR (66-100%)
        - **Scale Mapping**:
          - NEAR: 0.8x - 3.0x (large objects)
          - MID: 0.4x - 1.5x (medium objects)
          - FAR: 0.05x - 0.4x (small objects)
        - **Automatic Positioning**: Objects placed in appropriate depth zones

        ### Validation

        **Quality Metrics:**
        - LPIPS perceptual quality (0-1, higher = better)
        - Anomaly detection (Isolation Forest)
        - Composition plausibility score

        **Physics Validation:**
        - Gravity/buoyancy checks for underwater scenes
        - Material density database (50+ materials)
        - Scale plausibility verification
        """)

    with tab3:
        st.markdown("""
        ## Quick Start Guide

        ### 1. Dataset Analysis

        1. Go to **Dataset Analysis** tab
        2. Upload your COCO JSON file
        3. Review class distribution
        4. Select balancing strategy:
           - **Complete**: Balance all classes to maximum
           - **Partial**: Balance to 75% of maximum
           - **Minority**: Only balance underrepresented classes
        5. Click "Proceed to Generation"

        ### 2. Single Image Generation

        1. Go to **Generation** > **Single Image** tab
        2. Select background (upload, path, or browse)
        3. Select object classes to place
        4. Configure effects in sidebar
        5. Click "Generate Image"
        6. Download result

        ### 3. Batch Generation

        1. Go to **Generation** > **Batch Generation** tab
        2. Set directories:
           - Backgrounds_filtered directory
           - Objects directory (organized by class)
           - Output directory
        3. Set number of images or use analysis targets
        4. Click "Start Batch Generation"
        5. Monitor progress
        6. Download COCO JSON when complete

        ### Directory Structure

        ```
        /app/datasets/
        ├── Backgrounds_filtered/
        │   ├── image001.jpg
        │   ├── image002.jpg
        │   └── ...
        └── Objects/
            ├── fish/
            │   ├── fish_001.png
            │   └── ...
            ├── coral/
            │   └── ...
            └── debris/
                └── ...

        /app/output/
        └── batch/
            ├── images/
            │   ├── synthetic_00000.jpg
            │   └── ...
            └── synthetic_dataset.json
        ```

        ### Tips

        - Use **PNG with alpha** for object images
        - Enable **depth-aware placement** for realistic scaling
        - Start with **color_correction + blur_matching** effects
        - Add **caustics** for underwater scenes
        - Enable **validation** for high-quality datasets
        """)


if __name__ == "__main__":
    main()
