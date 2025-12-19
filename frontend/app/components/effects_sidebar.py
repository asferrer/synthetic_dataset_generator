"""
Effects Sidebar Component
=========================
Professional configurable sidebar for effect settings.
"""

import streamlit as st
from typing import Dict, List, Tuple


def render_effects_sidebar() -> Tuple[Dict, List[str], Dict]:
    """
    Render the effects configuration sidebar with enhanced UI.

    Returns:
        Tuple of (generation_config, effects_list, effects_config)
    """
    # Sidebar header
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <div style="font-size: 1.25rem; font-weight: 700; color: var(--color-text-primary);">
            ⚙️ Configuration
        </div>
        <div style="font-size: 0.75rem; color: var(--color-text-muted);">
            Adjust generation settings and effects
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Service Status Section
    st.markdown("""
    <div style="font-size: 0.7rem; color: var(--color-text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
        Service Status
    </div>
    """, unsafe_allow_html=True)

    from app.components.api_client import get_api_client
    client = get_api_client()

    health = client.get_health()

    if health and "services" in health:
        # Compact service status grid
        services = health.get("services", [])
        service_count = len(services)

        if service_count > 0:
            # Create compact status indicators
            st.markdown('<div style="display: flex; flex-wrap: wrap; gap: 0.25rem; margin-bottom: 0.5rem;">', unsafe_allow_html=True)

            for service in services:
                status = service.get("status", "unknown")
                name = service.get("name", "unknown")

                if status == "healthy":
                    color = "var(--color-success)"
                    icon = "🟢"
                elif status == "degraded":
                    color = "var(--color-warning)"
                    icon = "🟡"
                else:
                    color = "var(--color-error)"
                    icon = "🔴"

                st.markdown(f"""
                <div style="display: inline-flex; align-items: center; gap: 0.25rem; padding: 0.25rem 0.5rem;
                            background: var(--color-bg-secondary); border-radius: var(--radius-full);
                            font-size: 0.7rem; border: 1px solid var(--color-border);">
                    <span>{icon}</span>
                    <span style="color: var(--color-text-secondary);">{name[:4].capitalize()}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="padding: 0.5rem; background: var(--color-error-bg); border-radius: var(--radius-md);
                    font-size: 0.75rem; color: var(--color-error); text-align: center;">
            🔴 Gateway unavailable
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border: none; border-top: 1px solid var(--color-border); margin: 1rem 0;'>", unsafe_allow_html=True)

    # Generation Settings Section
    st.markdown("""
    <div style="font-size: 0.7rem; color: var(--color-text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.75rem;">
        🎨 Generation Settings
    </div>
    """, unsafe_allow_html=True)

    depth_aware = st.checkbox(
        "Depth-aware placement",
        value=True,
        key="depth_aware",
        help="Position objects based on background depth map"
    )

    max_objects = st.slider(
        "Max objects per image",
        min_value=1,
        max_value=10,
        value=5,
        key="max_objects",
        help="Maximum number of objects to place in each image"
    )

    overlap_threshold = st.slider(
        "Overlap threshold",
        min_value=0.0,
        max_value=0.5,
        value=0.1,
        key="overlap",
        help="Maximum allowed overlap between objects (0 = no overlap)"
    )

    st.markdown("<hr style='border: none; border-top: 1px solid var(--color-border); margin: 1rem 0;'>", unsafe_allow_html=True)

    # Effects Selection Section
    st.markdown("""
    <div style="font-size: 0.7rem; color: var(--color-text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.75rem;">
        ✨ Effects
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        color_correction = st.checkbox("🎨 Color", value=True, key="fx_color", help="Color correction")
        blur_matching = st.checkbox("🔍 Blur", value=True, key="fx_blur", help="Blur matching")
        shadows = st.checkbox("🌑 Shadow", value=True, key="fx_shadows", help="Shadow generation")
        caustics = st.checkbox("💫 Caustics", value=True, key="fx_caustics", help="Light caustics")

    with col2:
        underwater = st.checkbox("🌊 Water", value=True, key="fx_underwater", help="Underwater effect")
        lighting = st.checkbox("💡 Light", value=False, key="fx_lighting", help="Additional lighting")
        motion_blur = st.checkbox("💨 Motion", value=False, key="fx_motion", help="Motion blur")
        edge_smoothing = st.checkbox("✂️ Edge", value=True, key="fx_edge", help="Edge smoothing")

    st.markdown("<hr style='border: none; border-top: 1px solid var(--color-border); margin: 1rem 0;'>", unsafe_allow_html=True)

    # Intensities Section
    st.markdown("""
    <div style="font-size: 0.7rem; color: var(--color-text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.75rem;">
        🎚️ Intensities
    </div>
    """, unsafe_allow_html=True)

    color_intensity = st.slider(
        "Color correction",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        key="int_color",
        help="Intensity of color transfer from background to objects"
    )

    blur_strength = st.slider(
        "Blur matching",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        key="int_blur",
        help="Strength of blur matching between objects and background"
    )

    underwater_intensity = st.slider(
        "Underwater effect",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        key="int_underwater",
        help="Intensity of underwater color tinting"
    )

    caustics_intensity = st.slider(
        "Caustics",
        min_value=0.0,
        max_value=0.5,
        value=0.15,
        key="int_caustics",
        help="Intensity of underwater light caustics"
    )

    shadow_opacity = st.slider(
        "Shadow opacity",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        key="int_shadow",
        help="Darkness of generated shadows"
    )

    st.markdown("<hr style='border: none; border-top: 1px solid var(--color-border); margin: 1rem 0;'>", unsafe_allow_html=True)

    # Validation Section
    st.markdown("""
    <div style="font-size: 0.7rem; color: var(--color-text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.75rem;">
        ✅ Validation
    </div>
    """, unsafe_allow_html=True)

    validate_quality = st.checkbox(
        "Quality validation (LPIPS)",
        value=False,
        key="val_quality",
        help="Run LPIPS perceptual quality check on generated images"
    )

    validate_physics = st.checkbox(
        "Physics validation",
        value=False,
        key="val_physics",
        help="Check for gravity/buoyancy violations in object placement"
    )

    st.markdown("<hr style='border: none; border-top: 1px solid var(--color-border); margin: 1rem 0;'>", unsafe_allow_html=True)

    # Advanced Settings (Collapsible)
    with st.expander("🌊 Water Settings", expanded=False):
        water_clarity = st.selectbox(
            "Water clarity",
            ["clear", "murky", "very_murky"],
            format_func=lambda x: {
                "clear": "🔵 Clear",
                "murky": "🟢 Murky",
                "very_murky": "🟤 Very Murky"
            }.get(x, x),
            key="water_clarity",
            help="Clarity level of the water"
        )

        st.markdown("""
        <div style="font-size: 0.75rem; color: var(--color-text-muted); margin: 0.5rem 0;">
            Water Color (RGB)
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            water_r = st.number_input("R", 0, 255, 120, key="water_r")
        with col2:
            water_g = st.number_input("G", 0, 255, 80, key="water_g")
        with col3:
            water_b = st.number_input("B", 0, 255, 20, key="water_b")

        # Color preview
        st.markdown(f"""
        <div style="width: 100%; height: 24px; border-radius: var(--radius-sm);
                    background-color: rgb({water_r}, {water_g}, {water_b});
                    border: 1px solid var(--color-border); margin-top: 0.5rem;">
        </div>
        """, unsafe_allow_html=True)

    with st.expander("💡 Lighting Settings", expanded=False):
        lighting_type = st.selectbox(
            "Lighting type",
            ["ambient", "spotlight", "gradient"],
            format_func=lambda x: {
                "ambient": "☀️ Ambient - Uniform soft lighting",
                "spotlight": "🔦 Spotlight - Directional focused light",
                "gradient": "🌅 Gradient - Smooth light transition"
            }.get(x, x),
            key="lighting_type",
            help="Type of additional lighting effect"
        )

        lighting_intensity = st.slider(
            "Lighting intensity",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            key="lighting_intensity",
            help="Intensity of the lighting effect"
        )

    # Build effects list
    effects = []
    if color_correction:
        effects.append("color_correction")
    if blur_matching:
        effects.append("blur_matching")
    if shadows:
        effects.append("shadows")
    if caustics:
        effects.append("caustics")
    if underwater:
        effects.append("underwater")
    if lighting:
        effects.append("lighting")
    if motion_blur:
        effects.append("motion_blur")
    if edge_smoothing:
        effects.append("edge_smoothing")

    # Build effects config
    effects_config = {
        "color_intensity": color_intensity,
        "blur_strength": blur_strength,
        "underwater_intensity": underwater_intensity,
        "caustics_intensity": caustics_intensity,
        "shadow_opacity": shadow_opacity,
        "lighting_type": lighting_type,
        "lighting_intensity": lighting_intensity,
        "water_color": (water_b, water_g, water_r),  # BGR format
        "water_clarity": water_clarity,
        "motion_blur_probability": 0.2,
        "edge_feather": 5,
    }

    # Build generation config
    generation_config = {
        "depth_aware": depth_aware,
        "max_objects": max_objects,
        "overlap_threshold": overlap_threshold,
        "validate_quality": validate_quality,
        "validate_physics": validate_physics,
    }

    return generation_config, effects, effects_config
