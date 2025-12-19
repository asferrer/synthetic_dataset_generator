"""
Effects Sidebar Component
=========================
Configurable sidebar for effect settings.
"""

import streamlit as st
from typing import Dict, List, Tuple


def render_effects_sidebar() -> Tuple[Dict, List[str], Dict]:
    """
    Render the effects configuration sidebar.

    Returns:
        Tuple of (generation_config, effects_list, effects_config)
    """
    with st.sidebar:
        st.header("Configuration")

        # Service Status
        st.subheader("Services")
        from app.components.api_client import get_api_client
        client = get_api_client()

        if st.button("Refresh Status", key="refresh_status"):
            st.rerun()

        health = client.get_health()
        if health and "services" in health:
            for service in health.get("services", []):
                status = service.get("status", "unknown")
                name = service.get("name", "unknown")

                if status == "healthy":
                    st.success(f"{name.capitalize()}")
                elif status == "degraded":
                    st.warning(f"{name.capitalize()}")
                else:
                    st.error(f"{name.capitalize()}")
        else:
            st.error("Gateway unavailable")

        st.divider()

        # Generation Settings
        st.subheader("Generation")
        depth_aware = st.checkbox("Depth-aware placement", value=True, key="depth_aware")
        max_objects = st.slider("Max objects per image", 1, 10, 5, key="max_objects")
        overlap_threshold = st.slider("Overlap threshold", 0.0, 0.5, 0.1, key="overlap")

        st.divider()

        # Effects Selection
        st.subheader("Effects")

        col1, col2 = st.columns(2)

        with col1:
            color_correction = st.checkbox("Color correction", value=True, key="fx_color")
            blur_matching = st.checkbox("Blur matching", value=True, key="fx_blur")
            shadows = st.checkbox("Shadows", value=True, key="fx_shadows")
            caustics = st.checkbox("Caustics", value=True, key="fx_caustics")

        with col2:
            underwater = st.checkbox("Underwater", value=True, key="fx_underwater")
            lighting = st.checkbox("Lighting", value=False, key="fx_lighting")
            motion_blur = st.checkbox("Motion blur", value=False, key="fx_motion")
            edge_smoothing = st.checkbox("Edge smoothing", value=True, key="fx_edge")

        st.divider()

        # Effect Intensities
        st.subheader("Intensities")

        color_intensity = st.slider(
            "Color correction", 0.0, 1.0, 0.7,
            key="int_color",
            help="Intensity of color transfer from background to objects"
        )

        blur_strength = st.slider(
            "Blur matching", 0.0, 3.0, 1.0,
            key="int_blur",
            help="Strength of blur matching between objects and background"
        )

        underwater_intensity = st.slider(
            "Underwater effect", 0.0, 1.0, 0.25,
            key="int_underwater",
            help="Intensity of underwater color tinting"
        )

        caustics_intensity = st.slider(
            "Caustics", 0.0, 0.5, 0.15,
            key="int_caustics",
            help="Intensity of underwater light caustics"
        )

        shadow_opacity = st.slider(
            "Shadow opacity", 0.0, 1.0, 0.4,
            key="int_shadow",
            help="Darkness of generated shadows"
        )

        st.divider()

        # Validation Settings
        st.subheader("Validation")

        validate_quality = st.checkbox(
            "Quality validation",
            value=False,
            key="val_quality",
            help="Run LPIPS perceptual quality check"
        )

        validate_physics = st.checkbox(
            "Physics validation",
            value=False,
            key="val_physics",
            help="Check for gravity/buoyancy violations"
        )

        st.divider()

        # Water Settings (collapsible)
        with st.expander("Water Settings"):
            water_clarity = st.selectbox(
                "Water clarity",
                ["clear", "murky", "very_murky"],
                key="water_clarity"
            )

            water_r = st.slider("Water R", 0, 255, 120, key="water_r")
            water_g = st.slider("Water G", 0, 255, 80, key="water_g")
            water_b = st.slider("Water B", 0, 255, 20, key="water_b")

        # Lighting Settings (collapsible)
        with st.expander("Lighting Settings"):
            lighting_type = st.selectbox(
                "Lighting type",
                ["ambient", "spotlight", "gradient"],
                key="lighting_type"
            )
            lighting_intensity = st.slider(
                "Lighting intensity", 0.0, 1.0, 0.5,
                key="lighting_intensity"
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
