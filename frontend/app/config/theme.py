"""
Theme Configuration System
===========================
Simplified theme system using Streamlit's native Light/Dark modes.
"""

import streamlit as st
from dataclasses import dataclass
from typing import Dict


@dataclass
class ThemeColors:
    """Theme color palette"""
    # Primary colors
    primary: str
    primary_hover: str
    primary_light: str

    # Background colors
    bg_primary: str
    bg_secondary: str
    bg_tertiary: str
    bg_card: str

    # Text colors
    text_primary: str
    text_secondary: str
    text_muted: str

    # Status colors
    success: str
    success_bg: str
    warning: str
    warning_bg: str
    error: str
    error_bg: str
    info: str
    info_bg: str

    # Border colors
    border: str
    border_light: str

    # Shadows
    shadow: str
    shadow_lg: str


# Light Theme
LIGHT_THEME = ThemeColors(
    primary="#0066FF",
    primary_hover="#0052CC",
    primary_light="#E6F0FF",

    bg_primary="#FFFFFF",
    bg_secondary="#F8FAFC",
    bg_tertiary="#F1F5F9",
    bg_card="#FFFFFF",

    text_primary="#1E293B",
    text_secondary="#475569",
    text_muted="#94A3B8",

    success="#10B981",
    success_bg="#ECFDF5",
    warning="#F59E0B",
    warning_bg="#FFFBEB",
    error="#EF4444",
    error_bg="#FEF2F2",
    info="#3B82F6",
    info_bg="#EFF6FF",

    border="#E2E8F0",
    border_light="#F1F5F9",

    shadow="0 1px 3px rgba(0, 0, 0, 0.1)",
    shadow_lg="0 4px 6px -1px rgba(0, 0, 0, 0.1)",
)


# Dark Theme
DARK_THEME = ThemeColors(
    primary="#3B82F6",
    primary_hover="#60A5FA",
    primary_light="#1E3A5F",

    bg_primary="#0F172A",
    bg_secondary="#1E293B",
    bg_tertiary="#334155",
    bg_card="#1E293B",

    text_primary="#F8FAFC",
    text_secondary="#CBD5E1",
    text_muted="#64748B",

    success="#34D399",
    success_bg="#064E3B",
    warning="#FBBF24",
    warning_bg="#78350F",
    error="#F87171",
    error_bg="#7F1D1D",
    info="#60A5FA",
    info_bg="#1E3A5F",

    border="#334155",
    border_light="#475569",

    shadow="0 1px 3px rgba(0, 0, 0, 0.3)",
    shadow_lg="0 4px 6px -1px rgba(0, 0, 0, 0.4)",
)


THEMES: Dict[str, ThemeColors] = {
    "light": LIGHT_THEME,
    "dark": DARK_THEME,
}


def get_current_theme() -> str:
    """Get current theme name from session state"""
    if "theme" not in st.session_state:
        st.session_state.theme = "light"
    return st.session_state.theme


def set_theme(theme_name: str) -> None:
    """Set theme in session state"""
    if theme_name in THEMES:
        st.session_state.theme = theme_name


def get_theme_colors() -> ThemeColors:
    """Get current theme colors"""
    return THEMES.get(get_current_theme(), LIGHT_THEME)


def render_theme_toggle() -> None:
    """Render a simple Light/Dark theme toggle"""
    current = get_current_theme()

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "☀️ Light",
            key="theme_light",
            use_container_width=True,
            type="primary" if current == "light" else "secondary"
        ):
            set_theme("light")
            st.rerun()

    with col2:
        if st.button(
            "🌙 Dark",
            key="theme_dark",
            use_container_width=True,
            type="primary" if current == "dark" else "secondary"
        ):
            set_theme("dark")
            st.rerun()


def get_colors_dict() -> Dict[str, str]:
    """Get current theme colors as a dictionary for inline styles.

    Use this for inline HTML styles instead of var() CSS variables.
    Example: style=f"color: {c['text_primary']}; background: {c['bg_card']};"
    """
    colors = get_theme_colors()
    return {
        # Primary
        "primary": colors.primary,
        "primary_hover": colors.primary_hover,
        "primary_light": colors.primary_light,
        # Backgrounds
        "bg_primary": colors.bg_primary,
        "bg_secondary": colors.bg_secondary,
        "bg_tertiary": colors.bg_tertiary,
        "bg_card": colors.bg_card,
        # Text
        "text_primary": colors.text_primary,
        "text_secondary": colors.text_secondary,
        "text_muted": colors.text_muted,
        # Status
        "success": colors.success,
        "success_bg": colors.success_bg,
        "warning": colors.warning,
        "warning_bg": colors.warning_bg,
        "error": colors.error,
        "error_bg": colors.error_bg,
        "info": colors.info,
        "info_bg": colors.info_bg,
        # Borders
        "border": colors.border,
        "border_light": colors.border_light,
        # Shadows
        "shadow": colors.shadow,
        "shadow_lg": colors.shadow_lg,
        # Accents
        "accent_1": "#EC4899",
        "accent_2": "#14B8A6",
        "accent_3": "#F97316",
    }


def get_theme_css() -> str:
    """Generate CSS variables for current theme"""
    colors = get_theme_colors()

    return f"""
    :root {{
        /* Primary */
        --color-primary: {colors.primary};
        --color-primary-hover: {colors.primary_hover};
        --color-primary-light: {colors.primary_light};

        /* Backgrounds */
        --color-bg-primary: {colors.bg_primary};
        --color-bg-secondary: {colors.bg_secondary};
        --color-bg-tertiary: {colors.bg_tertiary};
        --color-bg-card: {colors.bg_card};

        /* Text */
        --color-text-primary: {colors.text_primary};
        --color-text-secondary: {colors.text_secondary};
        --color-text-muted: {colors.text_muted};

        /* Status */
        --color-success: {colors.success};
        --color-success-bg: {colors.success_bg};
        --color-warning: {colors.warning};
        --color-warning-bg: {colors.warning_bg};
        --color-error: {colors.error};
        --color-error-bg: {colors.error_bg};
        --color-info: {colors.info};
        --color-info-bg: {colors.info_bg};

        /* Borders */
        --color-border: {colors.border};
        --color-border-light: {colors.border_light};

        /* Shadows */
        --shadow: {colors.shadow};
        --shadow-lg: {colors.shadow_lg};

        /* Accent colors (for gradients) */
        --color-accent-1: #EC4899;
        --color-accent-2: #14B8A6;
        --color-accent-3: #F97316;

        /* Spacing */
        --space-xs: 0.25rem;
        --space-sm: 0.5rem;
        --space-md: 1rem;
        --space-lg: 1.5rem;
        --space-xl: 2rem;
        --space-2xl: 3rem;

        /* Border radius */
        --radius-sm: 0.375rem;
        --radius-md: 0.5rem;
        --radius-lg: 0.75rem;
        --radius-xl: 1rem;
        --radius-full: 9999px;

        /* Typography */
        --font-size-xs: 0.75rem;
        --font-size-sm: 0.875rem;
        --font-size-md: 1rem;
        --font-size-lg: 1.125rem;
        --font-size-xl: 1.25rem;
        --font-size-2xl: 1.5rem;
        --font-size-3xl: 1.875rem;

        /* Transitions */
        --transition-fast: 150ms ease;
        --transition-normal: 250ms ease;
        --transition-slow: 350ms ease;
    }}
    """
