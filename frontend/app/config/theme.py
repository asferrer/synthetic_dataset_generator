"""
Theme Configuration System
===========================
Professional theme system with dark/light mode support.
"""

import streamlit as st
from dataclasses import dataclass
from typing import Dict, Any


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

    # Accent colors
    accent_1: str
    accent_2: str
    accent_3: str

    # Shadows
    shadow: str
    shadow_lg: str


# Light Theme
LIGHT_THEME = ThemeColors(
    # Primary - Ocean blue for underwater/synthetic data theme
    primary="#0066FF",
    primary_hover="#0052CC",
    primary_light="#E6F0FF",

    # Backgrounds
    bg_primary="#FFFFFF",
    bg_secondary="#F8FAFC",
    bg_tertiary="#F1F5F9",
    bg_card="#FFFFFF",

    # Text
    text_primary="#1E293B",
    text_secondary="#475569",
    text_muted="#94A3B8",

    # Status
    success="#10B981",
    success_bg="#ECFDF5",
    warning="#F59E0B",
    warning_bg="#FFFBEB",
    error="#EF4444",
    error_bg="#FEF2F2",
    info="#3B82F6",
    info_bg="#EFF6FF",

    # Borders
    border="#E2E8F0",
    border_light="#F1F5F9",

    # Accents
    accent_1="#8B5CF6",  # Purple
    accent_2="#06B6D4",  # Cyan
    accent_3="#F97316",  # Orange

    # Shadows
    shadow="0 1px 3px rgba(0, 0, 0, 0.1)",
    shadow_lg="0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
)


# Dark Theme
DARK_THEME = ThemeColors(
    # Primary - Bright blue for contrast
    primary="#3B82F6",
    primary_hover="#60A5FA",
    primary_light="#1E3A5F",

    # Backgrounds
    bg_primary="#0F172A",
    bg_secondary="#1E293B",
    bg_tertiary="#334155",
    bg_card="#1E293B",

    # Text
    text_primary="#F8FAFC",
    text_secondary="#CBD5E1",
    text_muted="#64748B",

    # Status
    success="#34D399",
    success_bg="#064E3B",
    warning="#FBBF24",
    warning_bg="#78350F",
    error="#F87171",
    error_bg="#7F1D1D",
    info="#60A5FA",
    info_bg="#1E3A5F",

    # Borders
    border="#334155",
    border_light="#475569",

    # Accents
    accent_1="#A78BFA",  # Purple
    accent_2="#22D3EE",  # Cyan
    accent_3="#FB923C",  # Orange

    # Shadows
    shadow="0 1px 3px rgba(0, 0, 0, 0.3)",
    shadow_lg="0 4px 6px -1px rgba(0, 0, 0, 0.4), 0 2px 4px -1px rgba(0, 0, 0, 0.2)",
)


# Ocean Theme (Custom underwater aesthetic)
OCEAN_THEME = ThemeColors(
    # Primary - Deep ocean blue
    primary="#0077B6",
    primary_hover="#0096C7",
    primary_light="#CAF0F8",

    # Backgrounds
    bg_primary="#03045E",
    bg_secondary="#023E8A",
    bg_tertiary="#0077B6",
    bg_card="#023E8A",

    # Text
    text_primary="#CAF0F8",
    text_secondary="#90E0EF",
    text_muted="#48CAE4",

    # Status
    success="#4ADE80",
    success_bg="#064E3B",
    warning="#FCD34D",
    warning_bg="#78350F",
    error="#FB7185",
    error_bg="#881337",
    info="#38BDF8",
    info_bg="#0C4A6E",

    # Borders
    border="#0077B6",
    border_light="#0096C7",

    # Accents
    accent_1="#C084FC",  # Purple
    accent_2="#22D3EE",  # Cyan
    accent_3="#FB923C",  # Orange

    # Shadows
    shadow="0 1px 3px rgba(0, 0, 0, 0.4)",
    shadow_lg="0 4px 6px -1px rgba(0, 0, 0, 0.5), 0 2px 4px -1px rgba(0, 0, 0, 0.3)",
)


THEMES = {
    "light": LIGHT_THEME,
    "dark": DARK_THEME,
    "ocean": OCEAN_THEME,
}


class ThemeManager:
    """Manages theme state and CSS generation"""

    @staticmethod
    def get_current_theme() -> str:
        """Get current theme name from session state"""
        if "theme" not in st.session_state:
            st.session_state.theme = "light"
        return st.session_state.theme

    @staticmethod
    def set_theme(theme_name: str) -> None:
        """Set theme in session state"""
        if theme_name in THEMES:
            st.session_state.theme = theme_name

    @staticmethod
    def get_theme_colors() -> ThemeColors:
        """Get current theme colors"""
        theme_name = ThemeManager.get_current_theme()
        return THEMES.get(theme_name, LIGHT_THEME)

    @staticmethod
    def render_theme_selector() -> None:
        """Render theme selector in sidebar"""
        current = ThemeManager.get_current_theme()

        theme_icons = {
            "light": "☀️",
            "dark": "🌙",
            "ocean": "🌊",
        }

        theme_labels = {
            "light": "Light",
            "dark": "Dark",
            "ocean": "Ocean",
        }

        # Create columns for theme buttons
        cols = st.columns(3)

        for i, (theme_name, icon) in enumerate(theme_icons.items()):
            with cols[i]:
                if st.button(
                    f"{icon}",
                    key=f"theme_btn_{theme_name}",
                    use_container_width=True,
                    type="primary" if current == theme_name else "secondary",
                    help=f"{theme_labels[theme_name]} theme"
                ):
                    ThemeManager.set_theme(theme_name)
                    st.rerun()


def get_theme_css() -> str:
    """Generate CSS variables for current theme"""
    colors = ThemeManager.get_theme_colors()

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

        /* Accents */
        --color-accent-1: {colors.accent_1};
        --color-accent-2: {colors.accent_2};
        --color-accent-3: {colors.accent_3};

        /* Shadows */
        --shadow: {colors.shadow};
        --shadow-lg: {colors.shadow_lg};

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
