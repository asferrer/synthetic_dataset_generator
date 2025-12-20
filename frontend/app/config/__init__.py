# Frontend configuration module
from .theme import (
    get_theme_css,
    get_theme_colors,
    get_colors_dict,
    get_current_theme,
    set_theme,
    render_theme_toggle,
    THEMES,
    LIGHT_THEME,
    DARK_THEME,
    ThemeColors,
)
from .styles import get_custom_css, inject_styles

__all__ = [
    "get_theme_css",
    "get_theme_colors",
    "get_colors_dict",
    "get_current_theme",
    "set_theme",
    "render_theme_toggle",
    "THEMES",
    "LIGHT_THEME",
    "DARK_THEME",
    "ThemeColors",
    "get_custom_css",
    "inject_styles",
]
