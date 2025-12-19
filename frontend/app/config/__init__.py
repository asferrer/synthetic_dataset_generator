# Frontend configuration module
from .theme import ThemeManager, get_theme_css, THEMES
from .styles import get_custom_css, inject_styles

__all__ = [
    "ThemeManager",
    "get_theme_css",
    "THEMES",
    "get_custom_css",
    "inject_styles",
]
