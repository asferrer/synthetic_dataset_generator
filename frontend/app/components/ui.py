"""
Reusable UI Components
======================
Professional, consistent UI components for the Streamlit frontend.
"""

import streamlit as st
from typing import Optional, List, Dict, Any, Union, Callable
from dataclasses import dataclass

from app.config.theme import get_colors_dict


# =============================================================================
# WORKFLOW STEPPER
# =============================================================================

WORKFLOW_STEPS = [
    ("① Análisis", "Analizar dataset"),
    ("② Configurar", "Configurar generación"),
    ("③ Generar", "Generar imágenes"),
    ("④ Exportar", "Exportar formatos"),
    ("⑤ Combinar", "Combinar datasets"),
    ("⑥ Splits", "Crear splits"),
]


def workflow_stepper(
    current_step: int,
    completed_steps: Optional[List[int]] = None,
    on_step_click: Optional[Callable[[int], None]] = None
) -> None:
    """
    Render a visual workflow stepper showing progress through 6 steps.
    Uses native Streamlit columns for better compatibility.

    Args:
        current_step: Current active step (1-6)
        completed_steps: List of completed step numbers
        on_step_click: Optional callback when a step is clicked
    """
    c = get_colors_dict()

    if completed_steps is None:
        completed_steps = []

    # Use native Streamlit columns for layout
    cols = st.columns(11)  # 6 steps + 5 connectors

    step_labels = ["Análisis", "Configurar", "Generar", "Exportar", "Combinar", "Splits"]

    col_idx = 0
    for i in range(1, 7):
        is_current = i == current_step
        is_completed = i in completed_steps

        if is_completed:
            bg_color = c['success']
            text_color = "white"
            icon = "✓"
        elif is_current:
            bg_color = c['primary']
            text_color = "white"
            icon = str(i)
        else:
            bg_color = c['bg_tertiary']
            text_color = c['text_muted']
            icon = str(i)

        opacity = "1" if (is_completed or i <= current_step) else "0.6"

        with cols[col_idx]:
            st.markdown(f"""<div style="text-align: center; opacity: {opacity};">
<div style="width: 32px; height: 32px; border-radius: 50%; background: {bg_color}; margin: 0 auto; line-height: 32px; color: {text_color}; font-weight: 600; font-size: 0.85rem;">{icon}</div>
<div style="font-size: 0.65rem; color: {c['text_muted']}; margin-top: 4px;">{step_labels[i-1]}</div>
</div>""", unsafe_allow_html=True)

        col_idx += 1

        # Add connector line between steps
        if i < 6:
            line_color = c['success'] if i in completed_steps else c['border']
            with cols[col_idx]:
                st.markdown(f"""<div style="height: 2px; background: {line_color}; margin-top: 15px;"></div>""", unsafe_allow_html=True)
            col_idx += 1


def workflow_navigation(
    current_step: int,
    total_steps: int = 6,
    can_go_next: bool = True,
    next_label: str = "Siguiente",
    on_next: Optional[str] = None,
    on_prev: Optional[str] = None,
    show_skip: bool = False
) -> Optional[str]:
    """
    Render workflow navigation buttons (Previous/Next).

    Args:
        current_step: Current step number (1-6)
        total_steps: Total number of steps
        can_go_next: Whether the next button should be enabled
        next_label: Label for the next button
        on_next: Navigation key for next step
        on_prev: Navigation key for previous step
        show_skip: Whether to show a skip button

    Returns:
        Action taken: 'prev', 'next', 'skip', or None
    """
    c = get_colors_dict()

    st.markdown(f"<hr style='border: none; border-top: 1px solid {c['border']}; margin: 1.5rem 0;'>",
                unsafe_allow_html=True)

    cols = st.columns([1, 2, 1]) if not show_skip else st.columns([1, 1, 1, 1])

    action = None

    with cols[0]:
        if current_step > 1:
            if st.button("← Anterior", key="wf_prev", use_container_width=True):
                if on_prev:
                    st.session_state.nav_menu = on_prev
                action = "prev"

    with cols[-1]:
        if current_step < total_steps:
            if st.button(
                f"{next_label} →",
                key="wf_next",
                type="primary",
                disabled=not can_go_next,
                use_container_width=True
            ):
                if on_next:
                    st.session_state.nav_menu = on_next
                action = "next"
        else:
            if st.button(
                "✓ Finalizar",
                key="wf_finish",
                type="primary",
                disabled=not can_go_next,
                use_container_width=True
            ):
                action = "finish"

    if show_skip and len(cols) > 2:
        with cols[2]:
            if st.button("Saltar", key="wf_skip", use_container_width=True):
                action = "skip"

    return action


# =============================================================================
# METRIC CARDS
# =============================================================================

def metric_card(
    title: str,
    value: Union[str, int, float],
    icon: str = "",
    delta: Optional[Union[str, int, float]] = None,
    delta_color: str = "normal",  # normal, inverse, off
    help_text: Optional[str] = None,
    color: str = "default"  # default, primary, success, warning, error
) -> None:
    """
    Render a professional metric card.

    Args:
        title: Card title/label
        value: Main value to display
        icon: Emoji or icon character
        delta: Change/delta value (optional)
        delta_color: Color mode for delta
        help_text: Tooltip help text
        color: Card accent color
    """
    c = get_colors_dict()

    border_colors = {
        "default": c['border'],
        "primary": c['primary'],
        "success": c['success'],
        "warning": c['warning'],
        "error": c['error'],
    }

    text_colors = {
        "default": c['text_primary'],
        "primary": c['primary'],
        "success": c['success'],
        "warning": c['warning'],
        "error": c['error'],
    }

    delta_html = ""
    if delta is not None:
        if isinstance(delta, (int, float)):
            delta_sign = "+" if delta > 0 else ""
            delta_cls_color = c['success'] if delta > 0 else c['error']
            if delta_color == "inverse":
                delta_cls_color = c['error'] if delta > 0 else c['success']
            elif delta_color == "off":
                delta_cls_color = c['text_muted']
            delta_html = f'<div style="font-size: 0.875rem; margin-top: 0.25rem; color: {delta_cls_color};">{delta_sign}{delta}</div>'
        else:
            delta_html = f'<div style="font-size: 0.875rem; margin-top: 0.25rem; color: {c["text_muted"]};">{delta}</div>'

    icon_html = f'<span style="font-size: 1.25rem;">{icon}</span>' if icon else ""
    border_color = border_colors.get(color, c['border'])
    value_color = text_colors.get(color, c['text_primary'])

    st.markdown(f"""
    <div style="background: {c['bg_card']}; border: 1px solid {border_color}; border-radius: 0.75rem; padding: 1rem;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="font-size: 0.75rem; color: {c['text_muted']}; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;">{title}</span>
            {icon_html}
        </div>
        <div style="font-size: 1.5rem; font-weight: 700; color: {value_color}; line-height: 1.2;">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def metric_row(metrics: List[Dict[str, Any]], columns: int = 4) -> None:
    """
    Render a row of metric cards.

    Args:
        metrics: List of metric configurations
        columns: Number of columns
    """
    cols = st.columns(columns)
    for i, metric in enumerate(metrics):
        with cols[i % columns]:
            metric_card(**metric)


# =============================================================================
# STATUS & ALERTS
# =============================================================================

def status_badge(
    text: str,
    status: str = "default",  # default, success, warning, error, info
    icon: str = ""
) -> str:
    """
    Generate HTML for a status badge.

    Returns:
        HTML string for the badge
    """
    c = get_colors_dict()
    icons = {
        "success": "✓",
        "warning": "⚠",
        "error": "✗",
        "info": "ℹ",
        "default": "",
    }
    colors = {
        "success": (c['success_bg'], c['success']),
        "warning": (c['warning_bg'], c['warning']),
        "error": (c['error_bg'], c['error']),
        "info": (c['info_bg'], c['info']),
        "default": (c['bg_tertiary'], c['text_secondary']),
    }

    badge_icon = icon or icons.get(status, "")
    bg, fg = colors.get(status, colors["default"])
    return f'<span style="display: inline-flex; align-items: center; padding: 0.25rem 0.5rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; background: {bg}; color: {fg};">{badge_icon} {text}</span>'


def alert_box(
    message: str,
    type: str = "info",  # info, success, warning, error
    icon: str = "",
    dismissible: bool = False
) -> None:
    """
    Render a custom alert box.

    Args:
        message: Alert message
        type: Alert type
        icon: Custom icon (optional)
        dismissible: Whether alert can be dismissed
    """
    c = get_colors_dict()
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
    }
    colors = {
        "info": (c['info_bg'], c['info']),
        "success": (c['success_bg'], c['success']),
        "warning": (c['warning_bg'], c['warning']),
        "error": (c['error_bg'], c['error']),
    }

    alert_icon = icon or icons.get(type, "ℹ️")
    bg, border = colors.get(type, colors["info"])

    st.markdown(f"""
    <div style="padding: 1rem; border-radius: 0.5rem; display: flex; align-items: center; gap: 0.5rem; background-color: {bg}; border-left: 4px solid {border};">
        <span style="font-size: 1.25rem;">{alert_icon}</span>
        <span style="color: {c['text_primary']};">{message}</span>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# SECTION HEADERS
# =============================================================================

def section_header(
    title: str,
    icon: str = "",
    subtitle: Optional[str] = None,
    divider: bool = True
) -> None:
    """
    Render a section header with icon and optional subtitle.

    Args:
        title: Section title
        icon: Section icon/emoji
        subtitle: Optional subtitle
        divider: Show bottom divider
    """
    c = get_colors_dict()
    subtitle_html = f'<p style="color: {c["text_muted"]}; margin-top: 0.25rem; font-size: 0.875rem;">{subtitle}</p>' if subtitle else ""

    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 0.5rem; margin: 1.5rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 2px solid {c['primary']};">
        <span style="font-size: 1.25rem;">{icon}</span>
        <h3 style="font-size: 1.125rem; font-weight: 600; color: {c['text_primary']}; margin: 0;">{title}</h3>
    </div>
    {subtitle_html}
    """, unsafe_allow_html=True)


def page_header(
    title: str,
    subtitle: Optional[str] = None,
    icon: str = ""
) -> None:
    """
    Render a page header.

    Args:
        title: Page title
        subtitle: Page subtitle
        icon: Page icon/emoji
    """
    c = get_colors_dict()
    icon_html = f'<span style="font-size: 2.5rem; margin-right: 0.5rem;">{icon}</span>' if icon else ""
    subtitle_html = f'<p style="font-size: 1rem; color: {c["text_muted"]}; margin-top: 0.5rem;">{subtitle}</p>' if subtitle else ""

    st.markdown(f"""
    <div style="margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid {c['border']};">
        <div style="display: flex; align-items: center;">
            {icon_html}
            <h1 style="font-size: 1.875rem; font-weight: 700; color: {c['text_primary']}; margin: 0;">{title}</h1>
        </div>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# SERVICE CARDS
# =============================================================================

def service_card(
    name: str,
    status: str,  # healthy, degraded, unhealthy
    latency: Optional[float] = None,
    port: Optional[int] = None,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Render a service status card.

    Args:
        name: Service name
        status: Service status
        latency: Response latency in ms
        port: Service port
        details: Additional service details
    """
    c = get_colors_dict()
    status_lower = status.lower()
    status_configs = {
        "healthy": ("🟢", c['success'], c['success_bg']),
        "degraded": ("🟡", c['warning'], c['warning_bg']),
        "unhealthy": ("🔴", c['error'], c['error_bg']),
    }
    status_icon, status_color, status_bg = status_configs.get(status_lower, ("⚪", c['text_muted'], c['bg_tertiary']))

    latency_html = f'<span style="font-size: 0.75rem; color: {c["text_muted"]};">{latency:.0f}ms</span>' if latency else ""
    port_html = f'<span style="font-size: 0.75rem; color: {c["text_muted"]};">:{port}</span>' if port else ""

    st.markdown(f"""
    <div style="background: {c['bg_card']}; border: 1px solid {c['border']}; border-radius: 0.75rem; padding: 1rem; position: relative; overflow: hidden;">
        <div style="position: absolute; top: 0; left: 0; right: 0; height: 3px; background: {status_color};"></div>
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="font-weight: 600; font-size: 1rem; color: {c['text_primary']};">{name.capitalize()}{port_html}</span>
            <span style="padding: 0.25rem 0.5rem; border-radius: 9999px; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; background: {status_bg}; color: {status_color};">{status_icon} {status.upper()}</span>
        </div>
        {latency_html}
    </div>
    """, unsafe_allow_html=True)

    # Show details in expander if provided
    if details:
        with st.expander("View Details"):
            st.json(details)


# =============================================================================
# PROGRESS COMPONENTS
# =============================================================================

def progress_card(
    title: str,
    current: int,
    total: int,
    status: str = "running",  # running, completed, failed
    subtitle: Optional[str] = None
) -> None:
    """
    Render a progress card with status.

    Args:
        title: Progress title
        current: Current progress value
        total: Total value
        status: Current status
        subtitle: Optional subtitle
    """
    c = get_colors_dict()
    percentage = (current / total * 100) if total > 0 else 0
    status_colors = {
        "running": c['primary'],
        "completed": c['success'],
        "failed": c['error'],
    }
    status_bgs = {
        "running": c['primary_light'],
        "completed": c['success_bg'],
        "failed": c['error_bg'],
    }
    color = status_colors.get(status, c['primary'])
    bg = status_bgs.get(status, c['primary_light'])

    st.markdown(f"""
    <div style="background: {c['bg_card']}; border: 1px solid {c['border']}; border-radius: 0.75rem; padding: 1rem;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="font-size: 0.75rem; color: {c['text_muted']}; font-weight: 500; text-transform: uppercase;">{title}</span>
            <span style="padding: 0.25rem 0.5rem; border-radius: 9999px; font-size: 0.7rem; font-weight: 600; background: {bg}; color: {color};">{status.upper()}</span>
        </div>
        <div style="font-size: 1.5rem; font-weight: 700; color: {c['text_primary']};">{current} / {total}</div>
        <div style="margin-top: 0.5rem;">
            <div style="background: {c['bg_tertiary']}; border-radius: 9999px; height: 8px; overflow: hidden;">
                <div style="background: {color}; width: {percentage}%; height: 100%; border-radius: 9999px;"></div>
            </div>
            <div style="font-size: 0.75rem; color: {c['text_muted']}; margin-top: 0.25rem;">{percentage:.1f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# DATA DISPLAY
# =============================================================================

def stat_grid(stats: List[Dict[str, Any]], columns: int = 4) -> None:
    """
    Render a grid of statistics.

    Args:
        stats: List of stat dictionaries with 'label', 'value', and optional 'icon'
        columns: Number of columns
    """
    c = get_colors_dict()
    cols = st.columns(columns)
    for i, stat in enumerate(stats):
        with cols[i % columns]:
            icon = stat.get('icon', '')
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem;">
                <div style="font-size: 2rem;">{icon}</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: {c['text_primary']};">{stat['value']}</div>
                <div style="font-size: 0.875rem; color: {c['text_muted']};">{stat['label']}</div>
            </div>
            """, unsafe_allow_html=True)


def info_table(data: Dict[str, Any], title: Optional[str] = None) -> None:
    """
    Render a simple info table.

    Args:
        data: Dictionary of key-value pairs
        title: Optional table title
    """
    c = get_colors_dict()
    if title:
        st.markdown(f"**{title}**")

    rows = "".join([
        f'<tr><td style="color: {c["text_muted"]}; padding: 0.5rem 1rem 0.5rem 0;">{k}</td>'
        f'<td style="font-weight: 500; color: {c["text_primary"]}; padding: 0.5rem 0;">{v}</td></tr>'
        for k, v in data.items()
    ])

    st.markdown(f"""
    <table style="width: 100%;">
        {rows}
    </table>
    """, unsafe_allow_html=True)


# =============================================================================
# EMPTY STATES
# =============================================================================

def empty_state(
    title: str,
    message: str,
    icon: str = "📭",
    action_label: Optional[str] = None,
    action_key: Optional[str] = None
) -> bool:
    """
    Render an empty state placeholder.

    Args:
        title: Empty state title
        message: Description message
        icon: Large icon/emoji
        action_label: Optional action button label
        action_key: Button key for action

    Returns:
        True if action button was clicked, False otherwise
    """
    c = get_colors_dict()
    st.markdown(f"""
    <div style="text-align: center; padding: 3rem 2rem; background: {c['bg_secondary']}; border-radius: 0.75rem; border: 1px dashed {c['border']};">
        <div style="font-size: 4rem; margin-bottom: 1rem;">{icon}</div>
        <h3 style="color: {c['text_primary']}; margin-bottom: 0.5rem;">{title}</h3>
        <p style="color: {c['text_muted']}; max-width: 400px; margin: 0 auto;">{message}</p>
    </div>
    """, unsafe_allow_html=True)

    if action_label and action_key:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            return st.button(action_label, key=action_key, use_container_width=True, type="primary")
    return False


# =============================================================================
# LOADING STATES
# =============================================================================

def loading_placeholder(message: str = "Loading...", height: int = 200) -> None:
    """
    Render a loading placeholder.

    Args:
        message: Loading message
        height: Placeholder height in pixels
    """
    c = get_colors_dict()
    st.markdown(f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: {height}px; background: {c['bg_secondary']}; border-radius: 0.75rem;">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">⏳</div>
        <p style="color: {c['text_muted']};">{message}</p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# NAVIGATION HELPERS
# =============================================================================

def tab_bar(tabs: List[str], default: int = 0, key: str = "tabs") -> int:
    """
    Create a custom styled tab bar.

    Args:
        tabs: List of tab names
        default: Default selected tab index
        key: Session state key

    Returns:
        Index of selected tab
    """
    if f"{key}_selected" not in st.session_state:
        st.session_state[f"{key}_selected"] = default

    cols = st.columns(len(tabs))
    for i, tab in enumerate(tabs):
        with cols[i]:
            is_selected = st.session_state[f"{key}_selected"] == i
            if st.button(
                tab,
                key=f"{key}_tab_{i}",
                use_container_width=True,
                type="primary" if is_selected else "secondary"
            ):
                st.session_state[f"{key}_selected"] = i
                st.rerun()

    return st.session_state[f"{key}_selected"]


# =============================================================================
# FORM HELPERS
# =============================================================================

def labeled_input(
    label: str,
    input_type: str = "text",
    help_text: Optional[str] = None,
    required: bool = False,
    **kwargs
) -> Any:
    """
    Render a labeled input with optional help text.

    Args:
        label: Input label
        input_type: Type of input (text, number, select, etc.)
        help_text: Help text tooltip
        required: Whether field is required
        **kwargs: Additional arguments for the input

    Returns:
        Input value
    """
    label_text = f"{label} {'*' if required else ''}"

    if input_type == "text":
        return st.text_input(label_text, help=help_text, **kwargs)
    elif input_type == "number":
        return st.number_input(label_text, help=help_text, **kwargs)
    elif input_type == "select":
        return st.selectbox(label_text, help=help_text, **kwargs)
    elif input_type == "multiselect":
        return st.multiselect(label_text, help=help_text, **kwargs)
    elif input_type == "textarea":
        return st.text_area(label_text, help=help_text, **kwargs)
    elif input_type == "slider":
        return st.slider(label_text, help=help_text, **kwargs)
    elif input_type == "checkbox":
        return st.checkbox(label_text, help=help_text, **kwargs)
    else:
        return st.text_input(label_text, help=help_text, **kwargs)


def form_section(title: str, icon: str = "") -> None:
    """
    Create a form section divider.

    Args:
        title: Section title
        icon: Section icon
    """
    c = get_colors_dict()
    st.markdown(f"""
    <div style="margin: 1.5rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid {c['border']};">
        <span style="font-weight: 600; color: {c['text_primary']};">{icon} {title}</span>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# DIVIDERS & SPACING
# =============================================================================

def spacer(height: int = 24) -> None:
    """Add vertical spacing."""
    st.markdown(f'<div style="height: {height}px;"></div>', unsafe_allow_html=True)


def divider_with_text(text: str) -> None:
    """Render a divider with centered text."""
    c = get_colors_dict()
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin: 1.5rem 0;">
        <div style="flex: 1; height: 1px; background: {c['border']};"></div>
        <span style="padding: 0 1rem; color: {c['text_muted']}; font-size: 0.875rem;">{text}</span>
        <div style="flex: 1; height: 1px; background: {c['border']};"></div>
    </div>
    """, unsafe_allow_html=True)
