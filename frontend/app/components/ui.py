"""
Reusable UI Components
======================
Professional, consistent UI components for the Streamlit frontend.
"""

import streamlit as st
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass


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
    color_classes = {
        "default": "",
        "primary": "border-primary",
        "success": "text-success",
        "warning": "text-warning",
        "error": "text-error",
    }

    delta_html = ""
    if delta is not None:
        if isinstance(delta, (int, float)):
            delta_sign = "+" if delta > 0 else ""
            delta_class = "positive" if delta > 0 else "negative"
            if delta_color == "inverse":
                delta_class = "negative" if delta > 0 else "positive"
            elif delta_color == "off":
                delta_class = ""
            delta_html = f'<div class="metric-card-delta {delta_class}">{delta_sign}{delta}</div>'
        else:
            delta_html = f'<div class="metric-card-delta">{delta}</div>'

    icon_html = f'<span class="metric-icon">{icon}</span>' if icon else ""
    help_attr = f'title="{help_text}"' if help_text else ""

    st.markdown(f"""
    <div class="metric-card {color_classes.get(color, '')}" {help_attr}>
        <div class="metric-card-header">
            <span class="metric-card-title">{title}</span>
            {icon_html}
        </div>
        <div class="metric-card-value">{value}</div>
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
    icons = {
        "success": "✓",
        "warning": "⚠",
        "error": "✗",
        "info": "ℹ",
        "default": "",
    }
    badge_icon = icon or icons.get(status, "")
    return f'<span class="badge {status}">{badge_icon} {text}</span>'


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
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
    }
    alert_icon = icon or icons.get(type, "ℹ️")

    st.markdown(f"""
    <div class="status-card {type}">
        <span class="status-icon">{alert_icon}</span>
        <span class="status-message">{message}</span>
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
    subtitle_html = f'<p class="section-subtitle text-muted">{subtitle}</p>' if subtitle else ""

    st.markdown(f"""
    <div class="section-header">
        <span class="section-icon">{icon}</span>
        <h3 class="section-title">{title}</h3>
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
    icon_html = f'<span style="font-size: 2.5rem; margin-right: 0.5rem;">{icon}</span>' if icon else ""
    subtitle_html = f'<p class="page-subtitle">{subtitle}</p>' if subtitle else ""

    st.markdown(f"""
    <div class="page-header">
        <div style="display: flex; align-items: center;">
            {icon_html}
            <h1 class="page-title">{title}</h1>
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
    status_lower = status.lower()
    status_icon = {
        "healthy": "🟢",
        "degraded": "🟡",
        "unhealthy": "🔴",
    }.get(status_lower, "⚪")

    latency_html = f'<span class="service-latency">{latency:.0f}ms</span>' if latency else ""
    port_html = f'<span class="service-port">:{port}</span>' if port else ""

    st.markdown(f"""
    <div class="service-card {status_lower}">
        <div class="service-card-header">
            <span class="service-name">{name.capitalize()}{port_html}</span>
            <span class="service-status {status_lower}">{status_icon} {status.upper()}</span>
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
    percentage = (current / total * 100) if total > 0 else 0
    status_colors = {
        "running": "var(--color-primary)",
        "completed": "var(--color-success)",
        "failed": "var(--color-error)",
    }
    color = status_colors.get(status, "var(--color-primary)")

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-header">
            <span class="metric-card-title">{title}</span>
            <span class="badge {'success' if status == 'completed' else 'primary' if status == 'running' else 'error'}">{status.upper()}</span>
        </div>
        <div class="metric-card-value">{current} / {total}</div>
        <div style="margin-top: 0.5rem;">
            <div style="background: var(--color-bg-tertiary); border-radius: 9999px; height: 8px; overflow: hidden;">
                <div style="background: {color}; width: {percentage}%; height: 100%; border-radius: 9999px; transition: width 0.3s ease;"></div>
            </div>
            <div class="text-muted" style="font-size: 0.75rem; margin-top: 0.25rem;">{percentage:.1f}%</div>
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
    cols = st.columns(columns)
    for i, stat in enumerate(stats):
        with cols[i % columns]:
            icon = stat.get('icon', '')
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem;">
                <div style="font-size: 2rem;">{icon}</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: var(--color-text-primary);">{stat['value']}</div>
                <div style="font-size: 0.875rem; color: var(--color-text-muted);">{stat['label']}</div>
            </div>
            """, unsafe_allow_html=True)


def info_table(data: Dict[str, Any], title: Optional[str] = None) -> None:
    """
    Render a simple info table.

    Args:
        data: Dictionary of key-value pairs
        title: Optional table title
    """
    if title:
        st.markdown(f"**{title}**")

    rows = "".join([
        f'<tr><td style="color: var(--color-text-muted); padding: 0.5rem 1rem 0.5rem 0;">{k}</td>'
        f'<td style="font-weight: 500; padding: 0.5rem 0;">{v}</td></tr>'
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
    st.markdown(f"""
    <div style="text-align: center; padding: 3rem 2rem; background: var(--color-bg-secondary); border-radius: var(--radius-lg); border: 1px dashed var(--color-border);">
        <div style="font-size: 4rem; margin-bottom: 1rem;">{icon}</div>
        <h3 style="color: var(--color-text-primary); margin-bottom: 0.5rem;">{title}</h3>
        <p style="color: var(--color-text-muted); max-width: 400px; margin: 0 auto;">{message}</p>
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
    st.markdown(f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: {height}px; background: var(--color-bg-secondary); border-radius: var(--radius-lg);">
        <div class="animate-pulse" style="font-size: 2rem; margin-bottom: 0.5rem;">⏳</div>
        <p style="color: var(--color-text-muted);">{message}</p>
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
    st.markdown(f"""
    <div style="margin: 1.5rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid var(--color-border);">
        <span style="font-weight: 600; color: var(--color-text-primary);">{icon} {title}</span>
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
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin: 1.5rem 0;">
        <div style="flex: 1; height: 1px; background: var(--color-border);"></div>
        <span style="padding: 0 1rem; color: var(--color-text-muted); font-size: 0.875rem;">{text}</span>
        <div style="flex: 1; height: 1px; background: var(--color-border);"></div>
    </div>
    """, unsafe_allow_html=True)
