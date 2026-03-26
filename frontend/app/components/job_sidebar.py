"""
Sidebar widget showing active jobs from all services.

Visible on every page so the user can track background jobs
without navigating to the Monitor tab.
"""

import streamlit as st
from typing import List, Dict, Any


# Job type icons
_JOB_ICONS = {
    "generation": "🎨",
    "extraction": "🎯",
    "sam3_conversion": "🔬",
    "labeling": "🤖",
    "relabeling": "🤖",
    "auto_tune": "🔧",
    "domain_analysis": "📊",
    "optimization": "⚡",
    "randomization_batch": "🎲",
    "style_transfer_batch": "🎨",
}


@st.cache_data(ttl=5)
def _fetch_active_jobs() -> List[Dict[str, Any]]:
    """Fetch active jobs from the unified API (cached for 5s)."""
    from app.components.api_client import get_api_client

    try:
        client = get_api_client()
        response = client.list_active_jobs()
        return response.get("jobs", [])
    except Exception:
        return []


def _calc_progress(job: Dict[str, Any]) -> float:
    """Calculate progress percentage from a job dict."""
    total = job.get("total_items", 0)
    processed = job.get("processed_items", 0)
    if total > 0:
        return min(100.0, (processed / total) * 100)
    return 0.0


def _check_job_completions(current_active_ids: set):
    """Detect newly completed jobs and show toast notifications."""
    prev_active = st.session_state.get("_prev_active_job_ids", set())

    newly_completed = prev_active - current_active_ids
    for job_id in newly_completed:
        short_id = job_id[:8] if len(job_id) > 8 else job_id
        st.toast(f"Job {short_id}... finalizado", icon="✅")

    st.session_state._prev_active_job_ids = current_active_ids


def render_active_jobs_sidebar():
    """Render compact active jobs indicator in the sidebar.

    Call this at the end of render_sidebar() in main.py.
    """
    try:
        active_jobs = _fetch_active_jobs()
    except Exception:
        return

    # Check for newly completed jobs (toast notifications)
    current_ids = {j.get("id", "") for j in active_jobs}
    _check_job_completions(current_ids)

    if not active_jobs:
        return

    with st.sidebar:
        st.markdown("---")
        st.markdown(f"**⚡ Jobs Activos ({len(active_jobs)})**")

        for job in active_jobs[:5]:
            job_id = job.get("id", "")
            job_type = job.get("job_type", "unknown")
            icon = _JOB_ICONS.get(job_type, "⏳")
            pct = _calc_progress(job)
            short_id = job_id[:8] if len(job_id) > 8 else job_id

            st.markdown(f"{icon} `{short_id}` {pct:.0f}%")
            st.progress(pct / 100.0)

        if len(active_jobs) > 5:
            st.caption(f"+{len(active_jobs) - 5} mas...")

        if st.button("📊 Abrir Monitor", key="sidebar_open_monitor"):
            st.session_state.nav_menu = "📊 Monitor"
            st.rerun()
