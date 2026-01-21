"""
Dataset Browser Component
=========================
Reusable component for browsing and selecting datasets.
"""

import streamlit as st
from datetime import datetime
from typing import Optional, Dict, Any, Callable
from pathlib import Path

from app.components.api_client import get_api_client
from app.config.theme import get_colors_dict


def render_dataset_browser(
    on_select: Callable[[Dict], None],
    dataset_type: Optional[str] = None,
    title: str = "Seleccionar Dataset",
    show_preview: bool = True
) -> Optional[Dict]:
    """
    Render dataset browser with selection functionality.

    Args:
        on_select: Callback when dataset is selected (receives metadata dict)
        dataset_type: Filter by type ('generation', 'extraction', 'combined')
        title: Browser title
        show_preview: Show image previews

    Returns:
        Selected dataset metadata or None
    """
    c = get_colors_dict()
    client = get_api_client()

    st.markdown(f"### {title}")

    # Fetch datasets
    with st.spinner("Cargando datasets..."):
        result = client.list_datasets(dataset_type=dataset_type, limit=50)

    if result.get("error"):
        st.error(f"Error cargando datasets: {result['error']}")
        return None

    datasets = result.get("datasets", [])

    if not datasets:
        st.info("No hay datasets disponibles. Genera uno primero en el paso ③ Generar.")
        return None

    # Dataset filter options
    col1, col2 = st.columns([3, 1])
    with col1:
        search = st.text_input("🔍 Buscar por nombre o job_id", key="dataset_search")
    with col2:
        sort_by = st.selectbox("Ordenar por", ["Más reciente", "Más antiguo", "Más imágenes"], key="dataset_sort")

    # Apply filters
    filtered = datasets
    if search:
        filtered = [d for d in datasets if search.lower() in d.get("dataset_name", "").lower()
                    or search.lower() in d.get("job_id", "").lower()]

    # Apply sorting
    if sort_by == "Más reciente":
        filtered.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    elif sort_by == "Más antiguo":
        filtered.sort(key=lambda x: x.get("created_at", ""))
    elif sort_by == "Más imágenes":
        filtered.sort(key=lambda x: x.get("num_images", 0), reverse=True)

    st.markdown(f"**{len(filtered)} datasets encontrados**")

    # Render dataset cards
    for dataset in filtered:
        _render_dataset_card(dataset, on_select, show_preview, c)

    return None


def _render_dataset_card(
    dataset: Dict,
    on_select: Callable,
    show_preview: bool,
    c: Dict
) -> None:
    """Render a single dataset card."""
    job_id = dataset.get("job_id", "")
    name = dataset.get("dataset_name", "Unknown")
    num_images = dataset.get("num_images", 0)
    num_annotations = dataset.get("num_annotations", 0)
    num_categories = dataset.get("num_categories", 0)
    created_at = dataset.get("created_at", "")
    file_size = dataset.get("file_size_mb", 0)
    class_dist = dataset.get("class_distribution", {})

    # Format timestamp
    try:
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        time_str = dt.strftime("%Y-%m-%d %H:%M")
    except:
        time_str = created_at[:16] if created_at else "Desconocido"

    with st.container():
        st.markdown(f"""
        <div style="background: {c['bg_card']}; border: 1px solid {c['border']};
                    border-radius: 0.75rem; padding: 1.25rem; margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
                <div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: {c['text_primary']}; margin-bottom: 0.25rem;">
                        {name}
                    </div>
                    <div style="font-size: 0.8rem; color: {c['text_muted']}; font-family: monospace;">
                        {job_id}
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.8rem; color: {c['text_muted']};">
                        📅 {time_str}
                    </div>
                    <div style="font-size: 0.8rem; color: {c['text_muted']};">
                        💾 {file_size:.1f} MB
                    </div>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;
                        text-align: center; margin-bottom: 1rem; padding: 0.75rem;
                        background: {c['bg_secondary']}; border-radius: 0.5rem;">
                <div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: {c['primary']};">{num_images:,}</div>
                    <div style="font-size: 0.75rem; color: {c['text_muted']};">Imágenes</div>
                </div>
                <div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: {c['text_primary']};">{num_annotations:,}</div>
                    <div style="font-size: 0.75rem; color: {c['text_muted']};">Anotaciones</div>
                </div>
                <div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: {c['text_primary']};">{num_categories}</div>
                    <div style="font-size: 0.75rem; color: {c['text_muted']};">Clases</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Class distribution expander
        if class_dist:
            with st.expander("📊 Distribución de clases", expanded=False):
                import pandas as pd
                df = pd.DataFrame([
                    {"Clase": k, "Anotaciones": v}
                    for k, v in sorted(class_dist.items(), key=lambda x: -x[1])
                ])
                st.dataframe(df, use_container_width=True, hide_index=True)

        # Preview images
        if show_preview and dataset.get("preview_images"):
            with st.expander("🖼️ Vista previa", expanded=False):
                cols = st.columns(5)
                for idx, img_path in enumerate(dataset.get("preview_images", [])[:5]):
                    if Path(img_path).exists():
                        with cols[idx]:
                            try:
                                from PIL import Image
                                img = Image.open(img_path)
                                st.image(img, use_container_width=True)
                            except:
                                st.caption("Error cargando imagen")

        # Select button
        if st.button(
            "✅ Seleccionar este dataset",
            key=f"select_{job_id}",
            type="primary",
            use_container_width=True
        ):
            on_select(dataset)
