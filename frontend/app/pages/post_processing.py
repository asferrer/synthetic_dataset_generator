"""
Post-Processing Page
====================
Dataset processing, label management, export, and splits.
"""

import os
import json
import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from app.utils import (
    LabelManager, ExportManager, DatasetSplitter,
    KFoldGenerator, KFoldConfig,
    ClassBalancer, ClassWeightsCalculator, BalancingConfig
)
from app.components.ui import page_header, section_header, metric_card, alert_box, empty_state, spacer
from app.config.theme import get_colors_dict


def render_post_processing_page():
    """Render the post-processing page with horizontal sub-navigation."""
    page_header(
        title="Post-Processing",
        subtitle="Procesa cualquier dataset COCO: etiquetas, exportación, splits y balanceo",
        icon="🛠️"
    )

    # Initialize session state
    if 'pp_coco_data' not in st.session_state:
        st.session_state.pp_coco_data = None

    # =========================================================================
    # ALWAYS VISIBLE: Load Dataset Section
    # =========================================================================
    _render_load_dataset_section()

    spacer(16)

    # =========================================================================
    # HORIZONTAL SUB-NAVIGATION
    # =========================================================================
    if st.session_state.pp_coco_data:
        nav_options = ["🏷️ Etiquetas", "📤 Exportar", "✂️ Splits", "⚖️ Balanceo"]

        selected_section = st.radio(
            "Herramientas de Post-Processing",
            options=nav_options,
            horizontal=True,
            key="pp_nav_section",
            label_visibility="collapsed"
        )

        c = get_colors_dict()
        st.markdown(f"<hr style='border: none; border-top: 1px solid {c['border']}; margin: 1rem 0;'>", unsafe_allow_html=True)

        # Route to the appropriate section
        if selected_section == "🏷️ Etiquetas":
            _render_labels_section()
        elif selected_section == "📤 Exportar":
            _render_export_section()
        elif selected_section == "✂️ Splits":
            _render_splits_section()
        elif selected_section == "⚖️ Balanceo":
            _render_balancing_section()

        # Download always at the bottom when data is loaded
        spacer(24)
        _render_download_section()
    else:
        empty_state(
            title="No hay dataset cargado",
            message="Carga un archivo COCO JSON para acceder a las herramientas de post-processing.",
            icon="📂"
        )


# =============================================================================
# LOAD DATASET SECTION (Always visible)
# =============================================================================

def _render_load_dataset_section():
    """Render the dataset loading section."""
    section_header("Cargar Dataset", icon="📁")

    # Check if there's data from analysis
    has_analyzed = 'coco_data' in st.session_state and st.session_state.coco_data is not None

    data_source_options = ["Subir archivo COCO JSON"]
    if has_analyzed:
        data_source_options.append("Usar dataset analizado")

    col1, col2 = st.columns([2, 1])

    with col1:
        data_source = st.radio(
            "Origen de datos",
            options=data_source_options,
            horizontal=True,
            key='pp_data_source',
            label_visibility="collapsed"
        )

    if data_source == "Subir archivo COCO JSON":
        uploaded = st.file_uploader(
            "Archivo COCO JSON",
            type=['json'],
            key='pp_upload',
            label_visibility="collapsed"
        )
        if uploaded:
            try:
                st.session_state.pp_coco_data = json.load(uploaded)
                n_images = len(st.session_state.pp_coco_data.get('images', []))
                n_anns = len(st.session_state.pp_coco_data.get('annotations', []))
                n_cats = len(st.session_state.pp_coco_data.get('categories', []))
                st.success(f"Cargado: {n_images} imágenes, {n_anns} anotaciones, {n_cats} categorías")
            except Exception as e:
                st.error(f"Error al cargar JSON: {e}")

    elif data_source == "Usar dataset analizado" and has_analyzed:
        st.session_state.pp_coco_data = st.session_state.coco_data
        n_images = len(st.session_state.pp_coco_data.get('images', []))
        n_anns = len(st.session_state.pp_coco_data.get('annotations', []))
        st.success(f"Usando dataset analizado: {n_images} imágenes, {n_anns} anotaciones")

    # Show current dataset status
    if st.session_state.pp_coco_data:
        _render_dataset_overview()


def _render_dataset_overview():
    """Render compact dataset overview."""
    c = get_colors_dict()
    with st.expander("📊 Vista general del dataset", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div style="background: {c['bg_card']}; border: 1px solid {c['border']}; border-radius: 0.5rem; padding: 1rem; text-align: center;">
                <div style="font-size: 1.5rem; font-weight: 700; color: {c['text_primary']};">{len(st.session_state.pp_coco_data.get('images', []))}</div>
                <div style="font-size: 0.75rem; color: {c['text_muted']};">Imágenes</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="background: {c['bg_card']}; border: 1px solid {c['border']}; border-radius: 0.5rem; padding: 1rem; text-align: center;">
                <div style="font-size: 1.5rem; font-weight: 700; color: {c['text_primary']};">{len(st.session_state.pp_coco_data.get('annotations', []))}</div>
                <div style="font-size: 0.75rem; color: {c['text_muted']};">Anotaciones</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div style="background: {c['bg_card']}; border: 1px solid {c['border']}; border-radius: 0.5rem; padding: 1rem; text-align: center;">
                <div style="font-size: 1.5rem; font-weight: 700; color: {c['text_primary']};">{len(st.session_state.pp_coco_data.get('categories', []))}</div>
                <div style="font-size: 0.75rem; color: {c['text_muted']};">Categorías</div>
            </div>
            """, unsafe_allow_html=True)

        # Class distribution
        cat_id_to_name = {
            cat['id']: cat['name']
            for cat in st.session_state.pp_coco_data.get('categories', [])
        }
        class_counts = {}
        for ann in st.session_state.pp_coco_data.get('annotations', []):
            cat_name = cat_id_to_name.get(ann['category_id'], f"Unknown_{ann['category_id']}")
            class_counts[cat_name] = class_counts.get(cat_name, 0) + 1

        if class_counts:
            df = pd.DataFrame([
                {'Clase': k, 'Cantidad': v}
                for k, v in sorted(class_counts.items(), key=lambda x: -x[1])
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)


# =============================================================================
# LABELS SECTION (Etiquetas)
# =============================================================================

def _render_labels_section():
    """Render label management section."""
    section_header("Gestión de Etiquetas", icon="🏷️")

    label_op = st.radio(
        "Operación",
        ["Renombrar", "Eliminar", "Fusionar", "Añadir", "Segmentación → BBox"],
        horizontal=True,
        key='pp_label_op'
    )

    current_labels = [cat['name'] for cat in st.session_state.pp_coco_data.get('categories', [])]

    spacer(8)

    if label_op == "Renombrar":
        col1, col2 = st.columns(2)
        with col1:
            old_label = st.selectbox("Etiqueta a renombrar", options=current_labels, key='pp_rename_old')
        with col2:
            new_name = st.text_input("Nuevo nombre", value=old_label if old_label else "", key='pp_rename_new')

        if st.button("Renombrar", key='pp_rename_btn', type="primary"):
            if old_label and new_name and old_label != new_name:
                st.session_state.pp_coco_data = LabelManager.rename_label(
                    st.session_state.pp_coco_data, old_label, new_name
                )
                st.success(f"Renombrado '{old_label}' a '{new_name}'")
                st.rerun()

    elif label_op == "Eliminar":
        labels_to_delete = st.multiselect(
            "Selecciona etiquetas a eliminar",
            options=current_labels,
            key='pp_delete_labels'
        )
        delete_anns = st.checkbox("También eliminar anotaciones", value=True, key='pp_delete_anns')

        if st.button("Eliminar seleccionadas", key='pp_delete_btn', type="primary"):
            if labels_to_delete:
                st.session_state.pp_coco_data = LabelManager.delete_labels_batch(
                    st.session_state.pp_coco_data, labels_to_delete, delete_anns
                )
                st.success(f"Eliminadas {len(labels_to_delete)} etiquetas")
                st.rerun()

    elif label_op == "Fusionar":
        source_labels = st.multiselect(
            "Etiquetas a fusionar (serán eliminadas)",
            options=current_labels,
            key='pp_merge_sources'
        )
        target_label = st.text_input("Nombre de etiqueta destino", key='pp_merge_target')

        if st.button("Fusionar", key='pp_merge_btn', type="primary"):
            if source_labels and target_label:
                st.session_state.pp_coco_data = LabelManager.merge_labels(
                    st.session_state.pp_coco_data, source_labels, target_label
                )
                st.success(f"Fusionadas {len(source_labels)} etiquetas en '{target_label}'")
                st.rerun()

    elif label_op == "Añadir":
        new_labels = st.text_area(
            "Nombres de nuevas etiquetas (una por línea)",
            height=100,
            key='pp_new_labels'
        )

        if st.button("Añadir etiquetas", key='pp_add_btn', type="primary"):
            if new_labels.strip():
                labels_list = [l.strip() for l in new_labels.strip().split('\n') if l.strip()]
                st.session_state.pp_coco_data = LabelManager.add_labels_batch(
                    st.session_state.pp_coco_data, labels_list
                )
                st.success(f"Añadidas {len(labels_list)} nuevas etiquetas")
                st.rerun()

    elif label_op == "Segmentación → BBox":
        overwrite = st.checkbox("Sobrescribir bboxes existentes", value=False, key='pp_overwrite_bbox')

        if st.button("Convertir", key='pp_seg_bbox_btn', type="primary"):
            st.session_state.pp_coco_data = LabelManager.segmentation_to_bbox(
                st.session_state.pp_coco_data, overwrite
            )
            st.success("Segmentaciones convertidas a bounding boxes")
            st.rerun()

    # Show label statistics
    spacer(16)
    with st.expander("📊 Estadísticas de etiquetas"):
        stats = LabelManager.get_label_statistics(st.session_state.pp_coco_data)
        if stats:
            stats_df = pd.DataFrame([
                {
                    'Etiqueta': name,
                    'ID': s['id'],
                    'Anotaciones': s['annotation_count'],
                    'Imágenes': s['image_count'],
                    '%': f"{s['percentage']:.1f}%"
                }
                for name, s in sorted(stats.items(), key=lambda x: -x[1]['annotation_count'])
            ])
            st.dataframe(stats_df, use_container_width=True, hide_index=True)


# =============================================================================
# EXPORT SECTION (Exportar)
# =============================================================================

def _render_export_section():
    """Render export section with dataset and splits export."""
    # Sub-tabs for export
    export_tab = st.radio(
        "Tipo de exportación",
        ["📦 Dataset completo", "📂 Exportar Splits"],
        horizontal=True,
        key="pp_export_tab",
        label_visibility="collapsed"
    )

    spacer(8)

    if export_tab == "📦 Dataset completo":
        _render_dataset_export()
    else:
        _render_splits_export()


def _render_dataset_export():
    """Render dataset export controls."""
    section_header("Exportar Dataset", icon="📤")

    col1, col2 = st.columns(2)

    with col1:
        export_formats = st.multiselect(
            "Formatos de exportación",
            options=['coco', 'yolo', 'pascal_voc'],
            default=['coco', 'yolo'],
            key='pp_export_formats'
        )

        output_dir = st.text_input(
            "Directorio de salida",
            value=os.environ.get("OUTPUT_PATH", "/app/output/processed"),
            key='pp_output_dir'
        )

    with col2:
        images_dir = st.text_input(
            "Directorio de imágenes origen",
            value=os.environ.get("BACKGROUNDS_PATH", "/app/datasets"),
            key='pp_images_dir'
        )

        copy_images = st.checkbox("Copiar imágenes a salida", value=False, key='pp_copy_images')

    if st.button("Exportar Dataset", type="primary", key='pp_export_btn', use_container_width=True):
        if export_formats:
            with st.spinner(f"Exportando a {', '.join(export_formats)}..."):
                results = ExportManager.export(
                    st.session_state.pp_coco_data,
                    output_dir,
                    export_formats,
                    copy_images,
                    images_dir
                )

                for fmt, result in results.items():
                    if result.get('success'):
                        st.success(f"**{fmt.upper()}**: {result.get('output_path')}")
                    else:
                        st.error(f"**{fmt.upper()}** falló: {result.get('error')}")
        else:
            st.warning("Selecciona al menos un formato de exportación")


def _render_splits_export():
    """Render splits export controls."""
    section_header("Exportar Splits", icon="📂")

    if 'pp_splits' not in st.session_state:
        alert_box(
            "Primero debes crear splits en la sección 'Splits' para poder exportarlos.",
            type="info",
            icon="ℹ️"
        )
        return

    split_export_formats = st.multiselect(
        "Formatos de exportación para splits",
        options=['coco', 'yolo', 'pascal_voc'],
        default=['coco', 'yolo'],
        key='pp_split_export_formats'
    )

    splits_output_dir = st.text_input(
        "Directorio de splits",
        value=os.environ.get("OUTPUT_PATH", "/app/output") + "/splits",
        key='pp_splits_export_dir'
    )

    if st.button("Exportar Todos los Splits", type="primary", key='pp_export_splits_btn', use_container_width=True):
        splits_base = Path(splits_output_dir)

        with st.spinner("Exportando splits..."):
            for split_name, split_data in st.session_state.pp_splits.items():
                split_output = str(splits_base / split_name)

                results = ExportManager.export(
                    split_data,
                    split_output,
                    split_export_formats,
                    copy_images=False
                )

                st.write(f"**{split_name.capitalize()}**:")
                for fmt, result in results.items():
                    if result.get('success'):
                        st.success(f"  {fmt}: OK")
                    else:
                        st.error(f"  {fmt}: {result.get('error')}")


# =============================================================================
# SPLITS SECTION (Splits)
# =============================================================================

def _render_splits_section():
    """Render splits section with train/val/test and K-Fold."""
    # Sub-tabs for splits
    splits_tab = st.radio(
        "Tipo de split",
        ["📊 Train/Val/Test", "🔄 K-Fold Cross-Validation"],
        horizontal=True,
        key="pp_splits_tab",
        label_visibility="collapsed"
    )

    spacer(8)

    if splits_tab == "📊 Train/Val/Test":
        _render_train_val_test_splits()
    else:
        _render_kfold_splits()


def _render_train_val_test_splits():
    """Render train/val/test splits controls."""
    c = get_colors_dict()
    section_header("Crear Splits Train/Val/Test", icon="✂️")

    col1, col2, col3 = st.columns(3)

    with col1:
        train_ratio = st.slider("Train %", 0, 100, 70, key='pp_train') / 100
    with col2:
        val_ratio = st.slider("Val %", 0, 100, 20, key='pp_val') / 100
    with col3:
        test_ratio = 1 - train_ratio - val_ratio
        st.markdown(f"""
        <div style="background: {c['bg_card']}; border: 1px solid {c['border']}; border-radius: 0.5rem; text-align: center; padding: 0.5rem;">
            <div style="font-size: 1.25rem; font-weight: 700; color: {c['text_primary']};">{test_ratio*100:.0f}%</div>
            <div style="font-size: 0.75rem; color: {c['text_muted']};">Test</div>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        strategy = st.radio(
            "Estrategia de split",
            ['stratified', 'random'],
            horizontal=True,
            key='pp_split_strategy',
            format_func=lambda x: {'stratified': '📊 Estratificado', 'random': '🎲 Aleatorio'}.get(x, x)
        )
    with col2:
        random_seed = st.number_input("Semilla aleatoria", value=42, key='pp_seed')

    splits_output_dir = st.text_input(
        "Directorio de salida para splits",
        value=os.environ.get("OUTPUT_PATH", "/app/output") + "/splits",
        key='pp_splits_dir'
    )

    if st.button("Crear Splits", type="primary", key='pp_splits_btn', use_container_width=True):
        with st.spinner("Creando splits..."):
            splits = DatasetSplitter.split_dataset(
                st.session_state.pp_coco_data,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                strategy=strategy,
                random_seed=int(random_seed)
            )

            # Save each split
            splits_path = Path(splits_output_dir)
            splits_path.mkdir(parents=True, exist_ok=True)

            for split_name, split_data in splits.items():
                split_file = splits_path / f"{split_name}.json"
                with open(split_file, 'w') as f:
                    json.dump(split_data, f, indent=2)

            # Show results
            st.success("¡Splits creados!")

            stats = DatasetSplitter.get_split_statistics(splits)

            split_df = pd.DataFrame([
                {
                    'Split': name.capitalize(),
                    'Imágenes': s['num_images'],
                    'Anotaciones': s['num_annotations']
                }
                for name, s in stats.items()
            ])
            st.dataframe(split_df, use_container_width=True, hide_index=True)

            st.info(f"Guardado en: {splits_output_dir}")

            # Store splits in session for potential export
            st.session_state.pp_splits = splits


def _render_kfold_splits():
    """Render K-Fold cross-validation controls."""
    section_header("K-Fold Cross-Validation", icon="🔄")

    st.markdown("""
    Genera splits K-Fold para entrenamiento con validación cruzada.
    Cada fold proporciona una partición train/validation diferente.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        n_folds = st.number_input("Número de folds (K)", min_value=2, max_value=20, value=5, key='pp_n_folds')

    with col2:
        kfold_stratified = st.checkbox("Estratificado", value=True, key='pp_kfold_stratified',
                                       help="Mantener distribución de clases en cada fold")

    with col3:
        kfold_seed = st.number_input("Semilla aleatoria", value=42, key='pp_kfold_seed')

    kfold_output_dir = st.text_input(
        "Directorio de salida K-Fold",
        value=os.environ.get("OUTPUT_PATH", "/app/output") + "/kfold",
        key='pp_kfold_dir'
    )

    if st.button("Generar K-Folds", type="primary", key='pp_kfold_btn', use_container_width=True):
        try:
            with st.spinner(f"Generando {n_folds}-fold splits..."):
                config = KFoldConfig(
                    n_folds=int(n_folds),
                    stratified=kfold_stratified,
                    random_seed=int(kfold_seed)
                )
                generator = KFoldGenerator(config)

                images = st.session_state.pp_coco_data['images']
                annotations = st.session_state.pp_coco_data['annotations']

                folds = generator.get_all_folds(images, annotations)

                # Save each fold
                kfold_path = Path(kfold_output_dir)
                kfold_path.mkdir(parents=True, exist_ok=True)

                fold_stats = []
                for fold in folds:
                    train_data, val_data = generator.export_fold(fold, st.session_state.pp_coco_data)

                    # Save train
                    train_file = kfold_path / f"fold_{fold.fold_number}_train.json"
                    with open(train_file, 'w') as f:
                        json.dump(train_data, f, indent=2)

                    # Save val
                    val_file = kfold_path / f"fold_{fold.fold_number}_val.json"
                    with open(val_file, 'w') as f:
                        json.dump(val_data, f, indent=2)

                    fold_stats.append({
                        'Fold': fold.fold_number,
                        'Train Imgs': fold.n_train,
                        'Val Imgs': fold.n_val,
                        'Train Anns': len(train_data['annotations']),
                        'Val Anns': len(val_data['annotations'])
                    })

                st.success(f"¡Generados {n_folds} folds!")

                # Show statistics
                st.dataframe(pd.DataFrame(fold_stats), use_container_width=True, hide_index=True)

                st.info(f"Guardado en: {kfold_output_dir}")

        except Exception as e:
            st.error(f"Error generando K-Folds: {e}")


# =============================================================================
# BALANCING SECTION (Balanceo)
# =============================================================================

def _render_balancing_section():
    """Render class balancing section."""
    # Sub-tabs for balancing
    balance_tab = st.radio(
        "Herramienta de balanceo",
        ["⚖️ Balanceo de Clases", "🧮 Pesos de Clases"],
        horizontal=True,
        key="pp_balance_tab",
        label_visibility="collapsed"
    )

    spacer(8)

    if balance_tab == "⚖️ Balanceo de Clases":
        _render_class_balancing()
    else:
        _render_class_weights()


def _render_class_balancing():
    """Render class balancing controls."""
    c = get_colors_dict()
    section_header("Balanceo de Clases", icon="⚖️")

    st.markdown("""
    Balancea tu dataset para manejar el desbalance de clases:
    - **Sobremuestreo**: Duplicar muestras de clases minoritarias
    - **Submuestreo**: Eliminar muestras de clases mayoritarias
    - **Híbrido**: Combinar ambos enfoques (objetivo = mediana)
    """)

    col1, col2 = st.columns(2)

    with col1:
        balance_strategy = st.selectbox(
            "Estrategia de balanceo",
            options=['oversample', 'undersample', 'hybrid'],
            format_func=lambda x: {
                'oversample': '📈 Sobremuestreo',
                'undersample': '📉 Submuestreo',
                'hybrid': '⚖️ Híbrido'
            }.get(x, x),
            key='pp_balance_strategy'
        )

    with col2:
        target_mode = st.radio(
            "Cantidad objetivo",
            ["Auto", "Personalizado"],
            horizontal=True,
            key='pp_target_mode'
        )

    custom_target = None
    if target_mode == "Personalizado":
        custom_target = st.number_input(
            "Cantidad objetivo por clase",
            min_value=1,
            value=100,
            key='pp_custom_target'
        )

    if st.button("Aplicar Balanceo", type="primary", key='pp_balance_btn', use_container_width=True):
        try:
            with st.spinner(f"Aplicando balanceo {balance_strategy}..."):
                balancer = ClassBalancer()
                balanced_data, result = balancer.balance(
                    st.session_state.pp_coco_data,
                    strategy=balance_strategy,
                    target_count=custom_target
                )

                # Show results
                st.success("¡Balanceo completado!")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div style="background: {c['bg_card']}; border: 1px solid {c['border']}; border-radius: 0.5rem; padding: 1rem; text-align: center;">
                        <div style="font-size: 1.5rem; font-weight: 700; color: {c['text_primary']};">{result.total_original}</div>
                        <div style="font-size: 0.75rem; color: {c['text_muted']};">Anotaciones originales</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div style="background: {c['bg_card']}; border: 1px solid {c['border']}; border-radius: 0.5rem; padding: 1rem; text-align: center;">
                        <div style="font-size: 1.5rem; font-weight: 700; color: {c['text_primary']};">{result.total_balanced}</div>
                        <div style="font-size: 0.75rem; color: {c['text_muted']};">Anotaciones balanceadas</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Show class counts comparison
                comparison_data = []
                cat_id_to_name = {cat['id']: cat['name'] for cat in st.session_state.pp_coco_data['categories']}

                for cat_id in result.original_counts:
                    comparison_data.append({
                        'Clase': cat_id_to_name.get(cat_id, f"ID_{cat_id}"),
                        'Original': result.original_counts[cat_id],
                        'Balanceado': result.balanced_counts.get(cat_id, 0)
                    })

                st.dataframe(
                    pd.DataFrame(comparison_data).sort_values('Clase'),
                    use_container_width=True,
                    hide_index=True
                )

                # Store for potential apply
                st.session_state.pp_balanced_data = balanced_data

        except Exception as e:
            st.error(f"Error en balanceo: {e}")

    # Apply balanced data button
    if 'pp_balanced_data' in st.session_state:
        spacer(8)
        if st.button("✅ Aplicar al dataset", key='pp_apply_balance', use_container_width=True):
            st.session_state.pp_coco_data = st.session_state.pp_balanced_data
            del st.session_state.pp_balanced_data
            st.success("¡Dataset actualizado con datos balanceados!")
            st.rerun()


def _render_class_weights():
    """Render class weights calculator."""
    section_header("Pesos de Clases para Entrenamiento", icon="🧮")

    st.markdown("""
    Calcula pesos de clases para funciones de pérdida ponderadas durante el entrenamiento.
    Los pesos ayudan a los modelos a enfocarse más en clases subrepresentadas.
    """)

    col1, col2 = st.columns(2)

    with col1:
        weight_method = st.selectbox(
            "Método de ponderación",
            options=[
                'inverse_frequency',
                'effective_samples',
                'focal',
                'sqrt_inverse'
            ],
            format_func=lambda x: {
                'inverse_frequency': '📊 Frecuencia Inversa (estándar)',
                'effective_samples': '🔬 Muestras Efectivas (Cui et al.)',
                'focal': '🎯 Pesos Focal Loss',
                'sqrt_inverse': '📐 Raíz Cuadrada Inversa'
            }.get(x, x),
            key='pp_weight_method'
        )

    with col2:
        export_format = st.selectbox(
            "Formato de exportación",
            options=['dict', 'pytorch', 'tensorflow', 'list'],
            format_func=lambda x: {
                'dict': '📖 Diccionario',
                'pytorch': '🔥 PyTorch',
                'tensorflow': '🧠 TensorFlow',
                'list': '📋 Lista'
            }.get(x, x),
            key='pp_weight_format'
        )

    if st.button("Calcular Pesos", type="primary", key='pp_calc_weights_btn', use_container_width=True):
        try:
            weights = ClassWeightsCalculator.from_coco_data(
                st.session_state.pp_coco_data,
                method=weight_method
            )

            exported = ClassWeightsCalculator.export_weights(
                weights,
                st.session_state.pp_coco_data['categories'],
                format=export_format
            )

            st.success("¡Pesos calculados!")

            # Display weights
            if export_format == 'dict':
                weights_df = pd.DataFrame([
                    {'Clase': k, 'Peso': f"{v:.4f}"}
                    for k, v in sorted(exported.items(), key=lambda x: -x[1])
                ])
                st.dataframe(weights_df, use_container_width=True, hide_index=True)

            elif export_format == 'pytorch':
                st.json({
                    'weights': [round(w, 4) for w in exported['weights']],
                    'class_to_idx': exported['class_to_idx']
                })

            elif export_format == 'list':
                st.code(f"weights = {[round(w, 4) for w in exported]}")

            else:
                st.json(exported)

            # Download button
            weights_json = json.dumps(exported, indent=2, default=str)
            st.download_button(
                "📥 Descargar Pesos JSON",
                data=weights_json,
                file_name=f"class_weights_{weight_method}.json",
                mime="application/json",
                key='pp_download_weights'
            )

        except Exception as e:
            st.error(f"Error calculando pesos: {e}")


# =============================================================================
# DOWNLOAD SECTION
# =============================================================================

def _render_download_section():
    """Render download section for modified dataset."""
    c = get_colors_dict()
    st.markdown(f"<hr style='border: none; border-top: 1px solid {c['border']}; margin: 1rem 0;'>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.pp_coco_data:
            json_data = json.dumps(st.session_state.pp_coco_data, indent=2)
            st.download_button(
                "📥 Descargar Dataset Modificado (COCO JSON)",
                data=json_data,
                file_name="processed_dataset.json",
                mime="application/json",
                key='pp_download_btn',
                use_container_width=True
            )


# =============================================================================
# PUBLIC FUNCTION: Standalone Labels Section (for main.py)
# =============================================================================

def render_labels_section():
    """
    Public function to render labels section as standalone page.
    Includes dataset loading and labels management.
    """
    # Initialize session state for labels page
    if 'pp_coco_data' not in st.session_state:
        st.session_state.pp_coco_data = None

    # Dataset loading section
    _render_load_dataset_section()

    spacer(16)

    # If data is loaded, show labels section
    if st.session_state.pp_coco_data:
        _render_labels_section()
        spacer(24)
        _render_download_section()
    else:
        empty_state(
            title="No hay dataset cargado",
            message="Carga un archivo COCO JSON para gestionar las etiquetas.",
            icon="📂"
        )
