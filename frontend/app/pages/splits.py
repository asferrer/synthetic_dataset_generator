"""
Splits Page (Step 6)
====================
Create train/val/test splits or K-fold cross-validation.
"""

import os
import json
import streamlit as st
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

from app.components.ui import (
    page_header, section_header, spacer, alert_box, empty_state,
    workflow_stepper, workflow_navigation
)
from app.utils import DatasetSplitter, KFoldGenerator, KFoldConfig, ExportManager
from app.config.theme import get_colors_dict


def render_splits_page():
    """Render the splits page (Step 6 of workflow)"""
    c = get_colors_dict()

    # Workflow stepper
    completed = st.session_state.get("workflow_completed", [])
    workflow_stepper(current_step=6, completed_steps=completed)

    page_header(
        title="Crear Splits de Entrenamiento",
        subtitle="Paso 6: Divide el dataset en train/validation/test",
        icon="✂️"
    )

    # Get the dataset to split (prefer combined, then generated, then source)
    dataset = _get_dataset_to_split()

    if not dataset:
        alert_box(
            "No hay dataset disponible para dividir. Completa los pasos anteriores.",
            type="warning",
            icon="⚠️"
        )

        action = workflow_navigation(
            current_step=6,
            can_go_next=False,
            on_prev="⑤ Combinar"
        )

        if action == "prev":
            st.session_state.nav_menu = "⑤ Combinar"
            st.rerun()
        return

    # Dataset summary
    _render_dataset_summary(dataset)

    spacer(16)

    # Split type selection
    split_type = st.radio(
        "Tipo de Split",
        ["📊 Train/Val/Test", "🔄 K-Fold Cross-Validation"],
        horizontal=True,
        key="split_type"
    )

    spacer(16)

    if split_type == "📊 Train/Val/Test":
        _render_train_val_test_section(dataset)
    else:
        _render_kfold_section(dataset)

    spacer(24)

    # Workflow completion
    has_splits = st.session_state.get("final_splits") is not None

    if has_splits:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {c['success']}, #059669);
                    border-radius: 0.75rem; padding: 2rem; text-align: center; color: white;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">🎉</div>
            <div style="font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                ¡Workflow Completado!
            </div>
            <div style="opacity: 0.9;">
                Has generado, exportado y dividido tu dataset sintético exitosamente.
            </div>
        </div>
        """, unsafe_allow_html=True)

        spacer(16)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🏠 Volver a Home", use_container_width=True, key="back_home"):
                st.session_state.nav_menu = "🏠 Home"
                st.rerun()

        with col2:
            if st.button("🔄 Nuevo Workflow", use_container_width=True, key="new_workflow"):
                _reset_workflow()
                st.session_state.nav_menu = "🏠 Home"
                st.rerun()

    else:
        action = workflow_navigation(
            current_step=6,
            can_go_next=False,
            next_label="",
            on_prev="⑤ Combinar"
        )

        if action == "prev":
            st.session_state.nav_menu = "⑤ Combinar"
            st.rerun()


def _get_dataset_to_split() -> Optional[Dict]:
    """Get the best available dataset to split"""
    # Priority: combined > generated > source
    if st.session_state.get("combined_dataset"):
        return st.session_state.combined_dataset
    if st.session_state.get("generated_dataset"):
        return st.session_state.generated_dataset
    if st.session_state.get("source_dataset"):
        return st.session_state.source_dataset
    return None


def _render_dataset_summary(dataset: Dict) -> None:
    """Render summary of dataset to split"""
    c = get_colors_dict()
    n_images = len(dataset.get("images", []))
    n_annotations = len(dataset.get("annotations", []))
    n_categories = len(dataset.get("categories", []))

    # Determine source
    if st.session_state.get("combined_dataset") == dataset:
        source_name = "Dataset Combinado"
        source_icon = "🔗"
    elif st.session_state.get("generated_dataset") == dataset:
        source_name = "Dataset Sintético"
        source_icon = "🏭"
    else:
        source_name = "Dataset Original"
        source_icon = "📊"

    st.markdown(f"""
    <div style="background: {c['bg_secondary']}; border: 1px solid {c['border']};
                border-radius: 0.5rem; padding: 1rem; margin-bottom: 1rem;
                display: flex; align-items: center; justify-content: space-between;">
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <span style="font-size: 1.5rem;">{source_icon}</span>
            <div>
                <div style="font-weight: 600; color: {c['text_primary']};">{source_name}</div>
                <div style="font-size: 0.85rem; color: {c['text_muted']};">
                    {n_images:,} imágenes | {n_annotations:,} anotaciones | {n_categories} clases
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_train_val_test_section(dataset: Dict) -> None:
    """Render train/val/test split configuration"""
    c = get_colors_dict()
    section_header("Configuración Train/Val/Test", icon="✂️")

    col1, col2, col3 = st.columns(3)

    with col1:
        train_pct = st.slider(
            "Train %",
            min_value=10,
            max_value=90,
            value=70,
            step=5,
            key="split_train"
        )

    with col2:
        val_pct = st.slider(
            "Val %",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            key="split_val"
        )

    with col3:
        test_pct = 100 - train_pct - val_pct
        st.markdown(f"""
        <div style="background: {c['bg_card']}; border: 1px solid {c['border']};
                    border-radius: 0.5rem; padding: 1rem; text-align: center;">
            <div style="font-size: 2rem; font-weight: 700; color: {c['primary']};">{test_pct}%</div>
            <div style="font-size: 0.85rem; color: {c['text_muted']};">Test</div>
        </div>
        """, unsafe_allow_html=True)

    if test_pct < 0:
        alert_box("Los porcentajes de Train y Val suman más de 100%", type="error")
        return

    spacer(16)

    col1, col2 = st.columns(2)

    with col1:
        strategy = st.radio(
            "Estrategia de Split",
            ["stratified", "random"],
            format_func=lambda x: {
                "stratified": "📊 Estratificado (mantiene distribución de clases)",
                "random": "🎲 Aleatorio"
            }.get(x, x),
            key="split_strategy"
        )

    with col2:
        seed = st.number_input(
            "Semilla aleatoria",
            min_value=0,
            value=42,
            key="split_seed",
            help="Para resultados reproducibles"
        )

    spacer(8)

    output_dir = st.text_input(
        "Directorio de salida",
        value=os.environ.get("OUTPUT_PATH", "/app/output") + "/splits",
        key="splits_output_dir"
    )

    spacer(16)

    # Preview
    n_images = len(dataset.get("images", []))
    n_train = int(n_images * train_pct / 100)
    n_val = int(n_images * val_pct / 100)
    n_test = n_images - n_train - n_val

    st.markdown(f"""
    <div style="background: {c['bg_card']}; border: 1px solid {c['border']};
                border-radius: 0.5rem; padding: 1rem;">
        <div style="font-size: 0.7rem; color: {c['text_muted']}; text-transform: uppercase;
                    letter-spacing: 0.05em; margin-bottom: 0.75rem;">
            Preview de Splits
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; text-align: center;">
            <div style="background: {c['success_bg']}; border-radius: 0.375rem; padding: 0.75rem;">
                <div style="font-size: 1.5rem; font-weight: 700; color: {c['success']};">{n_train:,}</div>
                <div style="font-size: 0.8rem; color: {c['text_muted']};">Train</div>
            </div>
            <div style="background: {c['warning_bg']}; border-radius: 0.375rem; padding: 0.75rem;">
                <div style="font-size: 1.5rem; font-weight: 700; color: {c['warning']};">{n_val:,}</div>
                <div style="font-size: 0.8rem; color: {c['text_muted']};">Validation</div>
            </div>
            <div style="background: {c['primary_light']}; border-radius: 0.375rem; padding: 0.75rem;">
                <div style="font-size: 1.5rem; font-weight: 700; color: {c['primary']};">{n_test:,}</div>
                <div style="font-size: 0.8rem; color: {c['text_muted']};">Test</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    spacer(16)

    # Export format selection
    with st.expander("📤 Opciones de Exportación", expanded=False):
        export_formats = st.multiselect(
            "Formatos de exportación",
            options=["coco", "yolo", "pascal_voc"],
            default=["coco"],
            key="split_export_formats"
        )

    spacer(16)

    # Create splits button
    if st.button("✂️ Crear Splits", type="primary", use_container_width=True, key="create_splits"):
        _create_train_val_test_splits(
            dataset, train_pct/100, val_pct/100, test_pct/100,
            strategy, int(seed), output_dir, export_formats if 'export_formats' in dir() else ["coco"]
        )


def _create_train_val_test_splits(
    dataset: Dict,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    strategy: str,
    seed: int,
    output_dir: str,
    export_formats: List[str]
) -> None:
    """Create and save train/val/test splits"""
    with st.spinner("Creando splits..."):
        try:
            splits = DatasetSplitter.split_dataset(
                dataset,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                strategy=strategy,
                random_seed=seed
            )

            # Save splits
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            for split_name, split_data in splits.items():
                # Save COCO JSON
                split_file = output_path / f"{split_name}.json"
                with open(split_file, 'w') as f:
                    json.dump(split_data, f, indent=2)

                # Export to other formats if requested
                if export_formats and len(export_formats) > 1:
                    split_format_dir = output_path / split_name
                    ExportManager.export(
                        split_data,
                        str(split_format_dir),
                        [f for f in export_formats if f != "coco"],
                        copy_images=False
                    )

            st.session_state.final_splits = splits

            # Show statistics
            stats = DatasetSplitter.get_split_statistics(splits)
            st.success("✅ Splits creados exitosamente!")

            stats_df = pd.DataFrame([
                {
                    "Split": name.capitalize(),
                    "Imágenes": s["num_images"],
                    "Anotaciones": s["num_annotations"]
                }
                for name, s in stats.items()
            ])
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

            st.info(f"📁 Guardado en: {output_dir}")

            # Mark step as complete
            if 6 not in st.session_state.get("workflow_completed", []):
                st.session_state.workflow_completed = st.session_state.get("workflow_completed", []) + [6]

            st.rerun()

        except Exception as e:
            st.error(f"❌ Error creando splits: {e}")


def _render_kfold_section(dataset: Dict) -> None:
    """Render K-Fold configuration"""
    c = get_colors_dict()
    section_header("Configuración K-Fold Cross-Validation", icon="🔄")

    st.markdown(f"""
    <div style="background: {c['bg_secondary']}; padding: 1rem; border-radius: 0.5rem;
                border-left: 3px solid {c['primary']}; margin-bottom: 1rem;">
        <div style="font-size: 0.85rem; color: {c['text_secondary']};">
            K-Fold divide el dataset en K particiones. Cada fold usa una partición diferente
            para validación y el resto para entrenamiento.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        n_folds = st.number_input(
            "Número de Folds (K)",
            min_value=2,
            max_value=20,
            value=5,
            key="kfold_n"
        )

    with col2:
        stratified = st.checkbox(
            "Estratificado",
            value=True,
            key="kfold_stratified",
            help="Mantener distribución de clases en cada fold"
        )

    with col3:
        seed = st.number_input(
            "Semilla aleatoria",
            min_value=0,
            value=42,
            key="kfold_seed"
        )

    spacer(8)

    output_dir = st.text_input(
        "Directorio de salida",
        value=os.environ.get("OUTPUT_PATH", "/app/output") + "/kfold",
        key="kfold_output_dir"
    )

    spacer(16)

    # Preview
    n_images = len(dataset.get("images", []))
    fold_size = n_images // int(n_folds)

    st.markdown(f"""
    <div style="background: {c['bg_card']}; border: 1px solid {c['border']};
                border-radius: 0.5rem; padding: 1rem;">
        <div style="font-size: 0.7rem; color: {c['text_muted']}; text-transform: uppercase;
                    letter-spacing: 0.05em; margin-bottom: 0.75rem;">
            Preview K-Fold
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; text-align: center;">
            <div>
                <div style="font-size: 1.5rem; font-weight: 700; color: {c['primary']};">{int(n_folds)}</div>
                <div style="font-size: 0.8rem; color: {c['text_muted']};">Folds</div>
            </div>
            <div>
                <div style="font-size: 1.5rem; font-weight: 700; color: {c['text_primary']};">~{fold_size:,}</div>
                <div style="font-size: 0.8rem; color: {c['text_muted']};">Val/Fold</div>
            </div>
            <div>
                <div style="font-size: 1.5rem; font-weight: 700; color: {c['text_primary']};">~{n_images - fold_size:,}</div>
                <div style="font-size: 0.8rem; color: {c['text_muted']};">Train/Fold</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    spacer(16)

    if st.button("🔄 Generar K-Folds", type="primary", use_container_width=True, key="create_kfold"):
        _create_kfold_splits(dataset, int(n_folds), stratified, int(seed), output_dir)


def _create_kfold_splits(
    dataset: Dict,
    n_folds: int,
    stratified: bool,
    seed: int,
    output_dir: str
) -> None:
    """Create and save K-Fold splits"""
    with st.spinner(f"Generando {n_folds}-fold splits..."):
        try:
            config = KFoldConfig(
                n_folds=n_folds,
                stratified=stratified,
                random_seed=seed
            )
            generator = KFoldGenerator(config)

            images = dataset["images"]
            annotations = dataset["annotations"]

            folds = generator.get_all_folds(images, annotations)

            # Save each fold
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            fold_stats = []
            all_folds = {}

            for fold in folds:
                train_data, val_data = generator.export_fold(fold, dataset)

                # Save train
                train_file = output_path / f"fold_{fold.fold_number}_train.json"
                with open(train_file, 'w') as f:
                    json.dump(train_data, f, indent=2)

                # Save val
                val_file = output_path / f"fold_{fold.fold_number}_val.json"
                with open(val_file, 'w') as f:
                    json.dump(val_data, f, indent=2)

                all_folds[f"fold_{fold.fold_number}_train"] = train_data
                all_folds[f"fold_{fold.fold_number}_val"] = val_data

                fold_stats.append({
                    "Fold": fold.fold_number,
                    "Train Imgs": fold.n_train,
                    "Val Imgs": fold.n_val,
                    "Train Anns": len(train_data["annotations"]),
                    "Val Anns": len(val_data["annotations"])
                })

            st.session_state.final_splits = all_folds

            st.success(f"✅ Generados {n_folds} folds exitosamente!")

            st.dataframe(pd.DataFrame(fold_stats), use_container_width=True, hide_index=True)

            st.info(f"📁 Guardado en: {output_dir}")

            # Mark step as complete
            if 6 not in st.session_state.get("workflow_completed", []):
                st.session_state.workflow_completed = st.session_state.get("workflow_completed", []) + [6]

            st.rerun()

        except Exception as e:
            st.error(f"❌ Error generando K-Folds: {e}")


def _reset_workflow() -> None:
    """Reset all workflow-related session state"""
    keys_to_clear = [
        'source_dataset', 'source_filename', 'workflow_step', 'workflow_completed',
        'analysis_result', 'balancing_targets', 'generation_config', 'current_job_id',
        'generated_dataset', 'generated_output_dir', 'export_results', 'export_completed',
        'combined_dataset', 'datasets_to_combine', '_combine_auto_added', 'final_splits'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
