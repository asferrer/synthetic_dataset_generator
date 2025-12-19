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

from app.utils import LabelManager, ExportManager, DatasetSplitter


def render_post_processing_page():
    """Render the post-processing page."""
    st.header("Dataset Post-Processing")
    st.markdown("""
    **Process any COCO dataset** - no synthetic generation required!

    - **Label Management**: Rename, delete, merge labels
    - **Multi-format Export**: COCO, YOLO, Pascal VOC
    - **Dataset Splits**: Train/Val/Test with stratification
    """)

    # Initialize session state
    if 'pp_coco_data' not in st.session_state:
        st.session_state.pp_coco_data = None

    st.divider()

    # ===== Section 1: Load Dataset =====
    st.subheader("1. Load Dataset")

    # Check if there's data from analysis
    has_analyzed = 'coco_data' in st.session_state and st.session_state.coco_data is not None

    data_source_options = ["Upload COCO JSON file"]
    if has_analyzed:
        data_source_options.append("Use analyzed dataset")

    data_source = st.radio(
        "Select data source",
        options=data_source_options,
        horizontal=True,
        key='pp_data_source'
    )

    if data_source == "Upload COCO JSON file":
        uploaded = st.file_uploader(
            "Upload COCO JSON",
            type=['json'],
            key='pp_upload'
        )
        if uploaded:
            try:
                st.session_state.pp_coco_data = json.load(uploaded)
                n_images = len(st.session_state.pp_coco_data.get('images', []))
                n_anns = len(st.session_state.pp_coco_data.get('annotations', []))
                n_cats = len(st.session_state.pp_coco_data.get('categories', []))
                st.success(f"Loaded: {n_images} images, {n_anns} annotations, {n_cats} categories")
            except Exception as e:
                st.error(f"Error loading JSON: {e}")

    elif data_source == "Use analyzed dataset" and has_analyzed:
        st.session_state.pp_coco_data = st.session_state.coco_data
        n_images = len(st.session_state.pp_coco_data.get('images', []))
        n_anns = len(st.session_state.pp_coco_data.get('annotations', []))
        st.success(f"Using analyzed dataset: {n_images} images, {n_anns} annotations")

    # Show current dataset status
    if st.session_state.pp_coco_data:
        with st.expander("Dataset Overview", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Images", len(st.session_state.pp_coco_data.get('images', [])))
            with col2:
                st.metric("Annotations", len(st.session_state.pp_coco_data.get('annotations', [])))
            with col3:
                st.metric("Categories", len(st.session_state.pp_coco_data.get('categories', [])))

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
                    {'Class': k, 'Count': v}
                    for k, v in sorted(class_counts.items(), key=lambda x: -x[1])
                ])
                st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    # ===== Section 2: Label Management =====
    st.subheader("2. Label Management")

    if st.session_state.pp_coco_data:
        label_op = st.radio(
            "Operation",
            ["Rename Labels", "Delete Labels", "Merge Labels", "Add New Labels", "Segmentation to BBox"],
            horizontal=True,
            key='pp_label_op'
        )

        current_labels = [cat['name'] for cat in st.session_state.pp_coco_data.get('categories', [])]

        if label_op == "Rename Labels":
            col1, col2 = st.columns(2)
            with col1:
                old_label = st.selectbox("Label to rename", options=current_labels, key='pp_rename_old')
            with col2:
                new_name = st.text_input("New name", value=old_label if old_label else "", key='pp_rename_new')

            if st.button("Rename", key='pp_rename_btn'):
                if old_label and new_name and old_label != new_name:
                    st.session_state.pp_coco_data = LabelManager.rename_label(
                        st.session_state.pp_coco_data, old_label, new_name
                    )
                    st.success(f"Renamed '{old_label}' to '{new_name}'")
                    st.rerun()

        elif label_op == "Delete Labels":
            labels_to_delete = st.multiselect(
                "Select labels to delete",
                options=current_labels,
                key='pp_delete_labels'
            )
            delete_anns = st.checkbox("Also delete annotations", value=True, key='pp_delete_anns')

            if st.button("Delete Selected", key='pp_delete_btn', type="primary"):
                if labels_to_delete:
                    st.session_state.pp_coco_data = LabelManager.delete_labels_batch(
                        st.session_state.pp_coco_data, labels_to_delete, delete_anns
                    )
                    st.success(f"Deleted {len(labels_to_delete)} labels")
                    st.rerun()

        elif label_op == "Merge Labels":
            source_labels = st.multiselect(
                "Labels to merge (will be deleted)",
                options=current_labels,
                key='pp_merge_sources'
            )
            target_label = st.text_input("Target label name", key='pp_merge_target')

            if st.button("Merge", key='pp_merge_btn'):
                if source_labels and target_label:
                    st.session_state.pp_coco_data = LabelManager.merge_labels(
                        st.session_state.pp_coco_data, source_labels, target_label
                    )
                    st.success(f"Merged {len(source_labels)} labels into '{target_label}'")
                    st.rerun()

        elif label_op == "Add New Labels":
            new_labels = st.text_area(
                "New label names (one per line)",
                height=100,
                key='pp_new_labels'
            )

            if st.button("Add Labels", key='pp_add_btn'):
                if new_labels.strip():
                    labels_list = [l.strip() for l in new_labels.strip().split('\n') if l.strip()]
                    st.session_state.pp_coco_data = LabelManager.add_labels_batch(
                        st.session_state.pp_coco_data, labels_list
                    )
                    st.success(f"Added {len(labels_list)} new labels")
                    st.rerun()

        elif label_op == "Segmentation to BBox":
            overwrite = st.checkbox("Overwrite existing bboxes", value=False, key='pp_overwrite_bbox')

            if st.button("Convert", key='pp_seg_bbox_btn'):
                st.session_state.pp_coco_data = LabelManager.segmentation_to_bbox(
                    st.session_state.pp_coco_data, overwrite
                )
                st.success("Converted segmentations to bounding boxes")
                st.rerun()

        # Show label statistics
        with st.expander("Label Statistics"):
            stats = LabelManager.get_label_statistics(st.session_state.pp_coco_data)
            if stats:
                stats_df = pd.DataFrame([
                    {
                        'Label': name,
                        'ID': s['id'],
                        'Annotations': s['annotation_count'],
                        'Images': s['image_count'],
                        '%': f"{s['percentage']:.1f}%"
                    }
                    for name, s in sorted(stats.items(), key=lambda x: -x[1]['annotation_count'])
                ])
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
    else:
        st.info("Load a dataset to enable label management")

    st.divider()

    # ===== Section 3: Export =====
    st.subheader("3. Export Dataset")

    if st.session_state.pp_coco_data:
        col1, col2 = st.columns(2)

        with col1:
            export_formats = st.multiselect(
                "Export formats",
                options=['coco', 'yolo', 'pascal_voc'],
                default=['coco', 'yolo'],
                key='pp_export_formats'
            )

            output_dir = st.text_input(
                "Output directory",
                value=os.environ.get("OUTPUT_PATH", "/app/output/processed"),
                key='pp_output_dir'
            )

        with col2:
            images_dir = st.text_input(
                "Source images directory",
                value=os.environ.get("BACKGROUNDS_PATH", "/app/datasets"),
                key='pp_images_dir'
            )

            copy_images = st.checkbox("Copy images to output", value=False, key='pp_copy_images')

        if st.button("Export Dataset", type="primary", key='pp_export_btn'):
            if export_formats:
                with st.spinner(f"Exporting to {', '.join(export_formats)}..."):
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
                            st.error(f"**{fmt.upper()}** failed: {result.get('error')}")
            else:
                st.warning("Select at least one export format")
    else:
        st.info("Load a dataset to enable export")

    st.divider()

    # ===== Section 4: Dataset Splits =====
    st.subheader("4. Create Train/Val/Test Splits")

    if st.session_state.pp_coco_data:
        col1, col2, col3 = st.columns(3)

        with col1:
            train_ratio = st.slider("Train %", 0, 100, 70, key='pp_train') / 100
        with col2:
            val_ratio = st.slider("Val %", 0, 100, 20, key='pp_val') / 100
        with col3:
            test_ratio = 1 - train_ratio - val_ratio
            st.metric("Test %", f"{test_ratio*100:.0f}%")

        col1, col2 = st.columns(2)
        with col1:
            strategy = st.radio(
                "Split strategy",
                ['stratified', 'random'],
                horizontal=True,
                key='pp_split_strategy'
            )
        with col2:
            random_seed = st.number_input("Random seed", value=42, key='pp_seed')

        splits_output_dir = st.text_input(
            "Splits output directory",
            value=os.environ.get("OUTPUT_PATH", "/app/output") + "/splits",
            key='pp_splits_dir'
        )

        if st.button("Create Splits", type="primary", key='pp_splits_btn'):
            with st.spinner("Creating splits..."):
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
                st.success("Splits created!")

                stats = DatasetSplitter.get_split_statistics(splits)

                split_df = pd.DataFrame([
                    {
                        'Split': name.capitalize(),
                        'Images': s['num_images'],
                        'Annotations': s['num_annotations']
                    }
                    for name, s in stats.items()
                ])
                st.dataframe(split_df, use_container_width=True, hide_index=True)

                st.info(f"Saved to: {splits_output_dir}")

                # Store splits in session for potential export
                st.session_state.pp_splits = splits
    else:
        st.info("Load a dataset to enable splits")

    st.divider()

    # ===== Section 5: Export Splits =====
    if 'pp_splits' in st.session_state:
        st.subheader("5. Export Splits to Multiple Formats")

        split_export_formats = st.multiselect(
            "Export formats for splits",
            options=['coco', 'yolo', 'pascal_voc'],
            default=['coco', 'yolo'],
            key='pp_split_export_formats'
        )

        if st.button("Export All Splits", type="primary", key='pp_export_splits_btn'):
            splits_base = Path(splits_output_dir)

            with st.spinner("Exporting splits..."):
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

    st.divider()

    # ===== Download Modified Dataset =====
    st.subheader("Download Modified Dataset")

    if st.session_state.pp_coco_data:
        json_data = json.dumps(st.session_state.pp_coco_data, indent=2)
        st.download_button(
            "Download COCO JSON",
            data=json_data,
            file_name="processed_dataset.json",
            mime="application/json",
            key='pp_download_btn'
        )
