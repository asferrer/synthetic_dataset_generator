"""
Generation Page
===============
Synthetic image generation (single and batch).
"""

import os
import time
from datetime import datetime
import streamlit as st
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional

from app.components.api_client import get_api_client


# Try to import streamlit_autorefresh, if not available, use workaround
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False


def render_single_generation(
    effects: List[str],
    effects_config: Dict,
    generation_config: Dict,
):
    """Render single image generation UI"""
    st.subheader("Single Image Generation")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Input**")

        # Background selection
        input_method = st.radio(
            "Background source",
            ["Upload file", "Path input", "Browse datasets"],
            horizontal=True,
            key="single_input_method"
        )

        background_path = None

        if input_method == "Upload file":
            uploaded = st.file_uploader(
                "Upload background",
                type=["jpg", "jpeg", "png"],
                key="single_bg_upload"
            )

            if uploaded:
                # Save to shared volume
                shared_dir = Path("/shared/images/input")
                shared_dir.mkdir(parents=True, exist_ok=True)

                save_path = shared_dir / uploaded.name
                with open(save_path, "wb") as f:
                    f.write(uploaded.getvalue())

                background_path = str(save_path)
                st.image(uploaded, caption="Background", width="stretch")

        elif input_method == "Path input":
            background_path = st.text_input(
                "Background path",
                placeholder="/app/datasets/Backgrounds_filtered/image.jpg",
                key="single_bg_path"
            )

            if background_path and Path(background_path).exists():
                try:
                    img = Image.open(background_path)
                    st.image(img, caption="Background preview", width="stretch")
                except Exception as e:
                    st.warning(f"Cannot preview: {e}")

        else:
            # Browse datasets
            datasets_dir = Path(os.environ.get("BACKGROUNDS_PATH", "/app/datasets/Backgrounds_filtered"))

            if datasets_dir.exists():
                bg_files = list(datasets_dir.glob("*.jpg")) + list(datasets_dir.glob("*.png"))
                bg_names = [f.name for f in bg_files[:50]]  # Limit to 50

                if bg_names:
                    selected = st.selectbox(
                        "Select background",
                        options=bg_names,
                        key="single_bg_select"
                    )
                    if selected:
                        background_path = str(datasets_dir / selected)
                        try:
                            img = Image.open(background_path)
                            st.image(img, caption=selected, width="stretch")
                        except:
                            pass
                else:
                    st.warning("No backgrounds found in datasets")
            else:
                st.warning(f"Datasets directory not found: {datasets_dir}")

        # Objects selection
        st.markdown("**Objects**")

        objects_dir = Path(os.environ.get("OBJECTS_PATH", "/app/datasets/Objects"))
        objects_to_place = []

        if objects_dir.exists():
            class_dirs = [d for d in objects_dir.iterdir() if d.is_dir()]

            if class_dirs:
                selected_classes = st.multiselect(
                    "Select classes to place",
                    options=[d.name for d in class_dirs],
                    key="single_obj_classes"
                )

                for cls in selected_classes:
                    cls_dir = objects_dir / cls
                    obj_files = list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg"))

                    if obj_files:
                        # Add random object from class
                        import random
                        obj_file = random.choice(obj_files)
                        objects_to_place.append({
                            "image_path": str(obj_file),
                            "class_name": cls,
                            "position": None,
                            "scale": None,
                            "rotation": None,
                        })

                if selected_classes:
                    st.info(f"Will place {len(objects_to_place)} objects")

        # Output path
        output_dir = st.text_input(
            "Output directory",
            value="/app/output/single",
            key="single_output_dir"
        )

        output_filename = st.text_input(
            "Output filename",
            value="synthetic_001.jpg",
            key="single_output_name"
        )

        output_path = str(Path(output_dir) / output_filename)

        # Generate button
        if st.button(
            "Generate Image",
            type="primary",
            disabled=not background_path,
            key="single_generate_btn"
        ):
            client = get_api_client()

            with st.spinner("Generating..."):
                start = time.time()

                result = client.compose_image(
                    background_path=background_path,
                    objects=objects_to_place,
                    effects=effects,
                    effects_config=effects_config,
                    output_path=output_path,
                    validate_quality=generation_config.get("validate_quality", False),
                    validate_physics=generation_config.get("validate_physics", False),
                )

                elapsed = time.time() - start

            st.session_state["single_result"] = result
            st.session_state["single_elapsed"] = elapsed

    with col2:
        st.markdown("**Output**")

        if "single_result" in st.session_state:
            result = st.session_state["single_result"]
            elapsed = st.session_state.get("single_elapsed", 0)

            if result.get("success"):
                st.success(f"Generated in {elapsed:.1f}s")

                output = result.get("output_path", "")
                if output and Path(output).exists():
                    try:
                        img = Image.open(output)
                        st.image(img, caption="Generated image", width="stretch")
                    except Exception as e:
                        st.error(f"Cannot display: {e}")

                # Metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Processing time", f"{result.get('processing_time_ms', 0):.0f}ms")
                    st.metric("Objects placed", result.get("objects_placed", 0))

                with col_b:
                    st.metric("Depth used", "Yes" if result.get("depth_used") else "No")
                    st.metric("Valid", "Yes" if result.get("is_valid", True) else "No")

                # Effects applied
                fx = result.get("effects_applied", [])
                if fx:
                    st.info(f"Effects: {', '.join(fx)}")

                # Validation results
                if result.get("quality_score"):
                    with st.expander("Quality Score"):
                        qs = result["quality_score"]
                        st.json(qs)

                if result.get("physics_violations"):
                    with st.expander("Physics Violations"):
                        for v in result["physics_violations"]:
                            st.warning(f"{v.get('violation_type')}: {v.get('description')}")

                # Annotations
                if result.get("annotations"):
                    with st.expander("Annotations"):
                        st.json(result["annotations"])

                # Download
                if output and Path(output).exists():
                    with open(output, "rb") as f:
                        st.download_button(
                            "Download Image",
                            f,
                            file_name=Path(output).name,
                            mime="image/jpeg",
                            key="single_download"
                        )

            else:
                st.error(f"Generation failed: {result.get('error', 'Unknown error')}")


def render_batch_generation(
    effects: List[str],
    effects_config: Dict,
    generation_config: Dict,
):
    """Render batch generation UI"""
    st.subheader("Batch Generation")

    # Check for targets from analysis
    targets = st.session_state.get("balancing_targets", {})

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Input Directories**")

        backgrounds_dir = st.text_input(
            "Backgrounds_filtered directory",
            value=os.environ.get("BACKGROUNDS_PATH", "/app/datasets/Backgrounds_filtered"),
            key="batch_bg_dir"
        )

        objects_dir = st.text_input(
            "Objects directory",
            value=os.environ.get("OBJECTS_PATH", "/app/datasets/Objects"),
            key="batch_obj_dir"
        )

        output_dir = st.text_input(
            "Output directory",
            value="/app/output/batch",
            key="batch_output_dir"
        )

        # Validate directories
        bg_ok = Path(backgrounds_dir).exists() if backgrounds_dir else False
        obj_ok = Path(objects_dir).exists() if objects_dir else False

        if not bg_ok:
            st.warning(f"Backgrounds_filtered directory not found")
        if not obj_ok:
            st.warning(f"Objects directory not found")

    with col2:
        st.markdown("**Generation Settings**")

        # Use targets from analysis if available
        if targets:
            st.success(f"Using targets from analysis: {sum(targets.values())} synthetic images")
            use_targets = st.checkbox("Use analysis targets", value=True, key="use_targets")

            if use_targets:
                st.json(targets)
                num_images = sum(targets.values())
            else:
                num_images = st.number_input(
                    "Number of images",
                    min_value=1,
                    max_value=10000,
                    value=100,
                    key="batch_num_images"
                )
                targets = None
        else:
            num_images = st.number_input(
                "Number of images",
                min_value=1,
                max_value=10000,
                value=100,
                key="batch_num_images_no_targets"
            )
            targets = None

        max_objects = st.slider(
            "Max objects per image",
            1, 10,
            value=generation_config.get("max_objects", 5),
            key="batch_max_objects"
        )

        depth_aware = st.checkbox(
            "Depth-aware placement",
            value=generation_config.get("depth_aware", True),
            key="batch_depth_aware"
        )

        save_pipeline_debug = st.checkbox(
            "Save pipeline debug images",
            value=False,
            help="Save intermediate pipeline images for the first iteration (for documentation/paper)",
            key="batch_pipeline_debug"
        )

    st.divider()

    # Start button
    can_generate = bg_ok and obj_ok and num_images > 0

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if st.button(
            "Start Batch Generation",
            type="primary",
            disabled=not can_generate,
            key="batch_start_btn"
        ):
            client = get_api_client()

            with st.spinner(f"Starting batch generation of {num_images} images..."):
                result = client.compose_batch(
                    backgrounds_dir=backgrounds_dir,
                    objects_dir=objects_dir,
                    output_dir=output_dir,
                    num_images=num_images,
                    targets_per_class=targets,
                    max_objects_per_image=max_objects,
                    effects=effects,
                    effects_config=effects_config,
                    depth_aware=depth_aware,
                    validate_quality=generation_config.get("validate_quality", False),
                    validate_physics=generation_config.get("validate_physics", False),
                    save_pipeline_debug=save_pipeline_debug,
                )

            if result.get("success"):
                st.session_state["batch_job_id"] = result.get("job_id")
                st.session_state["batch_result"] = result
                st.success(f"Job started: {result.get('job_id')}")
            else:
                st.error(f"Failed to start: {result.get('error')}")

    # Show job status
    if "batch_job_id" in st.session_state:
        job_id = st.session_state["batch_job_id"]

        st.divider()
        st.markdown(f"**Job Status:** `{job_id}`")

        with col2:
            if st.button("Refresh Status", key="batch_refresh"):
                client = get_api_client()
                result = client.get_job_status(job_id)
                st.session_state["batch_result"] = result

        with col3:
            if st.button("Clear Job", key="batch_clear"):
                del st.session_state["batch_job_id"]
                if "batch_result" in st.session_state:
                    del st.session_state["batch_result"]
                st.rerun()

        if "batch_result" in st.session_state:
            result = st.session_state["batch_result"]

            status = result.get("status", "unknown")

            # Progress
            generated = result.get("images_generated", 0)
            rejected = result.get("images_rejected", 0)
            pending = result.get("images_pending", 0)
            total = generated + rejected + pending

            if total > 0:
                progress = generated / total
                st.progress(progress, text=f"{status}: {generated}/{total} generated")

            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.metric("Generated", generated)
            with col_b:
                st.metric("Rejected", rejected)
            with col_c:
                st.metric("Pending", pending)

            # Synthetic counts
            counts = result.get("synthetic_counts", {})
            if counts:
                with st.expander("Per-class counts"):
                    st.json(counts)

            # Output path
            coco_path = result.get("output_coco_path")
            if coco_path and Path(coco_path).exists():
                st.success(f"COCO JSON: {coco_path}")

                with open(coco_path, "rb") as f:
                    st.download_button(
                        "Download COCO JSON",
                        f,
                        file_name="synthetic_dataset.json",
                        mime="application/json",
                        key="batch_download_coco"
                    )

            if result.get("error"):
                st.error(f"Error: {result.get('error')}")


def render_jobs_panel():
    """Render the jobs panel showing all batch jobs"""
    st.subheader("Jobs Monitor")

    # Auto-refresh toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        auto_refresh = st.checkbox(
            "Auto-refresh every 5 seconds",
            value=st.session_state.get("jobs_auto_refresh", False),
            key="jobs_auto_refresh_toggle"
        )
        st.session_state["jobs_auto_refresh"] = auto_refresh

    with col2:
        if st.button("Refresh Now", key="jobs_refresh_now"):
            pass  # Will trigger refresh

    # Handle auto-refresh
    if auto_refresh and HAS_AUTOREFRESH:
        st_autorefresh(interval=5000, key="jobs_autorefresh")

    # Fetch jobs
    client = get_api_client()
    jobs_result = client.list_jobs()

    if jobs_result.get("error"):
        st.error(f"Failed to fetch jobs: {jobs_result.get('error')}")
        return

    jobs = jobs_result.get("jobs", [])
    total = jobs_result.get("total", 0)

    if not jobs:
        st.info("No jobs found. Start a batch generation to see jobs here.")
        return

    st.markdown(f"**Total Jobs: {total}**")

    # Display each job
    for job in jobs:
        job_id = job.get("job_id", "unknown")
        status = job.get("status", "unknown")
        generated = job.get("images_generated", 0)
        rejected = job.get("images_rejected", 0)
        pending = job.get("images_pending", 0)
        total_images = generated + rejected + pending
        created_at = job.get("created_at", "")
        output_dir = job.get("output_dir", "")
        error = job.get("error")

        with st.container():
            # Header row
            col1, col2, col3 = st.columns([3, 2, 1])

            with col1:
                st.markdown(f"**`{job_id}`**")
                if created_at:
                    # Parse and format timestamp
                    try:
                        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                        st.caption(f"Created: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                    except:
                        st.caption(f"Created: {created_at}")

            with col2:
                # Status badge with color
                if status == "completed":
                    st.success(f"Status: {status}")
                elif status == "processing":
                    st.warning(f"Status: {status}")
                elif status == "failed":
                    st.error(f"Status: {status}")
                elif status == "queued":
                    st.info(f"Status: {status}")
                elif status == "cancelled":
                    st.info(f"Status: {status}")
                elif status == "cancelling":
                    st.warning(f"Status: {status}")
                else:
                    st.write(f"Status: {status}")

            with col3:
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("Details", key=f"job_details_{job_id}"):
                        st.session_state["selected_job_id"] = job_id
                with btn_col2:
                    # Show Stop button only for running jobs
                    if status in ["processing", "queued"]:
                        if st.button("Stop", key=f"job_stop_{job_id}", type="secondary"):
                            result = client.cancel_job(job_id)
                            if result.get("success"):
                                st.toast(f"Cancellation requested for {job_id}")
                                st.rerun()
                            else:
                                st.error(f"Failed to cancel: {result.get('message', result.get('error', 'Unknown error'))}")

            # Progress bar
            if total_images > 0:
                progress = generated / total_images
                st.progress(progress, text=f"{generated}/{total_images} images ({progress*100:.1f}%)")

            # Metrics row
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("Generated", generated)
            with col_b:
                st.metric("Rejected", rejected)
            with col_c:
                st.metric("Pending", pending)
            with col_d:
                if output_dir:
                    st.caption(f"Output: {output_dir}")

            # Error display
            if error:
                st.error(f"Error: {error}")

            # Class counts (collapsed)
            counts = job.get("synthetic_counts", {})
            if counts:
                with st.expander("Per-class counts"):
                    st.json(counts)

            st.divider()

    # Job details modal/section
    if "selected_job_id" in st.session_state:
        selected_id = st.session_state["selected_job_id"]

        st.markdown("---")
        st.subheader(f"Job Details: `{selected_id}`")

        # Fetch full job details
        job_details = client.get_job_status(selected_id)

        if job_details.get("error"):
            st.error(f"Failed to fetch job details: {job_details.get('error')}")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.json(job_details)

            with col2:
                # Show output directory contents if available
                output_dir = job_details.get("output_dir", "")
                if output_dir:
                    output_path = Path(output_dir)
                    if output_path.exists():
                        st.markdown("**Output Directory Contents:**")

                        # Check for pipeline debug
                        debug_path = output_path / "pipeline_debug"
                        if debug_path.exists():
                            st.success("Pipeline debug images available!")
                            debug_files = sorted(debug_path.glob("*.jpg")) + sorted(debug_path.glob("*.png"))
                            if debug_files:
                                st.markdown("Pipeline steps:")
                                for f in debug_files:
                                    st.write(f"- {f.name}")

                        # Check for images
                        images_path = output_path / "images"
                        if images_path.exists():
                            img_count = len(list(images_path.glob("*.jpg")))
                            st.info(f"Images folder: {img_count} images")

                        # COCO JSON
                        coco_path = output_path / "synthetic_dataset.json"
                        if coco_path.exists():
                            st.success("COCO JSON available")
                            with open(coco_path, "rb") as f:
                                st.download_button(
                                    "Download COCO JSON",
                                    f,
                                    file_name=f"{selected_id}_dataset.json",
                                    mime="application/json",
                                    key=f"download_coco_{selected_id}"
                                )

        if st.button("Close Details", key="close_job_details"):
            del st.session_state["selected_job_id"]
            st.rerun()


def render_generation_page(
    effects: List[str],
    effects_config: Dict,
    generation_config: Dict,
):
    """Render the generation page"""
    st.header("Synthetic Image Generation")

    # Tabs for single vs batch vs jobs
    tab1, tab2, tab3 = st.tabs(["Single Image", "Batch Generation", "Jobs Monitor"])

    with tab1:
        render_single_generation(effects, effects_config, generation_config)

    with tab2:
        render_batch_generation(effects, effects_config, generation_config)

    with tab3:
        render_jobs_panel()
