"""Streamlit dashboard wiring that integrates the project's pipeline modules."""
from pathlib import Path
import time
import json

import streamlit as st

from ui.components import sidebar_inputs
from ui.helpers import save_uploaded_temp, get_video_metadata, ensure_dir
from ui.visualizations import save_vector_visualizations

from src.workflow import run_motion_estimation_workflow


def show_dashboard() -> None:
    st.set_page_config(page_title="Motion Estimation Comparison", layout="wide", initial_sidebar_state="expanded")

    # Simple glass-like styling (translucent panels)
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(135deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); }
        .glass { background: rgba(255,255,255,0.03); backdrop-filter: blur(8px) saturate(120%); border-radius:12px; padding:10px; }
        .stButton>button { background: rgba(255,255,255,0.06); border-radius:8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header
    st.title("Motion Estimation Algorithm Comparison")
    st.subheader("Full Search vs Diamond Search for Video Compression")
    st.markdown("Small demo dashboard to compare algorithms, visualize vectors and residuals.")

    # Sidebar inputs
    params = sidebar_inputs()

    col1, col2 = st.columns([2, 3])

    # Video preview and metadata
    with col1:
        st.header("Input Video")
        uploaded = params["uploaded_file"]
        if uploaded is None:
            st.info("Upload a video (mp4/avi/mov) to start")
        else:
            tmp_path = save_uploaded_temp(uploaded)
            st.video(str(tmp_path))
            try:
                meta = get_video_metadata(tmp_path)
                st.write(meta)
            except Exception as e:
                st.error(f"Unable to read video metadata: {e}")

    # Controls and run
    if params["run"] and uploaded is not None:
        # Create output root for this run
        out_root = Path("outputs") / f"streamlit_{int(time.time())}"
        ensure_dir(out_root)

        status = st.empty()
        progress = st.progress(0)

        try:
            status.info("✔ Extracting frames")
            progress.progress(10)

            results = run_motion_estimation_workflow(
                tmp_path,
                block_size=params["block_size"],
                search_range=params["search_range"],
                output_root=out_root,
            )

            progress.progress(60)
            status.info("✔ Running visualizations")

            # Create visualizations
            vis_out = Path(results["vector_visualizations"]) if "vector_visualizations" in results else out_root / "outputs" / "vector_visualizations"
            ensure_dir(vis_out)

            # reference frame path
            extracted = Path(results["extracted_path"])
            frame_files = sorted([p for p in extracted.glob("frame_*.png")])
            ref_frame = frame_files[0] if frame_files else None

            fs_vectors = results.get("fs_vectors", [])
            ds_vectors = results.get("diamond_vectors", [])

            vis_paths = save_vector_visualizations(ref_frame, vis_out, fs_vectors, ds_vectors)

            progress.progress(85)
            status.success("✔ Done")
            progress.progress(100)

            # Show results
            # Make motion vectors JSON-serializable
            serializable_results = dict(results)
            for key in ("fs_vectors", "diamond_vectors"):
                if key in serializable_results and serializable_results[key] is not None:
                    serializable_results[key] = [
                        {"x": int(mv.x), "y": int(mv.y), "dx": int(mv.dx), "dy": int(mv.dy)}
                        for mv in serializable_results[key]
                    ]

            with st.expander("Outputs", expanded=True):
                st.json(serializable_results)

            # Display all generated outputs (vectors, residuals, charts, motion vectors, logs)
            st.markdown("---")
            st.subheader("All Generated Outputs")
            output_dir = out_root / "outputs"

            tabs = st.tabs(["Vectors", "Residuals", "Charts", "Motion Vectors", "Logs"])

            # VECTORS
            with tabs[0]:
                vec_dir = output_dir / "vector_visualizations"
                if vec_dir.exists():
                    imgs = sorted(list(vec_dir.glob("**/*.*")))
                    if imgs:
                        for i in range(0, len(imgs), 3):
                            cols = st.columns(3)
                            for j, img in enumerate(imgs[i:i+3]):
                                cols[j].image(str(img), caption=img.name, width=360)
                    else:
                        st.info("No vector visualizations found.")
                else:
                    st.info("No vector visualizations directory.")

            # RESIDUALS
            with tabs[1]:
                res_dir = output_dir / "residual_frames"
                if res_dir.exists():
                    imgs = sorted(list(res_dir.glob("**/*.*")))
                    if imgs:
                        for img in imgs:
                            st.image(str(img), caption=img.name, width=640)
                    else:
                        st.info("No residual frames found.")
                else:
                    st.info("No residual frames directory.")

            # CHARTS
            with tabs[2]:
                charts_dir = output_dir / "charts"
                if charts_dir.exists():
                    imgs = sorted(list(charts_dir.glob("**/*.*")))
                    if imgs:
                        for img in imgs:
                            st.image(str(img), caption=img.name, width=640)
                    else:
                        st.info("No charts found.")
                else:
                    st.info("No charts directory.")

            # MOTION VECTORS (CSV)
            with tabs[3]:
                mv_dir = output_dir / "motion_vectors"
                if mv_dir.exists():
                    csvs = sorted(list(mv_dir.glob("**/*.csv")))
                    if csvs:
                        for csvf in csvs:
                            st.write(csvf.relative_to(output_dir))
                            with open(csvf, "rb") as fh:
                                st.download_button(f"Download {csvf.name}", data=fh, file_name=csvf.name, key=f"dl_mv_{csvf.name}")
                    else:
                        st.info("No motion vector CSV files found.")
                else:
                    st.info("No motion vectors directory.")

            # LOGS
            with tabs[4]:
                logs_dir = output_dir / "logs"
                if logs_dir.exists():
                    files = sorted(list(logs_dir.glob("**/*.*")))
                    if files:
                        for f in files:
                            if f.suffix.lower() in [".json", ".txt", ".log", ".csv"]:
                                st.write(f.relative_to(output_dir))
                                try:
                                    st.code(f.read_text(encoding='utf-8'))
                                except Exception:
                                    with open(f, "rb") as fh:
                                        st.download_button(f"Download {f.name}", data=fh, file_name=f.name, key=f"dl_log_{f.name}")
                            else:
                                with open(f, "rb") as fh:
                                    st.download_button(f"Download {f.name}", data=fh, file_name=f.name, key=f"dl_log_{f.name}")
                    else:
                        st.info("No logs found.")
                else:
                    st.info("No logs directory.")
            # Visualization columns
            v1, v2 = st.columns(2)
            if params["show_vectors"]:
                with v1:
                    st.subheader("Motion Vectors")
                    if "comparison" in vis_paths:
                        st.image(vis_paths["comparison"], width=640)
                    else:
                        if "fs" in vis_paths:
                            st.image(vis_paths["fs"], caption="Full Search", width=400)
                        if "ds" in vis_paths:
                            st.image(vis_paths["ds"], caption="Diamond Search", width=400)

            if params["show_residuals"]:
                with v2:
                    st.subheader("Residual Frame")
                    res_img = Path(results["residual_image_path"]) if "residual_image_path" in results else None
                    if res_img and res_img.exists():
                        st.image(str(res_img), width=640)
                    st.write("Residual stats:")
                    st.json(results.get("residual_stats", {}))

            # Metrics table
            st.header("Performance Metrics")
            fs_stats = results.get("fs_stats", {}) or {}
            ds_stats = results.get("ds_stats", {}) or {}
            residual = results.get("residual_stats", {}) or {}

            rows = []
            rows.append({
                "Algorithm": "Full Search",
                "Runtime (ms)": f"{fs_stats.get('time_ms', '-'):.2f}" if fs_stats else "-",
                "Residual Energy": f"{residual.get('energy', '-'):.2f}" if residual else "-",
                "MSE": f"{residual.get('mse', '-'):.2f}" if residual else "-",
                "Block Size": params["block_size"],
                "Search Range": params["search_range"],
            })
            rows.append({
                "Algorithm": "Diamond Search",
                "Runtime (ms)": f"{ds_stats.get('time_ms', '-'):.2f}" if ds_stats else "-",
                "Residual Energy": f"{residual.get('energy', '-'):.2f}" if residual else "-",
                "MSE": f"{residual.get('mse', '-'):.2f}" if residual else "-",
                "Block Size": params["block_size"],
                "Search Range": params["search_range"],
            })
            st.table(rows)

            # Downloads
            st.header("Download Outputs")
            if results.get("fs_vector_path"):
                st.download_button("Download motion vectors (FS)", data=open(results.get("fs_vector_path", ""), "rb"), file_name=Path(results.get("fs_vector_path", "")).name, key="dl_fs_vectors")
            if results.get("ds_vector_path"):
                st.download_button("Download motion vectors (DS)", data=open(results.get("ds_vector_path", ""), "rb"), file_name=Path(results.get("ds_vector_path", "")).name, key="dl_ds_vectors")
            if results.get("residual_image_path"):
                st.download_button("Download residual image", data=open(results.get("residual_image_path", ""), "rb"), file_name=Path(results.get("residual_image_path", "")).name, key="dl_residual_image")

        except Exception as e:
            status.error(f"Processing failed: {e}")

    else:
        st.info("Configure parameters in the sidebar and press RUN ANALYSIS.")
