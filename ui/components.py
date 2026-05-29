from typing import Tuple
import streamlit as st


def sidebar_inputs() -> dict:
    st.sidebar.title("Inputs & Parameters")

    st.sidebar.header("Video Input")
    uploaded_file = st.sidebar.file_uploader("Upload video", type=["mp4", "avi", "mov"])

    st.sidebar.header("Algorithm Selection")
    alg_choice = st.sidebar.multiselect("Algorithms", ["Full Search", "Diamond Search"], default=["Full Search", "Diamond Search"])

    st.sidebar.header("Parameters")
    block_size = st.sidebar.slider("Block Size", min_value=8, max_value=32, value=16, step=8)
    search_range = st.sidebar.slider("Search Range", min_value=3, max_value=15, value=7)
    frame_count = st.sidebar.number_input("Frame Range (frames)", min_value=2, max_value=500, value=20)
    grayscale = st.sidebar.checkbox("Grayscale Processing", value=True)

    st.sidebar.header("Visualization Options")
    show_vectors = st.sidebar.checkbox("Show motion vectors", value=True)
    show_residuals = st.sidebar.checkbox("Show residual frames", value=True)
    show_runtime = st.sidebar.checkbox("Show runtime comparison", value=True)
    show_charts = st.sidebar.checkbox("Show charts", value=True)
    use_numba = st.sidebar.checkbox("Use Numba acceleration", value=True)

    st.sidebar.markdown("---")
    run_button = st.sidebar.button("RUN ANALYSIS")

    return {
        "uploaded_file": uploaded_file,
        "alg_choice": alg_choice,
        "block_size": int(block_size),
        "search_range": int(search_range),
        "frame_count": int(frame_count),
        "grayscale": bool(grayscale),
        "show_vectors": bool(show_vectors),
        "show_residuals": bool(show_residuals),
        "show_runtime": bool(show_runtime),
        "show_charts": bool(show_charts),
        "use_numba": bool(use_numba),
        "run": run_button,
    }
