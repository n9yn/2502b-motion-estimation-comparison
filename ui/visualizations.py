from pathlib import Path
from typing import Optional

import cv2
from src.visualization import vector_field


def save_vector_visualizations(
    reference_frame_path: Path,
    output_path: Path,
    fs_vectors: list,
    ds_vectors: list,
) -> dict:
    """Create and save vector visualizations (single and comparison). Returns paths."""
    ref = cv2.imread(str(reference_frame_path), cv2.IMREAD_GRAYSCALE)
    results = {}
    if fs_vectors:
        fs_img = output_path / "vector_field_fs.png"
        vector_field.save_vector_field_visualization(ref, fs_vectors, fs_img, method_name="Full Search")
        results["fs"] = str(fs_img)
    if ds_vectors:
        ds_img = output_path / "vector_field_ds.png"
        vector_field.save_vector_field_visualization(ref, ds_vectors, ds_img, method_name="Diamond Search")
        results["ds"] = str(ds_img)
    if fs_vectors and ds_vectors:
        comp_img = output_path / "vector_field_comparison.png"
        vector_field.plot_vector_field_comparison(ref, ref, fs_vectors, ds_vectors, comp_img)
        results["comparison"] = str(comp_img)
    return results
