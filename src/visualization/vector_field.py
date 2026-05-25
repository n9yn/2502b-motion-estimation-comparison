"""Visualization for motion vector fields."""

from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.motion_estimation.motion_vector import MotionVector


def plot_motion_vectors(
    frame: Any,
    motion_vectors: list[MotionVector],
    output_path: str | Path,
    title: str = "Motion Vector Field",
    scale: float = 2.0,
) -> None:
    """Plot motion vectors on top of a reference frame.
    
    Args:
        frame: Input frame (grayscale or color)
        motion_vectors: List of MotionVector objects
        output_path: Path to save the visualization
        title: Title for the plot
        scale: Scale factor for vector visualization
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Handle different frame types
    if isinstance(frame, np.ndarray):
        if len(frame.shape) == 2:
            # Grayscale frame
            display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            # Color frame
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.shape[2] == 3 else frame
    else:
        display_frame = frame

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(display_frame, cmap='gray' if len(display_frame.shape) == 2 else None)
    
    # Plot motion vectors as arrows
    for mv in motion_vectors:
        # Draw arrows from (x, y) to (x + dx*scale, y + dy*scale)
        ax.arrow(
            mv.x, mv.y,
            mv.dx * scale, mv.dy * scale,
            head_width=3, head_length=2,
            fc='red', ec='red', alpha=0.7, linewidth=1.5
        )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Plotted {len(motion_vectors)} motion vectors to {output_path}")


def overlay_vectors_on_frame(
    frame: Any,
    motion_vectors: list[MotionVector],
    block_size: int = 16,
    scale: float = 1.0,
    color: tuple[int, int, int] = (0, 255, 0),
) -> Any:
    """Overlay motion vectors directly on a frame image.
    
    Args:
        frame: Input frame
        motion_vectors: List of MotionVector objects
        block_size: Size of motion blocks
        scale: Scale factor for vector visualization
        color: Color for vector arrows (BGR format)
    
    Returns:
        Frame with overlaid motion vectors
    """
    if isinstance(frame, np.ndarray):
        frame_with_vectors = frame.copy()
        
        # Convert to color if grayscale
        if len(frame_with_vectors.shape) == 2:
            frame_with_vectors = cv2.cvtColor(frame_with_vectors, cv2.COLOR_GRAY2BGR)
    else:
        frame_with_vectors = frame.copy()
    
    # Draw motion vectors as arrows
    for mv in motion_vectors:
        # Calculate starting point (center of block)
        start_x = int(mv.x + block_size // 2)
        start_y = int(mv.y + block_size // 2)
        
        # Calculate ending point with scaled displacement
        end_x = int(start_x + mv.dx * scale)
        end_y = int(start_y + mv.dy * scale)
        
        # Draw arrow
        cv2.arrowedLine(
            frame_with_vectors,
            (start_x, start_y),
            (end_x, end_y),
            color,
            thickness=2,
            tipLength=0.3
        )
    
    return frame_with_vectors


def plot_vector_field_comparison(
    frame_fs: Any,
    frame_diamond: Any,
    vectors_fs: list[MotionVector],
    vectors_diamond: list[MotionVector],
    output_path: str | Path,
    title: str = "Vector Field Comparison: Full Search vs Diamond Search",
) -> None:
    """Create a side-by-side comparison of vector fields.
    
    Args:
        frame_fs: Reference frame for Full Search
        frame_diamond: Reference frame for Diamond Search
        vectors_fs: Motion vectors from Full Search
        vectors_diamond: Motion vectors from Diamond Search
        output_path: Path to save the comparison image
        title: Title for the comparison plot
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Full Search visualization
    if isinstance(frame_fs, np.ndarray):
        if len(frame_fs.shape) == 2:
            display_fs = cv2.cvtColor(frame_fs, cv2.COLOR_GRAY2RGB)
        else:
            display_fs = cv2.cvtColor(frame_fs, cv2.COLOR_BGR2RGB) if frame_fs.shape[2] == 3 else frame_fs
    else:
        display_fs = frame_fs
    
    ax1.imshow(display_fs)
    for mv in vectors_fs:
        ax1.arrow(
            mv.x, mv.y,
            mv.dx * 2.0, mv.dy * 2.0,
            head_width=3, head_length=2,
            fc='red', ec='red', alpha=0.7, linewidth=1.5
        )
    ax1.set_title('Full Search Vector Field', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    
    # Diamond Search visualization
    if isinstance(frame_diamond, np.ndarray):
        if len(frame_diamond.shape) == 2:
            display_diamond = cv2.cvtColor(frame_diamond, cv2.COLOR_GRAY2RGB)
        else:
            display_diamond = cv2.cvtColor(frame_diamond, cv2.COLOR_BGR2RGB) if frame_diamond.shape[2] == 3 else frame_diamond
    else:
        display_diamond = frame_diamond
    
    ax2.imshow(display_diamond)
    for mv in vectors_diamond:
        ax2.arrow(
            mv.x, mv.y,
            mv.dx * 2.0, mv.dy * 2.0,
            head_width=3, head_length=2,
            fc='blue', ec='blue', alpha=0.7, linewidth=1.5
        )
    ax2.set_title('Diamond Search Vector Field', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison plot saved to {output_path}")


def save_vector_field_visualization(
    frame: Any,
    motion_vectors: list[MotionVector],
    output_path: str | Path,
    method_name: str = "Motion Estimation",
) -> None:
    """Save vector field visualization with method name.
    
    Args:
        frame: Input frame
        motion_vectors: List of MotionVector objects
        output_path: Path to save the visualization
        method_name: Name of the motion estimation method
    """
    plot_motion_vectors(
        frame, motion_vectors, output_path,
        title=f"Motion Vector Field - {method_name}"
    )
