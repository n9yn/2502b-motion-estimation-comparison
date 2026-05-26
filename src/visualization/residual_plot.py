"""Residual frame plotting utilities."""

from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_residual_frame(
    frame: Any,
    output_path: str | Path,
    title: str = "Residual Frame",
) -> None:
    """Plot and save a residual frame image.
    
    Args:
        frame: Residual frame (grayscale or color)
        output_path: Path to save the plot
        title: Title for the plot
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    if isinstance(frame, np.ndarray):
        if len(frame.shape) == 2:
            # Grayscale residual frame
            im = ax.imshow(frame, cmap='hot')
        else:
            # Color residual frame
            if frame.shape[2] == 3:
                # Convert BGR to RGB if necessary
                display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                display_frame = frame
            im = ax.imshow(display_frame)
    else:
        im = ax.imshow(frame, cmap='hot')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    # Add colorbar for grayscale
    if isinstance(frame, np.ndarray) and len(frame.shape) == 2:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Residual Energy', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"OK Residual frame saved to {output_path}")


def plot_residual_comparison(
    residual_fs: Any,
    residual_diamond: Any,
    output_path: str | Path,
    title: str = "Residual Comparison: Full Search vs Diamond Search",
) -> None:
    """Create a side-by-side comparison of residual frames.
    
    Args:
        residual_fs: Residual frame from Full Search
        residual_diamond: Residual frame from Diamond Search
        output_path: Path to save the comparison image
        title: Title for the comparison plot
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Full Search residual
    if isinstance(residual_fs, np.ndarray):
        if len(residual_fs.shape) == 2:
            im1 = ax1.imshow(residual_fs, cmap='hot')
        else:
            display_fs = cv2.cvtColor(residual_fs, cv2.COLOR_BGR2RGB) if residual_fs.shape[2] == 3 else residual_fs
            im1 = ax1.imshow(display_fs)
    else:
        im1 = ax1.imshow(residual_fs, cmap='hot')
    
    ax1.set_title('Full Search Residual', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Energy', fontsize=10)
    
    # Diamond Search residual
    if isinstance(residual_diamond, np.ndarray):
        if len(residual_diamond.shape) == 2:
            im2 = ax2.imshow(residual_diamond, cmap='hot')
        else:
            display_diamond = cv2.cvtColor(residual_diamond, cv2.COLOR_BGR2RGB) if residual_diamond.shape[2] == 3 else residual_diamond
            im2 = ax2.imshow(display_diamond)
    else:
        im2 = ax2.imshow(residual_diamond, cmap='hot')
    
    ax2.set_title('Diamond Search Residual', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Energy', fontsize=10)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"OK Residual comparison plot saved to {output_path}")
