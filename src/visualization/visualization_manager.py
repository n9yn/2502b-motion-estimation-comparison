"""Comprehensive visualization management and saving utilities."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.motion_estimation.motion_vector import MotionVector
from src.visualization.vector_field import (
    plot_motion_vectors,
    overlay_vectors_on_frame,
    plot_vector_field_comparison,
    save_vector_field_visualization,
)
from src.visualization.residual_plot import (
    plot_residual_frame,
    plot_residual_comparison,
)
from src.visualization.comparison_chart import (
    plot_runtime_comparison,
    plot_energy_comparison,
    plot_vector_statistics,
    plot_metrics_comparison,
)


class VisualizationManager:
    """Manager class for all visualization operations."""
    
    def __init__(self, output_dir: str | Path):
        """Initialize visualization manager.
        
        Args:
            output_dir: Base directory for all visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.vector_dir = self.output_dir / "vectors"
        self.residual_dir = self.output_dir / "residuals"
        self.comparison_dir = self.output_dir / "comparisons"
        self.charts_dir = self.output_dir / "charts"
        
        # Create directories
        for d in [self.vector_dir, self.residual_dir, self.comparison_dir, self.charts_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def save_vector_field(
        self,
        frame: Any,
        vectors: list[MotionVector],
        method_name: str,
        frame_idx: int = 0,
    ) -> Path:
        """Save vector field visualization for a single frame.
        
        Args:
            frame: Input frame
            vectors: List of motion vectors
            method_name: Name of the estimation method (e.g., "Full Search")
            frame_idx: Frame index for naming
        
        Returns:
            Path to saved visualization
        """
        output_path = self.vector_dir / f"{method_name.lower().replace(' ', '_')}_frame_{frame_idx:04d}.png"
        save_vector_field_visualization(frame, vectors, output_path, method_name)
        return output_path
    
    def save_vector_field_with_overlay(
        self,
        frame: Any,
        vectors: list[MotionVector],
        method_name: str,
        frame_idx: int = 0,
    ) -> Path:
        """Save frame with overlaid motion vectors.
        
        Args:
            frame: Input frame
            vectors: List of motion vectors
            method_name: Name of the estimation method
            frame_idx: Frame index for naming
        
        Returns:
            Path to saved image
        """
        frame_with_vectors = overlay_vectors_on_frame(frame, vectors)
        output_path = self.vector_dir / f"{method_name.lower().replace(' ', '_')}_overlay_frame_{frame_idx:04d}.png"
        cv2.imwrite(str(output_path), frame_with_vectors)
        print(f"✓ Vector overlay saved to {output_path}")
        return output_path
    
    def save_vector_comparison(
        self,
        frame_fs: Any,
        frame_diamond: Any,
        vectors_fs: list[MotionVector],
        vectors_diamond: list[MotionVector],
        frame_idx: int = 0,
    ) -> Path:
        """Save side-by-side comparison of vector fields.
        
        Args:
            frame_fs: Frame for Full Search
            frame_diamond: Frame for Diamond Search
            vectors_fs: Motion vectors from Full Search
            vectors_diamond: Motion vectors from Diamond Search
            frame_idx: Frame index for naming
        
        Returns:
            Path to saved comparison
        """
        output_path = self.comparison_dir / f"vector_field_comparison_frame_{frame_idx:04d}.png"
        plot_vector_field_comparison(
            frame_fs, frame_diamond,
            vectors_fs, vectors_diamond,
            output_path,
            f"Vector Field Comparison (Frame {frame_idx})"
        )
        return output_path
    
    def save_residual_frame(
        self,
        residual_frame: Any,
        method_name: str,
        frame_idx: int = 0,
    ) -> Path:
        """Save residual frame visualization.
        
        Args:
            residual_frame: Residual frame image
            method_name: Name of the estimation method
            frame_idx: Frame index for naming
        
        Returns:
            Path to saved visualization
        """
        output_path = self.residual_dir / f"{method_name.lower().replace(' ', '_')}_residual_frame_{frame_idx:04d}.png"
        plot_residual_frame(
            residual_frame, output_path,
            title=f"Residual Frame - {method_name}"
        )
        return output_path
    
    def save_residual_comparison(
        self,
        residual_fs: Any,
        residual_diamond: Any,
        frame_idx: int = 0,
    ) -> Path:
        """Save side-by-side comparison of residual frames.
        
        Args:
            residual_fs: Residual frame from Full Search
            residual_diamond: Residual frame from Diamond Search
            frame_idx: Frame index for naming
        
        Returns:
            Path to saved comparison
        """
        output_path = self.comparison_dir / f"residual_comparison_frame_{frame_idx:04d}.png"
        plot_residual_comparison(
            residual_fs, residual_diamond,
            output_path,
            f"Residual Comparison (Frame {frame_idx})"
        )
        return output_path
    
    def save_runtime_comparison(
        self,
        runtime_data: dict[str, float],
        chart_name: str = "runtime_comparison",
    ) -> Path:
        """Save runtime comparison chart.
        
        Args:
            runtime_data: Dictionary of algorithm names and runtimes
            chart_name: Name for the chart file
        
        Returns:
            Path to saved chart
        """
        output_path = self.charts_dir / f"{chart_name}.png"
        plot_runtime_comparison(runtime_data, output_path)
        return output_path
    
    def save_energy_comparison(
        self,
        energy_data: dict[str, float],
        chart_name: str = "energy_comparison",
    ) -> Path:
        """Save energy comparison chart.
        
        Args:
            energy_data: Dictionary of algorithm names and energy values
            chart_name: Name for the chart file
        
        Returns:
            Path to saved chart
        """
        output_path = self.charts_dir / f"{chart_name}.png"
        plot_energy_comparison(energy_data, output_path)
        return output_path
    
    def save_vector_statistics(
        self,
        stats_data: dict[str, dict[str, float]],
        chart_name: str = "vector_statistics",
    ) -> Path:
        """Save vector statistics comparison chart.
        
        Args:
            stats_data: Dictionary of algorithm stats
            chart_name: Name for the chart file
        
        Returns:
            Path to saved chart
        """
        output_path = self.charts_dir / f"{chart_name}.png"
        plot_vector_statistics(stats_data, output_path)
        return output_path
    
    def save_metrics_comparison(
        self,
        metrics_fs: dict[str, float],
        metrics_diamond: dict[str, float],
        chart_name: str = "metrics_comparison",
    ) -> Path:
        """Save metrics comparison chart.
        
        Args:
            metrics_fs: Full Search metrics
            metrics_diamond: Diamond Search metrics
            chart_name: Name for the chart file
        
        Returns:
            Path to saved chart
        """
        output_path = self.charts_dir / f"{chart_name}.png"
        plot_metrics_comparison(metrics_fs, metrics_diamond, output_path)
        return output_path
    
    def save_all_visualizations(
        self,
        frame_fs: Any,
        frame_diamond: Any,
        vectors_fs: list[MotionVector],
        vectors_diamond: list[MotionVector],
        residual_fs: Any | None = None,
        residual_diamond: Any | None = None,
        frame_idx: int = 0,
    ) -> dict[str, Path]:
        """Save all available visualizations for a frame pair.
        
        Args:
            frame_fs: Frame for Full Search
            frame_diamond: Frame for Diamond Search
            vectors_fs: Motion vectors from Full Search
            vectors_diamond: Motion vectors from Diamond Search
            residual_fs: Optional residual frame from Full Search
            residual_diamond: Optional residual frame from Diamond Search
            frame_idx: Frame index for naming
        
        Returns:
            Dictionary of visualization types and their save paths
        """
        results = {}
        
        # Save vector fields
        results['vector_fs'] = self.save_vector_field(frame_fs, vectors_fs, "Full Search", frame_idx)
        results['vector_diamond'] = self.save_vector_field(frame_diamond, vectors_diamond, "Diamond Search", frame_idx)
        
        # Save overlays
        results['overlay_fs'] = self.save_vector_field_with_overlay(frame_fs, vectors_fs, "Full Search", frame_idx)
        results['overlay_diamond'] = self.save_vector_field_with_overlay(frame_diamond, vectors_diamond, "Diamond Search", frame_idx)
        
        # Save comparisons
        results['vector_comparison'] = self.save_vector_comparison(
            frame_fs, frame_diamond, vectors_fs, vectors_diamond, frame_idx
        )
        
        # Save residuals if provided
        if residual_fs is not None:
            results['residual_fs'] = self.save_residual_frame(residual_fs, "Full Search", frame_idx)
        
        if residual_diamond is not None:
            results['residual_diamond'] = self.save_residual_frame(residual_diamond, "Diamond Search", frame_idx)
        
        if residual_fs is not None and residual_diamond is not None:
            results['residual_comparison'] = self.save_residual_comparison(
                residual_fs, residual_diamond, frame_idx
            )
        
        return results
    
    def get_summary(self) -> dict[str, int]:
        """Get summary of saved visualizations.
        
        Returns:
            Dictionary with counts of saved files in each directory
        """
        return {
            'vectors': len(list(self.vector_dir.glob('*.png'))),
            'residuals': len(list(self.residual_dir.glob('*.png'))),
            'comparisons': len(list(self.comparison_dir.glob('*.png'))),
            'charts': len(list(self.charts_dir.glob('*.png'))),
        }
