"""Comparison chart generation for algorithm analysis."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_runtime_comparison(
    data: dict[str, float],
    output_path: str | Path,
    title: str = "Algorithm Runtime Comparison",
) -> None:
    """Generate a runtime comparison chart for motion estimation algorithms.
    
    Args:
        data: Dictionary with algorithm names as keys and runtime values (ms) as values
        output_path: Path to save the comparison chart
        title: Title for the chart
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    algorithms = list(data.keys())
    runtimes = list(data.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(algorithms, runtimes, color=colors[:len(algorithms)], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, runtime in zip(bars, runtimes):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., height,
            f'{runtime:.2f}ms',
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )
    
    ax.set_ylabel('Runtime (ms)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"OK Runtime comparison chart saved to {output_path}")


def plot_energy_comparison(
    data: dict[str, float],
    output_path: str | Path,
    title: str = "Residual Energy Comparison",
) -> None:
    """Generate an energy comparison chart for residual analysis.
    
    Args:
        data: Dictionary with algorithm names as keys and energy values as values
        output_path: Path to save the comparison chart
        title: Title for the chart
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    algorithms = list(data.keys())
    energies = list(data.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(algorithms, energies, color=colors[:len(algorithms)], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, energy in zip(bars, energies):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., height,
            f'{energy:.2f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )
    
    ax.set_ylabel('Residual Energy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"OK Energy comparison chart saved to {output_path}")


def plot_vector_statistics(
    data: dict[str, dict[str, float]],
    output_path: str | Path,
    title: str = "Vector Statistics Comparison",
) -> None:
    """Generate a comprehensive statistics comparison chart.
    
    Args:
        data: Dictionary with algorithm names as keys and stats dicts as values
              Stats should contain keys like 'mean_magnitude', 'std_magnitude', etc.
        output_path: Path to save the comparison chart
        title: Title for the chart
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    algorithms = list(data.keys())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3']
    
    # Mean magnitude
    mean_magnitudes = [data[algo].get('mean_magnitude', 0) for algo in algorithms]
    axes[0, 0].bar(algorithms, mean_magnitudes, color=colors[:len(algorithms)], alpha=0.8, edgecolor='black')
    axes[0, 0].set_title('Mean Vector Magnitude', fontweight='bold')
    axes[0, 0].set_ylabel('Magnitude (pixels)')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Standard deviation of magnitude
    std_magnitudes = [data[algo].get('std_magnitude', 0) for algo in algorithms]
    axes[0, 1].bar(algorithms, std_magnitudes, color=colors[:len(algorithms)], alpha=0.8, edgecolor='black')
    axes[0, 1].set_title('Std Dev of Vector Magnitude', fontweight='bold')
    axes[0, 1].set_ylabel('Std Dev (pixels)')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Number of zero vectors
    zero_vectors = [data[algo].get('zero_vectors', 0) for algo in algorithms]
    axes[1, 0].bar(algorithms, zero_vectors, color=colors[:len(algorithms)], alpha=0.8, edgecolor='black')
    axes[1, 0].set_title('Number of Zero Vectors', fontweight='bold')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Processing time
    proc_times = [data[algo].get('processing_time', 0) for algo in algorithms]
    axes[1, 1].bar(algorithms, proc_times, color=colors[:len(algorithms)], alpha=0.8, edgecolor='black')
    axes[1, 1].set_title('Processing Time', fontweight='bold')
    axes[1, 1].set_ylabel('Time (ms)')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"OK Vector statistics chart saved to {output_path}")


def plot_metrics_comparison(
    metrics_fs: dict[str, float],
    metrics_diamond: dict[str, float],
    output_path: str | Path,
    title: str = "Full Search vs Diamond Search Metrics",
) -> None:
    """Generate a comprehensive metrics comparison between two algorithms.
    
    Args:
        metrics_fs: Metrics dictionary for Full Search
        metrics_diamond: Metrics dictionary for Diamond Search
        output_path: Path to save the comparison chart
        title: Title for the chart
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Get common metric keys
    metric_names = list(set(metrics_fs.keys()) & set(metrics_diamond.keys()))
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    fs_values = [metrics_fs[m] for m in metric_names]
    diamond_values = [metrics_diamond[m] for m in metric_names]
    
    bars1 = ax.bar(x - width/2, fs_values, width, label='Full Search', color='#FF6B6B', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, diamond_values, width, label='Diamond Search', color='#4ECDC4', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"OK Metrics comparison chart saved to {output_path}")


def plot_residual_energy_series(
    energy_data: dict[str, list[float]],
    output_path: str | Path,
    title: str = "Residual Energy Series Comparison",
) -> None:
    """Generate a line chart comparing residual energy across frame pairs.

    Args:
        energy_data: Dictionary mapping algorithm names to lists of energy values.
        output_path: Path to save the comparison chart.
        title: Title for the chart.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Determine number of points (frame pairs)
    first_vals = next(iter(energy_data.values()), [])
    x_values = list(range(len(first_vals)))

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3']

    for (algorithm, energies), color in zip(energy_data.items(), colors):
        ax.plot(
            x_values,
            energies,
            marker='o',
            linewidth=2.0,
            label=algorithm,
            color=color,
            alpha=0.9,
        )

    ax.set_xlabel('Frame Pair Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Residual Energy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"OK Residual energy series chart saved to {output_path}")
