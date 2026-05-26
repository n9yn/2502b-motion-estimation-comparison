"""Comprehensive comparison analysis for motion estimation algorithms."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.utils.metrics import compute_mse, compute_psnr


@dataclass
class AlgorithmStats:
    """Statistics for a motion estimation algorithm."""

    name: str
    execution_time_ms: float
    comparisons: int
    residual_energy: float
    mse: float
    psnr: float
    num_vectors: int

    def __str__(self) -> str:
        return (
            f"{self.name:15} | "
            f"Time: {self.execution_time_ms:7.2f}ms | "
            f"Comparisons: {self.comparisons:7d} | "
            f"Energy: {self.residual_energy:10.2f} | "
            f"MSE: {self.mse:10.2f} | "
            f"PSNR: {self.psnr:7.2f}dB"
        )


@dataclass
class ComparisonResult:
    """Complete comparison results between two algorithms."""

    full_search: AlgorithmStats
    diamond_search: AlgorithmStats

    def get_speedup(self) -> float:
        """Calculate Diamond Search speedup relative to Full Search."""
        return self.full_search.execution_time_ms / self.diamond_search.execution_time_ms

    def get_complexity_reduction(self) -> float:
        """Calculate reduction in comparisons for Diamond Search."""
        ratio = self.diamond_search.comparisons / self.full_search.comparisons
        return (1 - ratio) * 100

    def get_energy_ratio(self) -> float:
        """Get ratio of Diamond Search energy to Full Search energy."""
        if self.full_search.residual_energy == 0:
            return 1.0
        return self.diamond_search.residual_energy / self.full_search.residual_energy

    def get_accuracy_loss(self) -> float:
        """Get accuracy loss in MSE between the two algorithms."""
        return self.diamond_search.mse - self.full_search.mse

    def print_summary(self) -> None:
        """Print comparison summary."""
        print("\n" + "=" * 100)
        print("MOTION ESTIMATION ALGORITHM COMPARISON SUMMARY")
        print("=" * 100)

        print("\n[1] EXECUTION TIME Comparison")
        print("-" * 100)
        print(f"Full Search : {self.full_search.execution_time_ms:7.2f}ms")
        print(f"Diamond Search: {self.diamond_search.execution_time_ms:7.2f}ms")
        speedup = self.get_speedup()
        print(f"OK Diamond Search Speedup: {speedup:.2f}x faster")

        print("\n[2] SEARCH COMPLEXITY Comparison")
        print("-" * 100)
        print(f"Full Search : {self.full_search.comparisons:7d} comparisons")
        print(f"Diamond Search: {self.diamond_search.comparisons:7d} comparisons")
        reduction = self.get_complexity_reduction()
        print(f"OK Complexity Reduction: {reduction:.1f}% fewer comparisons")

        print("\n[3] MOTION ACCURACY Comparison")
        print("-" * 100)
        print(f"Full Search     - MSE: {self.full_search.mse:10.2f}, PSNR: {self.full_search.psnr:7.2f}dB")
        print(f"Diamond Search  - MSE: {self.diamond_search.mse:10.2f}, PSNR: {self.diamond_search.psnr:7.2f}dB")
        mse_loss = self.get_accuracy_loss()
        if abs(mse_loss) < 0.01:
            print("OK Nearly identical accuracy (MSE difference < 0.01)")
        elif mse_loss > 0:
            print(f"WARNING: Accuracy Loss: {mse_loss:.2f} (Diamond Search higher MSE)")
        else:
            print(f"OK Accuracy Gain: {abs(mse_loss):.2f} (Diamond Search better)")

        print("\n[4] RESIDUAL ENERGY Comparison")
        print("-" * 100)
        print(f"Full Search     : {self.full_search.residual_energy:10.2f}")
        print(f"Diamond Search  : {self.diamond_search.residual_energy:10.2f}")
        energy_ratio = self.get_energy_ratio()
        diff = abs(self.full_search.residual_energy - self.diamond_search.residual_energy)
        pct_diff = (diff / self.full_search.residual_energy * 100) if self.full_search.residual_energy > 0 else 0
        print(f"OK Energy Difference: {diff:.2f} ({pct_diff:.1f}%)")

        print("\n[5] OBSERVATIONS & CONCLUSIONS")
        print("-" * 100)
        self._print_observations()

    def _print_observations(self) -> None:
        """Print detailed observations."""
        observations = []

        speedup = self.get_speedup()
        if speedup > 5:
            observations.append(f"- Diamond Search is SIGNIFICANTLY FASTER ({speedup:.1f}x speedup)")
        elif speedup > 2:
            observations.append(f"- Diamond Search is moderately faster ({speedup:.1f}x speedup)")
        else:
            observations.append(f"- Diamond Search has modest speedup ({speedup:.1f}x)")

        complexity_reduction = self.get_complexity_reduction()
        observations.append(
            f"- Complexity reduction of {complexity_reduction:.1f}% "
            f"({self.diamond_search.comparisons:,d} vs {self.full_search.comparisons:,d} comparisons)"
        )

        mse_loss = self.get_accuracy_loss()
        if abs(mse_loss) < 0.01:
            observations.append("- Motion accuracy is VIRTUALLY IDENTICAL between algorithms")
        elif mse_loss < 0:
            observations.append(f"- Diamond Search produced BETTER motion vectors (MSE {abs(mse_loss):.2f} lower)")
        else:
            loss_pct = (mse_loss / self.full_search.mse * 100) if self.full_search.mse > 0 else 0
            observations.append(f"- Diamond Search has slight accuracy loss ({loss_pct:.1f}% higher MSE)")

        energy_ratio = self.get_energy_ratio()
        energy_diff = abs(self.full_search.residual_energy - self.diamond_search.residual_energy)
        energy_pct = (energy_diff / self.full_search.residual_energy * 100) if self.full_search.residual_energy > 0 else 0

        if abs(energy_ratio - 1.0) < 0.05:
            observations.append("- Residual energy is SUBSTANTIALLY SIMILAR (~same compression efficiency)")
        elif energy_ratio < 0.95:
            observations.append(f"- Diamond Search has LOWER energy ({energy_pct:.1f}% difference) - better compression")
        else:
            observations.append(f"- Diamond Search has slightly HIGHER energy ({energy_pct:.1f}% difference)")

        # Efficiency summary
        observations.append("\n- RECOMMENDATION:")
        if speedup > 3 and mse_loss <= 0.1:
            observations.append(
                "  Diamond Search is RECOMMENDED - excellent speedup with minimal accuracy trade-off"
            )
        elif speedup > 2 and mse_loss <= 0.05:
            observations.append("  Diamond Search is RECOMMENDED - good speedup with negligible accuracy loss")
        elif speedup > 1.5:
            observations.append("  Diamond Search is a viable choice - moderate speedup with acceptable accuracy")
        else:
            observations.append("  Full Search may be preferred - speedup does not justify potential accuracy loss")

        for obs in observations:
            print(obs)

        print("=" * 100)


def create_comparison_result(
    frame1: np.ndarray,
    frame2: np.ndarray,
    residual_fs: np.ndarray,
    residual_diamond: np.ndarray,
    energy_fs: float,
    energy_diamond: float,
    vectors_fs_with_stats: tuple[Any, dict],
    vectors_diamond_with_stats: tuple[Any, dict],
) -> ComparisonResult:
    """Create a comprehensive comparison result."""

    vectors_fs, stats_fs = vectors_fs_with_stats
    vectors_diamond, stats_diamond = vectors_diamond_with_stats

    # Compute accuracy metrics
    mse_fs = compute_mse(frame1, residual_fs)
    mse_diamond = compute_mse(frame1, residual_diamond)
    psnr_fs = compute_psnr(frame1, residual_fs)
    psnr_diamond = compute_psnr(frame1, residual_diamond)

    fs_stats = AlgorithmStats(
        name="Full Search",
        execution_time_ms=stats_fs["time_ms"],
        comparisons=stats_fs["comparisons"],
        residual_energy=energy_fs,
        mse=mse_fs,
        psnr=psnr_fs,
        num_vectors=len(vectors_fs),
    )

    diamond_stats = AlgorithmStats(
        name="Diamond Search",
        execution_time_ms=stats_diamond["time_ms"],
        comparisons=stats_diamond["comparisons"],
        residual_energy=energy_diamond,
        mse=mse_diamond,
        psnr=psnr_diamond,
        num_vectors=len(vectors_diamond),
    )

    return ComparisonResult(full_search=fs_stats, diamond_search=diamond_stats)
