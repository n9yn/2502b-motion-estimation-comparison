"""Unit tests for demo workflow utilities."""

from pathlib import Path

from src.demo.frame_selector import FrameSelector
from src.demo.run_demo import save_selected_frames


def test_select_uniform_distribution_returns_expected_indices(tmp_path: Path) -> None:
    for i in range(10):
        (tmp_path / f"frame_{i:05d}.png").write_text("")

    indices = FrameSelector.select_uniform_distribution(tmp_path, num_frames=4)

    assert indices == [2, 4, 6, 8]


def test_save_selected_frames_copies_selected_files(tmp_path: Path) -> None:
    source = tmp_path / "frame_00001.png"
    source.write_text("dummy")
    destination = tmp_path / "selected_frames"

    copied = save_selected_frames({"selected_frames": [str(source)]}, destination)

    assert len(copied) == 1
    assert (destination / source.name).exists()
    assert copied[0] == str(destination / source.name)
