"""Select representative frames from videos for analysis and comparison."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.preprocessing.frame_extractor import extract_frames_from_video
from src.utils.config import Config


class FrameSelector:
    """Utility class for selecting representative frames from videos."""
    
    @staticmethod
    def select_by_motion_intensity(
        frames_dir: str | Path,
        num_frames: int = 5,
        intensity_metric: str = "variance",
    ) -> list[tuple[int, float]]:
        """Select frames based on motion intensity.
        
        Args:
            frames_dir: Directory containing extracted frames
            num_frames: Number of frames to select
            intensity_metric: Metric to use ("variance", "diff_sum")
            
        Returns:
            List of (frame_index, intensity) tuples
        """
        frames_dir = Path(frames_dir)
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        
        if len(frame_files) < 2:
            return []
        
        frame_scores = []
        
        for idx in range(len(frame_files) - 1):
            f1 = cv2.imread(str(frame_files[idx]), cv2.IMREAD_GRAYSCALE)
            f2 = cv2.imread(str(frame_files[idx + 1]), cv2.IMREAD_GRAYSCALE)
            
            if f1 is None or f2 is None:
                continue
            
            if intensity_metric == "variance":
                diff = cv2.absdiff(f1, f2)
                score = np.var(diff)
            else:  # diff_sum
                diff = cv2.absdiff(f1, f2)
                score = np.sum(diff)
            
            frame_scores.append((idx, float(score)))
        
        # Sort by intensity and select top frames
        frame_scores.sort(key=lambda x: x[1], reverse=True)
        selected = sorted(frame_scores[:num_frames], key=lambda x: x[0])
        
        return selected
    
    @staticmethod
    def select_uniform_distribution(
        frames_dir: str | Path,
        num_frames: int = 5,
    ) -> list[int]:
        """Select frames uniformly distributed across the sequence.
        
        Args:
            frames_dir: Directory containing extracted frames
            num_frames: Number of frames to select
            
        Returns:
            List of frame indices
        """
        frames_dir = Path(frames_dir)
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        
        if len(frame_files) == 0:
            return []
        
        total_frames = len(frame_files)
        indices = []
        
        if num_frames == 1:
            indices = [total_frames // 2]
        else:
            step = total_frames // (num_frames + 1)
            indices = [step * (i + 1) for i in range(num_frames)]
        
        return [min(idx, total_frames - 1) for idx in indices]
    
    @staticmethod
    def select_first_last_middle(frames_dir: str | Path) -> list[int]:
        """Select first, last, and middle frames.
        
        Args:
            frames_dir: Directory containing extracted frames
            
        Returns:
            List of frame indices
        """
        frames_dir = Path(frames_dir)
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        total = len(frame_files)
        
        if total == 0:
            return []
        elif total == 1:
            return [0]
        else:
            return [0, total // 2, total - 1]
    
    @staticmethod
    def select_edges_and_peak(
        frames_dir: str | Path,
        num_frames: int = 7,
    ) -> list[int]:
        """Select frames from edges and peaks of motion intensity.
        
        Args:
            frames_dir: Directory containing extracted frames
            num_frames: Target number of frames
            
        Returns:
            List of frame indices
        """
        frames_dir = Path(frames_dir)
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        total = len(frame_files)
        
        if total < 3:
            return list(range(total))
        
        # Always include first and last
        selected = {0, total - 1}
        
        # Add middle frame
        selected.add(total // 2)
        
        # Calculate motion intensity for remaining selection
        frame_scores = []
        for idx in range(total - 1):
            f1 = cv2.imread(str(frame_files[idx]), cv2.IMREAD_GRAYSCALE)
            f2 = cv2.imread(str(frame_files[idx + 1]), cv2.IMREAD_GRAYSCALE)
            
            if f1 is None or f2 is None:
                continue
            
            diff = cv2.absdiff(f1, f2)
            score = np.sum(diff)
            frame_scores.append((idx, score))
        
        # Sort by score and add top frames
        frame_scores.sort(key=lambda x: x[1], reverse=True)
        for idx, _ in frame_scores[:num_frames - len(selected)]:
            selected.add(idx)
            if len(selected) >= num_frames:
                break
        
        return sorted(list(selected))[:num_frames]


def get_representative_frames(
    video_path: str | Path,
    num_frames: int = 5,
    selection_method: str = "motion",
) -> dict[str, Any]:
    """Extract and select representative frames from a video.
    
    Args:
        video_path: Path to source video
        num_frames: Number of frames to select
        selection_method: Selection strategy ("motion", "uniform", "first_last_middle", "edges")
        
    Returns:
        Dictionary with frame paths and metadata
    """
    video_path = Path(video_path)
    
    # Extract all frames
    motion_level = video_path.stem
    frames_dir = Config.EXTRACTED_FRAMES_DIR / motion_level / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    extract_frames_from_video(video_path, frames_dir)
    
    # Select representative frames
    selector = FrameSelector()
    
    if selection_method == "motion":
        selected = selector.select_by_motion_intensity(frames_dir, num_frames)
        indices = [idx for idx, _ in selected]
        scores = [score for _, score in selected]
    elif selection_method == "uniform":
        indices = selector.select_uniform_distribution(frames_dir, num_frames)
        scores = [None] * len(indices)
    elif selection_method == "first_last_middle":
        indices = selector.select_first_last_middle(frames_dir)
        scores = [None] * len(indices)
    elif selection_method == "edges":
        indices = selector.select_edges_and_peak(frames_dir, num_frames)
        scores = [None] * len(indices)
    else:
        indices = selector.select_uniform_distribution(frames_dir, num_frames)
        scores = [None] * len(indices)
    
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    selected_frames = []
    
    for idx in indices:
        if idx < len(frame_files):
            selected_frames.append(str(frame_files[idx]))
    
    return {
        "video_path": str(video_path),
        "motion_level": motion_level,
        "total_frames_extracted": len(frame_files),
        "selected_indices": indices,
        "selected_frames": selected_frames,
        "motion_scores": scores,
        "selection_method": selection_method,
    }
