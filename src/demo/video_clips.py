"""Create demo video clips showcasing motion estimation algorithms."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.utils.config import Config


def create_demo_video_clips(
    video_path: str | Path,
    output_dir: str | Path,
    num_clips: int = 3,
    clip_duration_seconds: float = 3.0,
) -> list[str]:
    """Create multiple short demo video clips from a source video.
    
    Args:
        video_path: Path to source video file
        output_dir: Directory to save demo clips
        num_clips: Number of clips to create
        clip_duration_seconds: Duration of each clip in seconds
        
    Returns:
        List of paths to created demo clips
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"✗ Video file not found: {video_path}")
        return []
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"✗ Cannot open video: {video_path}")
        return []
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames_per_clip = int(clip_duration_seconds * fps)
    
    if total_frames < frames_per_clip:
        print(f"⚠ Warning: Video is shorter than requested clip duration ({total_frames} frames)")
        num_clips = 1
        frames_per_clip = total_frames
    
    # Calculate start positions for clips distributed across the video
    clip_starts = []
    if num_clips == 1:
        clip_starts = [0]
    else:
        max_start = total_frames - frames_per_clip
        for i in range(num_clips):
            start_frame = int((max_start / (num_clips - 1)) * i) if num_clips > 1 else 0
            clip_starts.append(start_frame)
    
    created_clips = []
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    for clip_idx, start_frame in enumerate(clip_starts):
        clip_name = f"demo_clip_{clip_idx + 1:02d}.mp4"
        clip_path = output_path / clip_name
        
        out = cv2.VideoWriter(str(clip_path), fourcc, fps, (frame_width, frame_height))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames_written = 0
        
        for _ in range(frames_per_clip):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frames_written += 1
        
        out.release()
        created_clips.append(str(clip_path))
        print(f"✓ Created demo clip {clip_idx + 1}: {clip_name} ({frames_written} frames)")
    
    cap.release()
    return created_clips


def create_comparison_video(
    video_path: str | Path,
    output_path: str | Path,
    duration_seconds: float = 5.0,
    motion_type: str = "all",
) -> str:
    """Create a single comparison demo video with specific motion characteristics.
    
    Args:
        video_path: Source video path
        output_path: Output video path
        duration_seconds: Duration of the demo video
        motion_type: Type of motion to capture ("low", "medium", "high", "all")
        
    Returns:
        Path to created video
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"✗ Cannot open video: {video_path}")
        return ""
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames_to_extract = int(duration_seconds * fps)
    
    # Select start position based on motion type
    if motion_type == "low":
        start_frame = 0
    elif motion_type == "medium":
        start_frame = max(0, total_frames // 3)
    elif motion_type == "high":
        start_frame = max(0, (total_frames * 2) // 3)
    else:
        start_frame = 0
        frames_to_extract = min(frames_to_extract, total_frames)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
    
    frames_written = 0
    while frames_written < frames_to_extract:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frames_written += 1
    
    cap.release()
    out.release()
    
    print(f"✓ Created comparison video: {output_path.name} ({frames_written} frames, {motion_type} motion)")
    return str(output_path)


def create_all_demo_clips(output_base_dir: str | Path = None) -> dict[str, Any]:
    """Create a full set of demo video clips for all available videos.
    
    Args:
        output_base_dir: Base output directory
        
    Returns:
        Dictionary with created clips and metadata
    """
    if output_base_dir is None:
        output_base_dir = Config.OUTPUT_DIR / "demo_videos"
    
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "low_motion": [],
        "medium_motion": [],
        "high_motion": [],
        "comparisons": [],
    }
    
    # Create clips from available videos
    for motion_level in ["low_motion", "medium_motion", "high_motion"]:
        video_path = Config.RAW_VIDEOS_DIR / f"{motion_level}.mp4"
        if video_path.exists():
            clips_dir = output_base_dir / motion_level
            clips = create_demo_video_clips(video_path, clips_dir, num_clips=3, clip_duration_seconds=3.0)
            results[motion_level] = clips
    
    # Create comparison videos
    comparison_dir = output_base_dir / "comparisons"
    for motion_level in ["low_motion", "medium_motion", "high_motion"]:
        video_path = Config.RAW_VIDEOS_DIR / f"{motion_level}.mp4"
        if video_path.exists():
            comparison_video = create_comparison_video(
                video_path,
                comparison_dir / f"comparison_{motion_level}.mp4",
                duration_seconds=5.0,
                motion_type=motion_level.split("_")[0],
            )
            if comparison_video:
                results["comparisons"].append(comparison_video)
    
    return results
