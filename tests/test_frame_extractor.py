import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.preprocessing.frame_extractor import (
    extract_frames_from_video,
    verify_frame_sequence_integrity,
)


def test_extract_frames_from_video_creates_frames() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        video_file = tmp_path / "sample.avi"
        extracted_dir = tmp_path / "extracted"
        sample_dir = tmp_path / "sample_frames"

        frame_size = (32, 32)
        writer = cv2.VideoWriter(
            str(video_file),
            cv2.VideoWriter_fourcc(*"MJPG"),
            10.0,
            frame_size,
        )
        assert writer.isOpened(), "Video writer failed to open."

        for i in range(6):
            frame = np.full((frame_size[1], frame_size[0], 3), i * 40, dtype=np.uint8)
            writer.write(frame)

        writer.release()

        extract_frames_from_video(
            video_file,
            extracted_dir,
            sample_output_dir=sample_dir,
            sample_frame_count=3,
        )

        assert verify_frame_sequence_integrity(extracted_dir)
        extracted_files = sorted(extracted_dir.glob("frame_*.png"))
        sample_files = sorted(sample_dir.glob("frame_*.png"))

        assert len(extracted_files) == 6
        assert len(sample_files) == 3
        assert extracted_files[0].name == "frame_0000.png"
        assert sample_files[0].name == "frame_0000.png"
