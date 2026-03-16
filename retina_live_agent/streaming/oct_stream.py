"""
OCT Stream — simulates a real-time intraoperative OCT feed.

Reads frames from data/oct_demo_video.mp4 (user-provided).
If no video is found, generates synthetic grayscale frames for demo purposes.

Frame rate: one frame every 300 ms (~3.3 fps — realistic for intraoperative OCT).
"""

import time
import threading
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import cv2

VIDEO_PATH = Path(__file__).resolve().parents[1] / "data" / "oct_demo_video.mp4"
FRAME_INTERVAL_S = 0.30   # 300 ms between processed frames


class OCTStream:
    """
    Streams OCT frames from a video file (or synthetic source) and calls
    a user-supplied callback for each frame.

    Usage
    -----
    stream = OCTStream(on_frame=my_callback)
    stream.start()
    ...
    stream.stop()
    """

    def __init__(
        self,
        on_frame: Callable[[np.ndarray, int], None],
        video_path: Path = VIDEO_PATH,
        loop: bool = True,
    ):
        self.on_frame    = on_frame
        self.video_path  = Path(video_path)
        self.loop        = loop
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("[OCTStream] Streaming started.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        print("[OCTStream] Streaming stopped.")

    # ------------------------------------------------------------------
    def _run(self) -> None:
        if self.video_path.exists():
            self._stream_from_video()
        else:
            print(
                f"[OCTStream] Video not found at {self.video_path}. "
                "Streaming synthetic frames."
            )
            self._stream_synthetic()

    # ------------------------------------------------------------------
    def _stream_from_video(self) -> None:
        cap = cv2.VideoCapture(str(self.video_path))
        frame_idx = 0

        while not self._stop_event.is_set():
            ret, frame = cap.read()

            if not ret:
                if self.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            self.on_frame(frame, frame_idx)
            frame_idx += 1
            time.sleep(FRAME_INTERVAL_S)

        cap.release()

    # ------------------------------------------------------------------
    def _stream_synthetic(self) -> None:
        """
        Generate synthetic OCT-like greyscale frames with random noise.
        Used when no real video is available (demo / CI mode).
        """
        frame_idx = 0
        while not self._stop_event.is_set():
            # Simulate a 512×512 OCT B-scan with Gaussian noise + horizontal bands
            frame = np.zeros((512, 512, 3), dtype=np.uint8)
            noise = np.random.normal(80, 30, (512, 512)).clip(0, 255).astype(np.uint8)
            frame[:, :, 0] = noise
            frame[:, :, 1] = noise
            frame[:, :, 2] = noise

            # Add faint retinal layer lines
            for y in [180, 220, 280, 320]:
                cv2.line(frame, (0, y), (512, y), (140, 140, 140), 1)

            self.on_frame(frame, frame_idx)
            frame_idx += 1
            time.sleep(FRAME_INTERVAL_S)
