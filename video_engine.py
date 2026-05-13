"""
video_engine.py
---------------
Frame acquisition wrapper tuned for ADAS pipeline.

Sampling strategy
-----------------
For a 30-fps video we want exactly ONE analysed frame per second.
We achieve this by skipping (fps - 1) frames between reads so the
pipeline always works on the first frame of each new second bucket,
not every raw frame.  This matches the LLD requirement:
  "1 frame extracted from a 30-fps second, analysed; then 1 from the next second."

For a live webcam the same logic applies: we still only hand the
pipeline one frame per second worth of real time.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import cv2

VideoSource = Union[int, str]


@dataclass
class VideoFrame:
    frame: "cv2.Mat"
    timestamp: float
    frame_index: int = 0           # absolute frame counter inside the file
    second_index: int = 0          # which second this frame belongs to


class VideoEngine:
    """
    Lightweight frame acquisition wrapper — Streamlit-friendly.

    Create once → keep in st.session_state → call read() on every rerun.
    read() returns the NEXT 1-per-second frame, or None if the source
    has been exhausted / not yet due.
    """

    # Fallback assumed fps when the codec cannot report it
    _DEFAULT_FPS: float = 30.0

    def __init__(
        self,
        source: Optional[VideoSource] = None,
        prefer_file: str = "POV_driving_video_from_inside.mp4",
        target_size: Optional[Tuple[int, int]] = (960, 540),
    ) -> None:

        if source is None:
            source = prefer_file if os.path.exists(prefer_file) else 0

        self.source: VideoSource = source
        self.target_size = target_size

        self._cap: Optional[cv2.VideoCapture] = cv2.VideoCapture(source)

        # How many raw frames make up one second in this source
        reported = self._cap.get(cv2.CAP_PROP_FPS) if self._cap else 0.0
        self._source_fps: float = reported if reported > 1.0 else self._DEFAULT_FPS

        # Frames to skip so we process exactly 1 frame / second
        self._skip_frames: int = max(1, round(self._source_fps)) - 1

        # Internal counters
        self._raw_frame_index: int = 0      # counts every raw frame read
        self._second_index: int = 0         # counts delivered 1-per-sec frames

        # Wall-clock guard for webcam / real-time sources
        self._last_deliver_ts: float = 0.0

        # Always hold a copy of the most recently *delivered* frame so the
        # dashboard can show something before the first real second elapses.
        self._last_delivered: Optional[VideoFrame] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_opened(self) -> bool:
        return bool(self._cap) and self._cap.isOpened()

    @property
    def source_fps(self) -> float:
        return self._source_fps

    @property
    def skip_frames(self) -> int:
        return self._skip_frames

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self) -> Optional[VideoFrame]:
        """
        Return the next 1-per-second frame, or None if not yet due / EOF.

        For video files: skips (_skip_frames) raw frames then returns the
        next one — delivering exactly 1 frame per source-second.

        For webcams: additionally enforces a wall-clock guard so we never
        deliver faster than 1 frame / second of real time.
        """
        if not self.is_opened:
            return None

        # --- Skip the in-between frames of the current second ----------
        for _ in range(self._skip_frames):
            ok, _ = self._cap.read()
            if not ok:
                self._try_loop()
                return None
            self._raw_frame_index += 1

        # --- Read the representative frame for this second -------------
        ok, frame = self._cap.read()
        if not ok:
            self._try_loop()
            return None

        self._raw_frame_index += 1

        # Webcam real-time guard: don't deliver faster than 1 fps
        if isinstance(self.source, int):
            now = time.time()
            if now - self._last_deliver_ts < 1.0:
                return self._last_delivered      # not yet a new second
            self._last_deliver_ts = now

        # Resize
        if self.target_size is not None:
            frame = cv2.resize(frame, self.target_size)

        vf = VideoFrame(
            frame=frame,
            timestamp=time.time(),
            frame_index=self._raw_frame_index,
            second_index=self._second_index,
        )
        self._second_index += 1
        self._last_delivered = vf
        return vf

    def peek_last(self) -> Optional[VideoFrame]:
        """Return the last delivered frame without advancing the stream."""
        return self._last_delivered

    def release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_loop(self) -> None:
        """Loop video files back to frame 0; no-op for webcams."""
        if isinstance(self.source, str):
            try:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._raw_frame_index = 0
                self._second_index = 0
            except Exception:
                pass