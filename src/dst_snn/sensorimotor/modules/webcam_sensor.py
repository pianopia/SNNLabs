"""Webcam (or synthetic frame) sensor that emits frame-difference events.

When OpenCV is available and a camera opens, frames are captured live.
Otherwise the sensor falls back to a deterministic synthetic frame stream so
tests and headless CI remain offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..protocol import SensorimotorMessage, register_message


def opencv_available() -> bool:
    try:
        import cv2  # noqa: F401
    except Exception:
        return False
    return True


@dataclass
class WebcamSensor:
    id: str = "webcam-sensor"
    width: int = 64
    height: int = 48
    camera_index: int = 0
    use_camera: bool = True
    _previous: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _capture: Any = field(default=None, init=False, repr=False)
    _step: int = field(default=0, init=False)

    def register(self) -> SensorimotorMessage:
        return register_message(
            module_id=self.id,
            role="sensor",
            modality="vision",
            shape=[self.height, self.width],
        )

    def _synthetic_frame(self, step: int) -> np.ndarray:
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        cx = (self.width // 2) + int(8 * np.sin(step * 0.2))
        cy = self.height // 2
        frame = ((xx - cx) ** 2 + (yy - cy) ** 2 < 40).astype(np.float32)
        return frame

    def _open_camera(self) -> bool:
        if not self.use_camera or not opencv_available():
            return False
        try:
            import cv2

            capture = cv2.VideoCapture(self.camera_index)
            if not capture.isOpened():
                capture.release()
                return False
            self._capture = capture
            return True
        except Exception:
            self._capture = None
            return False

    def close(self) -> None:
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None

    def _read_frame(self) -> tuple[np.ndarray, str]:
        if self._capture is None and self.use_camera:
            self._open_camera()
        if self._capture is not None:
            import cv2

            ok, frame = self._capture.read()
            if ok and frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (self.width, self.height))
                return gray.astype(np.float32) / 255.0, "camera"
        return self._synthetic_frame(self._step), "synthetic"

    def observe(self, step: int | None = None) -> SensorimotorMessage:
        if step is not None:
            self._step = step
        frame, source = self._read_frame()
        if self._previous is None:
            events = np.zeros_like(frame)
            motion = 0.0
        else:
            diff = frame - self._previous
            events = (np.abs(diff) > 0.08).astype(np.float32) * np.sign(diff)
            motion = float(np.mean(np.abs(events)))
        self._previous = frame
        self._step += 1
        return SensorimotorMessage(
            type="observation",
            id=self.id,
            payload={
                "frame_mean": float(frame.mean()),
                "motion": motion,
                "event_density": float(np.mean(np.abs(events))),
                "source": source,
                "step": self._step - 1,
                "width": self.width,
                "height": self.height,
            },
        )
