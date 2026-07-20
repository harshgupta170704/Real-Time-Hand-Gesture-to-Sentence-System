"""
Camera capture module.
Supports Pi Camera v2 (via Picamera2) and USB webcam (via OpenCV).
"""

import cv2
import numpy as np
import config


class Camera:
    """Unified camera interface for Pi Camera v2 and USB webcams."""

    def __init__(self, source=None):
        """
        Initialize the camera.

        Args:
            source: "picamera" for Pi Camera v2, "webcam" for USB/desktop.
                    Defaults to config.CAMERA_SOURCE.
        """
        self.source = source or config.CAMERA_SOURCE
        self.width = config.CAMERA_WIDTH
        self.height = config.CAMERA_HEIGHT
        self.cap = None
        self.picam = None
        self._initialized = False

    def start(self):
        """Start the camera capture."""
        if self.source == "picamera":
            self._start_picamera()
        else:
            self._start_webcam()
        self._initialized = True
        print(f"[Camera] Started ({self.source}) at {self.width}x{self.height}")

    def _start_picamera(self):
        """Initialize Pi Camera v2 via Picamera2."""
        try:
            from picamera2 import Picamera2

            self.picam = Picamera2()
            camera_config = self.picam.create_video_configuration(
                main={
                    "size": (self.width, self.height),
                    "format": "RGB888",
                }
            )
            self.picam.configure(camera_config)
            self.picam.start()
        except ImportError:
            print("[Camera] Picamera2 not available, falling back to webcam...")
            self.source = "webcam"
            self._start_webcam()
        except Exception as e:
            print(f"[Camera] Pi Camera error: {e}, falling back to webcam...")
            self.source = "webcam"
            self._start_webcam()

    def _start_webcam(self):
        """Initialize USB webcam via OpenCV."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("[Camera] Could not open webcam!")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

    def read(self):
        """
        Read a frame from the camera.

        Returns:
            numpy.ndarray: RGB frame (H, W, 3), or None if capture failed.
        """
        if not self._initialized:
            raise RuntimeError("[Camera] Camera not started. Call start() first.")

        if self.source == "picamera" and self.picam is not None:
            frame = self.picam.capture_array()
            return frame  # Already RGB from Picamera2
        elif self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                return None
            # OpenCV captures in BGR, convert to RGB for MediaPipe
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        return None

    def stop(self):
        """Release camera resources."""
        if self.picam is not None:
            try:
                self.picam.stop()
            except Exception:
                pass
            self.picam = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._initialized = False
        print("[Camera] Stopped.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
