"""
Camera capture module with threaded frame grabbing.
Supports Pi Camera v2 (via Picamera2) and USB webcam (via OpenCV).
Threaded capture ensures the ML pipeline never blocks on camera I/O.
"""

import cv2
import threading
import time
import numpy as np
import config


class Camera:
    """Unified camera interface with threaded capture for maximum FPS."""

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
        self.mirror = getattr(config, "CAMERA_MIRROR", True)
        self.cap = None
        self.picam = None
        self._initialized = False

        # Threaded capture state
        self._frame = None
        self._frame_lock = threading.Lock()
        self._capture_thread = None
        self._running = False

    def start(self):
        """Start the camera capture (opens device + starts background thread)."""
        if self.source == "picamera":
            self._start_picamera()
        else:
            self._start_webcam()
        self._initialized = True

        # Start the background capture thread
        self._running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True
        )
        self._capture_thread.start()

        print(f"[Camera] Started ({self.source}) at {self.width}x{self.height}")
        if self.mirror:
            print("[Camera] Mirror mode enabled (natural selfie view)")

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
            # Try index 1 as fallback (some laptops use 1 for built-in cam)
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                raise RuntimeError("[Camera] Could not open webcam!")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        # Reduce buffer size so we always get the latest frame
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _capture_loop(self):
        """Background thread: continuously grabs frames from the camera."""
        while self._running:
            frame = self._grab_frame()
            if frame is not None:
                # Apply mirror if enabled
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                with self._frame_lock:
                    self._frame = frame
            else:
                time.sleep(0.001)

    def _grab_frame(self):
        """Grab a single frame from the active camera source."""
        if self.source == "picamera" and self.picam is not None:
            try:
                frame = self.picam.capture_array()
                return frame  # Already RGB from Picamera2
            except Exception:
                return None
        elif self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                return None
            # OpenCV captures in BGR, convert to RGB for MediaPipe
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        return None

    def read(self):
        """
        Read the latest frame from the camera (non-blocking).

        Returns:
            numpy.ndarray: RGB frame (H, W, 3), or None if no frame yet.
        """
        if not self._initialized:
            raise RuntimeError("[Camera] Camera not started. Call start() first.")

        with self._frame_lock:
            if self._frame is not None:
                return self._frame.copy()
        return None

    def stop(self):
        """Release camera resources and stop the capture thread."""
        self._running = False
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

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
