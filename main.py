"""
Real-Time Hand Gesture to Sentence System
==========================================
Main application — runs the processing pipeline and serves a web UI.

Optimized for laptop webcam with real-time performance.
Uses Google's free pre-trained MediaPipe Gesture Recognizer model.

View the output in your browser at:
    http://localhost:5000

Usage:
    python main.py                    # Run with laptop webcam (default)
    python main.py --camera picamera  # Run with Pi Camera v2
"""

import sys
import time
import argparse
import cv2
import numpy as np
import config
from camera import Camera
from gesture_classifier import GestureClassifier
from sentence_builder import SentenceBuilder
import web_server


def get_local_ip():
    """Try to detect the machine's local IP address."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


class GestureToSentenceApp:
    """Main application for real-time hand gesture to sentence conversion."""

    def __init__(self, camera_source=None):
        self.camera = Camera(source=camera_source)
        self.classifier = GestureClassifier()
        self.builder = SentenceBuilder()

        # HandDetector is only needed for rule-based fallback
        # (mp.solutions was removed in newer MediaPipe versions)
        self.detector = None
        if self.classifier.classifier_type == "rule":
            from hand_detector import HandDetector
            self.detector = HandDetector()

        # FPS tracking
        self.fps = 0.0
        self.frame_times = []

        # Visual feedback
        self.no_hand_frames = 0
        self.last_word_added = None

    def run(self):
        """Main processing loop."""
        ip = get_local_ip()
        print()
        print("+==================================================+")
        print("|   HAND GESTURE -> SENTENCE SYSTEM                |")
        print("|   Real-Time Laptop Webcam Edition                |")
        print("+==================================================+")
        print("|                                                  |")
        port = str(config.WEB_PORT)
        print(f"|   Open in your browser:                          |")
        print(f"|   http://localhost:{port:<29s}|")
        print(f"|   http://{ip}:{port:<29s}      |")
        print("|                                                  |")
        print("+--------------------------------------------------+")
        print("|   Gestures recognized:                           |")
        gestures = self.classifier.get_all_gestures()
        for gesture, word in gestures.items():
            print(f"|     {gesture:20s} -> {word:<20s}   |")
        print("|                                                  |")
        print("|   Press Ctrl+C to stop                           |")
        print("+==================================================+")
        print()

        # Start the web server (runs in background thread)
        web_server.start_server()

        # Start camera
        self.camera.start()

        try:
            while True:
                loop_start = time.time()

                # 1. Capture frame
                frame_rgb = self.camera.read()
                if frame_rgb is None:
                    time.sleep(0.01)
                    continue

                # 2. Detect & classify
                if self.classifier.classifier_type == "mediapipe_task":
                    gesture_result = self._process_with_task_api(frame_rgb)
                else:
                    gesture_result = self._process_with_rule_based(frame_rgb)

                # 3. Track hand presence (for duplicate word reset)
                if gesture_result.get("gesture") is None:
                    self.no_hand_frames += 1
                    if self.no_hand_frames > 5:
                        self.builder.reset_duplicate_block()
                else:
                    self.no_hand_frames = 0

                # 4. Handle web UI commands
                if web_server.check_clear_requested():
                    self.builder.clear()
                    self.last_word_added = None
                    print("  [Sentence cleared from web UI]")
                if web_server.check_undo_requested():
                    self.builder.undo_last_word()
                    print("  [Undo from web UI]")
                if web_server.check_speak_requested():
                    self.builder.speak()
                    print("  [Speaking sentence from web UI]")

                # 5. Feed to sentence builder
                builder_result = self.builder.update(gesture_result)

                # 6. Log word additions
                if builder_result["word_added"]:
                    self.last_word_added = builder_result["word_added"]
                    print(
                        f"  + \"{builder_result['word_added']}\" -> "
                        f"\"{builder_result['current_sentence']}\""
                    )

                # 7. Render the frame with overlay
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                self._draw_overlay(frame_bgr, gesture_result, builder_result)

                # 8. Push frame + state to web server
                web_server.update_frame(frame_bgr)
                web_server.update_sentence_data({
                    "current_sentence": builder_result["current_sentence"],
                    "gesture": gesture_result.get("gesture"),
                    "gesture_word": gesture_result.get("word"),
                    "confidence": gesture_result.get("confidence", 0),
                    "progress": builder_result["building_progress"],
                    "status": builder_result["status"],
                    "word_count": self.builder.get_word_count(),
                    "last_word_added": self.last_word_added,
                    "history": self.builder.get_history(),
                    "fps": self.fps,
                    "raw_words": self.builder.get_raw_words(),
                })

                # 9. Update FPS
                elapsed = time.time() - loop_start
                self._update_fps(elapsed)

        except KeyboardInterrupt:
            print("\n[App] Shutting down...")
        finally:
            self.camera.stop()
            if self.detector:
                self.detector.close()
            print("[App] Done.")

    def _process_with_task_api(self, frame_rgb):
        """Process a frame using MediaPipe Gesture Recognizer Task API."""
        result = self.classifier.classify_frame(frame_rgb)
        return result

    def _process_with_rule_based(self, frame_rgb):
        """Process a frame using MediaPipe Hands + rule-based classifier."""
        if self.detector is None:
            from hand_detector import HandDetector
            self.detector = HandDetector()
        detection = self.detector.detect(frame_rgb)
        finger_states = self.detector.get_finger_states(detection["landmarks"])
        result = self.classifier.classify(
            detection["landmarks"],
            finger_states,
            detection["normalized"],
        )
        # Attach landmarks for overlay drawing
        result["_detection"] = detection
        return result

    def _draw_overlay(self, frame_bgr, gesture_result, builder_result):
        """Draw a lightweight overlay on the video frame."""
        h, w = frame_bgr.shape[:2]

        # Draw hand landmarks (for rule-based mode)
        if self.classifier.classifier_type == "rule" and self.detector:
            detection = gesture_result.get("_detection")
            if detection and config.SHOW_LANDMARKS:
                self.detector.draw_landmarks(frame_bgr, detection["raw_results"])
        else:
            # For task API, draw landmarks manually (works across all MediaPipe versions)
            landmarks_data = self.classifier.get_last_hand_landmarks()
            if landmarks_data and config.SHOW_LANDMARKS:
                for hand_lms in landmarks_data:
                    points = []
                    for lm in hand_lms:
                        px = int(lm.x * w)
                        py = int(lm.y * h)
                        points.append((px, py))
                        cv2.circle(frame_bgr, (px, py), 3, (0, 255, 0), -1)
                    # Draw connections between landmarks
                    connections = [
                        (0,1),(1,2),(2,3),(3,4),        # thumb
                        (0,5),(5,6),(6,7),(7,8),        # index
                        (5,9),(9,10),(10,11),(11,12),   # middle
                        (9,13),(13,14),(14,15),(15,16), # ring
                        (13,17),(17,18),(18,19),(19,20),# pinky
                        (0,17),                         # palm
                    ]
                    for c in connections:
                        if c[0] < len(points) and c[1] < len(points):
                            cv2.line(frame_bgr, points[c[0]], points[c[1]],
                                     (0, 220, 0), 1, cv2.LINE_AA)

        # Detected gesture label
        gesture = gesture_result.get("gesture")
        word = gesture_result.get("word")
        if gesture and word:
            label = f"{gesture} -> {word}"
            cv2.putText(
                frame_bgr, label, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                config.COLOR_SUCCESS, 1, cv2.LINE_AA,
            )

        # FPS
        if config.SHOW_FPS:
            cv2.putText(
                frame_bgr, f"{self.fps:.0f} FPS", (w - 65, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 200, 0), 1, cv2.LINE_AA,
            )

        # Sentence at bottom
        sentence = builder_result["current_sentence"]
        if sentence:
            # Dark bar at bottom
            cv2.rectangle(frame_bgr, (0, h - 28), (w, h), (0, 0, 0), -1)
            cv2.putText(
                frame_bgr, sentence, (6, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA,
            )

    def _update_fps(self, frame_time):
        """Rolling average FPS."""
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        avg = sum(self.frame_times) / len(self.frame_times)
        self.fps = 1.0 / avg if avg > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Hand Gesture -> Sentence | Real-Time Laptop Webcam"
    )
    parser.add_argument(
        "--camera", choices=["picamera", "webcam"],
        default=config.CAMERA_SOURCE,
        help="Camera source (default: webcam)",
    )
    parser.add_argument(
        "--classifier", choices=["mediapipe_task", "rule"],
        default=config.CLASSIFIER_TYPE,
        help="Classifier type (default: mediapipe_task)",
    )
    parser.add_argument(
        "--port", type=int, default=config.WEB_PORT,
        help="Web server port (default: 5000)",
    )
    parser.add_argument(
        "--no-tts", action="store_true",
        help="Disable text-to-speech",
    )
    args = parser.parse_args()

    if args.classifier:
        config.CLASSIFIER_TYPE = args.classifier
    if args.port:
        config.WEB_PORT = args.port
    if args.no_tts:
        config.TTS_ENABLED = False

    app = GestureToSentenceApp(camera_source=args.camera)
    app.run()


if __name__ == "__main__":
    main()
