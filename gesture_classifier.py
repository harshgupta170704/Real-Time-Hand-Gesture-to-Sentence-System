"""
Gesture classifier module -- expanded with custom landmark analysis.
Supports:
  1. MediaPipe Gesture Recognizer (7 pre-trained gestures from Google)
  2. Custom landmark-based gestures (8 additional finger-state patterns)
  3. Rule-based fallback using finger extension states

The custom gestures analyze hand landmarks returned by the Task API
to detect additional finger patterns beyond the 7 pre-trained ones.
This gives us 15 total distinct gestures without any model training.
"""

import os
import time
import math
import collections
import numpy as np
import config


class GestureClassifier:
    """Classifies hand gestures into word labels with temporal smoothing."""

    def __init__(self, classifier_type=None):
        self.classifier_type = classifier_type or config.CLASSIFIER_TYPE
        self.recognizer = None
        self._last_result = None
        self._frame_count = 0

        # Rolling smoothing buffer for noise reduction
        window_size = getattr(config, "SMOOTHING_WINDOW", 5)
        self._gesture_buffer = collections.deque(maxlen=window_size)
        self._smoothing_threshold = getattr(config, "SMOOTHING_THRESHOLD", 0.6)

        if self.classifier_type == "mediapipe_task":
            self._init_mediapipe_task()

        print(f"[GestureClassifier] Ready ({self.classifier_type} mode).")

    # ─── MediaPipe Gesture Recognizer (pre-trained model) ────

    def _init_mediapipe_task(self):
        """Initialize the MediaPipe Gesture Recognizer task in VIDEO mode."""
        model_path = config.GESTURE_MODEL_PATH
        if not os.path.exists(model_path):
            if config.FALLBACK_TO_RULE:
                print(f"[GestureClassifier] Model not found: {model_path}")
                print("[GestureClassifier] Falling back to rule-based classifier.")
                print("[GestureClassifier] Run 'python download_model.py' to get the model.")
                self.classifier_type = "rule"
                return
            else:
                raise FileNotFoundError(
                    f"Gesture model not found: {model_path}\n"
                    f"Run: python download_model.py"
                )

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            base_options = mp_python.BaseOptions(
                model_asset_path=model_path
            )
            options = vision.GestureRecognizerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=config.MAX_NUM_HANDS,
                min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
            )
            self.recognizer = vision.GestureRecognizer.create_from_options(options)
            print("[GestureClassifier] MediaPipe Gesture Recognizer loaded (VIDEO mode).")

        except Exception as e:
            print(f"[GestureClassifier] Task API error: {e}")
            if config.FALLBACK_TO_RULE:
                print("[GestureClassifier] Falling back to rule-based classifier.")
                self.classifier_type = "rule"
            else:
                raise

    def classify_frame(self, frame_rgb):
        """
        Classify gesture from an RGB frame.
        Uses pre-trained model first, then falls back to custom landmark
        analysis for additional gestures.

        Args:
            frame_rgb: numpy array (H, W, 3) in RGB format.

        Returns:
            dict with gesture, word, confidence, and landmarks.
        """
        if self.recognizer is None:
            return {"gesture": None, "word": None, "confidence": 0.0, "landmarks": None}

        try:
            import mediapipe as mp

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb
            )

            self._frame_count += 1
            timestamp_ms = int(self._frame_count * (1000.0 / config.CAMERA_FPS))

            result = self.recognizer.recognize_for_video(mp_image, timestamp_ms)
            self._last_result = result

            # Extract landmarks (needed for both model and custom gestures)
            landmarks = None
            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                landmarks = [
                    (lm.x, lm.y, lm.z)
                    for lm in result.hand_landmarks[0]
                ]

            # 1. Try pre-trained model gesture first
            model_gesture = None
            model_confidence = 0.0
            if result.gestures and len(result.gestures) > 0:
                gesture = result.gestures[0][0]
                model_gesture = gesture.category_name
                model_confidence = gesture.score

            # 2. If model is confident, use its prediction
            if (model_gesture and model_gesture != "None"
                    and model_confidence >= config.GESTURE_CONFIDENCE_THRESHOLD):

                self._gesture_buffer.append(model_gesture)
                smoothed = self._get_smoothed_gesture()

                if smoothed:
                    word = config.GESTURE_WORD_MAP.get(smoothed, None)
                    if word:
                        return {
                            "gesture": smoothed,
                            "word": word,
                            "confidence": float(model_confidence),
                            "landmarks": landmarks,
                        }

            # 3. If model didn't detect or low confidence, try custom gestures
            if landmarks:
                custom_result = self._classify_custom_gesture(landmarks)
                if custom_result:
                    self._gesture_buffer.append(custom_result["gesture"])
                    smoothed = self._get_smoothed_gesture()

                    if smoothed:
                        # Look up word from either map
                        word = (config.GESTURE_WORD_MAP.get(smoothed)
                                or config.CUSTOM_GESTURE_WORD_MAP.get(smoothed))
                        if word:
                            return {
                                "gesture": smoothed,
                                "word": word,
                                "confidence": custom_result["confidence"],
                                "landmarks": landmarks,
                            }

            # No gesture detected
            self._gesture_buffer.append(None)
            return {"gesture": None, "word": None, "confidence": 0.0, "landmarks": landmarks}

        except Exception as e:
            print(f"[GestureClassifier] Recognition error: {e}")
            return {"gesture": None, "word": None, "confidence": 0.0, "landmarks": None}

    # ─── Custom gesture detection from landmarks ─────────────

    def _classify_custom_gesture(self, landmarks):
        """
        Detect custom gestures by analyzing finger states from landmarks.
        Uses the same landmarks returned by the Task API.

        Args:
            landmarks: list of 21 (x, y, z) tuples from MediaPipe.

        Returns:
            dict with gesture and confidence, or None.
        """
        if not landmarks or len(landmarks) < 21:
            return None

        finger_states = self._compute_finger_states(landmarks)
        if finger_states is None:
            return None

        best_gesture = None
        best_confidence = 0.0

        for gesture_name, expected_state in config.CUSTOM_GESTURE_FINGER_STATES.items():
            matches = sum(1 for a, e in zip(finger_states, expected_state) if a == e)
            confidence = matches / 5.0

            if confidence > best_confidence:
                best_confidence = confidence
                best_gesture = gesture_name

        # Require at least 4/5 fingers matching (80%)
        if best_confidence >= 0.8 and best_gesture:
            word = config.CUSTOM_GESTURE_WORD_MAP.get(best_gesture)
            if word:
                return {
                    "gesture": best_gesture,
                    "word": word,
                    "confidence": best_confidence,
                }

        return None

    def _compute_finger_states(self, landmarks):
        """
        Compute finger extension states from landmark positions.
        Works with landmarks from either the Task API or Hands API.

        Args:
            landmarks: list of 21 (x, y, z) tuples.

        Returns:
            tuple of 5 ints: (thumb, index, middle, ring, pinky)
            1 = extended, 0 = closed.
        """
        if not landmarks or len(landmarks) < 21:
            return None

        states = []

        # Thumb: check if tip is farther from middle MCP than IP is
        thumb_tip = landmarks[config.THUMB_TIP]
        thumb_ip = landmarks[config.THUMB_IP]
        middle_mcp = landmarks[config.MIDDLE_MCP]

        thumb_tip_dist = math.sqrt(
            (thumb_tip[0] - middle_mcp[0]) ** 2 +
            (thumb_tip[1] - middle_mcp[1]) ** 2
        )
        thumb_ip_dist = math.sqrt(
            (thumb_ip[0] - middle_mcp[0]) ** 2 +
            (thumb_ip[1] - middle_mcp[1]) ** 2
        )
        states.append(1 if thumb_tip_dist > thumb_ip_dist else 0)

        # Other fingers: tip.y < pip.y means extended (image coords: y grows downward)
        finger_tips = [config.INDEX_TIP, config.MIDDLE_TIP, config.RING_TIP, config.PINKY_TIP]
        finger_pips = [config.INDEX_PIP, config.MIDDLE_PIP, config.RING_PIP, config.PINKY_PIP]

        for tip_idx, pip_idx in zip(finger_tips, finger_pips):
            tip_y = landmarks[tip_idx][1]
            pip_y = landmarks[pip_idx][1]
            states.append(1 if tip_y < pip_y else 0)

        return tuple(states)

    # ─── Smoothing ───────────────────────────────────────────

    def _get_smoothed_gesture(self):
        """
        Get the dominant gesture from the smoothing buffer.
        Returns the gesture name if it appears in >= threshold fraction
        of recent frames, otherwise None.
        """
        if len(self._gesture_buffer) == 0:
            return None

        gesture_counts = {}
        for g in self._gesture_buffer:
            if g is not None:
                gesture_counts[g] = gesture_counts.get(g, 0) + 1

        if not gesture_counts:
            return None

        dominant = max(gesture_counts, key=gesture_counts.get)
        ratio = gesture_counts[dominant] / len(self._gesture_buffer)

        if ratio >= self._smoothing_threshold:
            return dominant
        return None

    # ─── Rule-based classifier (fallback) ────────────────────

    def classify(self, landmarks, finger_states, normalized_landmarks=None):
        """
        Rule-based classification using finger extension states.
        Used as fallback when mp.solutions is not available.
        """
        if landmarks is None or finger_states is None:
            return {"gesture": None, "word": None, "confidence": 0.0}

        best_gesture = None
        best_confidence = 0.0

        for gesture_name, expected_state in config.GESTURE_FINGER_STATES.items():
            matches = sum(1 for a, e in zip(finger_states, expected_state) if a == e)
            confidence = matches / 5.0
            if confidence > best_confidence:
                best_confidence = confidence
                best_gesture = gesture_name

        if best_confidence >= 0.8:
            word = config.RULE_GESTURE_WORD_MAP.get(best_gesture, None)
            return {
                "gesture": best_gesture,
                "word": word,
                "confidence": best_confidence,
            }

        return {"gesture": None, "word": None, "confidence": best_confidence}

    def get_last_hand_landmarks(self):
        """Get hand landmarks from the last MediaPipe Task recognition."""
        if self._last_result and self._last_result.hand_landmarks:
            return self._last_result.hand_landmarks
        return None

    def get_all_gestures(self):
        """Return all available gesture->word mappings (pre-trained + custom)."""
        if self.classifier_type == "mediapipe_task":
            combined = {}
            combined.update(config.GESTURE_WORD_MAP)
            combined.update(config.CUSTOM_GESTURE_WORD_MAP)
            return combined
        return config.RULE_GESTURE_WORD_MAP.copy()
