"""
Gesture classifier module.
Supports:
  1. MediaPipe Gesture Recognizer (pre-trained .task model from Google)
  2. Rule-based fallback using finger extension states
"""

import os
import numpy as np
import config


class GestureClassifier:
    """Classifies hand gestures into word labels."""

    def __init__(self, classifier_type=None):
        self.classifier_type = classifier_type or config.CLASSIFIER_TYPE
        self.recognizer = None
        self._last_result = None

        if self.classifier_type == "mediapipe_task":
            self._init_mediapipe_task()

        print(f"[GestureClassifier] Ready ({self.classifier_type} mode).")

    # ─── MediaPipe Gesture Recognizer (pre-trained model) ────

    def _init_mediapipe_task(self):
        """Initialize the MediaPipe Gesture Recognizer task."""
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

            # Configure the recognizer
            base_options = mp_python.BaseOptions(
                model_asset_path=model_path
            )
            options = vision.GestureRecognizerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=config.MAX_NUM_HANDS,
                min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
            )
            self.recognizer = vision.GestureRecognizer.create_from_options(options)
            print("[GestureClassifier] MediaPipe Gesture Recognizer loaded.")

        except Exception as e:
            print(f"[GestureClassifier] Task API error: {e}")
            if config.FALLBACK_TO_RULE:
                print("[GestureClassifier] Falling back to rule-based classifier.")
                self.classifier_type = "rule"
            else:
                raise

    def classify_frame(self, frame_rgb):
        """
        Classify gesture directly from an RGB frame using MediaPipe Task API.
        Only used when classifier_type == "mediapipe_task".

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
            result = self.recognizer.recognize(mp_image)
            self._last_result = result

            if result.gestures and len(result.gestures) > 0:
                gesture = result.gestures[0][0]
                gesture_name = gesture.category_name
                confidence = gesture.score

                # Extract landmarks if available
                landmarks = None
                if result.hand_landmarks and len(result.hand_landmarks) > 0:
                    landmarks = [
                        (lm.x, lm.y, lm.z)
                        for lm in result.hand_landmarks[0]
                    ]

                if confidence >= config.GESTURE_CONFIDENCE_THRESHOLD:
                    word = config.GESTURE_WORD_MAP.get(gesture_name, None)
                    return {
                        "gesture": gesture_name,
                        "word": word,
                        "confidence": float(confidence),
                        "landmarks": landmarks,
                    }

            return {"gesture": None, "word": None, "confidence": 0.0, "landmarks": None}

        except Exception as e:
            print(f"[GestureClassifier] Recognition error: {e}")
            return {"gesture": None, "word": None, "confidence": 0.0, "landmarks": None}

    # ─── Rule-based classifier (fallback) ────────────────────

    def classify(self, landmarks, finger_states, normalized_landmarks=None):
        """
        Rule-based classification using finger extension states.

        Args:
            landmarks: list of 21 (x, y, z) tuples.
            finger_states: tuple of 5 ints (thumb, index, middle, ring, pinky).
            normalized_landmarks: unused in rule mode.

        Returns:
            dict with gesture, word, confidence.
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
        """Return all available gesture→word mappings."""
        if self.classifier_type == "mediapipe_task":
            return config.GESTURE_WORD_MAP.copy()
        return config.RULE_GESTURE_WORD_MAP.copy()
