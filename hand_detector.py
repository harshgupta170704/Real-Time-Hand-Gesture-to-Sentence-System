"""
Hand detection module using MediaPipe Hands.
Extracts and normalizes 21 hand landmarks from camera frames.
"""

import cv2
import numpy as np
import mediapipe as mp
import config


class HandDetector:
    """Detects hands and extracts normalized landmarks using MediaPipe."""

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
            model_complexity=config.MODEL_COMPLEXITY,
        )
        print("[HandDetector] Initialized.")

    def detect(self, frame_rgb):
        """
        Detect hands in an RGB frame.

        Args:
            frame_rgb: numpy array (H, W, 3) in RGB format.

        Returns:
            dict with:
                - "landmarks": list of (x, y, z) tuples for 21 landmarks,
                               or None if no hand detected.
                - "normalized": list of 63 floats (normalized relative to wrist),
                                or None if no hand detected.
                - "handedness": "Left" or "Right", or None.
                - "raw_results": raw MediaPipe results for drawing.
        """
        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return {
                "landmarks": None,
                "normalized": None,
                "handedness": None,
                "raw_results": results,
            }

        # Take the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]

        # Extract raw landmarks as (x, y, z) tuples
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append((lm.x, lm.y, lm.z))

        # Normalize relative to wrist (landmark 0)
        normalized = self._normalize_landmarks(landmarks)

        # Get handedness
        handedness = None
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label

        return {
            "landmarks": landmarks,
            "normalized": normalized,
            "handedness": handedness,
            "raw_results": results,
        }

    def _normalize_landmarks(self, landmarks):
        """
        Normalize landmarks relative to wrist position and hand scale.
        This makes the features invariant to hand position and size in frame.

        Args:
            landmarks: list of 21 (x, y, z) tuples.

        Returns:
            list of 63 floats (flattened normalized landmarks).
        """
        wrist = np.array(landmarks[config.WRIST])

        # Shift all landmarks so wrist is at origin
        shifted = [np.array(lm) - wrist for lm in landmarks]

        # Scale by the max distance from wrist (to normalize hand size)
        distances = [np.linalg.norm(s) for s in shifted]
        max_dist = max(distances) if max(distances) > 0 else 1.0

        normalized = [s / max_dist for s in shifted]

        # Flatten to 63 values
        return [coord for point in normalized for coord in point]

    def draw_landmarks(self, frame_bgr, raw_results):
        """
        Draw hand landmarks and connections on a BGR frame.

        Args:
            frame_bgr: numpy array (H, W, 3) in BGR format.
            raw_results: raw MediaPipe results from detect().

        Returns:
            frame_bgr with landmarks drawn.
        """
        if raw_results.multi_hand_landmarks:
            for hand_landmarks in raw_results.multi_hand_landmarks:
                # Draw connections
                self.mp_drawing.draw_landmarks(
                    frame_bgr,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )
        return frame_bgr

    def get_finger_states(self, landmarks):
        """
        Determine which fingers are extended (up) or closed (down).

        Args:
            landmarks: list of 21 (x, y, z) tuples.

        Returns:
            tuple of 5 ints: (thumb, index, middle, ring, pinky)
            1 = extended, 0 = closed.
        """
        if landmarks is None:
            return None

        states = []

        # Thumb: compare x-position (thumb extends sideways)
        # For right hand: tip.x > ip.x means extended
        # For left hand: tip.x < ip.x means extended
        # We use a heuristic: compare thumb tip to thumb IP joint
        thumb_tip = landmarks[config.THUMB_TIP]
        thumb_ip = landmarks[config.THUMB_IP]
        thumb_mcp = landmarks[config.THUMB_MCP]

        # Determine hand orientation from wrist and middle MCP
        wrist = landmarks[config.WRIST]
        middle_mcp = landmarks[config.MIDDLE_MCP]

        # Check if thumb is extended using distance from palm center
        # Thumb tip should be farther from middle MCP than thumb IP is
        thumb_tip_dist = np.linalg.norm(
            np.array(thumb_tip[:2]) - np.array(middle_mcp[:2])
        )
        thumb_ip_dist = np.linalg.norm(
            np.array(thumb_ip[:2]) - np.array(middle_mcp[:2])
        )
        states.append(1 if thumb_tip_dist > thumb_ip_dist else 0)

        # For other fingers: tip.y < pip.y means extended (in image coords, y increases downward)
        for tip_idx, pip_idx in zip(config.FINGER_TIPS[1:], config.FINGER_PIPS[1:]):
            tip_y = landmarks[tip_idx][1]
            pip_y = landmarks[pip_idx][1]
            states.append(1 if tip_y < pip_y else 0)

        return tuple(states)

    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()
        print("[HandDetector] Closed.")
