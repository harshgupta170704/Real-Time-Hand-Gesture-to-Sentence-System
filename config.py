"""
Configuration for the Hand Gesture to Sentence System.
Optimized for Raspberry Pi 4 Model B + Camera v2.
Web-based output (view on laptop browser via network).
"""

# ─────────────────────────────────────────────
# Camera Settings
# ─────────────────────────────────────────────
# NOTE: Optimized for Raspberry Pi 4 Model B (1.5GHz, 2-8GB RAM)
# Pi 4 handles 640x480 comfortably at ~15-20 FPS.
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
# Set to "picamera" for Pi Camera v2, or "webcam" for USB/desktop testing
CAMERA_SOURCE = "picamera"

# ─────────────────────────────────────────────
# Web Server Settings
# ─────────────────────────────────────────────
WEB_HOST = "0.0.0.0"   # Accessible from any device on the network
WEB_PORT = 5000
JPEG_QUALITY = 75       # MJPEG stream quality (Pi 4 handles higher quality)

# ─────────────────────────────────────────────
# MediaPipe Settings
# ─────────────────────────────────────────────
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.6
MODEL_COMPLEXITY = 1  # 0 = lite, 1 = full (Pi 4 can handle full model)

# ─────────────────────────────────────────────
# Gesture Recognizer Model
# ─────────────────────────────────────────────
# "mediapipe_task" = pre-trained MediaPipe Gesture Recognizer (downloaded)
# "rule"           = rule-based using finger states (no download needed)
CLASSIFIER_TYPE = "mediapipe_task"
# Path to the downloaded .task model file
GESTURE_MODEL_PATH = "models/gesture_recognizer.task"
# Fallback to rule-based if model not found
FALLBACK_TO_RULE = True
# Confidence threshold for gesture recognition
GESTURE_CONFIDENCE_THRESHOLD = 0.5

# ─────────────────────────────────────────────
# Sentence Builder Settings
# ─────────────────────────────────────────────
DEBOUNCE_FRAMES = 15       # ~1 sec at 15 FPS
COOLDOWN_FRAMES = 10       # Cooldown after adding a word
MAX_SENTENCE_LENGTH = 20

# ─────────────────────────────────────────────
# Gesture → Word Mapping (MediaPipe pre-trained model gestures)
# ─────────────────────────────────────────────
# The pre-trained MediaPipe Gesture Recognizer recognizes these gestures:
#   None, Closed_Fist, Open_Palm, Pointing_Up,
#   Thumb_Down, Thumb_Up, Victory, ILoveYou
#
# We map each to a useful word for sentence building:
GESTURE_WORD_MAP = {
    "Pointing_Up":  "I",
    "Victory":      "want",
    "Open_Palm":    "hello",
    "Closed_Fist":  "no",
    "Thumb_Up":     "yes",
    "ILoveYou":     "love",
    "Thumb_Down":   "stop",
}

# ─────────────────────────────────────────────
# Rule-based fallback — finger state definitions
# ─────────────────────────────────────────────
# Format: (thumb, index, middle, ring, pinky) — 1=extended, 0=closed
GESTURE_FINGER_STATES = {
    "index_up":       (0, 1, 0, 0, 0),
    "peace":          (0, 1, 1, 0, 0),
    "open_palm":      (1, 1, 1, 1, 1),
    "fist":           (0, 0, 0, 0, 0),
    "thumbs_up":      (1, 0, 0, 0, 0),
    "rock":           (1, 1, 0, 0, 1),
    "call":           (1, 0, 0, 0, 1),
    "three_fingers":  (0, 1, 1, 1, 0),
    "four_fingers":   (0, 1, 1, 1, 1),
}

RULE_GESTURE_WORD_MAP = {
    "index_up":       "I",
    "peace":          "want",
    "open_palm":      "hello",
    "fist":           "no",
    "thumbs_up":      "yes",
    "rock":           "love",
    "call":           "help",
    "three_fingers":  "you",
    "four_fingers":   "please",
}

# ─────────────────────────────────────────────
# Display / UI Settings
# ─────────────────────────────────────────────
SHOW_LANDMARKS = True
SHOW_FPS = True

# Colors (BGR format for OpenCV overlay on stream)
COLOR_PRIMARY = (255, 191, 0)
COLOR_SECONDARY = (0, 255, 191)
COLOR_SUCCESS = (0, 255, 100)
COLOR_WARNING = (0, 165, 255)
COLOR_TEXT = (255, 255, 255)

# ─────────────────────────────────────────────
# MediaPipe Hand Landmark Indices
# ─────────────────────────────────────────────
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIPS = [THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
FINGER_MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

# ─────────────────────────────────────────────
# Model Download URL
# ─────────────────────────────────────────────
GESTURE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "gesture_recognizer/gesture_recognizer/float16/latest/"
    "gesture_recognizer.task"
)
