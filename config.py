"""
Configuration for the Hand Gesture to Sentence System.
Optimized for Laptop Webcam -- real-time best-in-class edition.
Expanded gesture vocabulary with intelligent sentence framing.
"""

# ─────────────────────────────────────────────
# Camera Settings
# ─────────────────────────────────────────────
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
CAMERA_SOURCE = "webcam"
CAMERA_MIRROR = True

# ─────────────────────────────────────────────
# Web Server Settings
# ─────────────────────────────────────────────
WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
JPEG_QUALITY = 85

# ─────────────────────────────────────────────
# MediaPipe Settings
# ─────────────────────────────────────────────
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.5
MODEL_COMPLEXITY = 1

# ─────────────────────────────────────────────
# Gesture Recognizer Model
# ─────────────────────────────────────────────
CLASSIFIER_TYPE = "mediapipe_task"
GESTURE_MODEL_PATH = "models/gesture_recognizer.task"
FALLBACK_TO_RULE = True
GESTURE_CONFIDENCE_THRESHOLD = 0.55

# ─────────────────────────────────────────────
# Gesture Smoothing
# ─────────────────────────────────────────────
SMOOTHING_WINDOW = 5
SMOOTHING_THRESHOLD = 0.6

# ─────────────────────────────────────────────
# Sentence Builder Settings
# ─────────────────────────────────────────────
DEBOUNCE_FRAMES = 20
COOLDOWN_FRAMES = 8
MAX_SENTENCE_LENGTH = 20

# ─────────────────────────────────────────────
# Text-to-Speech Settings
# ─────────────────────────────────────────────
TTS_ENABLED = True
TTS_RATE = 150
TTS_VOLUME = 0.9

# ─────────────────────────────────────────────
# PRE-TRAINED MODEL gestures (7 from MediaPipe)
# ─────────────────────────────────────────────
# These are recognized by the MediaPipe Gesture Recognizer model:
#   Closed_Fist, Open_Palm, Pointing_Up,
#   Thumb_Down, Thumb_Up, Victory, ILoveYou
GESTURE_WORD_MAP = {
    "Pointing_Up":  "I",
    "Victory":      "want",
    "Open_Palm":    "hello",
    "Closed_Fist":  "no",
    "Thumb_Up":     "yes",
    "ILoveYou":     "love",
    "Thumb_Down":   "don't",
}

# ─────────────────────────────────────────────
# CUSTOM GESTURES (detected from landmarks)
# ─────────────────────────────────────────────
# Format: (thumb, index, middle, ring, pinky) — 1=extended, 0=closed
# These are checked when the pre-trained model has low confidence
# or returns "None", but landmarks are available.
CUSTOM_GESTURE_FINGER_STATES = {
    "three_fingers":  (0, 1, 1, 1, 0),   # Index+Middle+Ring up
    "four_fingers":   (0, 1, 1, 1, 1),   # All except thumb
    "pinky_up":       (0, 0, 0, 0, 1),   # Pinky only
    "hang_loose":     (1, 0, 0, 0, 1),   # Thumb+Pinky (shaka)
    "gun_shape":      (1, 1, 0, 0, 0),   # Thumb+Index (L-shape)
    "middle_up":      (0, 0, 1, 0, 0),   # Middle finger only
    "ring_pinky":     (0, 0, 0, 1, 1),   # Ring+Pinky up
    "index_pinky":    (0, 1, 0, 0, 1),   # Index+Pinky (spider)
}

CUSTOM_GESTURE_WORD_MAP = {
    "three_fingers":  "you",
    "four_fingers":   "please",
    "pinky_up":       "help",
    "hang_loose":     "thank",
    "gun_shape":      "need",
    "middle_up":      "go",
    "ring_pinky":     "food",
    "index_pinky":    "water",
}

# ─────────────────────────────────────────────
# COMBINED gesture info for display
# ─────────────────────────────────────────────
ALL_GESTURE_DISPLAY = {
    # Pre-trained model gestures
    "Pointing_Up":    {"word": "I",      "emoji": "pointing_up",   "source": "model"},
    "Victory":        {"word": "want",   "emoji": "victory",       "source": "model"},
    "Open_Palm":      {"word": "hello",  "emoji": "open_palm",     "source": "model"},
    "Closed_Fist":    {"word": "no",     "emoji": "closed_fist",   "source": "model"},
    "Thumb_Up":       {"word": "yes",    "emoji": "thumbs_up",     "source": "model"},
    "ILoveYou":       {"word": "love",   "emoji": "i_love_you",    "source": "model"},
    "Thumb_Down":     {"word": "don't",  "emoji": "thumbs_down",   "source": "model"},
    # Custom landmark-based gestures
    "three_fingers":  {"word": "you",    "emoji": "three_fingers",  "source": "custom"},
    "four_fingers":   {"word": "please", "emoji": "four_fingers",   "source": "custom"},
    "pinky_up":       {"word": "help",   "emoji": "pinky_up",       "source": "custom"},
    "hang_loose":     {"word": "thank",  "emoji": "hang_loose",     "source": "custom"},
    "gun_shape":      {"word": "need",   "emoji": "gun_shape",      "source": "custom"},
    "middle_up":      {"word": "go",     "emoji": "middle_up",      "source": "custom"},
    "ring_pinky":     {"word": "food",   "emoji": "ring_pinky",     "source": "custom"},
    "index_pinky":    {"word": "water",  "emoji": "index_pinky",    "source": "custom"},
}

# ─────────────────────────────────────────────
# GRAMMAR ENGINE rules
# ─────────────────────────────────────────────
# Word categories for grammar rules
WORD_CATEGORIES = {
    # Pronouns
    "I": "pronoun",
    "you": "pronoun",
    # Verbs
    "want": "verb",
    "need": "verb",
    "love": "verb",
    "help": "verb",
    "go": "verb",
    "thank": "verb",
    # Adjectives / Adverbs
    "good": "adjective",
    # Nouns
    "food": "noun",
    "water": "noun",
    # Interjections / Standalone
    "hello": "interjection",
    "yes": "interjection",
    "no": "interjection",
    "please": "adverb",
    "don't": "negation",
}

# Auto-insert rules: (before_category, after_category) -> word to insert
# e.g., "I food" -> "I need food" (pronoun before noun inserts "need")
GRAMMAR_AUTO_INSERT = {
    ("pronoun", "noun"): "need",
    ("verb", "pronoun"): None,     # "want you" is fine
    ("negation", "noun"): "need",  # "don't food" -> "don't need food"
}

# Word pairs that should auto-insert "to" between them
VERB_PAIRS_NEED_TO = [
    ("want", "go"),
    ("want", "help"),
    ("need", "go"),
    ("need", "help"),
]

# ─────────────────────────────────────────────
# Rule-based fallback — legacy finger state definitions
# ─────────────────────────────────────────────
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
    "call":           "thank",
    "three_fingers":  "you",
    "four_fingers":   "please",
}

# ─────────────────────────────────────────────
# Display / UI Settings
# ─────────────────────────────────────────────
SHOW_LANDMARKS = True
SHOW_FPS = True

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
