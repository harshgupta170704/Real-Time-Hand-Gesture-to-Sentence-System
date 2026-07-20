"""
Sentence builder module.
Buffers recognized gesture words with debouncing and forms meaningful sentences.
"""

import time
import config


class SentenceBuilder:
    """
    Buffers gesture words and builds sentences with debouncing,
    cooldown, and basic grammar correction.
    """

    def __init__(self):
        self.sentence_words = []        # Current sentence word list
        self.current_gesture = None     # Currently detected gesture
        self.gesture_count = 0          # Consecutive frame count for current gesture
        self.last_added_word = None     # Last word added (prevent duplicates)
        self.cooldown_counter = 0       # Cooldown frames after adding a word
        self.last_add_time = 0          # Timestamp of last word addition
        self.history = []               # Past sentences

        print("[SentenceBuilder] Initialized.")

    def update(self, gesture_result):
        """
        Process a new gesture detection result. Call this every frame.

        Args:
            gesture_result: dict with "gesture", "word", "confidence" from classifier.

        Returns:
            dict with:
                - "word_added": the word just added, or None.
                - "current_sentence": the full sentence string so far.
                - "building_gesture": gesture being held (not yet confirmed).
                - "building_progress": float 0-1 progress toward confirmation.
                - "status": "idle", "building", "added", "cooldown".
        """
        gesture = gesture_result.get("gesture")
        word = gesture_result.get("word")

        # Handle cooldown period
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return {
                "word_added": None,
                "current_sentence": self.get_sentence(),
                "building_gesture": None,
                "building_progress": 0.0,
                "status": "cooldown",
            }

        # No gesture detected — reset counter
        if gesture is None or word is None:
            self.current_gesture = None
            self.gesture_count = 0
            return {
                "word_added": None,
                "current_sentence": self.get_sentence(),
                "building_gesture": None,
                "building_progress": 0.0,
                "status": "idle",
            }

        # Same gesture as previous frame — increment counter
        if gesture == self.current_gesture:
            self.gesture_count += 1
        else:
            # New gesture — reset counter
            self.current_gesture = gesture
            self.gesture_count = 1

        progress = min(self.gesture_count / config.DEBOUNCE_FRAMES, 1.0)

        # Check if gesture has been held long enough to register
        if self.gesture_count >= config.DEBOUNCE_FRAMES:
            # Don't add the same word consecutively (unless reset between)
            if word != self.last_added_word:
                return self._add_word(word, gesture)
            else:
                # Same word again — user needs to break gesture first
                return {
                    "word_added": None,
                    "current_sentence": self.get_sentence(),
                    "building_gesture": gesture,
                    "building_progress": 1.0,
                    "status": "duplicate_blocked",
                }

        # Still building up to confirmation
        return {
            "word_added": None,
            "current_sentence": self.get_sentence(),
            "building_gesture": gesture,
            "building_progress": progress,
            "status": "building",
        }

    def _add_word(self, word, gesture):
        """Add a confirmed word to the sentence."""
        if len(self.sentence_words) >= config.MAX_SENTENCE_LENGTH:
            return {
                "word_added": None,
                "current_sentence": self.get_sentence(),
                "building_gesture": gesture,
                "building_progress": 1.0,
                "status": "sentence_full",
            }

        self.sentence_words.append(word)
        self.last_added_word = word
        self.cooldown_counter = config.COOLDOWN_FRAMES
        self.gesture_count = 0
        self.current_gesture = None
        self.last_add_time = time.time()

        return {
            "word_added": word,
            "current_sentence": self.get_sentence(),
            "building_gesture": None,
            "building_progress": 0.0,
            "status": "added",
        }

    def force_add_word(self, word):
        """Manually add a word (for testing or special commands)."""
        if len(self.sentence_words) < config.MAX_SENTENCE_LENGTH:
            self.sentence_words.append(word)
            self.last_added_word = word

    def get_sentence(self):
        """
        Get the current sentence with basic grammar formatting.

        Returns:
            str: Formatted sentence.
        """
        if not self.sentence_words:
            return ""

        sentence = " ".join(self.sentence_words)

        # Capitalize first letter
        sentence = sentence[0].upper() + sentence[1:]

        return sentence

    def get_sentence_final(self):
        """
        Get the final sentence with proper punctuation.

        Returns:
            str: Final formatted sentence.
        """
        sentence = self.get_sentence()
        if sentence and not sentence.endswith((".", "!", "?")):
            sentence += "."
        return sentence

    def clear(self):
        """Clear the current sentence."""
        if self.sentence_words:
            # Save to history before clearing
            final = self.get_sentence_final()
            if final.strip("."):
                self.history.append(final)
        self.sentence_words = []
        self.current_gesture = None
        self.gesture_count = 0
        self.last_added_word = None
        self.cooldown_counter = 0
        print("[SentenceBuilder] Sentence cleared.")

    def undo_last_word(self):
        """Remove the last word from the sentence."""
        if self.sentence_words:
            removed = self.sentence_words.pop()
            self.last_added_word = self.sentence_words[-1] if self.sentence_words else None
            print(f"[SentenceBuilder] Undone: '{removed}'")

    def get_word_count(self):
        """Return the number of words in the current sentence."""
        return len(self.sentence_words)

    def get_history(self):
        """Return past completed sentences."""
        return self.history.copy()

    def reset_duplicate_block(self):
        """
        Reset the duplicate blocker, allowing the same word to be added again.
        Called when the user briefly removes their hand or shows a different gesture.
        """
        self.last_added_word = None
