"""
Sentence builder module with Grammar Engine and Text-to-Speech.
Buffers recognized gesture words with debouncing, applies intelligent
grammar rules to auto-frame proper sentences, and speaks them aloud.
"""

import time
import threading
import config


class GrammarEngine:
    """
    Applies grammar rules to transform a list of gesture words
    into a grammatically correct sentence.

    Rules applied:
    1. Auto-capitalize first word and "I"
    2. Insert "to" between verb pairs (e.g., "want go" -> "want to go")
    3. Insert missing words between incompatible pairs
       (e.g., "I food" -> "I need food")
    4. Handle "don't" + verb correctly
    5. "thank you" auto-correction
    6. Add period at end
    """

    def __init__(self):
        self.word_categories = getattr(config, "WORD_CATEGORIES", {})
        self.verb_pairs_need_to = getattr(config, "VERB_PAIRS_NEED_TO", [])
        self.auto_insert_rules = getattr(config, "GRAMMAR_AUTO_INSERT", {})

    def get_category(self, word):
        """Get the grammatical category of a word."""
        return self.word_categories.get(word, "unknown")

    def build_sentence(self, words):
        """
        Take a raw list of gesture words and produce a grammatically
        improved sentence string.

        Args:
            words: list of word strings from gesture recognition.

        Returns:
            str: Grammatically improved sentence.
        """
        if not words:
            return ""

        # Work on a copy
        processed = list(words)

        # Step 1: Apply auto-insert rules (insert missing linking words)
        processed = self._apply_auto_inserts(processed)

        # Step 2: Insert "to" between verb pairs
        processed = self._insert_to_between_verbs(processed)

        # Step 3: Fix "thank" -> "thank you" if followed by nothing or non-pronoun
        processed = self._fix_thank_you(processed)

        # Step 4: Fix "don't" placement
        processed = self._fix_negation(processed)

        # Step 5: Join and capitalize
        sentence = " ".join(processed)

        # Capitalize first letter
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]

        # Always capitalize standalone "I"
        sentence = self._capitalize_i(sentence)

        return sentence

    def finalize_sentence(self, words):
        """Build sentence and add ending punctuation."""
        sentence = self.build_sentence(words)
        if sentence and not sentence.endswith((".", "!", "?")):
            # Questions: starts with helping words
            if any(sentence.lower().startswith(w) for w in
                   ["do ", "can ", "will ", "is ", "are "]):
                sentence += "?"
            else:
                sentence += "."
        return sentence

    def _apply_auto_inserts(self, words):
        """Insert linking words between incompatible adjacent categories."""
        if len(words) < 2:
            return words

        result = [words[0]]
        for i in range(1, len(words)):
            prev_word = words[i - 1]
            curr_word = words[i]
            prev_cat = self.get_category(prev_word)
            curr_cat = self.get_category(curr_word)

            key = (prev_cat, curr_cat)
            if key in self.auto_insert_rules:
                insert_word = self.auto_insert_rules[key]
                if insert_word and insert_word != prev_word and insert_word != curr_word:
                    result.append(insert_word)

            result.append(curr_word)

        return result

    def _insert_to_between_verbs(self, words):
        """Insert 'to' between verb pairs like 'want go' -> 'want to go'."""
        if len(words) < 2:
            return words

        result = [words[0]]
        for i in range(1, len(words)):
            prev = words[i - 1].lower()
            curr = words[i].lower()

            # Check if this pair needs "to"
            needs_to = False
            for v1, v2 in self.verb_pairs_need_to:
                if prev == v1 and curr == v2:
                    needs_to = True
                    break

            if needs_to:
                result.append("to")

            result.append(words[i])

        return result

    def _fix_thank_you(self, words):
        """Auto-complete 'thank' to 'thank you' when appropriate."""
        result = []
        for i, word in enumerate(words):
            result.append(word)
            if word.lower() == "thank":
                # If next word isn't "you", insert it
                next_word = words[i + 1] if i + 1 < len(words) else None
                if next_word is None or next_word.lower() != "you":
                    result.append("you")
        return result

    def _fix_negation(self, words):
        """Ensure 'don't' is followed by a verb, not a noun directly."""
        if len(words) < 2:
            return words

        result = [words[0]]
        for i in range(1, len(words)):
            prev = words[i - 1].lower()
            curr = words[i]
            curr_cat = self.get_category(curr)

            # "don't" + noun -> "don't want" + noun
            if prev == "don't" and curr_cat == "noun":
                result.append("want")

            result.append(curr)

        return result

    def _capitalize_i(self, sentence):
        """Capitalize standalone 'I' in a sentence."""
        words = sentence.split(" ")
        result = []
        for w in words:
            if w.lower() == "i" and len(w) == 1:
                result.append("I")
            else:
                result.append(w)
        return " ".join(result)


class SentenceBuilder:
    """
    Buffers gesture words and builds sentences with debouncing,
    cooldown, grammar correction, and text-to-speech.
    """

    def __init__(self):
        self.sentence_words = []
        self.current_gesture = None
        self.gesture_count = 0
        self.last_added_word = None
        self.cooldown_counter = 0
        self.last_add_time = 0
        self.history = []

        # Grammar engine
        self.grammar = GrammarEngine()

        # Text-to-Speech engine
        self._tts_engine = None
        self._tts_lock = threading.Lock()
        if getattr(config, "TTS_ENABLED", False):
            self._init_tts()

        print("[SentenceBuilder] Initialized with Grammar Engine.")

    def _init_tts(self):
        """Initialize the text-to-speech engine in a dedicated worker thread."""
        try:
            import pyttsx3
            import queue
            self._tts_queue = queue.Queue()
            
            def _tts_worker():
                try:
                    # Initialize COM object in this thread
                    engine = pyttsx3.init()
                    engine.setProperty("rate", getattr(config, "TTS_RATE", 150))
                    engine.setProperty("volume", getattr(config, "TTS_VOLUME", 0.9))
                    voices = engine.getProperty("voices")
                    if voices and len(voices) > 1:
                        for v in voices:
                            if "english" in v.name.lower() or "en" in str(v.id).lower():
                                engine.setProperty("voice", v.id)
                                break
                    
                    while True:
                        text = self._tts_queue.get()
                        if text is None: # poison pill
                            break
                        try:
                            engine.say(text)
                            engine.runAndWait()
                        except Exception as e:
                            print(f"[SentenceBuilder] TTS error during speech: {e}")
                except Exception as e:
                    print(f"[SentenceBuilder] TTS worker thread error: {e}")

            self._tts_thread = threading.Thread(target=_tts_worker, daemon=True)
            self._tts_thread.start()
            print("[SentenceBuilder] Text-to-Speech enabled (worker thread started).")
        except ImportError:
            print("[SentenceBuilder] pyttsx3 not installed -- TTS disabled.")
            print("[SentenceBuilder] Install with: pip install pyttsx3")
            self._tts_engine = None
        except Exception as e:
            print(f"[SentenceBuilder] TTS initialization error: {e}")
            self._tts_engine = None

    def speak(self, text=None):
        """Speak the given text or the current sentence by sending it to the TTS worker thread."""
        if not hasattr(self, '_tts_queue'):
            return

        sentence = text or self.get_sentence_final()
        if not sentence or sentence == ".":
            return

        self._tts_queue.put(sentence)

    def update(self, gesture_result):
        """
        Process a new gesture detection result. Call this every frame.

        Returns:
            dict with word_added, current_sentence, building_gesture,
            building_progress, status.
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

        # No gesture detected
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

        # Same gesture as previous frame
        if gesture == self.current_gesture:
            self.gesture_count += 1
        else:
            self.current_gesture = gesture
            self.gesture_count = 1

        progress = min(self.gesture_count / config.DEBOUNCE_FRAMES, 1.0)

        # Check if gesture held long enough
        if self.gesture_count >= config.DEBOUNCE_FRAMES:
            if word != self.last_added_word:
                return self._add_word(word, gesture)
            else:
                return {
                    "word_added": None,
                    "current_sentence": self.get_sentence(),
                    "building_gesture": gesture,
                    "building_progress": 1.0,
                    "status": "duplicate_blocked",
                }

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
        """Manually add a word."""
        if len(self.sentence_words) < config.MAX_SENTENCE_LENGTH:
            self.sentence_words.append(word)
            self.last_added_word = word

    def get_sentence(self):
        """
        Get the current sentence with grammar correction applied.
        The Grammar Engine transforms raw words into a proper sentence.
        """
        return self.grammar.build_sentence(self.sentence_words)

    def get_sentence_final(self):
        """Get the final sentence with punctuation."""
        return self.grammar.finalize_sentence(self.sentence_words)

    def get_raw_words(self):
        """Get the raw word list without grammar processing."""
        return list(self.sentence_words)

    def clear(self):
        """Clear the current sentence."""
        if self.sentence_words:
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
        return len(self.sentence_words)

    def get_history(self):
        return self.history.copy()

    def reset_duplicate_block(self):
        """Allow the same word to be added again after hand removal."""
        self.last_added_word = None
