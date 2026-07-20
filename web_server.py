"""
Web server for the Hand Gesture to Sentence System.
Serves a live camera feed (MJPEG) + real-time sentence data.
View from your laptop browser at http://<pi-ip>:5000
"""

import threading
import time
import json
import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template
import config


app = Flask(__name__, template_folder="templates", static_folder="static")

# ─── Shared state (thread-safe) ──────────────────────────
_lock = threading.Lock()
_latest_frame = None          # BGR numpy array (with landmarks drawn)
_sentence_data = {
    "current_sentence": "",
    "gesture": None,
    "gesture_word": None,
    "confidence": 0.0,
    "progress": 0.0,
    "status": "idle",
    "word_count": 0,
    "last_word_added": None,
    "history": [],
    "fps": 0.0,
}


def update_frame(frame_bgr):
    """Called by the processing thread to push a new frame."""
    global _latest_frame
    with _lock:
        _latest_frame = frame_bgr.copy()


def update_sentence_data(data):
    """Called by the processing thread to push sentence state."""
    global _sentence_data
    with _lock:
        _sentence_data.update(data)


def get_latest_frame():
    """Get the most recent processed frame."""
    with _lock:
        return _latest_frame.copy() if _latest_frame is not None else None


def get_sentence_data():
    """Get the current sentence state."""
    with _lock:
        return _sentence_data.copy()


# ─── Flask Routes ────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main web page."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG video stream endpoint."""
    return Response(
        _generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/state")
def api_state():
    """JSON API for current gesture + sentence state."""
    return jsonify(get_sentence_data())


@app.route("/api/gestures")
def api_gestures():
    """JSON API for available gesture→word mappings."""
    return jsonify(config.GESTURE_WORD_MAP)


@app.route("/api/clear", methods=["POST"])
def api_clear():
    """Clear the current sentence (called from web UI)."""
    # This sets a flag that main.py's processing loop will pick up
    global _clear_requested
    _clear_requested = True
    return jsonify({"status": "ok"})


@app.route("/api/undo", methods=["POST"])
def api_undo():
    """Undo last word (called from web UI)."""
    global _undo_requested
    _undo_requested = True
    return jsonify({"status": "ok"})


# ─── Command flags (set by web UI, consumed by main loop) ─
_clear_requested = False
_undo_requested = False


def check_clear_requested():
    global _clear_requested
    if _clear_requested:
        _clear_requested = False
        return True
    return False


def check_undo_requested():
    global _undo_requested
    if _undo_requested:
        _undo_requested = False
        return True
    return False


# ─── MJPEG Generator ────────────────────────────────────

def _generate_mjpeg():
    """Generate MJPEG stream from latest processed frames."""
    while True:
        frame = get_latest_frame()
        if frame is not None:
            # Encode as JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY]
            ret, buffer = cv2.imencode(".jpg", frame, encode_params)
            if ret:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )
        # Small sleep to avoid spinning (match ~15 fps max)
        time.sleep(0.05)


def start_server():
    """Start the Flask web server in a background thread."""
    server_thread = threading.Thread(
        target=lambda: app.run(
            host=config.WEB_HOST,
            port=config.WEB_PORT,
            debug=False,
            threaded=True,
            use_reloader=False,
        ),
        daemon=True,
    )
    server_thread.start()
    print(f"[WebServer] Started at http://0.0.0.0:{config.WEB_PORT}")
    print(f"[WebServer] Open in your laptop browser:")
    print(f"[WebServer]   http://<your-pi-ip>:{config.WEB_PORT}")
    return server_thread
