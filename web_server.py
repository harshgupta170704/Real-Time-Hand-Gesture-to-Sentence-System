"""
Web server for the Hand Gesture to Sentence System.
Serves a live camera feed (MJPEG) + real-time sentence data via SSE.
View from your browser at http://localhost:5000
"""

import threading
import time
import json
import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
import config


app = Flask(__name__, template_folder="templates", static_folder="static")

# ─── Shared state (thread-safe) ──────────────────────────
_lock = threading.Lock()
_latest_frame = None
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
    "raw_words": [],
}


def update_frame(frame_bgr):
    global _latest_frame
    with _lock:
        _latest_frame = frame_bgr.copy()


def update_sentence_data(data):
    global _sentence_data
    with _lock:
        _sentence_data.update(data)


def get_latest_frame():
    with _lock:
        return _latest_frame.copy() if _latest_frame is not None else None


def get_sentence_data():
    with _lock:
        return _sentence_data.copy()


# ─── Flask Routes ────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        _generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/state")
def api_state():
    return jsonify(get_sentence_data())


@app.route("/api/stream")
def api_stream():
    """Server-Sent Events stream for real-time state updates."""
    def generate():
        last_data = None
        while True:
            data = get_sentence_data()
            data_str = json.dumps(data, default=str)
            if data_str != last_data:
                yield f"data: {data_str}\n\n"
                last_data = data_str
            time.sleep(0.05)
    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/api/gestures")
def api_gestures():
    """Return all gesture->word mappings (pre-trained + custom)."""
    all_gestures = getattr(config, "ALL_GESTURE_DISPLAY", {})
    return jsonify(all_gestures)


@app.route("/api/clear", methods=["POST"])
def api_clear():
    global _clear_requested
    _clear_requested = True
    return jsonify({"status": "ok"})


@app.route("/api/undo", methods=["POST"])
def api_undo():
    global _undo_requested
    _undo_requested = True
    return jsonify({"status": "ok"})


@app.route("/api/speak", methods=["POST"])
def api_speak():
    global _speak_requested
    _speak_requested = True
    return jsonify({"status": "ok"})


# ─── Command flags ───────────────────────────────────────
_clear_requested = False
_undo_requested = False
_speak_requested = False


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


def check_speak_requested():
    global _speak_requested
    if _speak_requested:
        _speak_requested = False
        return True
    return False


# ─── MJPEG Generator ────────────────────────────────────

def _generate_mjpeg():
    while True:
        frame = get_latest_frame()
        if frame is not None:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY]
            ret, buffer = cv2.imencode(".jpg", frame, encode_params)
            if ret:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )
        time.sleep(0.05)


def start_server():
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
    print(f"[WebServer] Open in your browser:")
    print(f"[WebServer]   http://localhost:{config.WEB_PORT}")
    return server_thread
