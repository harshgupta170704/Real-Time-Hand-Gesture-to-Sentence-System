"""
Download the pre-trained MediaPipe Gesture Recognizer model.
This model is provided by Google and recognizes 7 hand gestures.

Run:  python download_model.py
"""

import os
import sys
import urllib.request
import config


def download_model():
    """Download the gesture recognizer .task model file."""
    url = config.GESTURE_MODEL_URL
    dest = config.GESTURE_MODEL_PATH

    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"Model already exists: {dest} ({size_mb:.1f} MB)")
        response = input("Re-download? (y/n): ").strip().lower()
        if response != 'y':
            print("Skipped.")
            return

    print(f"Downloading MediaPipe Gesture Recognizer model...")
    print(f"  URL:  {url}")
    print(f"  Dest: {dest}")
    print()

    try:
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(downloaded / total_size * 100, 100)
                mb_done = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                bar_len = 40
                filled = int(bar_len * percent / 100)
                bar = '█' * filled + '░' * (bar_len - filled)
                sys.stdout.write(
                    f"\r  [{bar}] {percent:5.1f}% ({mb_done:.1f}/{mb_total:.1f} MB)"
                )
                sys.stdout.flush()

        urllib.request.urlretrieve(url, dest, reporthook=progress_hook)
        print()

        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"\n✓ Download complete! ({size_mb:.1f} MB)")
        print(f"  Saved to: {dest}")
        print(f"\n  The model recognizes these gestures:")
        for gesture, word in config.GESTURE_WORD_MAP.items():
            print(f"    {gesture:15s} → \"{word}\"")

    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("\n  Manual download:")
        print(f"    1. Visit: {url}")
        print(f"    2. Save to: {os.path.abspath(dest)}")
        sys.exit(1)


if __name__ == "__main__":
    download_model()
