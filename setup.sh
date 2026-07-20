#!/bin/bash
# ═══════════════════════════════════════════════════════════
# Setup Script — Hand Gesture to Sentence System
# Raspberry Pi 4 Model B + Camera Module v2
# ═══════════════════════════════════════════════════════════
#
# PREREQUISITES:
#   - Raspberry Pi OS (64-bit) — required for MediaPipe
#   - Pi Camera v2 connected and enabled
#   - Internet connection for package downloads + model download
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# ═══════════════════════════════════════════════════════════

set -e

echo "╔══════════════════════════════════════════════════╗"
echo "║  🤟 Hand Gesture → Sentence — Setup              ║"
echo "║  Target: Raspberry Pi 4 Model B                  ║"
echo "╚══════════════════════════════════════════════════╝"

# ─── 1. Check architecture ──────────────────────────────
ARCH=$(uname -m)
echo ""
echo "[1/7] Checking architecture: $ARCH"
if [ "$ARCH" != "aarch64" ]; then
    echo "  ⚠  WARNING: Detected $ARCH architecture."
    echo "     MediaPipe requires 64-bit OS (aarch64)."
    echo "     Download 64-bit Pi OS: https://www.raspberrypi.com/software/"
    read -p "  Continue anyway? (y/n) " -n 1 -r
    echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
else
    echo "  ✓ 64-bit OS detected."
fi

# ─── 2. System packages ─────────────────────────────────
echo ""
echo "[2/7] Installing system dependencies..."
sudo apt update -qq
sudo apt install -y -qq \
    python3-pip \
    python3-venv \
    python3-opencv \
    python3-picamera2 \
    libatlas-base-dev \
    libhdf5-dev \
    libharfbuzz0b \
    liblapack-dev \
    libcblas-dev
echo "  ✓ System packages installed."

# ─── 3. Virtual environment ─────────────────────────────
echo ""
echo "[3/7] Setting up Python virtual environment..."
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv --system-site-packages "$VENV_DIR"
    echo "  ✓ Virtual environment created."
else
    echo "  ✓ Virtual environment already exists."
fi
source "$VENV_DIR/bin/activate"

# ─── 4. Python packages ─────────────────────────────────
echo ""
echo "[4/7] Installing Python packages..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "  ✓ Python packages installed."

# ─── 5. Create directories ──────────────────────────────
echo ""
echo "[5/7] Creating project directories..."
mkdir -p models
mkdir -p training_data
echo "  ✓ Directories ready."

# ─── 6. Download pre-trained model ──────────────────────
echo ""
echo "[6/7] Downloading MediaPipe Gesture Recognizer model..."
python3 download_model.py
echo "  ✓ Model ready."

# ─── 7. Test camera ─────────────────────────────────────
echo ""
echo "[7/7] Testing camera..."
python3 -c "
try:
    from picamera2 import Picamera2
    cam = Picamera2()
    cam.configure(cam.create_still_configuration())
    cam.start()
    import time; time.sleep(1)
    cam.stop()
    print('  ✓ Pi Camera v2 working!')
except Exception as e:
    print(f'  ⚠ Camera test failed: {e}')
    print('  Run: sudo raspi-config → Interface Options → Camera → Enable')
"

# ─── Done ────────────────────────────────────────────────
IP=$(hostname -I | awk '{print $1}')
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  ✓ SETUP COMPLETE!                               ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║                                                  ║"
echo "║  To start:                                       ║"
echo "║    source venv/bin/activate                      ║"
echo "║    python main.py                                ║"
echo "║                                                  ║"
echo "║  Then open on your laptop:                       ║"
echo "║    http://${IP}:5000                              ║"
echo "║                                                  ║"
echo "╚══════════════════════════════════════════════════╝"
