#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Raspberry Pi Optimized Launcher
# ─────────────────────────────────────────────────────────────
# This script launches the Hand Gesture to Sentence System
# with settings optimized for the Raspberry Pi's lower CPU power
# and the official Pi Camera Module.
# 
# Usage:
#   chmod +x run_pi.sh
#   ./run_pi.sh
# ─────────────────────────────────────────────────────────────

echo "=================================================="
echo "🍓 Starting Raspberry Pi Edition..."
echo "=================================================="

# Check if picamera library is installed
if python3 -c "import picamera" &> /dev/null; then
    echo "✓ PiCamera module found"
else
    echo "⚠️  PiCamera module not found! Installing..."
    pip3 install picamera
fi

echo "🚀 Launching system with PiCamera module..."

# We launch the main app but force it to use the picamera
# and disable TTS if it causes too much lag on the Pi
python3 main.py --camera picamera
