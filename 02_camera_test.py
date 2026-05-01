"""
02_camera_test.py
Tests RPi camera at 96x96 resolution matching model input.
Terminal: (rpi-cv-env) rpi10@raspberrypi:~/Downloads $ python 02_camera_test.py
"""

import subprocess, os, sys
import numpy as np

CAPTURE_PATH = os.path.expanduser("~/Downloads/camera_test_output.jpg")
# Capture at higher res, model preprocessing will resize to 96
CAPTURE_W    = 640
CAPTURE_H    = 640
MODEL_SIZE   = 96

print("\n" + "="*55)
print("  RPi Camera Test  (model input: 96x96)")
print("="*55)

# ── Test 1: Detect camera ─────────────────────────────────
print("\nTest 1: Detecting camera...")
r = subprocess.run(
    ["rpicam-hello", "--list-cameras"],
    capture_output=True, text=True
)
out = r.stdout + r.stderr
print(out)

if "No cameras" in out or r.returncode != 0:
    print("ERROR: No camera detected.")
    print("  → Check ribbon cable connection (both ends)")
    print("  → Run: sudo raspi-config → Interface Options → Camera → Enable")
    print("  → sudo reboot")
    sys.exit(1)
print("  ✓ Camera detected")

# ── Test 2: Capture at 640x640 (native res) ───────────────
# We always capture at higher res for quality,
# then resize to 96x96 only for model input.
print(f"\nTest 2: Capturing {CAPTURE_W}x{CAPTURE_H} image...")
r = subprocess.run([
    "rpicam-still",
    "-o",       CAPTURE_PATH,
    "--width",  str(CAPTURE_W),
    "--height", str(CAPTURE_H),
    "-t",       "2000",
    "--nopreview"
], capture_output=True, text=True)

if r.returncode != 0:
    print(f"ERROR: {r.stderr}")
    sys.exit(1)

kb = os.path.getsize(CAPTURE_PATH) / 1024
print(f"  Saved : {CAPTURE_PATH}  ({kb:.1f} KB)")
print("  ✓ Capture OK")

# ── Test 3: OpenCV read + resize to 96x96 ─────────────────
print("\nTest 3: Reading and resizing to model input size...")
try:
    import cv2
except ImportError:
    print("Installing opencv-python-headless...")
    subprocess.run(["pip", "install", "opencv-python-headless"], check=True)
    import cv2

img = cv2.imread(CAPTURE_PATH)
if img is None:
    print("ERROR: OpenCV could not read image")
    sys.exit(1)

print(f"  Original shape : {img.shape[1]}x{img.shape[0]}")

# Resize to model input — this is what the model sees
small = cv2.resize(img, (MODEL_SIZE, MODEL_SIZE))
print(f"  Resized shape  : {small.shape[1]}x{small.shape[0]}  ← model input")

# ── Test 4: Image quality ─────────────────────────────────
print("\nTest 4: Image quality check...")
gray    = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
mean_v  = gray.mean()
std_v   = gray.std()
print(f"  At 96x96 — mean brightness : {mean_v:.1f}")
print(f"  At 96x96 — std deviation   : {std_v:.1f}")

if mean_v < 5:
    print("  WARNING: Image nearly black — check lighting/lens")
elif std_v < 3:
    print("  WARNING: Very flat image — camera may not be working")
else:
    print("  ✓ Image quality looks healthy")

# Note about resolution
print("\n  NOTE: Camera always captures at 640x640 for quality.")
print("  The resize to 96x96 happens inside preprocessing only.")
print("  This gives better results than capturing at 96x96 directly.")

print("\n" + "="*55)
print("  Camera test passed.")
print("="*55 + "\n")