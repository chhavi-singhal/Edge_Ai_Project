"""
Livestream Hard Hat Detection — displays directly on Pi screen (HDMI monitor)
Run this when a monitor is connected to the Raspberry Pi via HDMI.
"""

import cv2
import numpy as np
import subprocess
import os
import time
from datetime import datetime
from helmet_detector import HardHatDetector

TMP_FRAME = "/tmp/live_frame.jpg"
MODEL_PATH = "best_int8.tflite"
CONF_THRESHOLD = 0.5
RESOLUTION = (640, 640)
WINDOW_NAME = "Hard Hat Detection — Press Q to quit"


def capture_frame():
    """Capture a single frame using rpicam-still"""
    r = subprocess.run([
        "rpicam-still",
        "-o", TMP_FRAME,
        "--width",  str(RESOLUTION[0]),
        "--height", str(RESOLUTION[1]),
        "-t", "100",          # 100ms = fastest possible capture
        "--nopreview",
        "--immediate"         # skip autofocus delay
    ], capture_output=True)
    if r.returncode != 0:
        return None
    return cv2.imread(TMP_FRAME)


def draw_overlay(frame, detections, fps, inference_ms, total_counts):
    """Draw bounding boxes + info panel on frame"""

    # ── Bounding boxes ────────────────────────────────────────
    for det in detections:
        class_id, conf, x1, y1, x2, y2 = det
        class_id = int(class_id)

        label     = f"{'Hardhat' if class_id == 0 else 'NO-Hardhat'}: {conf:.2f}"
        color     = (0, 255, 0) if class_id == 0 else (0, 0, 255)

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame,
                      (int(x1), int(y1) - lh - 10),
                      (int(x1) + lw, int(y1)),
                      color, -1)
        cv2.putText(frame, label,
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ── Top info panel ────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 110), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    fscale = 0.62
    thick  = 2
    white  = (255, 255, 255)
    green  = (0, 255, 0)
    red    = (0, 80, 255)

    hardhat_now    = sum(1 for d in detections if int(d[0]) == 0)
    nohardhat_now  = sum(1 for d in detections if int(d[0]) == 1)

    # Left column
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 28),  font, fscale, white, thick)
    cv2.putText(frame, f"Inference: {inference_ms:.0f} ms",
                (10, 58),  font, fscale, white, thick)
    cv2.putText(frame, f"Hardhat: {hardhat_now}",
                (10, 88),  font, fscale, green, thick)

    # Right column
    cv2.putText(frame, f"NO-Hardhat: {nohardhat_now}",
                (frame.shape[1] - 280, 28), font, fscale, red, thick)
    cv2.putText(frame, f"Total Hardhat: {total_counts['Hardhat']}",
                (frame.shape[1] - 280, 58), font, fscale, green, thick)
    cv2.putText(frame, f"Total NO-Hardhat: {total_counts['NO-Hardhat']}",
                (frame.shape[1] - 280, 88), font, fscale, red, thick)

    # Bottom timestamp
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame, ts,
                (10, frame.shape[0] - 10),
                font, 0.5, (180, 180, 180), 1)

    return frame


def main():
    print("\n" + "="*55)
    print("  Hard Hat Livestream — Pi Screen Display")
    print("="*55)
    print("  Make sure a monitor is connected via HDMI!")
    print("  Press  Q  in the window to quit")
    print("  Press  S  in the window to save a snapshot")
    print("="*55 + "\n")

    # Load model
    print("🔧 Loading model...")
    detector = HardHatDetector(MODEL_PATH, conf_threshold=CONF_THRESHOLD)
    print("✅ Model ready\n")

    # Check camera
    r = subprocess.run(["rpicam-still", "--list-cameras"],
                       capture_output=True, text=True)
    if "No cameras" in r.stdout + r.stderr:
        print("❌ No camera detected!")
        return

    print("🎥 Camera ready — opening display window...\n")

    # Create fullscreen window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME,
                          cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    total_counts = {"Hardhat": 0, "NO-Hardhat": 0}
    fps_history  = []
    snapshot_n   = 0
    frame_n      = 0

    os.makedirs("captures", exist_ok=True)

    while True:
        t0 = time.time()

        # Capture
        frame = capture_frame()
        if frame is None:
            print("❌ Capture failed — retrying...")
            time.sleep(0.2)
            continue

        frame_n += 1

        # Detect
        detections, inf_time = detector.detect(frame)

        # Update totals
        for d in detections:
            name = detector.class_names[int(d[0])]
            total_counts[name] = total_counts.get(name, 0) + 1

        # FPS
        elapsed = time.time() - t0
        fps = 1.0 / elapsed if elapsed > 0 else 0
        fps_history.append(fps)
        avg_fps = np.mean(fps_history[-30:])

        # Draw
        display = draw_overlay(frame.copy(), detections,
                               avg_fps, inf_time * 1000, total_counts)

        # Show
        cv2.imshow(WINDOW_NAME, display)

        # Terminal feedback
        hh  = sum(1 for d in detections if int(d[0]) == 0)
        noh = sum(1 for d in detections if int(d[0]) == 1)
        print(f"[{frame_n:04d}] FPS:{avg_fps:.1f} | "
              f"Inf:{inf_time*1000:.0f}ms | "
              f"Hardhat:{hh} NO-Hardhat:{noh}")

        # Key handling  (waitKey needed for imshow to refresh)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:   # Q or ESC
            print("\n👋 Quit by user")
            break
        elif key == ord('s'):
            snapshot_n += 1
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"captures/snapshot_{snapshot_n:03d}_{ts}.jpg"
            cv2.imwrite(path, display)
            print(f"📸 Snapshot saved: {path}")

    cv2.destroyAllWindows()
    if os.path.exists(TMP_FRAME):
        os.remove(TMP_FRAME)

    print(f"\n{'='*55}")
    print("FINAL COUNTS")
    print(f"{'='*55}")
    print(f"  Hardhat:    {total_counts['Hardhat']}")
    print(f"  NO-Hardhat: {total_counts['NO-Hardhat']}")
    print(f"  Frames:     {frame_n}")
    print(f"  Avg FPS:    {np.mean(fps_history):.2f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()