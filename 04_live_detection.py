"""
Live Hard Hat Detection using Camera Feed
Real-time detection on Raspberry Pi camera (Pi 5 compatible)
Uses rpicam-still instead of cv2.VideoCapture
"""

import cv2
import numpy as np
from datetime import datetime
import time
import subprocess
import os
from collections import deque
import sys

# Import the detector class
from helmet_detector import HardHatDetector

TMP_FRAME = "/tmp/live_frame.jpg"

class LiveDetector:
    def __init__(self, model_path="best_int8.tflite",
                 resolution=(640, 640),
                 conf_threshold=0.5,
                 display=False):
        """
        Initialize live detector

        Args:
            model_path: Path to TFLite model
            resolution: Camera resolution (width, height)
            conf_threshold: Confidence threshold
            display: Whether to display video (False for headless)
        """
        self.resolution = resolution
        self.display = display

        print(f"\n{'='*60}")
        print(f"INITIALIZING LIVE HARD HAT DETECTOR")
        print(f"{'='*60}\n")

        # Initialize detector
        print(f"🔧 Loading model...")
        self.detector = HardHatDetector(model_path, conf_threshold=conf_threshold)

        # Check camera
        print(f"\n🎥 Checking Pi camera...")
        r = subprocess.run(["rpicam-still", "--list-cameras"],
                           capture_output=True, text=True)
        out = r.stdout + r.stderr
        if "No cameras" in out or r.returncode != 0:
            raise RuntimeError("No camera detected. Check ribbon cable and camera enable.")
        print(f"✅ Camera ready at {resolution[0]}x{resolution[1]}")

        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.detection_counts = {"Hardhat": 0, "NO-Hardhat": 0}

        print(f"\n{'='*60}\n")

    def capture_frame(self):
        """Capture a single frame using rpicam-still"""
        r = subprocess.run([
            "rpicam-still",
            "-o", TMP_FRAME,
            "--width",  str(self.resolution[0]),
            "--height", str(self.resolution[1]),
            "-t", "200",
            "--nopreview"
        ], capture_output=True)

        if r.returncode != 0:
            return None

        frame = cv2.imread(TMP_FRAME)
        return frame

    def run(self, save_video=False, video_path="output.avi"):
        """
        Run live detection loop

        Args:
            save_video: Whether to save annotated frames as video
            video_path: Path to save video
        """
        print("🚀 Starting live detection...")
        print("\nControls:")
        print("  Ctrl+C - Quit")
        print("  (headless mode — no display window)")
        if save_video:
            print(f"  Video will be saved to: {video_path}")
        print()

        # Video writer setup
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(
                video_path, fourcc, 1,
                (self.resolution[0], self.resolution[1])
            )

        snapshot_count = 0
        frame_count = 0

        try:
            while True:
                loop_start = time.time()

                # Capture frame
                frame = self.capture_frame()
                if frame is None:
                    print("❌ Error capturing frame")
                    break

                frame_count += 1

                # Run detection
                detections, inference_time = self.detector.detect(frame)

                # Draw detections
                result_frame = self.detector.draw_detections(frame, detections)

                # Update detection counts
                for detection in detections:
                    class_id = int(detection[0])
                    class_name = self.detector.class_names[class_id]
                    self.detection_counts[class_name] = \
                        self.detection_counts.get(class_name, 0) + 1

                # Calculate FPS
                elapsed = time.time() - loop_start
                fps = 1.0 / elapsed if elapsed > 0 else 0
                self.fps_history.append(fps)
                avg_fps = np.mean(self.fps_history)

                # Add info overlay
                result_frame = self.add_info_overlay(
                    result_frame, detections, avg_fps, inference_time
                )

                # Save video frame
                if save_video and video_writer is not None:
                    video_writer.write(result_frame)

                # Save each frame as snapshot in captures folder
                os.makedirs("captures", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_path = f"captures/frame_{frame_count:04d}_{timestamp}.jpg"
                cv2.imwrite(snap_path, result_frame)

                # Print stats every frame (headless feedback)
                hardhat_count   = sum(1 for d in detections if int(d[0]) == 0)
                no_hardhat_count = sum(1 for d in detections if int(d[0]) == 1)
                print(f"[Frame {frame_count:04d}] FPS: {avg_fps:.2f} | "
                      f"Inference: {inference_time*1000:.1f}ms | "
                      f"Hardhat: {hardhat_count} | "
                      f"NO-Hardhat: {no_hardhat_count} | "
                      f"Saved: {snap_path}")

        except KeyboardInterrupt:
            print("\n⚠️  Stopped by user")

        finally:
            if video_writer is not None:
                video_writer.release()
                print(f"💾 Video saved to: {video_path}")

            # Clean up tmp frame
            if os.path.exists(TMP_FRAME):
                os.remove(TMP_FRAME)

            self.print_statistics(frame_count)

    def add_info_overlay(self, frame, detections, fps, inference_time):
        """Add information overlay to frame"""
        overlay = frame.copy()
        panel_height = 120
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_height),
                      (0, 0, 0), -1)
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color      = (255, 255, 255)
        thickness  = 2
        y_offset   = 25
        line_height = 30

        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, y_offset), font, font_scale, color, thickness)
        cv2.putText(frame, f"Inference: {inference_time*1000:.1f}ms",
                    (10, y_offset + line_height), font, font_scale, color, thickness)

        hardhat_count    = sum(1 for d in detections if int(d[0]) == 0)
        no_hardhat_count = sum(1 for d in detections if int(d[0]) == 1)

        cv2.putText(frame, f"Hardhat: {hardhat_count}",
                    (10, y_offset + 2*line_height), font, font_scale,
                    (0, 255, 0), thickness)
        cv2.putText(frame, f"NO-Hardhat: {no_hardhat_count}",
                    (10, y_offset + 3*line_height), font, font_scale,
                    (0, 0, 255), thickness)

        cv2.putText(frame, f"Total Hardhat: {self.detection_counts.get('Hardhat', 0)}",
                    (frame.shape[1] - 300, y_offset), font, font_scale,
                    (0, 255, 0), thickness)
        cv2.putText(frame, f"Total NO-Hardhat: {self.detection_counts.get('NO-Hardhat', 0)}",
                    (frame.shape[1] - 300, y_offset + line_height), font, font_scale,
                    (0, 0, 255), thickness)

        return frame

    def print_statistics(self, total_frames):
        """Print final statistics"""
        print(f"\n{'='*60}")
        print("FINAL STATISTICS")
        print(f"{'='*60}")
        print(f"Total frames processed: {total_frames}")
        avg = np.mean(self.fps_history) if self.fps_history else 0
        print(f"Average FPS: {avg:.2f}")
        print(f"\nDetection Counts:")
        print(f"  Hardhat:    {self.detection_counts.get('Hardhat', 0)}")
        print(f"  NO-Hardhat: {self.detection_counts.get('NO-Hardhat', 0)}")
        print(f"{'='*60}\n")


def main():
    save_video = False
    headless   = True   # always headless on Pi 5 (no display)

    if len(sys.argv) > 1:
        if '--save-video' in sys.argv:
            save_video = True

    try:
        detector = LiveDetector(
            model_path="best_int8.tflite",
            resolution=(640, 640),
            conf_threshold=0.5,
            display=False
        )
        detector.run(save_video=save_video)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())