# Edge AI Project Report - Safety Helmet Detection System

**Course:** Edge AI  
**Project Title:**Safety Helmet Detection System
**Platform:** Raspberry Pi 4 Model B (4 GB RAM)  
**Model:** YOLOv8m fine-tuned, INT8 TFLite

---

## 1. Problem Statement, Motivation & Objectives

Construction and industrial sites must enforce Personal Protective Equipment (PPE) compliance, especially the use of safety helmets, to avoid head injuries and fatalities. Traditionally, human safety officers perform periodic visual checks for compliance. This method has limitations. It cannot provide constant coverage across the entire site, is subject to human fatigue and mistakes, and does not scale well to large or multi-zone worksites. Missing even one violation can lead to serious injury or legal issues for the site operator.

This project aims to replace manual monitoring with an always-on, automated AI-powered camera system that detects in real time if each person in the frame is wearing a hard hat. Edge AI is the best choice for three reasons: (1) **Latency** — a cloud-based system would introduce unacceptable delays before raising an alert, (2) **Privacy** — sending live video of workers to a remote server raises significant data privacy and legal issues, and (3) **Reliability** — construction sites often have poor or no internet connectivity, so a fully offline on-device system is crucial.

**Key Project Objectives:**

- Fine-tune a YOLOv8m object detection model on a labeled hard-hat dataset to distinguish between workers wearing (Hardhat) and not wearing (NO-Hardhat) safety helmets with ≥ 95% mAP@0.50.
- Compress the trained model using INT8 post-training quantization and export it to TFLite format, aiming for a model size under 25 MB suitable for embedded deployment.
- Run the quantized model on a Raspberry Pi 4 and set up a real-time camera inference loop with live HDMI display output, bounding box overlays, and session-level detection logging.
- Show end-to-end operation — from raw camera frame to annotated on-screen alert — entirely offline on the edge device without relying on the cloud.
- Measure the system's inference latency, throughput (FPS), and resource usage, and identify ways to improve it in the future.

---

## 2. Proposed Solution (Overview)

The system is a computer vision pipeline that continuously captures frames from a Raspberry Pi Camera Module. It processes these frames with a fine-tuned YOLOv8m object detector compressed to INT8 TFLite format, displaying results in real time on an HDMI-connected monitor. Each frame is analyzed to identify all people in the scene and classify each detected head region as either **Hardhat**  or **NO-Hardhat** . The system keeps cumulative detection counts and allows snapshot capture for audit purposes.

**End-to-end pipeline:**

```
1.[Roboflow Dataset]
      
2.[EDA & Class Distribution Analysis]
      
3.[YOLOv8m Fine-tuning — Google Colab / GPU]
      (Transfer learning from COCO pre-trained weights)
      
4.[Model Validation — mAP, Precision, Recall]
      
5.[TFLite INT8 Export via Ultralytics]
      
6.[Copy best_int8.tflite → Raspberry Pi 4]
      
7.[Live Camera Capture — rpicam-still / Picamera2]
      
8.[Preprocessing — resize 640×640, BGR→RGB, normalize]
      
9.[TFLite Inference — HardHatDetector class]
      
10.[Post-processing — decode [1,6,8400] tensor, NMS, coordinate rescaling]
      
11.[HDMI Display — OpenCV annotated frame + dashboard overlay]
```

The system consists of two Python modules: `helmet_detector.py`, which is a reusable, hardware-independent `HardHatDetector` class containing all inference logic, and `livestream.py`, which runs the application loop that manages the camera, display, keyboard input, and session state.

---

## 3. Hardware & Software Setup

### Hardware

 Component  Details 

* Edge Device : Raspberry Pi 4 Model B — 4 GB RAM, ARM Cortex-A72 quad-core @ 1.8 GHz 
* Camera : Raspberry Pi Camera Module v2 (8 MP, MIPI CSI-2 interface) 
* Display : HDMI monitor connected to Pi HDMI port for live output 
* Storage : 32 GB microSD card (Class 10) 
* Power : 5V / 3A USB-C power supply 
* Training Machine : Google Colab (NVIDIA T4 GPU) 

### Software

 Component  Details 

* OS : Raspberry Pi OS (64-bit, Bookworm) 
* Training Framework : Ultralytics YOLOv8 (`ultralytics` Python package) 
* Inference Runtime : TFLite Runtime (`tflite-runtime` 2.14 for ARM64) 
* Computer Vision : OpenCV 4.9 (`cv2`) 
* Dataset Management : Roboflow Python SDK 
* Camera Interface : `rpicam-still` (libcamera stack) 
* Numerical Computing : NumPy 1.26 
* Training Environment : Python 3.10, Google Colab 
* Deployment Environment : Python 3.11, Raspberry Pi OS 

---

## 4. Data Collection & Dataset Preparation

### Data Source

The dataset was sourced from **Roboflow Universe** — specifically the *Hard Hat Universe* public dataset, accessed through the Roboflow Python API. This is a curated collection of real construction site photographs with manually annotated bounding boxes, making it suitable for production-quality PPE detection.

### Dataset Statistics

| Split | Images | Hardhat Instances | NO-Hardhat Instances |
|---|---|---|---|
| Train | ~2,800 | ~5,200 | ~1,900 |
| Validation | ~400 | ~750 | ~280 |
| Test | ~200 | ~370 | ~140 |



### Preprocessing Steps

1. **Format standardization** — All images were downloaded in YOLOv8 format: images in `/images/`, annotations in `/labels/` as `.txt` files with `class x_center y_center width height` (normalized 0–1).
2. **Configuration file** — `data.yaml` generated by Roboflow specifies class names (`Hardhat`, `NO-Hardhat`) and split paths — used directly by the Ultralytics trainer.
3. **EDA** — A class distribution bar chart was plotted to verify no severe imbalance. Bounding box aspect ratio and size distribution were plotted to ensure that 640×640 input resolution was appropriate (no significant population of boxes smaller than 10px).
4. **Dynamic augmentation during training** — Mosaic (1.0), Mixup (0.1), CopyPaste (0.1), random horizontal flip, HSV color jitter — applied on the fly by the Ultralytics data loader, not pre-applied to disk.
5. **Normalization** — At inference time, pixel values were divided by 255.0 to fall within the [0.0, 1.0] float32 range. Input shape was expanded to [1, 640, 640, 3] for TFLite batch dimension.

---

## 5. Model Design, Training & Evaluation

### Model Architecture

**YOLOv8m (medium)** was chosen as the base architecture. YOLOv8 is a single-stage anchor-free object detector. The network consists of:
- **Backbone** — CSPDarknet with C2f bottleneck modules; extracts multi-scale feature maps.
- **Neck** — PANet (Path Aggregation Network); fuses features from multiple backbone levels to handle objects at different scales.
- **Head** — Decoupled detection head that outputs an [N, 6, 8400] tensor per image containing 8,400 anchor-free predictions each encoding `[x_center, y_center, width, height, conf_class0, conf_class1]`.

The model was initialized from `yolov8m.pt`, which includes weights pre-trained on the COCO dataset (80 classes, 330K images). The detection head was replaced to output 2 classes while keeping all backbone and neck weights unchanged and fine-tuned through transfer learning.

### Training Setup

| Hyperparameter | Value | Rationale |
|---|---|---|
| Epochs | 100 | Enough for convergence with early stopping as a safety net |
| Patience | 20 | Stops training if validation mAP does not improve for 20 epochs |
| Batch size | 16 | Typical for 8 GB GPU; provides stable gradient estimation |
| Image size | 640 × 640 | Native YOLOv8 input; ensures sufficient resolution for head detection |
| Optimiser | SGD | Default Ultralytics SGD with momentum 0.937 |
| Initial LR (lr0) | 0.01 | Standard for fine-tuning |
| Warmup epochs | 3 | Gradual LR ramp-up to protect pre-trained weights |
| Mosaic | 1.0 | Full mosaic augmentation used throughout training |
| Mixup | 0.1 | Slight blending to improve regularization |
| CopyPaste | 0.1 | Synthetic object placement for edge-case coverage |
| Device | NVIDIA T4 (Colab) | ~40 min total training time |

### Evaluation Metrics

The model was evaluated on the held-out test split using `model.val()`:

| Metric | Value |
|---|---|
| mAP@0.50 | **97.8%** |
| mAP@0.50:0.95 | **72.4%** |
| Precision | **96.5%** |
| Recall | **95.2%** |
| AP@0.50 — Hardhat | 98.2% |
| AP@0.50 — NO-Hardhat | 97.4% |

The high recall (95.2%) is crucial for this safety application, as the model misses fewer than 5 out of 100 real violations, reducing the risk of undetected non-compliance.

---

## 6. Model Compression & Efficiency Metrics

### Technique: Post-Training INT8 Quantization

The trained `best.pt` (FP32 PyTorch checkpoint) was exported to TFLite INT8 format using the Ultralytics export utility:

```python
from ultralytics import YOLO

model = YOLO('best.pt')

model.export(format='tflite', imgsz=640)

# Output: best_int8.tflite
```

Ultralytics performs post-training quantization (PTQ) during export. All weight tensors and activations are converted from 32-bit float (FP32) to 8-bit integer (INT8) representation. A calibration pass over representative data determines per-tensor quantization scale factors and zero-points. This ensures the INT8 value range accurately maps to the original float range.

### Compression Results

| Metric | FP32 PyTorch (`best.pt`) | INT8 TFLite (`best_int8.tflite`) | Change |
|---|---|---|---|
| Model size (disk) | ~100 MB | ~25 MB | **−75% (4× smaller)** |
| Inference time (Pi 4) | Not runnable (no PyTorch on Pi) | ~800 ms | Baseline for Pi |
| Inference time (GPU, Colab) | ~12 ms | N/A | - |
| RAM usage (Pi 4) | - | ~180–220 MB | Within 4 GB budget |
| Accuracy (mAP@0.50) | 97.8% | ~96.1% (estimated) | **−1.7% degradation** |

### Trade-offs Observed

- **Size vs. accuracy**: A 4× size reduction results in a cost of ~1–2% mAP degradation, which is acceptable for a binary safety classifier.
  
- **Speed vs. model choice**: INT8 helps, but the model remains medium-sized. YOLOv8n INT8 could be ~3× faster at a ~5–8% lower mAP.

- **No XNNPACK delegate enabled**: The default TFLite runtime was used without the XNNPACK or NNAPI acceleration delegates. This means the ARM NEON SIMD units on the Pi were not fully used, which represents a known optimization gap.

---

## 7. Model Deployment & On-Device Performance

### Deployment Steps

1. **Export**: `best_int8.tflite` was created on Colab using `model.export(format='tflite', imgsz=640)`.
   
2. **Transfer**: The model file was copied to the Raspberry Pi via `scp` over the local network.
   
3. **Dependency install**: `tflite-runtime`, `opencv-python-headless`, and `numpy` were installed via pip on the Pi. No PyTorch or Ultralytics was required at runtime.
   
4. **Module deployment**: `helmet_detector.py` and `livestream.py` were transferred to the Pi working directory.
   
5. **Camera verification**: The command `rpicam-still --list-cameras` confirmed the camera hardware was detected.
   
6. **Execution**: The command `python livestream.py` was run, which opened the OpenCV window in fullscreen on the HDMI monitor.

### On-Device Performance

| Metric | Value |
|---|---|
| Inference time per frame | ~750–900 ms |
| Frame capture time (rpicam-still) | ~600–800 ms |
| Total loop time per frame | ~1,800–2,000 ms |
| Effective throughput | **~0.5 FPS** |
| RAM usage (inference) | ~200 MB |
| CPU utilization (1 thread) | ~85–95% on one core |
| Model file size on device | 12 MB |

### Real-Time Behavior

The system works in a continuous loop. Each iteration captures a frame, runs detection, draws bounding boxes, and overlays a semi-transparent HUD showing FPS, inference latency, per-class counts, and timestamps. It refreshes the HDMI display through `cv2.imshow()`. Keyboard shortcuts allow quitting (`Q`/`ESC`) and saving annotated snapshots (`S`) to a `captures/` directory.

The 0.5 FPS throughput works for monitoring a fixed-camera entrance where a person pauses at a checkpoint. However, it does not support fast motion tracking in an open area.

---

## 8. System Prototype (Pictures / Figures)

> *Note: Actual photos of the hardware setup and live detection output will be inserted here in the final submission. The descriptions below indicate what each figure should show.*

**Figure 1 — Hardware setup**: ![alt text](image-1.png)

**Figure 2 — Training loss curves**: ![alt text](image.png)

**Figure 3 — Saved snapshot**: ![alt text](frame_0008_20260422_181836.jpg)
---

## 9. Conclusions & Limitations

### Key Outcomes

This project shows a complete end-to-end Edge AI pipeline for real-time PPE compliance monitoring. A YOLOv8m model fine-tuned on construction site data achieved **97.8% mAP@0.50**. After INT8 post-training quantization, the model size dropped from ~102 MB to **~25 MB**, a 4× reduction, while retaining about 98% of its original accuracy. The model was deployed on a Raspberry Pi 4 with no cloud dependency, running fully offline with live HDMI display output, bounding box annotations, and session-level violation logging.

### Limitations

- **Low throughput (0.5 FPS)**: The main limitation. This is due to `rpicam-still` starting a new OS process per frame (~700 ms overhead), single-threaded TFLite inference without hardware acceleration delegates (~800 ms), and sequential blocking I/O. The system is not suitable for fast-moving subjects or wide-area scenes.

- **2-class only**: The model detects helmets and their absence but does not detect other PPE items like vests, gloves, or goggles. Expanding to full PPE detection would need a larger, multi-class dataset.

- **No alert output**: The current prototype only displays results on-screen. A production system would need integration for buzzer, SMS, or network alerts.

- **Fixed camera position**: The system assumes a controlled field of view. Performance drops with extreme angles, poor lighting, or heavy occlusion.


---

## 10. Future Work

- **Picamera2 for continuous streaming**: Replace `rpicam-still` subprocess calls with the Picamera2 library to keep the camera open and deliver frames as NumPy arrays directly in RAM. This would remove ~700 ms of process startup time per frame.

- **Producer-consumer threading**: Separate capture and inference into different threads with a frame queue. This would allow camera capture and model inference to happen simultaneously, which might double effective throughput.

- **Downscale to YOLOv8n at 320×320**: Sacrifice ~5% mAP for a ~10× speedup in inference, which is suitable for entrance-gate scenarios where subjects are large in frame.

- **Multi-class PPE detection**: Update the dataset and model to detect safety vests, gloves, and eye protection along with helmets for thorough PPE auditing.

- **Alert system integration**: Add GPIO-driven buzzer, LED indicator, or network webhook to send real-time alerts to a site supervisor's phone when a violation occurs.

---

## 11. Challenges & Mitigation

| Challenge | Description | Mitigation |
|---|---|---|
| Camera incompatibility with OpenCV | The Pi Camera Module uses the MIPI CSI-2 interface and libcamera stack; `cv2.VideoCapture()` fails on modern Pi OS. | Used `rpicam-still` CLI tool via `subprocess.run()` to capture frames to `/tmp/` and read back with `cv2.imread()`. |
| No PyTorch on Raspberry Pi | Ultralytics YOLOv8 needs PyTorch, which is hard to install on ARM64 Pi OS in a lightweight form. | Exported the model to `best_int8.tflite` on Colab; used only `tflite-runtime` (lightweight ARM64 wheel) on the Pi. |
| Post-processing from raw TFLite tensor | TFLite output is a raw [1, 6, 8400] tensor — no built-in YOLO decoding. Custom NMS and coordinate rescaling had to be created manually. | Implemented `postprocess_predictions()` and `non_max_suppression()` using `cv2.dnn.NMSBoxes()` in the `HardHatDetector` class. |
| Low inference FPS | The sequential loop with `rpicam-still` and single-threaded inference gave only 0.5 FPS, which is visually laggy. | This is accepted as a known limitation for this prototype. Bottlenecks have been diagnosed, and fixes are noted in Future Work. |
| Class imbalance in dataset | Hardhat instances are about 2.7 times more common than NO-Hardhat, risking lower recall on the minority class. | Ultralytics augmentation (Mosaic, CopyPaste) artificially increases minority class variety. Per-class AP was monitored: NO-Hardhat AP@0.50 = 97.4% confirmed there was no recall drop. |
| Coordinate space mismatch | TFLite outputs normalized coordinates in a 640×640 model input space instead of the original image resolution. | Calculated `scale_x = orig_width / 640` and `scale_y = orig_height / 640` in `preprocess_image()` and applied it at decode time. |

---

## 12. References

1. **Ultralytics YOLOv8 Documentation** — Model training, export, and validation.
   https://docs.ultralytics.com

2. **Roboflow Universe — Hard Hat Universe Dataset** — Labeled PPE dataset used for training.
   https://universe.roboflow.com/hard-hat-universe

3. **TensorFlow Lite Runtime** — Interpreter API and INT8 inference on ARM.
   https://www.tensorflow.org/lite/guide/python

