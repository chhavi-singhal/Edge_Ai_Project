# Edge_Ai_Project
# 🪖 PPE Hard Hat Detection — Edge AI on Raspberry Pi 5

A real-time **Personal Protective Equipment (PPE) detection system** that runs entirely on a **Raspberry Pi 5** using a YOLOv8 model trained on Roboflow and deployed as a quantized TFLite model. The system detects whether workers are wearing hard hats using a Pi Camera Module, with no cloud dependency.

---

## ✨ Highlights

- **True Edge AI deployment** — All inference runs locally on the Raspberry Pi 5 with no cloud or internet dependency after setup.
- **int8 Quantized YOLOv8** — Trained in float32, exported as int8 TFLite for fast, memory-efficient inference on the Pi.
- **Pi 5 camera compatible** — Uses `rpicam-still` instead of OpenCV's VideoCapture (which fails on Pi 5 due to GStreamer issues).
- **Headless + display modes** — Run detection headlessly and view annotated captures, or connect an HDMI monitor for a live fullscreen stream.
- **Two detection classes** — `Hardhat` (green box) and `NO-Hardhat` (red box).

---

## 📁 Repository Structure

```
hardhat-detection/
├── 01_inspect_model.py        # Inspect TFLite model input/output details
├── 02_camera_test.py          # Test Pi camera using rpicam-still
├── 03_helmet_detector.py      # Run detection on a single saved image
├── 04_live_detection.py       # Headless live detection — saves annotated frames
├── helmet_detector.py         # Core HardHatDetector class (used by all scripts)
├── livestream.py              # Live detection with display on HDMI monitor
├── launcher.py                # Interactive menu to launch any script easily
├── ppe_yolov8_roboflow.ipynb  # Google Colab notebook — train & export model
├── setup_rpi.sh               # Original setup script (see Pi setup notes below)
├── requirements.txt           # Python dependencies
├── test_installation.py       # Verify package imports
└── captures/                  # Auto-created — annotated detection frames saved here
```

> ⚠️ The `venv/` folder is **not included** in this repository.  
> You must create it manually on your Raspberry Pi — full steps are below.

---

## 🧠 Model

| Property | Value |
|---|---|
| Architecture | YOLOv8m |
| Training framework | Ultralytics |
| Dataset | Roboflow (Hard Hat Universe) |
| Classes | `Hardhat`, `NO-Hardhat` |
| Input size | 640 × 640 × 3 |
| Export format | TFLite int8 quantized |
| Model size | ~25 MB |
| Inference time (Pi 5) | ~1200 ms per frame |

---

## 🔧 Hardware Required

- **Raspberry Pi 5** (4 GB or 8 GB)
- **Pi Camera Module** (IMX219 or compatible, connected via CSI ribbon cable)
- **MicroSD card** (32 GB+, Raspberry Pi OS Bookworm)
- **HDMI monitor** *(optional — only needed for `livestream.py`)*
- **Keyboard + Mouse** *(optional — SSH via VS Code works fine)*

---

## 🚀 Raspberry Pi Setup

### ⚠️ Important — Do NOT use `setup_rpi.sh` directly

Raspberry Pi OS Bookworm enforces **PEP 668**, which blocks system-wide `pip` installs. The shell script fails silently and leaves packages uninstalled. Follow the steps below instead, which use a **virtual environment**.

---

### Step 1 — Install system dependencies

```bash
sudo apt update
sudo apt install -y python3-full python3-venv libhdf5-dev libatlas-base-dev libcamera-apps
```

### Step 2 — Clone the repository

```bash
cd ~
git clone https://github.com/YOUR_USERNAME/hardhat-detection.git
cd hardhat-detection
```

### Step 3 — Create the Python virtual environment

```bash
python3 -m venv venv
```

This creates a `venv/` folder inside the project directory. It is excluded from Git. **You must recreate it on every new Pi you deploy to.**

### Step 4 — Activate the virtual environment

```bash
source venv/bin/activate
```

Your prompt will change to show `(venv)`:
```
(venv) rpi80@rpi80:~/hardhat-detection $
```

> 🔴 **Critical:** You must run `source venv/bin/activate` every time you open a new terminal before running any script. Without it, Python will not find the installed packages and you will get `ModuleNotFoundError`.

### Step 5 — Install Python packages inside the venv

```bash
pip install "numpy<2"
pip install tflite-runtime
```

For OpenCV, use the system package (much faster to install than pip):
```bash
sudo apt install -y python3-opencv
cp -r /usr/lib/python3/dist-packages/cv2* venv/lib/python3*/site-packages/ 2>/dev/null || true
```

If the copy fails, install the headless pip version instead:
```bash
pip install opencv-python-headless
pip install "numpy<2"   # re-pin after opencv upgrades numpy
```

### Step 6 — Verify everything is working

```bash
python3 -c "
import cv2
import tflite_runtime.interpreter
import numpy
print('cv2:', cv2.__version__)
print('numpy:', numpy.__version__)
print('tflite_runtime: OK')
"
```

Expected output:
```
cv2: 4.x.x
numpy: 1.26.4
tflite_runtime: OK
```

---

## 📦 Adding the Model File

The `best_int8.tflite` model is **not included** in this repo due to its size (~25 MB). You must export and copy it manually.

### Export from Google Colab

Open `ppe_yolov8_roboflow.ipynb` in Google Colab, train the model, then run:

```python
from ultralytics import YOLO

model = YOLO('runs/ppe/train/weights/best.pt')
tflite_path = model.export(format='tflite', imgsz=640, int8=True)
print(f'Exported to: {tflite_path}')
```

Then download the file:
```python
from google.colab import files
files.download('best_int8.tflite')
```

If you already have `best.pt` (or a `best.pt.zip`), upload it to Colab and export:

```python
# Upload the file
from google.colab import files
uploaded = files.upload()   # select best.pt or best.pt.zip

# If zipped, extract first
import zipfile, glob
with zipfile.ZipFile(list(uploaded.keys())[0], 'r') as z:
    z.extractall('.')

# Find and export
pt_path = glob.glob('**/best.pt', recursive=True)[0]
from ultralytics import YOLO
model = YOLO(pt_path)
model.export(format='tflite', imgsz=640, int8=True)

from google.colab import files
files.download('best_int8.tflite')
```

### Copy model to Raspberry Pi

Run this on your **laptop**:

```bash
scp best_int8.tflite rpi80@<PI_IP_ADDRESS>:~/hardhat-detection/
```

### Verify the model is correct

```bash
source venv/bin/activate

python3 -c "
import tflite_runtime.interpreter as tflite
interp = tflite.Interpreter('best_int8.tflite')
interp.allocate_tensors()
print('Input shape :', interp.get_input_details()[0]['shape'])
print('Output shape:', interp.get_output_details()[0]['shape'])
"
```

Expected:
```
Input shape : [  1 640 640   3]
Output shape: [   1    6 8400]
```

> If input shape shows `[1, 32, 32, 3]`, the model was exported incorrectly. Re-export using `imgsz=640` as shown above.

---

## 🎥 Running the System

**Always activate the venv first:**

```bash
cd ~/hardhat-detection
source venv/bin/activate
```

### Test the camera

```bash
python3 02_camera_test.py
```

Checks that `rpicam-still` can detect and capture from the Pi camera.

### Inspect the model

```bash
python3 01_inspect_model.py
```

Prints input/output shapes, tensor details, and model size.

### Run detection on a saved image

```bash
python3 03_helmet_detector.py /path/to/image.jpg
```

Saves an annotated result image in the current directory.

### Live detection — headless (no monitor needed)

```bash
python3 04_live_detection.py
```

Captures frames continuously using `rpicam-still`, runs detection, and saves annotated frames to the `captures/` folder. Open this folder in VS Code to see results. Press `Ctrl+C` to stop.

### Live detection with video output

```bash
python3 04_live_detection.py --save-video
```

Also writes an `output.avi` video file.

### Fullscreen livestream on HDMI monitor

```bash
python3 livestream.py
```

Opens a fullscreen window on the Pi's HDMI display showing the live camera feed with bounding boxes and stats overlay. Press `Q` or `ESC` to quit. Press `S` to save a snapshot.

### Interactive menu launcher

```bash
python3 launcher.py
```

Provides a numbered menu to launch any script without remembering commands.

---

## 📷 Camera Notes — Pi 5 Compatibility

Raspberry Pi 5 with the IMX219 camera does **not** work with `cv2.VideoCapture()`. You will see this error:

```
GStreamer warning: Embedded video playback halted; module v4l2src0 reported: Failed to allocate required memory.
GStreamer warning: unable to start pipeline
```

All camera capture in this project uses `rpicam-still` subprocess calls instead, which works reliably. The camera is detected as:

```
0 : imx219 [3280x2464 10-bit RGGB]
```

---

## ⚠️ Common Issues & Fixes

| Error | Cause | Fix |
|---|---|---|
| `No module named 'cv2'` | OpenCV not installed in venv | `sudo apt install python3-opencv` then copy into venv |
| `No module named 'tensorflow'` | Scripts use `tflite_runtime`, not full TensorFlow | `pip install tflite-runtime` inside venv |
| `numpy.core.multiarray failed to import` | NumPy 2.x installed, OpenCV needs 1.x | `pip install "numpy<2"` inside venv |
| `GStreamer warning: unable to start pipeline` | Expected on Pi 5 | Scripts already use `rpicam-still` — ignore this warning |
| `AttributeError: _ARRAY_API not found` | Ran pip/python outside the venv | Activate venv first: `source venv/bin/activate` |
| Input shape `[1, 32, 32, 3]` | Model exported with wrong settings | Re-export with `imgsz=640` in Colab |
| 0 detections | Camera not pointed at a person | Check `captures/` folder in VS Code to see what camera sees |
| `libcamera-hello: command not found` | libcamera-apps not installed | `sudo apt install -y libcamera-apps` |

---

## 🏋️ Training Pipeline Summary

The full pipeline is in `ppe_yolov8_roboflow.ipynb` (run in Google Colab):

1. Configure Roboflow API credentials
2. Download dataset in YOLOv8 format
3. Exploratory data analysis (class distribution, bounding box stats)
4. Train YOLOv8m for 100 epochs at 640×640 with augmentation
5. Validate on test split — prints mAP@0.50, precision, recall
6. Export to int8 TFLite for Raspberry Pi deployment

---

## 🔁 End-to-End Workflow

```
Google Colab
    ↓  Train YOLOv8m on Roboflow dataset
    ↓  Export → best_int8.tflite  (int8 quantized, ~25 MB)
    ↓
Laptop
    ↓  scp best_int8.tflite → Raspberry Pi
    ↓
Raspberry Pi 5
    ↓  cd ~/hardhat-detection
    ↓  source venv/bin/activate      ← always do this first
    ↓  python3 04_live_detection.py
    ↓  check captures/ folder for annotated results
```

---

## 📄 License

This project is for educational and research purposes.
