# 🚗 FogVision - AI-Powered ADAS Road Safety Dashboard

FogVision is an advanced **AI-Powered Driver Assistance System (ADAS)** dashboard. It processes video data (either uploaded files or live USB/network camera feeds) at 1 frame per second to perform real-time fog density analysis, adaptive contrast dehazing, multi-object tracking, time-to-collision warning logic, and contextual AI recommendation.

The dashboard features a premium, immersive **Royal Midnight Blue** dark theme with glowing Gold and Electric Teal highlights, designed for optimized visibility in simulated vehicle head-up displays.

---

## 🎯 Core Features

- **Dual Input Modes:**
  - **Upload Video:** Supports processing raw video files. Under this mode, dynamic distance estimation is bypassed, and all object distances are constrained to a constant `20.0m` (labeled as `constant_video`) for safety standard calibrations.
  - **Live Feed (USB Cam / DroidCam / ESP32 Cam):** Automatically connects to your local USB webcam (default index `0` or secondary indexes) or captures IP/MJPEG network streams from DroidCam or ESP32 Cam servers (e.g. `http://192.168.1.100/stream`).
- **Dynamic Preprocessing & Dehazing:**
  - **Fog Analysis:** Computes fog density percentage using the Dark Channel Prior (DCP) algorithm.
  - **Adaptive Contrast:** Automatically determines visibility levels, reduces speed thresholds, and applies CLAHE or full DCP dehazing.
  - **ROI Masking:** Uses a trapezoidal mask to filter out dashboard reflections and self-car body detections.
- **Computer Vision & Tracking:**
  - **YOLOv8 Detection:** Detects cars, trucks, pedestrians, cyclists, and traffic signs/lights.
  - **HSV Signal Analysis:** Classifies traffic light states (RED, GREEN, YELLOW).
  - **Brake-Light Monitoring:** Scans bounding boxes for red-glowing brake indicators on preceding vehicles.
  - **IoU Tracker:** Maintains unique object tracks over time (requires `min_hits = 3` to confirm tracks).
  - **Motion TTC Analyzer:** Computes relative velocities and Time-to-Collision (TTC) values.
- **Accident Blackspot Geospatial Maps:**
  - Embeds interactive Leaflet map overlays pointing out red-highlighted accident-prone zones (blackspots) within proximity to vehicle GPS locations.
- **Voice Recommendation Engine:**
  - Periodically queries a local Ollama LLM to synthesize driving hazard explanations.
  - Speaks warnings aloud using a multi-threaded TTS voice coordinator.

---

## 📁 Project Structure

```
fodVision/
├── DashBoard.py                # Main Streamlit dashboard application
├── pipeline.py                 # Core ADAS processing pipeline coordinator
├── video_engine.py             # 1-FPS frame acquisition engine (USB webcam / IP Stream / Video)
│
├── roi_mask.py                 # Region-of-Interest trapezoidal masking
├── fog_density.py              # Dark Channel Prior fog calculator
├── dehaze.py                   # Image dehazing model
├── fog_aware.py                # Fog-adaptive confidence NMS thresholds & filters
│
├── object_detect.py            # YOLOv8 object detection & HSV traffic light classifier
├── object_tracker.py           # Multi-object IoU tracker
├── red_glow.py                 # Active brake light glow detector
├── lane_detection.py           # Road lane peripheral limits polygon
│
├── improved_distance.py        # Monocular depth class-height calibrator
├── lidar_sensor.py             # LiDAR hardware scanner simulator wrapper
├── sensor_fusion.py            # Camera-LiDAR fusion module
│
├── motion_ttc.py               # Relative speed & Time-to-Collision estimator
├── risk_alerts.py              # Risk alarm thresholds checker
├── alert_hysteresis.py         # Cool-down and alert debouncing logic
├── confidence_gated_alerts.py  # Alert gating filters
├── temporal_fog_predictor.py   # Fog density trend forecaster
├── risk_score.py               # Combined 0-100 hazard risk engine
│
├── llm.py                      # Local Ollama LLM inference query handler
├── scene_memory_buffer.py      # Moving temporal window memory for LLM context
├── voice_alert.py              # Multi-threaded pyttsx3 Audio alert runner
├── road_context.py             # Reverse-geocoding road & blackspot lookup
├── evaluation_metrics.py       # Serializing performance and stage latencies
│
├── Requirements.txt            # Python package dependencies
├── .gitignore                  # Git file exclude listings
├── config/                     # Configuration parameters
│   └── blackspots.json         # Accident-prone zones database
└── models/                     # YOLO model weights storage
```

---

## 🚀 Setup & Installation

### Prerequisites
- **Python 3.8 to 3.13** installed.
- **NVIDIA GPU** with CUDA toolkit installed (recommended for smooth real-time YOLO/dehazing inference, falls back to CPU automatically).
- **Ollama** installed on your system (for local LLM advisory recommendations).

---

### Step-by-Step Installation

#### 1. Clone & Navigate to Project Directory
Open PowerShell or your command terminal and enter the project folder:
```powershell
cd "c:\Users\Puneeth Kumar\OneDrive\Desktop\IDP\Project\fodVision"
```

#### 2. Create and Activate Virtual Environment
```powershell
# Create environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate on Linux/macOS
source venv/bin/activate
```

#### 3. Install Dependencies
```powershell
pip install -r Requirements.txt
```

#### 4. Configure & Start Ollama
Start the local Ollama daemon service and pull the default reasoning model:
1. Open a separate terminal and start Ollama:
   ```bash
   ollama serve
   ```
2. Pull the default reasoning model in your active terminal:
   ```powershell
   ollama pull qwen3:1.7b
   ```
   *(Note: You can optionally pull `deepseek-r1:1.5b` and update the `MODEL_NAME` configuration inside `llm.py`)*

---

## 🎮 Running the Application

### 1. Launch the Streamlit Dashboard
With your virtual environment active, run:
```powershell
python -m streamlit run DashBoard.py
```

### 2. Operating the UI

#### 🎥 Video Mode
1. Set the radio selector to **Upload Video**.
2. Drag and drop your sample video (`Realistic_forward_moving_road.mp4` or other formats) into the file uploader.
3. The dashboard will automatically cache the file and begin processing.
4. Press **START** and **STOP** to control playback. Detections will show a constant `20.0m` distance tracking.

#### 🔌 Live Feed (USB Webcam / ESP32 Cam / DroidCam)
1. Set the radio selector to **Live Feed (Sensor + USB Cam/DroidCam)**.
2. In the text field, configure your source:
   - For a local **USB Webcam**, enter the camera index (e.g. `0` or `1`).
   - For a **DroidCam/ESP32 Cam** stream, enter the HTTP network URL (e.g., `http://192.168.1.100/stream`).
3. Click the **🔌 CONNECT FEED** button.
4. Once connected, press **START**. Distances will be calculated dynamically in real-time as objects move.

---

## ⚙️ Testing & Verification
You can verify the pipeline logic and distance overrides using the automated test suite script:
```powershell
# Run the test verification script
.\venv\Scripts\python.exe -m unittest test_pipeline_modes.py
```
*(Or run the custom mock verify script located inside the agent scratch subdirectory to check distance outputs under both modes).*
