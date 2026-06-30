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

## 📁 Complete File-by-File Codebase Map

Here is the detailed functional responsibility of every single file in the project:

### Main Core Orchestrators
* **[DashBoard.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/DashBoard.py):** The master user interface built in Streamlit. Renders the layout, safety panels, Folium maps, graphs, and handles initial file caching/pre-dehazing thresholds.
* **[pipeline.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/pipeline.py):** Core ADAS processing pipeline coordinator. Sequences visual processing, merges data, feeds context logs, checks alerts, and handles keyboard manual driving/ACC logic.
* **[video_engine.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/video_engine.py):** Direct video reader and webcam handler. Implements the standard 1-FPS frame skipping logic.

### Environment & Weather Processing
* **[fog_density.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/fog_density.py):** Dark Channel Prior (DCP) computer that parses average atmospheric light scattering to estimate fog density percentage.
* **[fog_aware.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/fog_aware.py):** Fog-driven confidence score and NMS threshold scaler, adjusting perception thresholds dynamically in low visibility.
* **[dehaze.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/dehaze.py):** Contrast recovery module applying CLAHE or adaptive contrast fallbacks to restore visibility.
* **[temporal_fog_predictor.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/temporal_fog_predictor.py):** Smoothes out weather fluctuations and runs an EMA regression over a 10-frame window to forecast incoming weather fronts.
* **[roi_mask.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/roi_mask.py):** Trapezoidal visual mask applied to ignore the vehicle hood and dashboard reflections.

### Vision & Tracking Layers
* **[object_detect.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/object_detect.py):** Runs YOLOv8 object boundaries and extracts detected traffic light bounding boxes, mapping their pixels into HSV space to classify colors (Red, Green, Yellow).
* **[object_tracker.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/object_tracker.py):** Simple Intersection-over-Union (IoU) object tracker maintaining target continuity across frames.
* **[red_glow.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/red_glow.py):** Pixel-intensity brake-light scanner checking for sudden bright red glows in preceding vehicle bounding boxes.
* **[lane_detection.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/lane_detection.py):** Traces a trapezoidal region representing active driving lanes, distinguishing target threats in-lane from peripheral hazards.

### Spatial Depth & Sensor Fusion
* **[improved_distance.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/improved_distance.py):** Estimates physical depth using monocular pinhole calculations and bounding-box class heights.
* **[lidar_sensor.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/lidar_sensor.py):** Simulates standard hardware LiDAR laser scans.
* **[sensor_fusion.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/sensor_fusion.py):** Cross-references camera bounding boxes with simulated LiDAR returns to output fused targets.

### Vehicle Dynamics & Control
* **[esp32_module.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/esp32_module.py):** Network wrapper for sending motor signals (`POST /motor`) and pulling distance sensors (`GET /sensors`) to/from the physical ESP32 car.
* **[simulation.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/simulation.py):** Pygame-based cognitive ADAS simulator displaying dashboard telemetry and road line flow animations.
* **[motion_ttc.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/motion_ttc.py):** Computes relative obstacle velocity ($v_{\text{rel}}$) and predicts collision times (TTC).

### Safety Warnings & Cognitive Advisors
* **[risk_score.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/risk_score.py):** Consolidates speed, proximity, weather, and traffic lights into a unified risk rating ($0$ to $100$).
* **[risk_alerts.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/risk_alerts.py):** Compares physical hazards against warnings and critical trigger distances.
* **[alert_hysteresis.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/alert_hysteresis.py):** Prevents alert spamming through configured warning and critical duration locks.
* **[confidence_gated_alerts.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/confidence_gated_alerts.py):** Prevents alerts for targets with low vision-detection confidence.
* **[llm.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/llm.py):** Serializes pipeline logs into a JSON context and queries the local Ollama LLM for advisory text.
* **[scene_memory_buffer.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/scene_memory_buffer.py):** Keeps a moving history log of target scenes.
* **[voice_alert.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/voice_alert.py):** Runs multi-threaded speech synthesis using a non-blocking voice worker.
* **[road_context.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/road_context.py):** GIS mapping server checking coordinate logs against the blackspots database.
* **[evaluation_metrics.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/evaluation_metrics.py):** Benchmarks frames-per-second, processing latencies, and output details.

### Test & Scratch Files
* **[test_esp32.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/test_esp32.py):** Minimal standalone utility for checking the status and communication response of the ESP32 server.
* **[test.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/test.py):** Quick validation scratch script.


---

## 🔄 End-to-End System Workflow

Each analysis cycle executes the following stages across the visual perception and physical driving control loop:

### 1. Initial Fog Check & Pre-dehaze Gating
* **Uploaded Videos:** The dashboard performs a Dark Channel Prior (DCP) check on the first frame. If the initial fog is $> 35.0\%$, the video is completely pre-dehazed, and the `VideoEngine` loads the dehazed file. The initial fog value is cached and set on the `ADASPipeline` instance to keep displaying the original weather conditions during playback.
* **Live Streams:** Pre-dehazing is bypassed. The pipeline runs a synchronous fog calculation on the first frame to display correct initial metrics immediately, followed by asynchronous calculations running in a background thread every 2 minutes (120 seconds).

### 2. Standardizing & Visual Tracking
* **Skipping Frames:** In video mode, intermediate frames are skipped to process exactly 1 frame per source-second. Frames are resized to a standard coordinate space of $640 \times 480$ pixels.
* **Perception & Zones:** Confirmed objects are mapped to Left ($X < 213$), Middle/In-lane ($213 \le X \le 426$), and Right ($X > 426$) zones. 

### 3. ESP32 Sensor Fusion & Velocity Estimation
* **Sensor Queries:** In live mode, zone distances are fused with physical VL53L0X distance readings fetched from `GET http://{esp32_ip}/sensors`.
* **Relative Velocity:** The center distance sensor readings are mapped over time:
  $$\Delta d = d_{\text{current}} - d_{\text{previous}}, \quad \Delta t = t_{\text{current}} - t_{\text{previous}}$$
  $$v_{\text{rel\_raw}} = \frac{\Delta d}{\Delta t}$$
  $$v_{\text{rel}} = 0.3 \times v_{\text{rel\_raw}} + 0.7 \times v_{\text{rel\_prev}}$$
  Negative $v_{\text{rel}}$ states trigger **Closing** warnings on the safety panel.

### 4. Drive Control Integration (Manual / ACC)
* **Arrow Key Capture:** In Live Feed mode, manual arrow keys are hooked with zero latency using the `keyboard` module (Up for forward, Down for reverse, Left/Right for steering). Releasing keys resets speed/turn to 0.
* **Adaptive Cruise Control (ACC):** The longitudinal controller uses a PD-controller to follow leading obstacles:
  $$v_{\text{motor}} = v_{\text{cruising}} + K_p \times e_{\text{dist}} + K_d \times v_{\text{rel}}$$
  where follow target is 0.50m, $K_p = 80.0$, and $K_d = 30.0$.
* **LLM Speed Cap:** Extracts the `"recommended_speed"` from the cached Ollama local decision payload, mapping it to the motor scale ($0-100$) and enforcing it as a ceiling.
* **Emergency Stop:** Enforces a hard speed-override to $0$ if center distance $< 0.25$m.
* **Control Signaling:** Motor commands are POSTed asynchronously to `POST http://{esp32_ip}/motor` using `{"speed": speed, "turn": turn}`.
* **Simulator Feedback:** The Pygame dashboard simulator renders animations using signed speed (scrolling the road lines direction) and active turn rates (displacing the simulated car horizontally by up to 35px).

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

## 📦 Local Folders (Excluded from Git)

The following folders are used locally for algorithm execution and model training but are excluded from Git tracking to bypass file size limits:
- **`PyFADE/`:** Fast and Deep Fog visibility analysis sub-library. Contains core pixel calculations and statistical models for weather classification, which are wrapped into `fog_density.py` and `fog_aware.py` inside the main application.
- **`Adonet/`:** Contains alternative dehazing models and configs (e.g., Fast Filter Adaptive methods) along with heavy `.pk` training weight files (such as `ots_train_ffa_3_19.pk` and `its_train_ffa_3_19.pk`) used for comparison and evaluation benchmarks.

---

## ⚙️ Testing & Verification
You can verify the pipeline logic and distance overrides using the automated test suite script:
```powershell
# Run the test verification script
.\venv\Scripts\python.exe -m unittest test_pipeline_modes.py
```
*(Or run the custom mock verify script located inside the agent scratch subdirectory to check distance outputs under both modes).*

---

## 🔄 System Workflow Summary

The **FogVision ADAS** operates on a modular frame-by-frame processing pipeline designed for low-visibility road safety:

1. **Frame Ingestion:** Frames are captured from an uploaded video (10 FPS Phase 2 playback) or live webcam stream (1 FPS locks) using [video_engine.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/video_engine.py).
2. **Fog Assessment:** Raw frames are evaluated for fog density percentage using the `pyfade` library in [fog_density.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/fog_density.py).
3. **Adaptive Dehazing:** If fog density is high ($>35\%$), a Dark Channel Prior (DCP) restoration model in [dehaze.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/dehaze.py) is applied to recover details.
4. **Lane Boundaries:** Driving lane boundaries are mapped in [lane_detection.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/lane_detection.py) to isolate in-lane targets.
5. **Visual Perception:** YOLOv8n object detection classifies road targets (vehicles, people, traffic lights). Cropped traffic signals are classified in HSV space.
6. **Distance Estimation:**
   - **Video Mode:** Relative depth mapping via the `MiDaS` model estimates distance.
   - **Live Mode (Plan A):** Physical VL53L0X distance sensors on an ESP32 microcontroller measure zone depth (left, middle, right) and compare them with MiDaS.
7. **Threat Assessment:** Relative approach speed and Time-To-Collision (TTC) are calculated in [motion_ttc.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/motion_ttc.py). Active alerts are filtered via confidence gating and alert hysteresis.
8. **Cognitive Advisory:** current driving context is structured and sent to local Ollama LLMs (e.g., `qwen3:1.7b` or `deepseek-r1:1.5b`), which output JSON safety guides spoken aloud via a threaded TTS engine ([voice_alert.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/voice_alert.py)).
