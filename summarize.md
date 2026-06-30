# FogVision ADAS: Project File-by-File Summary

This document provides a comprehensive, file-by-file breakdown of the **FogVision ADAS (Advanced Driver Assistance System)** codebase. The system is designed to run real-time video analytics on standard video files or live network streams (such as ESP32 Cam servers) at 1 frame per second, providing fog dehazing, object tracking, time-to-collision alerts, risk assessment, and AI-powered recommendations.

---

## Table of Contents
1. [Core Dashboard & Controller Layer](#1-core-dashboard--controller-layer)
2. [Vision & Preprocessing Layer](#2-vision--preprocessing-layer)
3. [Object Perception Layer](#3-object-perception-layer)
4. [Distance & Sensor Fusion Layer](#4-distance--sensor-fusion-layer)
5. [Dynamics & Risk Analytics Layer](#5-dynamics--risk-analytics-layer)
6. [AI Reasoning & Feedback Layer](#6-ai-reasoning--feedback-layer)
7. [System Utilities](#7-system-utilities)

---

## 1. Core Dashboard & Controller Layer

### 📄 [DashBoard.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/DashBoard.py)
* **Role:** Primary User Interface & Execution Loop
* **Key Features:**
  * Uses Streamlit to build a responsive light-theme dashboard.
  * Displays a dual input selector: **Upload Video** (with constant `20.0m` distance constraints) and **Live Feed** (integrates with local webcams or DroidCam/ESP32 Cam streams).
  * Renders a live processed frame overlay (dehazed, annotated with bounding boxes, active tracking IDs, and road lane boundaries).
  * Embeds a geospatial map using `folium` displaying vehicle location and nearby red-highlighted accident blackspot hazard zones.
  * Visualizes real-time metrics including fog density, visibility level, recommended speeds, pipeline FPS, and nearest hazard distance.
  * Injects a flashing full-screen header banner during high-risk scenarios and renders detailed LLM driving advisories.

### 📄 [pipeline.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/pipeline.py)
* **Role:** Main LLD Processing Orchestrator
* **Key Features:**
  * Coordinates frame processing step-by-step: resizes frame, estimates and smoothes fog density, applies dynamic dehazing contrast enhancement, masks the dashboard out, runs object detection, maps track IDs, estimates distance (dynamically or locks at `20.0m` for video mode), fuses LiDAR/camera readings, analyzes TTC threat vectors, filters objects in/out of lanes, computes risk score, and invokes LLM advice.
  * Utilizes time-gating intervals for computationally heavy components (queries the local LLM daemon every 5 seconds and updates road context/geospatial geocodes every 10 seconds).

### 📄 [video_engine.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/video_engine.py)
* **Role:** Frame Acquisition Wrapper
* **Key Features:**
  * Implements frame sampling contracts: extracts exactly 1 frame per source-second (skipping $FPS - 1$ frames in videos) to match LLD efficiency requirements.
  * Supports local video file paths and live feeds (integers for webcams or HTTP/HTTPS strings for ESP32 Cam/IP streams).
  * Enforces a wall-clock guard for live feeds to prevent rendering faster than 1 FPS, and disables loopbacks for live streams.

---

## 2. Vision & Preprocessing Layer

### 📄 [roi_mask.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/roi_mask.py)
* **Role:** Region-of-Interest Filter
* **Key Features:**
  * Generates a trapezoidal mask filtering out the bottom 20% of the image (configurable via `roi_bottom_ratio = 0.80`) to ignore vehicle hood reflections and self-car dashboard detections.
  * Filters raw YOLO detection arrays, keeping only objects whose bounding box centers lie within the active trapezoid.

### 📄 [fog_density.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/fog_density.py)
* **Role:** Fog Quantification
* **Key Features:**
  * Estimates local fog presence in the image using the Dark Channel Prior (DCP) algorithm. 
  * Analyzes standard transmission maps to output a raw density percentage.

### 📄 [dehaze.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/dehaze.py)
* **Role:** Image Restoration
* **Key Features:**
  * Implements a DCP-based dehazing algorithm.
  * Estimates global atmospheric light and computes transmission matrices.
  * Recovers scene radiance to generate enhanced, clear output frames under heavy haze/fog.

### 📄 [fog_aware.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/fog_aware.py)
* **Role:** Visibility Adaptive Controller
* **Key Features:**
  * Adapts pipeline parameters dynamically based on fog.
  * Automatically raises detection confidence thresholds (NMS) under heavy fog to prevent false positive detections.
  * Adjusts detection confidence score multipliers and determines whether to apply fast contrast enhancement (CLAHE) or full DCP dehazing.

---

## 3. Object Perception Layer

### 📄 [object_detect.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/object_detect.py)
* **Role:** YOLOv8 Object Detection & Traffic Signal Recognition
* **Key Features:**
  * Loads a pre-trained `yolov8n.pt` model to run detection on enhanced frames, filtering results by confidence.
  * Detects traffic lights and isolates their bounding boxes, converting the region into HSV space to categorize active signals (RED, GREEN, YELLOW, or UNKNOWN).

### 📄 [object_tracker.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/object_tracker.py)
* **Role:** Temporal Track Association
* **Key Features:**
  * Implements an IOU-based multi-object tracker that correlates detection bounding boxes across frames.
  * Tracks hits and ages, requiring at least 3 hits (`min_hits = 3`) to confirm and display an active track.
  * Computes object velocity vectors in pixels and stores historical motion paths.

### 📄 [red_glow.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/red_glow.py)
* **Role:** Brake-Light Indicator Detection
* **Key Features:**
  * Scans bounding boxes of preceding vehicles (cars, trucks, buses) for red-glowing light patches.
  * Helps identify brake lights or active red hazards ahead, feeding alert indicators to the risk evaluation engine.

### 📄 [lane_detection.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/lane_detection.py)
* **Role:** Road Lane Alignment
* **Key Features:**
  * Defines a constant lane boundary trapezoid representing the primary front lane and peripheral lane edges.
  * Evaluates if detected object coordinates fall inside the active driving lane.

---

## 4. Distance & Sensor Fusion Layer

### 📄 [improved_distance.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/improved_distance.py)
* **Role:** Camera-Based Distance Estimator
* **Key Features:**
  * Computes monocular depth estimates using camera pinhole geometry and class-specific height calibrations (e.g., standard heights for cars, trucks, pedestrians, and traffic signs).
  * Formulates depth confidence scores based on bounding box aspect ratios and height positions.

### 📄 [lidar_sensor.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/lidar_sensor.py)
* **Role:** Physical Sensor Interface
* **Key Features:**
  * Simulates/manages a connections interface to physical LiDAR sensor modules.
  * Provides algorithms to map 3D point cloud scans back to camera bounding boxes and extract precise spatial distances.

### 📄 [sensor_fusion.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/sensor_fusion.py)
* **Role:** Camera-LiDAR Data Merger
* **Key Features:**
  * Fuses camera detections with LiDAR distance outputs.
  * Automatically falls back to monocular camera-based estimations if no hardware LiDAR sensor is connected.

---

## 5. Dynamics & Risk Analytics Layer

### 📄 [motion_ttc.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/motion_ttc.py)
* **Role:** Dynamics Analyzer
* **Key Features:**
  * Computes relative velocity in meters per second using track displacements and ego-vehicle speed.
  * Calculates Time-to-Collision (TTC) in seconds.
  * Categorizes threats (e.g., critical if $TTC < 1.5s$).

### 📄 [risk_alerts.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/risk_alerts.py)
* **Role:** Hazard Validator
* **Key Features:**
  * Monitors targets located inside the ego lane.
  * Assesses threat levels based on warning/critical distance limits and current TTC.

### 📄 [alert_hysteresis.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/alert_hysteresis.py)
* **Role:** Alert Debouncer
* **Key Features:**
  * Prevents rapid alert toggling (oscillation) by enforcing safety warning cooldown periods (e.g., 5 seconds) and critical cooldowns (e.g., 10 seconds).

### 📄 [confidence_gated_alerts.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/confidence_gated_alerts.py)
* **Role:** False Positive Filter
* **Key Features:**
  * Gates safety alerts, ensuring objects are tracked across multiple frames with high confidence before warnings are dispatched.

### 📄 [temporal_fog_predictor.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/temporal_fog_predictor.py)
* **Role:** Weather Trend forecaster
* **Key Features:**
  * Uses temporal exponential moving average (EMA) smoothing of fog density logs to predict future visibility trends.

### 📄 [risk_score.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/risk_score.py)
* **Role:** Integrated Risk Engine
* **Key Features:**
  * Combines various metrics (fog density, target distance, speed deviations, nearby blackspots, brake light states) to calculate a unified risk score from `0` to `100` and assigns warning thresholds.

---

## 6. AI Reasoning & Feedback Layer

### 📄 [llm.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/llm.py)
* **Role:** AI Recommendation System
* **Key Features:**
  * Periodically invokes a local Ollama daemon (defaults to model `qwen3:1.7b` or `deepseek-r1:1.5b`) to synthesize driving advisories based on the driving environment context.
  * Implements a deterministic rule-based safety decision engine fallback on service errors, ensuring continuous operations.

### 📄 [scene_memory_buffer.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/scene_memory_buffer.py)
* **Role:** Temporal Memory Buffer
* **Key Features:**
  * Maintains a moving window history of recent frames and detections.
  * Supplies historical sequence summaries to the LLM.

### 📄 [voice_alert.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/voice_alert.py)
* **Role:** Audio Feedback
* **Key Features:**
  * Runs a background thread using `pyttsx3` to announce warnings without locking the main UI.

---

## 7. System Utilities

### 📄 [evaluation_metrics.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/evaluation_metrics.py)
* **Role:** Performance Profiler
* **Key Features:**
  * Measures processing latency in milliseconds across the pipeline stages.
  * Computes overall latency, frame processing rates, and serializes profiling reports.

### 📄 [road_context.py](file:///c:/Users/Puneeth%20Kumar/OneDrive/Desktop/IDP/Project/fodVision/road_context.py)
* **Role:** Geospatial Lookup
* **Key Features:**
  * Simulates/queries reverse-geolocation APIs to fetch current road names and locate nearby accident-prone zones (blackspots).

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

