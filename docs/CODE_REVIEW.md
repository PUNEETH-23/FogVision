# FodVision Project - Code Review & Issues Report

## 🔄 Latest Updates (May 7, 2026)
- ✅ All critical issues **RESOLVED**
- ✅ Dehaze model **ENABLED** (DehazeFormer-S)
- ✅ Ollama models configured: **Qwen3:1.7b** or **DeepSeek-r1:1.5b**
- ✅ Documentation updated with quick start guide
- ✅ Error handling and fallbacks implemented

## ✅ What's Working
- Clean project structure with modular components
- No Python syntax errors found
- GPU integration properly configured (NVIDIA RTX 3050)
- Object detection pipeline with traffic light color detection
- Fog density calculation using FADE algorithm
- **Image dehazing with DehazeFormer** ✅ ENABLED
- GPS integration with road context
- Voice alerts system
- LLM integration with Ollama (Qwen3:1.7b or DeepSeek-r1:1.5b)

---

## 🔴 Critical Issues to Fix

### 1. **Missing Dependencies in Requirements.txt**
The following packages are imported but NOT in Requirements.txt:

```
geopy                    # Used in road_context.py
pyttsx3                  # Used in voice_alert.py
ollama                   # Used in llm.py
streamlit                # Used in DashBoard.py
streamlit-folium         # Used in DashBoard.py
streamlit-js-eval        # Used in DashBoard.py
folium                   # Used in DashBoard.py
requests                 # Used in road_context.py
```

**Action Required:** Add these to Requirements.txt

---

### 2. **Model Name in llm.py** ✅ FIXED
```python
MODEL_NAME = "qwen3:1.7b"  # ✅ FIXED - Can also use "deepseek-r1:1.5b"
```

**Available Models:**
- **Qwen3:1.7b** (Recommended) - Faster, better for real-time
- **DeepSeek-r1:1.5b** - Better reasoning capability

To switch models, just edit the `MODEL_NAME` variable in `llm.py` or update your `.env` file.

---

### 3. **Model Mismatch in object_detect.py (Line 8)**
```python
model = YOLO("yolov8s.pt")  # ❌ ISSUE: Code loads 'yolov8s' but workspace has 'yolov8n.pt'
```

**Options:**
- Change to: `model = YOLO("yolov8n.pt")` (smaller, faster)
- Or download: `yolov8s.pt` if you want the small model

---

### 4. **Uninitialized road_context in pipeline.py**
When `pipeline.process()` is first called, `self.cached_road_context` is None. This causes `.get()` calls to fail on line ~155.

**Fix:** Initialize in `__init__`:
```python
def __init__(self):
    self.last_llm_time = 0
    self.last_gps_time = 0
    self.cached_road_context = {}  # Initialize as empty dict
    self.cached_llm_response = None
```

---

### 5. **Missing Video File in DashBoard.py**
```python
cap = cv2.VideoCapture("fog_video.mp4")  # ❌ File not found in workspace
```

**Options:**
- Create/add a video file: `fog_video.mp4`
- Or use webcam: `cv2.VideoCapture(0)`
- Or use test video from sample images

---

### 6. **Traffic Light Detection Issue in object_detect.py**
Traffic light color detection may not work correctly because the detected ROI might be too small. Add validation:

```python
if label == "traffic light":
    roi = frame[y1:y2, x1:x2]
    
    if roi.size != 0 and roi.shape[0] > 10 and roi.shape[1] > 10:  # Add minimum size check
        traffic_state = detect_traffic_light_color(roi)
```

---

### 7. **Missing Error Handling for GPS Access**
DashBoard.py requires GPS access but doesn't have fallback if GPS is unavailable:

```python
location = get_geolocation()
if not (location and "coords" in location):
    st.warning("Please allow GPS access")
    st.stop()
```

**Improvement:** Add mock GPS coordinates for testing:
```python
if not (location and "coords" in location):
    st.warning("Using mock GPS for testing")
    lat, lon = 28.6139, 77.2090  # Mock coordinates (Delhi)
else:
    lat = location["coords"]["latitude"]
    lon = location["coords"]["longitude"]
```

---

### 8. **Missing .env Configuration**
No `.env` file exists but might be needed for:
- API keys (OpenStreetMap, etc.)
- Ollama host/port configuration
- Model paths

---

## 🟡 Recommendations for Improvement

### Performance
1. **Cache YOLO model loading** - Currently loads on every frame
2. **Use half-precision more consistently** - Currently only in `process_frame()`
3. **Add frame skipping option** for lower-end devices

### Robustness  
1. **Add exception handling** in main loops
2. **Add logging** instead of just print statements
3. **Add configuration file** (JSON/YAML) for parameters

### Features
1. **Save alerts to log file** for accident investigation
2. **Add database** to store traffic patterns over time
3. **Add video export** of detected hazards
4. **Add performance metrics dashboard**

### Testing
1. Create unit tests for fog detection
2. Create unit tests for traffic light detection
3. Add integration tests for full pipeline

---

## 📋 Summary of Required Actions

| Issue | Severity | Status |
|-------|----------|--------|
| Add missing dependencies | 🔴 Critical | ✅ FIXED |
| Fix model name in llm.py | 🔴 Critical | ✅ FIXED |
| Fix model mismatch in object_detect.py | 🔴 Critical | ✅ FIXED |
| Initialize road_context | 🔴 Critical | ✅ FIXED |
| Add video file or webcam source | 🔴 Critical | ✅ FIXED |
| Add GPS fallback | 🟡 Important | ✅ FIXED |
| Add error handling | 🟡 Important | ✅ FIXED |
| Enable dehaze model | 🟢 Enhancement | ✅ ENABLED |
| Create .env file | 🟡 Important | ✅ CREATED |

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


