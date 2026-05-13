```md id="a91kzc"
# 16. COMPLETE SYSTEM WORKFLOW

The system is designed as a real-time intelligent driving assistant.

The expected runtime workflow is:

Video Feed
→ Frame Capture
→ Dehazing
→ Fog Density Estimation
→ Object Detection
→ Traffic Signal Detection
→ GPS Awareness
→ Road Context Analysis
→ Rule-Based Safety Engine
→ Ollama AI Reasoning
→ Dashboard Rendering
→ Voice Alerts

All future development MUST preserve this workflow.

---

# 17. VIDEO PROCESSING WORKFLOW

The dashboard must process video continuously.

Expected behavior:

- video playback should appear smooth
- AI inference should run periodically
- dashboard should update dynamically
- GPU load must remain stable

Processing flow:

continuous video playback
+
AI analysis every 1–2 seconds

The UI must NEVER freeze during inference.

---

# 18. FRAME ANALYSIS PIPELINE

Each frame analysis cycle must follow:

1. Capture frame
2. Resize frame
3. Apply dehazing
4. Save temporary enhanced frame
5. Run fog density estimation
6. Run object detection
7. Run traffic signal detection
8. Build object metadata
9. Retrieve road context
10. Build unified context JSON
11. Apply rule engine
12. Generate alerts
13. Send context to LLM
14. Update dashboard
15. Trigger voice alerts

Do NOT skip context generation.

---

# 19. DASHBOARD BEHAVIOR RULES

Dashboard layout requirements:

LEFT PANEL:
- live video/dehaze feed
- YOLO detections
- traffic signal overlays
- fog status overlays

RIGHT PANEL:
- live GPS map
- blackspots
- current location
- road awareness

BOTTOM PANEL:
- fog density
- risk level
- recommended speed
- FPS
- detected objects
- alerts
- AI recommendations

The dashboard should look like a real ADAS console.

---

# 20. LIVE MAP REQUIREMENTS

The map must:
- use Folium/Leaflet
- display current GPS location
- display nearby blackspots
- update dynamically
- support future route overlays

Road awareness must include:
- road name
- road type
- blackspot proximity

---

# 21. OBJECT DETECTION REQUIREMENTS

The system must detect:
- cars
- trucks
- buses
- motorcycles
- pedestrians
- traffic lights

Traffic signal states:
- RED
- YELLOW
- GREEN

Detection workflow:

YOLO detection
→ crop traffic light ROI
→ HSV color classification

---

# 22. FOG ANALYSIS REQUIREMENTS

Fog analysis uses:
- PyFADE
- optional future ML models

Fog output must include:
- fog density percentage
- visibility level
- risk level
- recommended speed

Fog categories:

LOW
MEDIUM
HIGH
CRITICAL

---

# 23. RULE ENGINE REQUIREMENTS

Hard safety rules MUST exist.

The rule engine decides:
- alerts
- warnings
- recommended speed
- risk escalation

Examples:

IF fog_density > 70:
    reduce speed

IF red traffic signal:
    trigger stop warning

IF blackspot nearby:
    trigger caution alert

IF vehicle ahead in dense fog:
    trigger high-risk warning

The rule engine always overrides LLM advice.

---

# 24. LLM WORKFLOW REQUIREMENTS

The LLM receives ONLY structured context JSON.

The LLM must NOT:
- process raw video
- process raw images
- perform object detection

LLM responsibilities:
- explain risk
- summarize driving conditions
- generate driving recommendations
- produce human-friendly warnings

LLM output should remain concise.

---

# 25. VOICE ALERT WORKFLOW

Voice alerts must:
- use cooldown timers
- avoid repetition
- prioritize critical events

Voice alerts trigger from:
- rule engine
- NOT directly from YOLO

Example:

YOLO detects traffic light
→ rule engine decides danger
→ voice alert triggers

---

# 26. CONTEXT FUSION REQUIREMENTS

pipeline.py is the ONLY valid sensor fusion layer.

All modules must send data into pipeline.py.

No module should directly communicate with:
- dashboard
- LLM
- alerts

without going through pipeline.py.

pipeline.py responsibilities:
- synchronize modules
- merge outputs
- build context JSON
- coordinate inference timing

---

# 27. TIMING RULES

Required timing:

Video rendering:
~30 FPS

Object detection:
every 1–2 sec

Fog analysis:
every 2 sec

LLM reasoning:
every 5 sec

GPS update:
every 10 sec

Voice alerts:
event-driven with cooldown

Do NOT violate timing architecture.

---

# 28. MODEL LOADING RULES

Heavy models must load ONCE only.

Correct:
initialize globally

Incorrect:
reloading inside loops

Applies to:
- YOLO
- DehazeFormer
- MiDaS
- Ollama clients

---

# 29. FUTURE MODULE INTEGRATION

Future modules should integrate through pipeline.py only.

Planned modules:

- lane detection
- depth estimation
- collision prediction
- drowsiness detection
- rain detection
- driver monitoring
- route prediction
- weather API integration

All future features must remain modular.

---

# 30. TARGET SYSTEM BEHAVIOR

Final system should behave like:

a real-time intelligent AI driving assistant.

Expected user experience:

- smooth live dashboard
- real-time road awareness
- fog-aware analytics
- AI-generated driving recommendations
- contextual alerts
- stable performance on RTX 3050

The system should resemble:
ADAS + AI Copilot + Fog Safety Assistant

NOT a simple object detection demo.
```