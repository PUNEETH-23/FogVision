# 🚗 FodVision - AI-Powered Road Safety Dashboard

Advanced Driver Assistance System (ADAS) using AI to detect fog density, traffic objects, and provide real-time safety recommendations.

## 🎯 Features

- **Real-time Fog Detection** - Monitor haziness and visibility levels
- **Image Dehazing** - DehazeFormer model for enhanced visibility
- **Object Detection** - YOLO-based detection of vehicles, pedestrians, traffic lights
- **Traffic Signal Recognition** - Automatic red/green/yellow detection
- **GPS Integration** - Road context and accident-prone area alerts
- **Voice Alerts** - Audio warnings for critical situations
- **AI Reasoning** - Ollama-powered LLM for intelligent recommendations
- **Live Dashboard** - Streamlit-based interactive monitoring interface

## 📁 Project Structure

```
fodVision/
├── DashBoard.py                # Main Streamlit application
├── pipeline.py                 # Core AI processing pipeline
├── object_detect.py           # YOLO object detection module
├── Density.py                 # Fog density calculation
├── dehaze.py                  # Image dehazing (DehazeFormer)
├── road_context.py            # GPS & road analysis
├── llm.py                     # Ollama LLM integration
├── voice_alert.py             # Voice alert system
├── Requirements.txt           # Python dependencies
│
├── assets/                    # Test images & samples
├── config/                    # Configuration files
│   ├── blackspots.json        # High-risk road locations
│   └── .env.example           # Environment template
├── models/                    # Model weights
│   └── yolov8n.pt            # YOLO detection model
├── docs/                      # Documentation
│   ├── QUICKSTART.md          # Quick start guide
│   ├── CODE_REVIEW.md         # Code analysis & issues
│   └── README.md              # This file
│
├── DehazeFormer/             # Dehazing models & configs
├── PyFADE/                   # Fog detection library
├── Adonet/                   # Alternative dehaze models
└── venv/                     # Python virtual environment
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies
```bash
pip install -r Requirements.txt
```

### 3. Configure Ollama
```bash
ollama serve  # In one terminal
ollama pull qwen3:1.7b  # In another terminal
```

### 4. Run Dashboard
```bash
streamlit run DashBoard.py
```

Visit `http://localhost:8501` in your browser.

## 📋 Requirements

- Python 3.8+
- CUDA 11.8+ (GPU recommended)
- Ollama for LLM inference
- 4GB+ VRAM (GPU) or 8GB+ RAM (CPU)

## 🔧 Configuration

### Models Available
- **Qwen3:1.7b** (Default) - Fast, real-time
- **DeepSeek-r1:1.5b** - Better reasoning

Switch in `llm.py`:
```python
MODEL_NAME = "qwen3:1.7b"  # or "deepseek-r1:1.5b"
```

### Video Source
- Uses `fog_video.mp4` if available
- Falls back to webcam (device 0)
- Edit `DashBoard.py` for custom source

## 📊 Pipeline Flow

```
Input Video
    ↓
[DEHAZE] - Remove haze for clarity
    ↓
[FOG DETECTION] - Calculate density
    ↓
[OBJECT DETECTION] - Find vehicles/pedestrians
    ↓
[TRAFFIC LIGHT] - Recognize signal state
    ↓
[ROAD CONTEXT] - GPS & accident zones
    ↓
[LLM DECISION] - AI recommendations
    ↓
[OUTPUT] - Dashboard & Alerts
```

## 🎨 Dashboard Features

- **Live Video Stream** - Processed frames with detections
- **Safety Metrics** - Fog density, risk level, recommended speed, FPS
- **Interactive Map** - Current location, blackspots, road info
- **Real-time Alerts** - Voice + visual warnings
- **AI Insights** - Intelligent driving recommendations

## 🔍 Key Components

### Fog Detection (`Density.py`)
- Uses PyFADE (Fast and Deep) algorithm
- Outputs: fog density, visibility level, recommended speed

### Object Detection (`object_detect.py`)
- YOLOv8 nano model (3.2M parameters)
- Detects: cars, pedestrians, cyclists, traffic lights
- Traffic light color classification (HSV-based)

### Dehaze Module (`dehaze.py`)
- DehazeFormer-S model
- Improves visibility in hazy conditions
- GPU-optimized inference

### LLM Module (`llm.py`)
- Ollama integration for offline inference
- Generates contextual driving recommendations
- No external API calls required

## 📝 Notes

- All paths are relative to project root
- Model weights auto-download on first run
- Ollama must be running for LLM features
- GPS requires location permission in browser
- Video processing: ~2-3 FPS on GPU, slower on CPU

## 📚 Documentation

- [QUICKSTART.md](docs/QUICKSTART.md) - Detailed setup guide
- [CODE_REVIEW.md](docs/CODE_REVIEW.md) - Code analysis & improvements
- [Agent.md](docs/Agent.md) - Agent customization

## ⚠️ Important

- Use responsibly - not a replacement for driver attention
- Verify alerts before acting
- Keep GPU cooled during long runs
- Monitor memory usage (GPU might hit ceiling)

## 🤝 Support

For issues or questions:
1. Check `docs/QUICKSTART.md`
2. Review `docs/CODE_REVIEW.md` for known issues
3. Verify all dependencies are installed
4. Check GPU memory with `nvidia-smi`

---

**Last Updated:** May 7, 2026  
**Python:** 3.13.1  
**GPU:** NVIDIA RTX 3050 (4GB)
