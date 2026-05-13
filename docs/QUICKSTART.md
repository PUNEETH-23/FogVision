# FodVision - Quick Start Guide

## Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU support - optional but recommended)
- Ollama (Download from https://ollama.ai)

## Step 1: Set Up Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

## Step 2: Install Dependencies
```bash
# Install required packages
pip install -r Requirements.txt

# Optional: Install PyTorch with CUDA support (for faster GPU inference)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Step 3: Configure Ollama
```bash
# Start Ollama service
ollama serve

# In another terminal, pull the model (choose one):
# Option 1: Qwen (faster, recommended for real-time)
ollama pull qwen3:1.7b

# Option 2: DeepSeek (good reasoning)
ollama pull deepseek-r1:1.5b

# Verify installation
ollama list
```

## Step 4: Set Up Configuration
```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your settings (optional)
# Or use defaults for testing
```

## Step 5: Prepare Video Source
Choose one of these options:

### Option A: Use Video File
- Place your `fog_video.mp4` in the project root
- The app will automatically detect it

### Option B: Use Webcam
- Video file is optional
- The app will fall back to webcam if `fog_video.mp4` not found

### Option C: Use Sample Images
- Create test video from included JPG files (if any)

## Step 6: Run the Dashboard
```bash
# Start Streamlit app
streamlit run DashBoard.py

# The app will open at http://localhost:8501
```

## Switching Ollama Models

To switch between different LLM models:

### Option 1: Edit llm.py directly
```python
# llm.py, line 10
MODEL_NAME = "qwen3:1.7b"      # Current (fast, recommended)
# or
MODEL_NAME = "deepseek-r1:1.5b" # Better reasoning
```

### Option 2: Use .env file
```bash
# Edit .env
OLLAMA_MODEL=deepseek-r1:1.5b
```

**Model Comparison:**
| Model | Speed | Reasoning | VRAM | Recommended For |
|-------|-------|-----------|------|-----------------|
| Qwen3:1.7b | ⚡ Fast | Good | ~3GB | Real-time driving |
| DeepSeek-r1:1.5b | 🔄 Slower | ⭐ Excellent | ~4GB | Complex decisions |

**Performance Notes:**
- Qwen3:1.7b: ~50-100ms per inference
- DeepSeek-r1:1.5b: ~200-400ms per inference (running at 5-sec intervals, not a bottleneck)

## Testing Without GPS
If you don't have GPS access or want to test:
- The app uses mock GPS coordinates (Delhi, India) as fallback
- Edit coordinates in DashBoard.py if needed

## Troubleshooting

### Issue: Module not found errors
```bash
# Reinstall dependencies
pip install -r Requirements.txt --upgrade
pip install -e .
```

### Issue: CUDA/GPU not detected
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Run on CPU instead (slower)
# Edit object_detect.py and change device=0 to device='cpu'
```

### Issue: Ollama connection failed
```bash
# Make sure Ollama service is running
ollama serve

# Check connection
ollama list
```

### Issue: Camera/Video not opening
```bash
# Try webcam (device 0)
# Or provide full path to video file
```

### Issue: Out of memory errors
- Use smaller YOLO model: `yolov8n.pt` (already configured)
- Reduce frame size in pipeline.py
- Close other applications

## Project Structure

```
fodVision/
├── DashBoard.py              # Main Streamlit application
├── pipeline.py               # AI processing pipeline
├── object_detect.py          # YOLO object detection
├── Density.py                # Fog density calculation
├── road_context.py           # GPS & road analysis
├── llm.py                    # Ollama LLM integration
├── voice_alert.py            # Voice alert system
├── dehaze.py                 # Image dehazing (optional)
├── Requirements.txt          # Python dependencies
├── .env.example              # Configuration template
├── blackspots.json           # High-risk road locations
├── yolov8n.pt                # YOLO model weights
├── DehazeFormer/             # Dehazing model
├── PyFADE/                   # Fog detection library
└── Adonet/                   # Alternative dehaze model
```

## Key Features
- ✅ Real-time fog/haze detection
- ✅ **Image dehazing** (DehazeFormer enabled)
- ✅ Object detection (vehicles, traffic lights, pedestrians)
- ✅ Traffic signal recognition
- ✅ GPS-based road hazard detection
- ✅ Voice alerts for drivers
- ✅ AI-powered decision making (Qwen3 or DeepSeek LLM)
- ✅ Live map display
- ✅ Performance metrics

## Next Steps
1. Start with `streamlit run DashBoard.py`
2. Allow GPS access when prompted
3. Point camera at road
4. Check console for alerts and warnings
5. Review metrics in dashboard

## Performance Tips
- GPU with CUDA is 5-10x faster than CPU
- Use `yolov8n.pt` for real-time processing
- Reduce frame processing frequency for lower-end hardware
- Enable caching in pipeline (already implemented)

## Support
Check `CODE_REVIEW.md` for detailed analysis and recommendations.
