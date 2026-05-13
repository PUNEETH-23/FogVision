# Fog Impact Testing Framework

This folder contains tools to test how fog affects object detection performance.

## Overview

The framework takes a **clear/fog-free video file** as input and:
1. Extracts frames from the video
2. Applies synthetic fog at 6 different levels (0%, 20%, 40%, 60%, 80%, 100%)
3. Runs YOLO object detection on each foggy version
4. Compares detection metrics (confidence, detection count, etc.)
5. Generates comprehensive visualizations and reports

## Files

- `test_fog_impact.py` - Main testing script
- `README.md` - This file

## Requirements

Make sure these packages are installed:
```bash
pip install opencv-python numpy matplotlib
```

These should already be installed from the main project requirements.

## Usage

### Basic Usage
```bash
python test_fog_impact.py <path_to_video_file>
```

### Examples

```bash
# Test with a 30-frame sample (default)
python test_fog_impact.py "C:\path\to\clear_video.mp4"

# Test with specific number of frames
python test_fog_impact.py "C:\path\to\clear_video.mp4" --frames 50

# Save results to custom directory
python test_fog_impact.py "C:\path\to\clear_video.mp4" --output "my_test_results"
```

### Command Line Options

- `video_path` (required): Path to your clear/fog-free video file
- `--frames` (optional): Maximum frames to process (default: 30)
- `--output` (optional): Output directory for results (default: `results`)

## Input Requirements

- **Video Format**: MP4, AVI, MOV, or any format supported by OpenCV
- **Content**: Clear video (no fog) recommended as baseline
- **Resolution**: Any resolution (will be resized to 640x480)
- **Min Length**: At least 1 second of video

## Output Files

The framework generates:

### 1. **fog_impact_analysis.png**
   Four-panel visualization showing:
   - Detection confidence vs fog level
   - Total detection count vs fog level
   - Average objects per frame vs fog level
   - Confidence standard deviation vs fog level

### 2. **fog_impact_report.txt**
   Detailed text report including:
   - Test summary and baseline metrics
   - Frame-by-frame results for each fog level
   - Key findings and critical degradation points

### 3. **fog_impact_results.json**
   Raw results in JSON format for:
   - Further analysis
   - Integration with other tools
   - Data persistence

### 4. **sample_fog_*.jpg**
   Sample frames showing the foggy versions at each level:
   - `sample_fog_0.0%.jpg` - Clear frame (baseline)
   - `sample_fog_0.2%.jpg` - Light fog
   - `sample_fog_0.4%.jpg` - Moderate fog
   - `sample_fog_0.6%.jpg` - Heavy fog
   - `sample_fog_0.8%.jpg` - Dense fog
   - `sample_fog_1.0%.jpg` - Very dense fog

## Interpretation Guide

### Key Metrics

**Average Confidence**: How confident the model is in its detections
- **Good**: > 0.7 (high confidence)
- **Acceptable**: 0.5 - 0.7 (medium confidence)
- **Poor**: < 0.5 (low confidence)

**Objects per Frame**: Average number of objects detected per frame
- **Degradation indicates**: Missing detections due to fog obscuration

**Confidence Std Dev**: Variability in confidence scores
- **Low (<0.1)**: Consistent detection quality
- **High (>0.3)**: Inconsistent/unreliable detections

### Example Analysis

```
Fog Level: 0.0% (Clear)
  Average Confidence: 0.856
  Objects per Frame:  3.45

Fog Level: 0.6% (Heavy)
  Average Confidence: 0.612 (-28.5% from baseline)
  Objects per Frame:  2.10 (-39.1% from baseline)
```

This shows that at 60% fog:
- Detection confidence drops ~28%
- Detection rate drops ~39% (fewer objects found)

## Performance Considerations

- **Processing Time**: ~2-3 seconds per frame (depends on video resolution and system)
- **Memory Usage**: Moderate (~1-2 GB for 30 frames at 480p)
- **GPU**: Optional but recommended for faster processing

## Troubleshooting

### Issue: "Cannot open video"
- **Solution**: Verify the video file path is correct and file exists

### Issue: "No frames extracted"
- **Solution**: Ensure video file is valid and contains video streams

### Issue: "ImportError: No module named 'object_detect'"
- **Solution**: Run script from the `fog_impact_testing` folder or adjust PYTHONPATH

### Issue: Very slow processing
- **Solution**: Reduce `--frames` parameter to process fewer frames

## Advanced Usage

### Custom Fog Levels

Edit the `fog_levels` in `test_fog_impact.py`:
```python
self.fog_levels = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]  # Custom levels
```

### Batch Testing

Create a script to test multiple videos:
```python
import subprocess

videos = ["video1.mp4", "video2.mp4", "video3.mp4"]
for video in videos:
    subprocess.run(["python", "test_fog_impact.py", video, "--output", f"results_{video}"])
```

## Example Workflow

1. **Prepare a clear video** (without fog)
   ```bash
   # Record or obtain a clear traffic/driving video
   ```

2. **Run the test**
   ```bash
   python test_fog_impact.py "my_video.mp4" --frames 50
   ```

3. **Review results**
   ```bash
   # Charts: fog_impact_analysis.png
   # Report: fog_impact_report.txt
   # Data: fog_impact_results.json
   ```

4. **Analyze findings**
   - Identify at what fog level performance degrades
   - Plan system improvements
   - Set operational thresholds

## Future Enhancements

Potential improvements:
- Real-time fog animation preview
- Comparison between multiple videos
- Automatic threshold detection
- Integration with model retraining pipeline
- Export to CSV for spreadsheet analysis

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all input files are valid
3. Check system requirements are met
4. Review console output for error messages