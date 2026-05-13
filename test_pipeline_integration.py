"""
Test Pipeline Integration
-------------------------
Tests that all modules work together properly.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("TESTING PIPELINE INTEGRATION")
print("=" * 60)

# Test 1: Import all modules
print("\n[1] Testing imports...")
try:
    from pipeline import ADASPipeline
    print("  - pipeline.py: OK")
except Exception as e:
    print(f"  - pipeline.py: FAILED - {e}")

try:
    from video_engine import VideoEngine
    print("  - video_engine.py: OK")
except Exception as e:
    print(f"  - video_engine.py: FAILED - {e}")

try:
    from dehaze import DehazeModel
    print("  - dehaze.py: OK")
except Exception as e:
    print(f"  - dehaze.py: FAILED - {e}")

try:
    from fog_density import estimate_fog_density
    print("  - fog_density.py: OK")
except Exception as e:
    print(f"  - fog_density.py: FAILED - {e}")

try:
    from object_detect import process_frame
    print("  - object_detect.py: OK")
except Exception as e:
    print(f"  - object_detect.py: FAILED - {e}")

try:
    from sensor_fusion import SensorFusion
    print("  - sensor_fusion.py: OK")
except Exception as e:
    print(f"  - sensor_fusion.py: FAILED - {e}")

# Test 2: Create dummy frame and test pipeline
print("\n[2] Testing pipeline with dummy frame...")
try:
    pipeline = ADASPipeline()

    # Create dummy frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add some random content
    cv2.rectangle(dummy_frame, (100, 200), (300, 400), (128, 128, 128), -1)

    # Process frame
    result = pipeline.process(dummy_frame, lat=12.9348, lon=77.6101)

    print(f"  - Pipeline process: OK")
    print(f"  - FPS: {result.get('fps', 'N/A')}")
    print(f"  - Fog density: {result.get('fog_data', {}).get('fog_density', 'N/A')}")
    print(f"  - Risk level: {result.get('risk_level', 'N/A')}")
    print(f"  - Detections: {len(result.get('detections', []))}")

except Exception as e:
    print(f"  - Pipeline process: FAILED - {e}")
    import traceback
    traceback.print_exc()

# Test 3: Test dehaze
print("\n[3] Testing dehaze...")
try:
    dehazer = DehazeModel()
    test_frame = cv2.imread("fog_impact_testing/mp_.mp4")
    if test_frame is None:
        # Create dummy
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    dehazed = dehazer.process(test_frame)
    print(f"  - Dehaze: OK (output shape: {dehazed.shape})")
except Exception as e:
    print(f"  - Dehaze: FAILED - {e}")

# Test 4: Check sensor_fusion with lidar
print("\n[4] Testing sensor_fusion...")
try:
    sf = SensorFusion()
    # Test with empty detections
    result = sf.fuse_detections([], (480, 640))
    print(f"  - SensorFusion: OK")
    print(f"  - Fusion stats: {result.get('fusion_stats', {})}")
except Exception as e:
    print(f"  - SensorFusion: FAILED - {e}")

print("\n" + "=" * 60)
print("INTEGRATION TEST COMPLETE")
print("=" * 60)