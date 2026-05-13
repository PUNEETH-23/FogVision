import cv2
import numpy as np
import tempfile
from pathlib import Path

prototxt_template_path = Path("test/test_template.prototxt")
caffemodel_path = Path("AOD_Net.caffemodel")
image_path = Path("test_frames/kaka.png")

# 1. Load and prepare the frame.
frame = cv2.imread(str(image_path))
if frame is None:
    raise FileNotFoundError(f"Could not read input image: {image_path}")
h, w = frame.shape[:2]

# 2. Render the Caffe deploy prototxt with the current image size.
prototxt = prototxt_template_path.read_text().format(height=h, width=w)
with tempfile.NamedTemporaryFile("w", suffix=".prototxt", delete=False) as tmp:
    tmp.write(prototxt)
    prototxt_path = tmp.name

# 3. Load the original Caffe model via OpenCV. No Caffe installation required.
net = cv2.dnn.readNetFromCaffe(prototxt_path, str(caffemodel_path))

# (Optional) Tell OpenCV to use your NVIDIA GPU if available.
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Convert the image to a float32 blob. AOD-Net expects values between 0 and 1.
blob = cv2.dnn.blobFromImage(frame, 1.0/255.0, (w, h), (0, 0, 0), swapRB=False, crop=False)

# 4. Run the forward pass.
net.setInput(blob)
output = net.forward()

# 5. Post-process the output from (1, 3, H, W) back to (H, W, 3).
dehazed_frame = output[0].transpose(1, 2, 0)

# Clip values to ensure they stay in valid image bounds [0, 1].
dehazed_frame = np.clip(dehazed_frame, 0.0, 1.0)

# Convert back to standard 8-bit image [0, 255].
final_image = (dehazed_frame * 255).astype(np.uint8)

# 6. Save the result.
cv2.imwrite("original_caffe_output.jpg", final_image)
print("Dehazed frame saved successfully!")
