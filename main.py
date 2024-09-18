import os
from ultralytics import YOLO
import cv2

# List of your model files
model_files = ['trashoverflow2.pt']  # Add all your .pt files here

# Source video
source_video = 'vids n pics/Gorgeous Drone Footage of the Garbage Behind My House.mp4'

# Process with each model
for model_file in model_files:
    print(f"Processing with model: {model_file}")
    model = YOLO(model_file)
    results = model(source=source_video, show=True, conf=0.4, save=True)

    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs

        # You can add custom processing here if needed
        result.show()  # display to screen (this will show each model's results sequentially)
        print(f"Pothole detected with model {model_file}")

    print(f"Finished processing with model: {model_file}\n")