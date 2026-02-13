# project1_basic.py
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

# Load image as NUMPY ARRAY first
img = cv2.imread("image.png")  # Returns numpy array (H,W,C)

# Now pass numpy array
results = model(img, save=True,
                project="./output/",
                show=True)
print("Results saved to runs/detect/predict/")