#!/usr/bin/env python3

from ultralytics import YOLO

vehicle_model = YOLO("./yolov8n.pt")
plate_model = YOLO("./detection-plate/runs/detect/train/weights/best.pt")

vehicle_model.export(
    format="coreml", project="./coreml_models/", name="vehicle_detection", exist_ok=True
)
plate_model.export(
    format="coreml", project="./coreml_models/", name="plate_detection", exist_ok=True
)
