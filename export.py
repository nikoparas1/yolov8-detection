# !/usr/bin/env python3
from ultralytics import YOLO

def yolo_to_coreml(yolo_model: str):
    model = YOLO(yolo_model)
    model.export(format="coreml", nms=False)

yolo_to_coreml("./license_plate_detector.pt")
