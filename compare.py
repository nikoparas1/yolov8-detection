# !/usr/bin/env python3
import os
from typing import List, Tuple
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from open_image_models import LicensePlateDetector

CONF_THRESH = 0.25  # same default you’re using
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS = 1
LINE_HEIGHT = 15  # pixels between lines in header

PALETTE = [
    (0, 255, 0),  # green
    (0, 0, 255),  # red
    (255, 0, 0),  # blue
    (0, 255, 255),  # yellow
    (255, 255, 0),  # cyan
]


def draw_boxes(
    img: cv2.Mat,
    boxes: List[Tuple[float, float, float, float]],
    scores: List[float],
    color: Tuple[int, int, int],
    conf_thresh: float = CONF_THRESH,
):
    h0, w0 = img.shape[:2]
    for (xc, yc, w, h), score in zip(boxes, scores):
        if score < conf_thresh:
            continue
        x1 = int(xc - w / 2)
        y1 = int(yc - h / 2)
        x2 = int(xc + w / 2)
        y2 = int(yc + h / 2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w0 - 1, x2), min(h0 - 1, y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            f"{score:.2f}",
            (x1, y1 - 6),
            FONT,
            FONT_SCALE,
            color,
            THICKNESS,
            cv2.LINE_AA,
        )


def draw_multi_header(img: cv2.Mat, scores_list: List[List[float]], labels: List[str]):
    """Draw one line per model: <label>: <count> [scores...]"""
    h0, w0 = img.shape[:2]
    n = len(labels)
    header_h = LINE_HEIGHT * n + 4
    # white background
    cv2.rectangle(img, (0, 0), (w0, header_h), (255, 255, 255), -1)
    # draw each line
    for i, (lbl, scores) in enumerate(zip(labels, scores_list)):
        line = (
            f"{lbl}: {len(scores)} " + "[" + ", ".join(f"{s:.2f}" for s in scores) + "]"
        )
        y = LINE_HEIGHT * (i + 1)
        cv2.putText(
            img,
            line,
            (4, y),
            FONT,
            FONT_SCALE,
            PALETTE[i % len(PALETTE)],  # match box color
            THICKNESS,
            cv2.LINE_AA,
        )


def compare_models(
    models: List[str],
    conf: float = CONF_THRESH,
    iou: float = 0.45,
    inp_dir: str = "./test/images",
    out_dir: str = "test_results",
) -> List[str]:

    detectors = []
    labels = []
    for p in models:
        stem = Path(p).stem
        labels.append(stem)
        if p.endswith(".onnx"):
            detectors.append(
                LicensePlateDetector(
                    detection_model="yolo-v9-t-640-license-plate-end2end"
                )
            )
        else:
            detectors.append(YOLO(p, task="detect"))

    os.makedirs(out_dir, exist_ok=True)
    test_set = sorted(Path(inp_dir).glob("*.*"))
    mismatches = []

    for img_path in test_set:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ Could not read {img_path}")
            continue

        boxes_list = []
        scores_list = []

        # 2) run each detector, branch on its type
        for p, det in zip(models, detectors):
            if isinstance(det, LicensePlateDetector):
                # ONNX path via open_image_models
                preds = det.predict(img)  # returns list of dicts
                # convert to xywh + score arrays
                boxes = np.array(
                    [
                        [
                            (x1 + x2) / 2,  # xc
                            (y1 + y2) / 2,  # yc
                            (x2 - x1),  # w
                            (y2 - y1),  # h
                        ]
                        for (x1, y1, x2, y2) in (d.bounding_box for d in preds)
                    ]
                )

                # pull confidence instead of score
                scores = np.array([d.confidence for d in preds])
            else:
                # PyTorch or CoreML path via Ultralytics YOLO
                # note: .onxx _could_ be run here too if you wanted, but we special-case above
                res = det.predict(source=str(img_path), conf=conf, iou=iou)[0]
                boxes = res.boxes.xywh.cpu().numpy()
                scores = res.boxes.conf.cpu().numpy()

            boxes_list.append(boxes)
            scores_list.append(scores)

        # 3) draw header + boxes
        draw_multi_header(img, scores_list, labels)
        for idx, (boxes, scores) in enumerate(zip(boxes_list, scores_list)):
            draw_boxes(
                img, boxes, scores, PALETTE[idx % len(PALETTE)], conf_thresh=conf
            )

        # 4) save + record mismatches
        out_path = Path(out_dir) / img_path.name
        cv2.imwrite(str(out_path), img)
        counts = [b.shape[0] for b in boxes_list]
        if len(set(counts)) != 1:
            mismatches.append(
                f"{img_path.name}: "
                + ", ".join(f"{lbl}={cnt}" for lbl, cnt in zip(labels, counts))
            )

    return mismatches


models = [
    "./detection_models/license_plate_detector.pt",
    "./detection_models/license_plate_detector.mlpackage",
    "./detection_models/yolo-v9-t-640-license-plates-end2end.onnx",
]

mismatches = compare_models(models=models)
print(f"Mismatches: {mismatches}")
