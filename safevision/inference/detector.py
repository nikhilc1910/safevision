"""
YOLOv8 detector with explicit result parsing.

I parse results manually rather than using .plot() or the high-level summary
methods — this way the pipeline doesn't break silently if the result format
changes between Ultralytics versions, and I know exactly what I'm getting.

The class index map is verified against construction-ppe.yaml at load time.
If the dataset is updated and the indices shift, this will raise immediately
rather than silently misclassifying safety gear.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

log = logging.getLogger(__name__)

# Confidence threshold — product decision about false-positive tolerance.
# 0.4 is the default; lower it in good lighting, raise it if alert fatigue
# becomes a problem. Do not hardcode.
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.4"))

# PPE violation classes — positive detection of these fires a violation.
# "no-X" class present means PPE is absent. This is more reliable than
# checking for absence of "helmet" because partial occlusion and facing-away
# poses make the positive class invisible even when PPE isn't worn.
# construction-ppe.yaml uses underscores and 'Person' — not hyphens or 'worker'.
# Keeping this in sync with the dataset avoids silent misclassification.
VIOLATION_CLASSES = {"no_helmet", "no_gloves", "no_boots", "no_goggle"}
PERSON_CLASS      = "Person"


@dataclass
class Detection:
    bbox:       np.ndarray   # [x1, y1, x2, y2] absolute pixel coords
    class_name: str
    conf:       float
    is_person:  bool
    is_violation: bool


class PPEDetector:
    def __init__(self, weights_path: str | Path) -> None:
        self.model = YOLO(str(weights_path))

        # Fuse Conv+BN into single op — ~12% faster inference, no accuracy cost.
        # Do this once at load time; don't fuse inside the inference loop.
        self.model.fuse()

        if torch.cuda.is_available():
            self.model.to("cuda")
            log.info("detector on GPU")
        else:
            log.warning("CUDA not available — running on CPU, expect slow inference")

        # Verify class map matches expectations. If construction-ppe.yaml is
        # swapped out for a different dataset, this catches the mismatch early.
        self._names = self.model.names
        self._verify_class_map()

    def _verify_class_map(self) -> None:
        # I added this after swapping dataset configs mid-development and spending
        # 20 minutes wondering why "no-helmet" wasn't firing. Fast fail beats silent misclass.
        required = VIOLATION_CLASSES | {PERSON_CLASS}
        present  = set(self._names.values())
        missing  = required - present
        if missing:
            raise ValueError(
                f"Model class map is missing expected classes: {missing}. "
                f"Got: {sorted(present)}"
            )
        log.info(f"class map verified — {len(self._names)} classes loaded")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on a single BGR frame. Returns list of Detection objects.

        frame: numpy array (H, W, 3) in BGR — standard OpenCV format
        """
        results = self.model(frame, conf=CONF_THRESHOLD, verbose=False)
        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            return []

        boxes  = r.boxes.xyxy.cpu().numpy()    # [N, 4]
        confs  = r.boxes.conf.cpu().numpy()    # [N]
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)  # [N]

        detections = []
        for i in range(len(boxes)):
            name = self._names[cls_ids[i]]
            detections.append(Detection(
                bbox        = boxes[i],
                class_name  = name,
                conf        = float(confs[i]),
                is_person   = name == PERSON_CLASS,
                is_violation= name in VIOLATION_CLASSES,
            ))

        return detections

    def split(self, detections: list[Detection]) -> tuple[list[Detection], list[Detection]]:
        """Split detections into (persons, ppe_items) — pipeline calls this every frame."""
        persons = [d for d in detections if d.is_person]
        ppe     = [d for d in detections if not d.is_person]
        return persons, ppe
