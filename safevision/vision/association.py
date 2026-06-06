"""
PPE-to-person association via expanded IoU.

I chose expanded IoU over centroid distance after centroid distance broke down
whenever two workers stood side by side — a glove detection midway between them
would get assigned to the wrong person. IoU with a vertically expanded person
bbox handles the ambiguity because spatial overlap is directional in a way
centroid proximity isn't.

The vertical expansion specifically handles headwear: a hard hat sits above the
shoulder line and won't overlap the person's torso bbox at all without it.
"""

from __future__ import annotations

import numpy as np

VERTICAL_EXPANSION   = 0.30   # expand person bbox 30% upward for headwear coverage
ASSOCIATION_IOU_THRESH = 0.10  # low threshold intentional — partial overlap is enough
                               # when workers are separated. Raise if false associations
                               # appear in crowded scenes.


def expand_bbox_vertical(bbox: np.ndarray, ratio: float) -> np.ndarray:
    """Expand bbox upward by ratio * height. Returns new [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    return np.array([x1, y1 - ratio * h, x2, y2], dtype=float)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Standard IoU between two [x1, y1, x2, y2] boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def associate(
    person_bboxes: list[np.ndarray],
    ppe_detections: list[dict],
) -> dict[int, list[dict]]:
    """
    Match each PPE detection to the best-overlapping person via expanded IoU.
    Person bboxes are expanded 30% upward before matching — handles headwear above the shoulder line.
    Detections with no overlapping person are silently dropped (equipment resting on surfaces).
    """
    # NOTE: person index here is frame-local (position in this call's person list),
    # not a persistent track ID. Workers who move between frames may get a different
    # index, which resets downstream dwell counters. Acceptable for MVP.
    # ByteTrack resolves this in v1.1.

    result: dict[int, list[dict]] = {i: [] for i in range(len(person_bboxes))}

    expanded = [expand_bbox_vertical(b, VERTICAL_EXPANSION) for b in person_bboxes]

    for det in ppe_detections:
        best_person = -1
        best_iou    = ASSOCIATION_IOU_THRESH  # must beat threshold to associate

        for i, exp_bbox in enumerate(expanded):
            score = iou(det["bbox"], exp_bbox)
            if score > best_iou:
                best_iou    = score
                best_person = i

        if best_person >= 0:
            result[best_person].append(det)
        # else: PPE with no overlapping person is silently dropped — it's
        # probably equipment resting on a surface, not a worn item.

    # TODO: handle cases where >1 PPE detection of the same class maps to the
    # same person (e.g., two "helmet" detections on one person). For now,
    # taking all matches; the violation check will just see duplicates.

    # NOTE: association accuracy degrades when >~6 workers are in frame
    # simultaneously. Expanded bboxes start overlapping and the greedy
    # best-IoU assignment can misfire. Document this in README known limitations.

    return result
