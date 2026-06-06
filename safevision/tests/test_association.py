"""
Tests for PPE-to-person association.

Focused on the two cases that actually matter: headwear above the shoulder
line (why we expand vertically), and two workers standing close together
(why we use IoU instead of centroid distance).
"""

import numpy as np
import pytest

from vision.association import associate, expand_bbox_vertical, iou, VERTICAL_EXPANSION


# ── iou ───────────────────────────────────────────────────────────────────────

def test_iou_perfect_overlap():
    box = np.array([0, 0, 100, 100], dtype=float)
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_no_overlap():
    a = np.array([0, 0, 50, 50], dtype=float)
    b = np.array([60, 60, 100, 100], dtype=float)
    assert iou(a, b) == 0.0


def test_iou_partial():
    a = np.array([0, 0, 100, 100], dtype=float)
    b = np.array([50, 0, 150, 100], dtype=float)
    # intersection = 50*100 = 5000, union = 2*10000 - 5000 = 15000
    assert iou(a, b) == pytest.approx(5000 / 15000)


# ── vertical expansion ────────────────────────────────────────────────────────

def test_expansion_extends_upward():
    # Person bbox: x1=100, y1=200, x2=200, y2=400 (height=200)
    bbox = np.array([100, 200, 200, 400], dtype=float)
    expanded = expand_bbox_vertical(bbox, VERTICAL_EXPANSION)

    assert expanded[0] == 100   # x1 unchanged
    assert expanded[2] == 200   # x2 unchanged
    assert expanded[3] == 400   # y2 (bottom) unchanged
    # y1 should move up by 30% of height = 60px
    assert expanded[1] == pytest.approx(200 - 0.30 * 200)


def test_expansion_allows_helmet_above_torso():
    """
    Worker bbox ends at y=200 (shoulder line).
    Helmet is at y=150–180 — above the torso, won't overlap without expansion.
    After expansion it should overlap.
    """
    person = np.array([100, 200, 200, 400], dtype=float)   # torso
    helmet = np.array([110, 150, 190, 185], dtype=float)   # above shoulder

    # Without expansion: no overlap
    assert iou(person, helmet) == 0.0

    expanded = expand_bbox_vertical(person, VERTICAL_EXPANSION)
    # With expansion: y1 moves to 200 - 0.3*200 = 140 — helmet is inside
    assert iou(expanded, helmet) > 0.0


# ── association ───────────────────────────────────────────────────────────────

def test_helmet_assigned_to_correct_person():
    """
    Two workers side by side. Helmet is above the left worker's head.
    Should associate to left worker, not right.
    """
    left_person  = np.array([50,  150, 150, 400], dtype=float)
    right_person = np.array([200, 150, 300, 400], dtype=float)

    # Helmet directly above left person
    helmet_det = {"bbox": np.array([60, 100, 140, 145], dtype=float),
                  "class": "helmet", "conf": 0.85}

    result = associate([left_person, right_person], [helmet_det])

    assert len(result[0]) == 1, "helmet should be associated to left person (index 0)"
    assert len(result[1]) == 0, "right person should have no PPE"
    assert result[0][0]["class"] == "helmet"


def test_ppe_on_surface_not_assigned():
    """
    A glove sitting on the floor with no person bbox nearby should be dropped.
    """
    person = np.array([50, 50, 150, 300], dtype=float)
    floor_glove = {"bbox": np.array([400, 350, 440, 390], dtype=float),
                   "class": "gloves", "conf": 0.6}

    result = associate([person], [floor_glove])
    assert len(result[0]) == 0


def test_no_persons_no_crash():
    ppe = [{"bbox": np.array([10, 10, 50, 50], dtype=float),
            "class": "no-helmet", "conf": 0.7}]
    result = associate([], ppe)
    assert result == {}


def test_no_ppe_empty_lists():
    person = np.array([50, 50, 150, 300], dtype=float)
    result = associate([person], [])
    assert result == {0: []}
