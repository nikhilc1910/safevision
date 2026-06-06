"""
Violation classification and event deduplication.

Alert fatigue is a documented safety failure mode — operators who see 50 alerts
in an hour start ignoring them. Deduplication is a safety feature, not a
noise filter. I treat it with the same care as the detection threshold.

Dedup key encodes: camera + zone + violation type + position bucket.
Position bucket prevents the same worker at slightly different pixel positions
from generating separate events on every frame.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# I landed on 10s from factory floor intuition — workers don't sprint in and out of zones.
# If this produces duplicate alerts in practice, bump to 30s. Needs real-world calibration.
DEDUP_WINDOW_S   = 10     # seconds before the same event can re-fire

POSITION_BUCKET  = 50     # pixels — spatial resolution for dedup key

# Classes that indicate a violation when detected (positive detection = violation).
# Violation fires on presence of "no-X", not absence of "X" —
# the model trained to detect both, and absence-based logic is unreliable
# when the person is partially occluded or facing away.
VIOLATION_CLASSES = {"no_helmet", "no_gloves", "no_boots", "no_goggle"}


@dataclass
class ViolationEvent:
    camera_id:   str
    zone_id:     str
    person_id:   int
    vtype:       str           # e.g. "no-helmet"
    bbox:        tuple         # (x1, y1, x2, y2) of the PPE detection
    confidence:  float
    timestamp:   float = field(default_factory=time.time)
    frame_snap:  bytes | None = None   # JPEG bytes of the frame, set by caller if needed


class EventClassifier:
    def __init__(self) -> None:
        # key → last fire timestamp
        self._last_fired: dict[str, float] = {}

    def _pos_bucket(self, bbox: tuple) -> tuple[int, int]:
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        # snap to nearest bucket centre — floor division creates hard boundaries
        # that cause double-fires when a worker straddles the bucket edge.
        bx = round(cx / POSITION_BUCKET) * POSITION_BUCKET
        by = round(cy / POSITION_BUCKET) * POSITION_BUCKET
        return bx, by

    def _dedup_key(self, camera_id: str, zone_id: str, vtype: str, bbox: tuple) -> str:
        bx, by = self._pos_bucket(bbox)
        return f"{camera_id}:{zone_id}:{vtype}:{bx},{by}"

    def classify(
        self,
        camera_id:   str,
        zone_id:     str,
        person_id:   int,
        ppe_detections: list[dict],   # dicts with 'class', 'bbox', 'conf'
        now: float | None = None,
    ) -> list[ViolationEvent]:
        """
        Given PPE detections associated to one person in one zone,
        return ViolationEvents that pass the dedup window.

        Violation fires on positive detection of a "no-X" class only.
        """
        if now is None:
            now = time.time()

        events = []

        for det in ppe_detections:
            cls = det["class"]
            if cls not in VIOLATION_CLASSES:
                continue

            bbox = tuple(det["bbox"])
            key  = self._dedup_key(camera_id, zone_id, cls, bbox)

            last = self._last_fired.get(key, 0.0)
            if now - last < DEDUP_WINDOW_S:
                continue   # same event within dedup window — skip

            self._last_fired[key] = now
            events.append(ViolationEvent(
                camera_id  = camera_id,
                zone_id    = zone_id,
                person_id  = person_id,
                vtype      = cls,
                bbox       = bbox,
                confidence = float(det["conf"]),
                timestamp  = now,
            ))

        return events

    def clear_stale(self, max_age_s: float = 300.0) -> None:
        """Prune entries older than max_age_s. Call periodically to avoid memory growth."""
        cutoff = time.time() - max_age_s
        stale  = [k for k, t in self._last_fired.items() if t < cutoff]
        for k in stale:
            del self._last_fired[k]
