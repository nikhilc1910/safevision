"""
Polygon zone management and dwell-threshold intrusion detection.

Single-frame positives are noise — someone walking past a zone boundary
at 15 FPS would fire a constant stream of alerts. The dwell threshold
filters these: 3 consecutive frames inside the polygon fires the event.

I used cv2.pointPolygonTest rather than a manual ray-casting check because
OpenCV's implementation handles floating-point precision on boundary cases
better than the naive version I'd write.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np

# I landed on 3 frames after trying 1 (too noisy) and 5 (catches events too late).
# At 15 FPS, 3 frames is ~0.2s — long enough to filter walk-bys, short enough to matter.
DWELL_THRESHOLD = 3   # consecutive frames before an intrusion event fires
                      # at 15 FPS: ~0.2 seconds of sustained presence
                      # configurable — raise for high-traffic zones, lower for critical ones


@dataclass
class Zone:
    zone_id:  str
    polygon:  np.ndarray   # shape (N, 1, 2), int32 — cv2.pointPolygonTest format
    label:    str = ""


@dataclass
class ZoneMonitor:
    zones: list[Zone] = field(default_factory=list)

    # dwell_counter[(person_id, zone_id)] = consecutive frames inside
    _dwell: dict[tuple, int] = field(default_factory=lambda: defaultdict(int))

    def add_zone(self, zone_id: str, points: list[tuple[int, int]], label: str = "") -> None:
        poly = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        self.zones.append(Zone(zone_id=zone_id, polygon=poly, label=label))

    def update(self, person_id: int, centroid: tuple[float, float]) -> list[str]:
        """
        Check centroid against all zones. Returns list of zone_ids where
        dwell threshold was just crossed (i.e., intrusion event fires this frame).

        person_id: frame-local index — see NOTE in association.py on ID stability.
        centroid: (cx, cy) in pixel coords
        """
        triggered = []

        for zone in self.zones:
            inside = cv2.pointPolygonTest(zone.polygon, centroid, measureDist=False) >= 0
            # measureDist=False returns +1/0/-1 only — faster than the signed distance
            # version and that's all we need here.

            key = (person_id, zone.zone_id)

            if inside:
                self._dwell[key] += 1
                if self._dwell[key] == DWELL_THRESHOLD:
                    # exactly DWELL_THRESHOLD: event fires once on threshold crossing,
                    # not every frame afterward. Counter keeps incrementing so we
                    # don't re-fire until the person exits and re-enters.
                    triggered.append(zone.zone_id)
            else:
                # reset on exit — don't decrement; partial dwell shouldn't carry over
                self._dwell[key] = 0

        return triggered

    def has_zones(self) -> bool:
        return len(self.zones) > 0

    def draw(self, frame: np.ndarray, color: tuple = (0, 165, 255)) -> np.ndarray:
        """Draw zone polygons onto frame in-place."""
        for zone in self.zones:
            cv2.polylines(frame, [zone.polygon], isClosed=True, color=color, thickness=2)
            if zone.label:
                # label at the first polygon vertex — not the centroid, but good enough
                x, y = zone.polygon[0][0]
                cv2.putText(frame, zone.label, (x, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame
