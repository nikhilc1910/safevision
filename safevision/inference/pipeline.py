"""
Frame-by-frame inference pipeline.

Wires together detector → association → zone checks → event classification.
Designed to be called from the Streamlit dashboard or from the command line.
Stateful across frames (dwell counters, dedup window) — create one Pipeline
instance per video/stream, don't reuse across sessions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import shutil
import subprocess

import cv2
import numpy as np

from inference.detector import PPEDetector
from vision.association import associate
from vision.events import EventClassifier, ViolationEvent
from vision.zones import ZoneMonitor

log = logging.getLogger(__name__)

# Skip every N-1 frames for inference — running the model at full 30 FPS is
# wasteful and the dwell threshold already handles transient detections.
# At FRAME_SKIP=2, effective inference rate is ~15 FPS on most footage.
FRAME_SKIP = 2


def _reencode_h264(src: Path, dst: Path) -> bool:
    """
    Re-encode src (mp4v) to H.264 at dst.
    Tries imageio-ffmpeg's bundled binary first (no system install needed),
    then falls back to system ffmpeg. Returns True on success.
    """
    ffmpeg_bin: str | None = None

    # imageio-ffmpeg bundles its own binary — preferred on Windows
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg_bin = get_ffmpeg_exe()
    except Exception:
        pass

    if not ffmpeg_bin:
        ffmpeg_bin = shutil.which("ffmpeg")

    if not ffmpeg_bin:
        log.warning("no ffmpeg found — annotated video may not play in browser; pip install imageio-ffmpeg")
        return False

    result = subprocess.run(
        [
            ffmpeg_bin, "-y",
            "-i", str(src),
            "-vcodec", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-movflags", "+faststart",   # web-optimised: moov atom at front
            str(dst),
        ],
        capture_output=True,
    )
    if result.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
        log.info(f"re-encoded to H.264 ({dst.stat().st_size // 1024} KB)")
        return True

    log.warning(f"H.264 re-encode failed: {result.stderr.decode()[:300]}")
    dst.unlink(missing_ok=True)
    return False


@dataclass
class FrameResult:
    frame_idx:  int
    annotated:  np.ndarray      # BGR frame with boxes + zone overlays drawn
    violations: list[ViolationEvent] = field(default_factory=list)
    person_count: int = 0


class Pipeline:
    def __init__(
        self,
        weights_path: str | Path,
        zone_config: list[dict] | None = None,
        camera_id: str = "cam0",
    ) -> None:
        self.detector   = PPEDetector(weights_path)
        self.zones      = ZoneMonitor()
        self.events     = EventClassifier()
        self.camera_id  = camera_id

        if zone_config:
            for z in zone_config:
                self.zones.add_zone(
                    zone_id=z["zone_id"],
                    points=z["points"],
                    label=z.get("label", ""),
                )

        self._frame_idx   = 0
        self._last_annotated: np.ndarray | None = None   # carry last annotation forward

    def run_frame(self, frame: np.ndarray) -> FrameResult:
        """Process one BGR frame. Returns annotated frame + any violation events."""
        self._frame_idx += 1

        # On skipped frames re-use the last annotation to avoid flickering.
        # Inference still runs at ~15 FPS; display is smooth.
        if self._frame_idx % FRAME_SKIP != 0:
            annotated = self._last_annotated if self._last_annotated is not None else frame.copy()
            return FrameResult(frame_idx=self._frame_idx, annotated=annotated)

        detections = self.detector.detect(frame)
        persons, ppe_items = self.detector.split(detections)

        # Associate PPE to persons by expanded IoU
        ppe_as_dicts = [
            {"bbox": d.bbox, "class": d.class_name, "conf": d.conf}
            for d in ppe_items
        ]
        person_bboxes = [d.bbox for d in persons]
        association   = associate(person_bboxes, ppe_as_dicts)

        all_violations: list[ViolationEvent] = []
        now = time.time()

        for person_idx, person_det in enumerate(persons):
            cx = float((person_det.bbox[0] + person_det.bbox[2]) / 2)
            cy = float((person_det.bbox[1] + person_det.bbox[3]) / 2)

            triggered_zones = self.zones.update(person_idx, (cx, cy))

            # No zones configured → full frame is the monitored area.
            # This is the expected state for single-camera demo use.
            if not triggered_zones and not self.zones.has_zones():
                triggered_zones = ["full_frame"]

            for zone_id in triggered_zones:
                events = self.events.classify(
                    camera_id      = self.camera_id,
                    zone_id        = zone_id,
                    person_id      = person_idx,
                    ppe_detections = association[person_idx],
                    now            = now,
                )
                all_violations.extend(events)

        annotated = self._draw(frame.copy(), persons, ppe_items, all_violations)
        self._last_annotated = annotated

        return FrameResult(
            frame_idx    = self._frame_idx,
            annotated    = annotated,
            violations   = all_violations,
            person_count = len(persons),
        )

    @staticmethod
    def _label(frame: np.ndarray, text: str, x1: int, y1: int, box_color: tuple) -> None:
        """Dark-background label chip — readable on any background."""
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.62
        thickness  = 2
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        pad = 5
        lx1, ly1 = x1, max(y1 - th - 2 * pad, 0)
        lx2, ly2 = x1 + tw + 2 * pad, y1
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (15, 15, 15), -1)   # near-black fill
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), box_color, 2)        # coloured border
        cv2.putText(frame, text, (lx1 + pad, ly2 - pad),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    @staticmethod
    def _draw_legend(frame: np.ndarray) -> None:
        """Bottom-right corner legend — tells viewer what colors mean at a glance."""
        h, w = frame.shape[:2]
        entries = [
            ((0, 0, 220),   "missing PPE"),   # red
            ((220, 160, 0), "PPE present"),   # amber
        ]
        font   = cv2.FONT_HERSHEY_SIMPLEX
        fs     = 0.45
        pad    = 8
        lh     = 20   # row height
        box_w  = 160
        box_h  = pad * 2 + lh * len(entries)
        bx     = w - box_w - 10
        by     = h - box_h - 10
        cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), (20, 20, 20), -1)
        cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), (50, 50, 50), 1)
        for i, (color, label) in enumerate(entries):
            y = by + pad + i * lh + lh // 2
            cv2.rectangle(frame, (bx + pad, y - 6), (bx + pad + 14, y + 6), color, -1)
            cv2.putText(frame, label, (bx + pad + 20, y + 4),
                        font, fs, (200, 200, 200), 1, cv2.LINE_AA)

    def _draw(
        self,
        frame:      np.ndarray,
        persons:    list,
        ppe_items:  list,
        violations: list[ViolationEvent],
    ) -> np.ndarray:
        violation_types = {v.vtype for v in violations}

        VIOLATION_COLOR = (0, 0, 220)     # red   — missing PPE
        PPE_OK_COLOR    = (220, 160, 0)   # amber — PPE present

        BOX_THICKNESS = 3

        # Person boxes skipped — full-body outlines clutter the frame.
        # The useful signal is PPE presence/absence, not person location.

        for det in ppe_items:
            # "none" is a dataset catch-all class — not meaningful to display
            if det.class_name == "none":
                continue
            x1, y1, x2, y2 = det.bbox.astype(int)
            color = VIOLATION_COLOR if det.is_violation else PPE_OK_COLOR
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

            # Violation boxes get a ⚠ prefix so they read as warnings even in small text
            prefix = "⚠ " if det.is_violation else ""
            label  = f"{prefix}{det.class_name}  {int(det.conf * 100)}"
            self._label(frame, label, x1, y1, color)

        self.zones.draw(frame)

        # Violation banner — only fires when zone + dwell threshold met
        if violation_types:
            msg = "  ⚠ VIOLATION: " + "  |  ".join(sorted(violation_types))
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (0, 0, 180), -1)
            cv2.putText(frame, msg, (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        self._draw_legend(frame)

        return frame

    def run_video(
        self,
        video_path: str | Path,
        output_path: str | Path | None = None,
        progress_cb=None,
    ) -> list[ViolationEvent]:
        """
        Process a video file end-to-end. Optionally write annotated output.
        progress_cb(frame_idx, total_frames) is called each frame if provided.

        Returns list of all ViolationEvents fired across the full video.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 15.0
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Always write mp4v (most portable writer), then re-encode to H.264 for browser playback.
        # OpenCV's H.264 writer is unreliable across builds; imageio-ffmpeg handles the encode step.
        writer = None
        raw_path: Path | None = None
        if output_path:
            output_path = Path(output_path)
            raw_path = output_path.with_suffix(".raw.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(raw_path), fourcc, fps, (w, h))

        all_violations: list[ViolationEvent] = []
        idx = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                result = self.run_frame(frame)
                all_violations.extend(result.violations)

                if writer:
                    writer.write(result.annotated)

                if progress_cb:
                    progress_cb(idx, total)

                idx += 1
        finally:
            cap.release()
            if writer:
                writer.release()

        if raw_path and raw_path.exists():
            ok = _reencode_h264(raw_path, output_path)
            if not ok:
                # ffmpeg unavailable — rename the mp4v file; won't play in browser but caller gets something
                raw_path.replace(output_path)
                log.warning("serving mp4v fallback — install imageio-ffmpeg for browser-compatible output")
            else:
                raw_path.unlink(missing_ok=True)

        log.info(f"processed {idx} frames — {len(all_violations)} violations")
        return all_violations


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    parser = argparse.ArgumentParser(description="Run SafeVision on a video file")
    parser.add_argument("video",   help="Input video path")
    parser.add_argument("weights", help="Model weights .pt path")
    parser.add_argument("--zones", help="JSON file with zone config", default=None)
    parser.add_argument("--out",   help="Annotated output video path", default=None)
    args = parser.parse_args()

    zones = None
    if args.zones:
        with open(args.zones) as f:
            zones = json.load(f)

    pipe = Pipeline(args.weights, zone_config=zones)
    violations = pipe.run_video(args.video, output_path=args.out)

    print(f"\n{len(violations)} violation event(s) detected")
    for v in violations:
        print(f"  {v.vtype:20s}  zone={v.zone_id}  conf={v.confidence:.2f}")
