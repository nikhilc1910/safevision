"""
Builds a demo violation clip from Construction-PPE test images.

Downloads the dataset if not already present (~178 MB, one-time).
Picks test images where the model fires violations, annotates them,
and writes a browser-playable MP4 to docs/demo_violations.mp4.

Usage:
    cd safevision/
    python make_demo_clip.py
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import DATASETS_DIR

MODEL_PATH = Path("runs/aug-freeze_final.pt")
OUT_PATH   = Path("../docs/demo_violations.mp4")   # → anx360/docs/
FPS        = 6      # slow enough to read the labels on each image
HOLD       = 36     # frames per image ≈ 6 s at 6 fps
MAX_IMAGES = 18     # cap so the clip stays under ~2 min
CONF       = 0.35   # slightly lower than prod to catch more violations on test images

VIOLATION_CLASSES = {"no_helmet", "no_gloves", "no_boots", "no_goggle"}


def _colour(label: str):
    if label in VIOLATION_CLASSES:
        return (0, 0, 220)        # red — violation
    if label == "Person":
        return (180, 180, 180)    # grey — person box
    return (50, 200, 50)          # green — PPE present


def _annotate(frame: np.ndarray, results) -> tuple[np.ndarray, bool]:
    names = results.names
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = names[int(box.cls[0])]
        conf  = float(box.conf[0])
        c = _colour(label)
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 2, y1), c, -1)
        cv2.putText(frame, text, (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    violations = sorted({
        names[int(b.cls[0])]
        for b in results.boxes
        if names[int(b.cls[0])] in VIOLATION_CLASSES
    })

    if violations:
        banner = "VIOLATION: " + ", ".join(violations)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (0, 0, 180), -1)
        cv2.putText(frame, banner, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame, bool(violations)


def _reencode_h264(src: Path, dst: Path) -> bool:
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg = get_ffmpeg_exe()
    except Exception:
        import shutil
        ffmpeg = shutil.which("ffmpeg")

    if not ffmpeg:
        print("imageio-ffmpeg not found — pip install imageio-ffmpeg for browser-compatible output")
        return False

    r = subprocess.run(
        [ffmpeg, "-y", "-i", str(src),
         "-vcodec", "libx264", "-preset", "ultrafast", "-crf", "23",
         "-movflags", "+faststart", str(dst)],
        capture_output=True,
    )
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def main():
    if not MODEL_PATH.exists():
        sys.exit(f"Model not found: {MODEL_PATH}. Copy aug-freeze_final.pt there first.")

    model = YOLO(str(MODEL_PATH))

    test_dir = DATASETS_DIR / "construction-ppe" / "images" / "test"
    if not test_dir.exists():
        print("Dataset not found locally — triggering download (~178 MB) ...")
        # val() download side-effect; suppress output
        model.val(data="construction-ppe.yaml", split="test",
                  verbose=False, save=False, plots=False)

    imgs = sorted(test_dir.glob("*.jpg"))
    print(f"Scanning {len(imgs)} test images for violations ...")

    violation_frames: list[tuple[np.ndarray, object]] = []
    for img_path in imgs:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        # resize to 1280×720 so the output isn't enormous
        frame = cv2.resize(frame, (1280, 720))
        r = model(frame, conf=CONF, verbose=False)[0]
        names = r.names
        has_v = any(names[int(b.cls[0])] in VIOLATION_CLASSES for b in r.boxes)
        if has_v:
            violation_frames.append((frame, r))
        if len(violation_frames) >= MAX_IMAGES:
            break

    print(f"{len(violation_frames)} violation images found")
    if not violation_frames:
        sys.exit("No violations detected. Check model path or lower CONF.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw = OUT_PATH.with_suffix(".raw.mp4")

    h, w = violation_frames[0][0].shape[:2]
    wr = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))

    for frame, results in violation_frames:
        ann, _ = _annotate(frame.copy(), results)
        for _ in range(HOLD):
            wr.write(ann)
    wr.release()

    print("Re-encoding to H.264 ...")
    ok = _reencode_h264(raw, OUT_PATH)
    if ok:
        raw.unlink(missing_ok=True)
        print(f"Done → {OUT_PATH}  ({OUT_PATH.stat().st_size // 1024} KB)")
    else:
        raw.replace(OUT_PATH)
        print(f"Done (mp4v fallback) → {OUT_PATH}")


if __name__ == "__main__":
    main()
