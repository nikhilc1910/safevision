"""
Quick demo runner — loads the fine-tuned model, annotates a video, writes output.
Not the full pipeline (no DB, no zones). Just inference + drawing for the demo GIF.

Usage:
    python demo_run.py <input_video> [--out output.mp4] [--conf 0.4]
"""

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

# violation classes — the ones that matter for the demo
VIOLATION_CLASSES = {"no_helmet", "no_safety_vest", "no_gloves", "no_boots", "no_goggle"}

# colours: violations in red, PPE present in green, person in grey
def _colour(label: str):
    if label in VIOLATION_CLASSES:
        return (0, 0, 220)   # red
    if label == "Person":
        return (160, 160, 160)
    return (50, 200, 50)     # green


def run(video_path: str, weights: str, out_path: str, conf: float):
    model = YOLO(weights)
    cap   = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        sys.exit(f"Cannot open {video_path}")

    fps  = cap.get(cv2.CAP_PROP_FPS) or 25
    w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out  = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    names = model.names   # id → class string

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, conf=conf, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = names[int(box.cls[0])]
            score = float(box.conf[0])
            colour = _colour(label)

            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            text = f"{label} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 2, y1), colour, -1)
            cv2.putText(frame, text, (x1 + 1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--weights", default="runs/aug-freeze_final.pt")
    p.add_argument("--out",     default="docs/demo_annotated.mp4")
    p.add_argument("--conf",    type=float, default=0.4)
    args = p.parse_args()

    run(args.video, args.weights, args.out, args.conf)
