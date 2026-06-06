"""
Held-out test set evaluation with per-class quality gate.

This runs against the test split, not the val split used during training.
The distinction matters — val metrics during training are optimistic because
the LR scheduler and early stopping are implicitly tuned against them.

Quality gate lives here; train.py calls this rather than duplicating it.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ultralytics import YOLO

log = logging.getLogger(__name__)

# Per-class minimum recall thresholds — below these the checkpoint doesn't ship.
# mAP50 aggregate can look fine while recall on safety-critical classes is low —
# the aggregate masks it because there are 12 classes and most aren't critical.
RECALL_GATE = {
    "no-helmet":      0.80,
    "no-safety-vest": 0.80,
    "no-gloves":      0.75,  # slightly looser — gloves are harder to detect (small, variable color)
}


def evaluate(weights_path: str | Path, dataset_yaml: str = "construction-ppe.yaml") -> dict:
    """
    Evaluate checkpoint on test split. Prints per-class metrics and runs quality gate.

    Returns per_class dict: {class_name: {precision, recall, ap50, ap50_95}}.
    Raises AssertionError if any RECALL_GATE class is below threshold.
    """
    model = YOLO(str(weights_path))

    log.info(f"evaluating: {weights_path}")
    log.info(f"dataset: {dataset_yaml}  split: test")

    results = model.val(data=dataset_yaml, split="test", verbose=False)

    per_class = {}
    for idx, class_idx in enumerate(results.box.ap_class_index):
        name = results.names[int(class_idx)]
        per_class[name] = {
            "precision": float(results.box.p[idx]),
            "recall":    float(results.box.r[idx]),
            "ap50":      float(results.box.ap50[idx]),
            "ap50_95":   float(results.box.ap[idx]),
        }

    # I print this as a table rather than logging.info lines — easier to scan
    # when reviewing Kaggle output after a long training run.
    header = f"{'class':<22}  {'P':>6}  {'R':>6}  {'AP50':>6}  {'AP50-95':>8}"
    print("\n" + header)
    print("-" * len(header))
    for name, m in sorted(per_class.items()):
        gate_marker = " ◄" if name in RECALL_GATE else ""
        print(
            f"{name:<22}  {m['precision']:>6.3f}  {m['recall']:>6.3f}"
            f"  {m['ap50']:>6.3f}  {m['ap50_95']:>8.3f}{gate_marker}"
        )
    print()

    # Aggregate
    log.info(
        f"aggregate — mAP50: {results.box.map50:.4f}  "
        f"mAP50-95: {results.box.map:.4f}  "
        f"P: {results.box.mp:.4f}  R: {results.box.mr:.4f}"
    )

    # Quality gate
    gate_passed = True
    for cls_name, min_recall in RECALL_GATE.items():
        if cls_name not in per_class:
            log.warning(f"quality gate: '{cls_name}' missing from results — not in test set?")
            continue
        actual = per_class[cls_name]["recall"]
        if actual < min_recall:
            log.error(f"GATE FAIL: {cls_name} recall={actual:.3f} < {min_recall}")
            gate_passed = False
        else:
            log.info(f"gate pass: {cls_name} recall={actual:.3f}")

    if not gate_passed:
        raise AssertionError(
            "Quality gate failed — checkpoint not promoted. "
            "Check per-class recall table above."
        )

    log.info("quality gate passed")
    return per_class


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("weights", help="Path to .pt checkpoint")
    parser.add_argument("--data", default="construction-ppe.yaml")
    args = parser.parse_args()

    evaluate(args.weights, args.data)
