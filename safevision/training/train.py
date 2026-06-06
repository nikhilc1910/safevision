"""
Fine-tuning pipeline for SafeVision PPE detector.

Backbone freeze warm-up is the main thing here — without it, 4k images
is not enough to fine-tune from scratch without destroying the low-level
features COCO pretraining built. I learned this the hard way trying a
full fine-tune first; val loss oscillated for 20 epochs before stabilizing.
"""

import logging
import os
from pathlib import Path

import torch
import wandb
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────

DATASET_YAML    = "construction-ppe.yaml"
BASE_WEIGHTS    = "yolov8n.pt"
EPOCHS          = 100
WARMUP_EPOCHS   = 10       # freeze backbone for this many epochs
FREEZE_UNTIL    = 10       # freeze layers 0–9 (inclusive)
IMG_SIZE        = 640
BATCH_SIZE      = 16       # safe for P100 16GB; don't push to 32 without checking
LR0             = 0.01
LR_FINAL_RATIO  = 0.01     # lrf — cosine schedule ends at LR0 * lrf
MOMENTUM        = 0.937
WEIGHT_DECAY    = 5e-4
CKPT_INTERVAL   = 5        # Kaggle kills sessions mid-training, so save often

CONF_THRESHOLD  = float(os.getenv("CONF_THRESHOLD", "0.4"))

# Safety-critical classes — recall threshold must be met before promoting ckpt.
# These are the ones that matter; missing a no-helmet is a real safety failure.
RECALL_GATE = {
    "no-helmet":      0.80,
    "no-safety-vest": 0.80,
}

# On Kaggle: outputs go here. Locally: creates ./runs/safevision/
if Path("/kaggle/working").exists():
    OUTPUT_DIR = Path("/kaggle/working/safevision")
else:
    OUTPUT_DIR = Path("runs/safevision")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = OUTPUT_DIR / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def freeze_backbone(model: YOLO, freeze_until: int) -> None:
    """Freeze layers 0–(freeze_until-1). Everything from freeze_until onward trains."""
    for i, (name, param) in enumerate(model.model.named_parameters()):
        param.requires_grad = i >= freeze_until
    frozen = sum(1 for p in model.model.parameters() if not p.requires_grad)
    log.info(f"backbone frozen: {frozen} params locked (layers 0–{freeze_until - 1})")


def unfreeze_all(model: YOLO) -> None:
    for param in model.model.parameters():
        param.requires_grad = True
    log.info("backbone unfrozen — full fine-tune begins")


def check_quality_gate(model: YOLO, dataset_yaml: str) -> dict:
    """
    Evaluate on held-out test split. Returns per-class recall dict.
    Raises AssertionError if any safety-critical class is below threshold.
    """
    results = model.val(data=dataset_yaml, split="test", verbose=False)

    per_class = {}
    for idx, class_idx in enumerate(results.box.ap_class_index):
        name = results.names[int(class_idx)]
        per_class[name] = {
            "precision": float(results.box.p[idx]),
            "recall":    float(results.box.r[idx]),
            "ap50":      float(results.box.ap50[idx]),
        }

    for cls_name, min_recall in RECALL_GATE.items():
        if cls_name not in per_class:
            log.warning(f"quality gate: '{cls_name}' not in eval results — skipping")
            continue
        actual = per_class[cls_name]["recall"]
        assert actual >= min_recall, (
            f"quality gate failed: {cls_name} recall={actual:.3f} < {min_recall}"
        )
        log.info(f"quality gate passed: {cls_name} recall={actual:.3f}")

    return per_class


# ── WandB callbacks ───────────────────────────────────────────────────────────

def make_callbacks(run) -> dict:
    """Returns per-epoch and end-of-training hooks for WandB logging."""

    def on_train_epoch_end(trainer):
        metrics = trainer.metrics or {}
        # Log components separately — if cls_loss dominates early it signals
        # class imbalance, not a learning rate problem.
        run.log({
            "epoch":             trainer.epoch,
            "train/box_loss":    trainer.loss_items[0] if trainer.loss_items is not None else None,
            "train/cls_loss":    trainer.loss_items[1] if trainer.loss_items is not None else None,
            "train/dfl_loss":    trainer.loss_items[2] if trainer.loss_items is not None else None,
            "metrics/mAP50":     metrics.get("metrics/mAP50(B)"),
            "metrics/mAP50-95":  metrics.get("metrics/mAP50-95(B)"),
            "metrics/precision": metrics.get("metrics/precision(B)"),
            "metrics/recall":    metrics.get("metrics/recall(B)"),
            "lr":                trainer.optimizer.param_groups[0]["lr"],
        })

    def on_train_end(trainer):
        run.summary["best_mAP50"] = trainer.best_fitness
        log.info(f"training done — best mAP50: {trainer.best_fitness:.4f}")

    return {
        "on_train_epoch_end": on_train_epoch_end,
        "on_train_end":       on_train_end,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def train(run_name: str = "aug-freeze") -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — check Kaggle GPU settings")

    log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    log.info(f"run: {run_name}  epochs: {EPOCHS}  img_size: {IMG_SIZE}")

    model = YOLO(BASE_WEIGHTS)

    # I log ultralytics version as a dataset proxy — construction-ppe.yaml
    # doesn't have its own version field, so the library version is the
    # closest thing to pinning what data was actually used.
    wandb_run = wandb.init(
        project="safevision-ppe",
        name=run_name,
        config={
            "model":           BASE_WEIGHTS,
            "dataset":         DATASET_YAML,
            "epochs":          EPOCHS,
            "warmup_epochs":   WARMUP_EPOCHS,
            "img_size":        IMG_SIZE,
            "batch_size":      BATCH_SIZE,
            "lr0":             LR0,
            "lrf":             LR_FINAL_RATIO,
            "momentum":        MOMENTUM,
            "weight_decay":    WEIGHT_DECAY,
            "conf_threshold":  CONF_THRESHOLD,
            "ultralytics_ver": __import__("ultralytics").__version__,
        },
    )

    # Attach per-epoch callbacks
    for event, fn in make_callbacks(wandb_run).items():
        model.add_callback(event, fn)

    # ── warm-up phase: frozen backbone ────────────────────────────────────────
    log.info(f"warm-up phase: {WARMUP_EPOCHS} epochs with frozen backbone")
    freeze_backbone(model, FREEZE_UNTIL)

    model.train(
        data=DATASET_YAML,
        epochs=WARMUP_EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        lr0=LR0,
        lrf=LR_FINAL_RATIO,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        project=str(OUTPUT_DIR),
        name="warmup",
        exist_ok=True,
        verbose=False,
    )

    # Save warm-up checkpoint before unfreezing
    warmup_ckpt = CKPT_DIR / "after_warmup.pt"
    torch.save(model.model.state_dict(), warmup_ckpt)
    log.info(f"warm-up checkpoint: {warmup_ckpt}")

    # ── full fine-tune ─────────────────────────────────────────────────────────
    unfreeze_all(model)
    log.info(f"full fine-tune: {EPOCHS - WARMUP_EPOCHS} epochs remaining")

    remaining = EPOCHS - WARMUP_EPOCHS

    # Checkpoint every CKPT_INTERVAL epochs via a callback — Ultralytics
    # only saves best by default, which is useless if Kaggle times out at epoch 87.
    def on_epoch_end(trainer):
        ep = trainer.epoch + WARMUP_EPOCHS  # offset for display
        if (ep + 1) % CKPT_INTERVAL == 0:
            ckpt_path = CKPT_DIR / f"epoch_{ep + 1:04d}.pt"
            torch.save(model.model.state_dict(), ckpt_path)
            log.info(f"interval checkpoint: {ckpt_path}")

    model.add_callback("on_train_epoch_end", on_epoch_end)

    model.train(
        data=DATASET_YAML,
        epochs=remaining,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        lr0=LR0 * 0.1,    # reduce LR after warm-up — don't blast the unfrozen backbone
        lrf=LR_FINAL_RATIO,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        project=str(OUTPUT_DIR),
        name=run_name,
        exist_ok=True,
        verbose=False,
    )

    # ── quality gate ──────────────────────────────────────────────────────────
    log.info("running quality gate on test split")
    try:
        per_class = check_quality_gate(model, DATASET_YAML)
        wandb_run.log({"quality_gate": "passed"})
        for cls_name, m in per_class.items():
            wandb_run.log({
                f"test/{cls_name}/recall":    m["recall"],
                f"test/{cls_name}/precision": m["precision"],
                f"test/{cls_name}/ap50":      m["ap50"],
            })
    except AssertionError as e:
        log.error(f"quality gate FAILED: {e}")
        wandb_run.log({"quality_gate": "failed", "gate_reason": str(e)})
        wandb_run.finish()
        raise

    # Save final weights and log as WandB artifact
    final_path = CKPT_DIR / "final.pt"
    model.save(str(final_path))
    artifact = wandb.Artifact("ppe-detector", type="model")
    artifact.add_file(str(final_path))
    wandb_run.log_artifact(artifact)

    wandb_run.finish()
    log.info(f"done — weights at {final_path}")
    return final_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="aug-freeze",
                        help="WandB run name — use to label ablation runs")
    args = parser.parse_args()

    train(run_name=args.run_name)
