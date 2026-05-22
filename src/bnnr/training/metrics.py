"""Metric aggregation utilities for training loop."""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

from bnnr.adapter import XAICapableModel
from bnnr.training.checkpoint import _is_ultralytics_tasks_backbone

if TYPE_CHECKING:
    from bnnr.trainer import BNNRTrainer


def average_metrics(all_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Average metric dicts across batches/epochs.

    Batches may omit keys (e.g. detection skips a bad batch and returns only
    ``loss`` + ``loss_non_finite``). Averaging must use the union of keys and
    only include finite values present for each key.
    """
    if not all_metrics:
        return {}
    keys: set[str] = set()
    for m in all_metrics:
        keys.update(m.keys())
    out: dict[str, float] = {}
    for k in keys:
        vals: list[float] = []
        for m in all_metrics:
            if k not in m:
                continue
            v = float(m[k])
            if math.isfinite(v):
                vals.append(v)
        if vals:
            out[k] = float(sum(vals) / len(vals))
    return out


def compute_classification_eval_details(
    preds: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict[str, dict[str, float | int]], dict[str, Any]]:
    """Per-class accuracy and confusion matrix for classification."""
    n_classes = int(max(int(np.max(preds)), int(np.max(labels)))) + 1
    per_class: dict[str, dict[str, float | int]] = {}
    for class_id in range(n_classes):
        mask = labels == class_id
        support = int(np.sum(mask))
        if support == 0:
            continue
        acc = float(np.mean(preds[mask] == labels[mask]))
        per_class[str(class_id)] = {"accuracy": acc, "support": support}
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(labels.tolist(), preds.tolist()):
        matrix[int(true_label), int(pred_label)] += 1
    confusion = {
        "labels": list(range(n_classes)),
        "matrix": matrix.tolist(),
    }
    return per_class, confusion


def compute_multilabel_eval_details(
    preds: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict[str, dict[str, float | int]], dict[str, Any]]:
    """Per-label precision/recall/f1 for multi-label classification."""
    n_labels = preds.shape[1] if preds.ndim == 2 else 0
    if n_labels == 0:
        return {}, {}

    per_label: dict[str, dict[str, float | int]] = {}
    confusion_per_label: list[dict[str, int]] = []
    for label_idx in range(n_labels):
        y_true = labels[:, label_idx]
        y_pred = preds[:, label_idx]
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        support = int(np.sum(y_true == 1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_label[str(label_idx)] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        confusion_per_label.append({"tp": tp, "fp": fp, "fn": fn})

    confusion = {
        "type": "multilabel_per_label",
        "labels": list(range(n_labels)),
        "per_label": confusion_per_label,
    }
    return per_label, confusion


def macro_mean_metric(
    per_class: dict[str, dict[str, float | int]],
    metric_key: str,
) -> float | None:
    """Return macro average of *metric_key* across per-class dict values."""
    values = [v.get(metric_key, 0.0) for v in per_class.values()]
    return float(np.mean(values)) if values else None


def collect_val_logits(
    trainer: BNNRTrainer,
    *,
    post_process: str = "argmax",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Run a forward pass over the val set and collect predictions/labels."""
    if trainer._last_eval_preds is not None and trainer._last_eval_labels is not None:
        return trainer._last_eval_preds, trainer._last_eval_labels

    if not isinstance(trainer.model, XAICapableModel):
        return None
    model_impl = trainer.model.get_model()
    device = next(model_impl.parameters()).device
    model_impl.eval()
    preds_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    with torch.no_grad():
        for raw_batch in trainer.val_loader:
            if len(raw_batch) == 3:
                images, labels_batch, _ = raw_batch
            else:
                images, labels_batch = raw_batch
            logits = model_impl(images.to(device))
            if post_process == "sigmoid":
                preds_rows.append(
                    (torch.sigmoid(logits) >= trainer.config.multilabel_threshold).int().cpu()
                )
            else:
                preds_rows.append(torch.argmax(logits, dim=1).cpu())
            label_rows.append(labels_batch.cpu())
    if not preds_rows or not label_rows:
        return None
    preds = torch.cat(preds_rows).numpy().astype(np.int64)
    labels = torch.cat(label_rows).numpy().astype(np.int64)
    return preds, labels


def compute_eval_class_details_detection(
    trainer: BNNRTrainer,
) -> tuple[dict[str, dict[str, float | int]], dict[str, Any]]:
    """Compute per-class AP for detection task."""
    from bnnr.detection_metrics import (
        calculate_detection_confusion_matrix,
        calculate_per_class_ap,
    )

    all_preds: list[dict[str, Tensor]] = getattr(trainer.model, "last_eval_preds", [])
    all_targets: list[dict[str, Tensor]] = getattr(trainer.model, "last_eval_targets", [])

    if not all_preds or not all_targets:
        all_preds = []
        all_targets = []
        model_obj = trainer.model.get_model() if hasattr(trainer.model, "get_model") else None
        if model_obj is None:
            return {}, {}

        use_ultra_fallback = _is_ultralytics_tasks_backbone(model_obj) and hasattr(
            trainer.model, "eval_step",
        )
        if use_ultra_fallback:
            trainer.model._eval_preds = []  # type: ignore[attr-defined]
            trainer.model._eval_targets = []  # type: ignore[attr-defined]
            with torch.no_grad():
                for raw_batch in trainer.val_loader:
                    if len(raw_batch) == 3:
                        images, targets, _ = raw_batch
                    else:
                        images, targets = raw_batch
                    trainer.model.eval_step((images, targets))
            all_preds = copy.deepcopy(trainer.model._eval_preds)  # type: ignore[attr-defined]
            all_targets = copy.deepcopy(trainer.model._eval_targets)  # type: ignore[attr-defined]
            trainer.model._eval_preds = []  # type: ignore[attr-defined]
            trainer.model._eval_targets = []  # type: ignore[attr-defined]
        else:
            model_obj.eval()
            with torch.no_grad():
                for raw_batch in trainer.val_loader:
                    if len(raw_batch) == 3:
                        images, targets, _ = raw_batch
                    else:
                        images, targets = raw_batch
                    device = next(model_obj.parameters()).device
                    images_list = [img.to(device) for img in images]
                    preds = model_obj(images_list)
                    for pred in preds:
                        all_preds.append({
                            "boxes": pred["boxes"].cpu(),
                            "scores": pred["scores"].cpu(),
                            "labels": pred["labels"].cpu(),
                        })
                    for target in targets:
                        all_targets.append({
                            "boxes": (
                                target["boxes"].cpu()
                                if isinstance(target["boxes"], Tensor)
                                else target["boxes"]
                            ),
                            "labels": (
                                target["labels"].cpu()
                                if isinstance(target["labels"], Tensor)
                                else target["labels"]
                            ),
                        })

    if not all_preds or not all_targets:
        return {}, {}

    class_names = trainer.config.detection_class_names
    known_classes: set[int] = set()
    for t in all_targets:
        if t.get("labels") is not None and len(t["labels"]) > 0:
            known_classes.update(int(x) for x in t["labels"].cpu().tolist())

    per_class_ap = calculate_per_class_ap(all_preds, all_targets, class_names=class_names)
    per_class: dict[str, dict[str, float | int]] = {}
    for cls_id, info in per_class_ap.items():
        per_class[cls_id] = {
            "accuracy": info["ap"],
            "ap_50": info["ap"],
            "support": info["support"],
        }

    confusion = calculate_detection_confusion_matrix(
        predictions=all_preds,
        targets=all_targets,
        num_classes=(
            (
                len(trainer.config.detection_class_names)
                if (
                    trainer.config.detection_class_names
                    and str(trainer.config.detection_class_names[0]).strip().lower()
                    in {"background", "bg", "__background__"}
                )
                else len(trainer.config.detection_class_names) + 1
            )
            if trainer.config.detection_class_names is not None
            else None
        ),
        iou_threshold=0.5,
    )

    if known_classes and confusion.get("labels"):
        allowed = known_classes | {0}
        old_labels = confusion["labels"]
        old_matrix = confusion["matrix"]
        keep_indices = [i for i, lbl in enumerate(old_labels) if lbl in allowed]
        if len(keep_indices) < len(old_labels):
            new_labels = [old_labels[i] for i in keep_indices]
            new_matrix = [
                [old_matrix[r][c] for c in keep_indices]
                for r in keep_indices
            ]
            confusion = {"labels": new_labels, "matrix": new_matrix}

    return per_class, confusion


def compute_eval_class_details(
    trainer: BNNRTrainer,
) -> tuple[dict[str, dict[str, float | int]], dict[str, Any]]:
    """Per-class eval details and confusion for the active task type."""
    if trainer._is_detection:
        return compute_eval_class_details_detection(trainer)
    if trainer._is_multilabel:
        result = collect_val_logits(trainer, post_process="sigmoid")
        if result is None:
            return {}, {}
        preds, labels = result
        return compute_multilabel_eval_details(preds, labels)
    result = collect_val_logits(trainer, post_process="argmax")
    if result is None:
        return {}, {}
    preds, labels = result
    return compute_classification_eval_details(preds, labels)


def compute_eval_analysis(trainer: BNNRTrainer) -> dict[str, Any]:
    """Build post-run evaluation analysis dict for the active task."""
    if trainer._is_detection:
        per_class, _ = compute_eval_class_details_detection(trainer)
        if not per_class:
            return {}
        ap_values = [v.get("accuracy", 0.0) for v in per_class.values()]
        return {
            "per_class_accuracy": per_class,
            "macro_per_class_accuracy": float(np.mean(ap_values)) if ap_values else None,
        }

    if trainer._is_multilabel:
        result = collect_val_logits(trainer, post_process="sigmoid")
        if result is None:
            return {}
        preds, labels = result
        per_label, _ = compute_multilabel_eval_details(preds, labels)
        if not per_label:
            return {}
        f1_values = [v.get("f1", 0.0) for v in per_label.values()]
        return {
            "per_label_f1": per_label,
            "macro_per_label_f1": float(np.mean(f1_values)) if f1_values else None,
        }

    if not isinstance(trainer.model, XAICapableModel):
        return {}

    model = trainer.model.get_model()
    device = next(model.parameters()).device
    model.eval()
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for raw_batch in trainer.val_loader:
            if len(raw_batch) == 3:
                images, labels, _ = raw_batch
            else:
                images, labels = raw_batch
            logits = model(images.to(device))
            preds = torch.argmax(logits, dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels.cpu())

    if not all_preds or not all_labels:
        return {}

    preds_cat = torch.cat(all_preds)
    labels_cat = torch.cat(all_labels)
    n_classes = int(max(int(preds_cat.max().item()), int(labels_cat.max().item()))) + 1

    per_class_accuracy: dict[str, dict[str, float | int]] = {}
    class_acc_values: list[float] = []
    for class_id in range(n_classes):
        mask = labels_cat == class_id
        support = int(mask.sum().item())
        if support == 0:
            continue
        class_acc = float((preds_cat[mask] == labels_cat[mask]).float().mean().item())
        per_class_accuracy[str(class_id)] = {
            "accuracy": class_acc,
            "support": support,
        }
        class_acc_values.append(class_acc)

    return {
        "per_class_accuracy": per_class_accuracy,
        "macro_per_class_accuracy": float(np.mean(class_acc_values)) if class_acc_values else None,
    }
