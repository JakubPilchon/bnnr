"""Dataset profiling utilities."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from bnnr.data_quality import run_data_quality_analysis


def count_labels(
    loader: DataLoader,
    *,
    is_detection: bool,
    is_multilabel: bool,
    capture_shape: bool = False,
) -> tuple[Counter[int], list[int], int]:
    """Count class labels in *loader* in a single pass.

    Returns ``(counter, image_shape, total_boxes)``.
    """
    counter: Counter[int] = Counter()
    image_shape: list[int] = []
    total_boxes = 0
    for raw_batch in loader:
        if is_detection:
            if len(raw_batch) == 3:
                images, targets, _ = raw_batch
            else:
                images, targets = raw_batch
            if capture_shape and not image_shape and images.ndim == 4:
                image_shape = list(images.shape[1:])
            for target in targets:
                for label in target["labels"].tolist():
                    counter[int(label)] += 1
                    total_boxes += 1
        elif is_multilabel:
            if len(raw_batch) == 3:
                images, labels, _ = raw_batch
            else:
                images, labels = raw_batch
            if capture_shape and not image_shape and images.ndim == 4:
                image_shape = list(images.shape[1:])
            for sample_labels in labels:
                for cls_idx in range(sample_labels.shape[0]):
                    if int(sample_labels[cls_idx]) == 1:
                        counter[cls_idx] += 1
        else:
            if len(raw_batch) == 3:
                images, labels, _ = raw_batch
            else:
                images, labels = raw_batch
            if capture_shape and not image_shape and images.ndim == 4:
                image_shape = list(images.shape[1:])
            labels = labels.squeeze()
            if labels.ndim == 0:
                labels = labels.unsqueeze(0)
            for label in labels.tolist():
                counter[int(label)] += 1
    return counter, image_shape, total_boxes


def compute_dataset_profile(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Any,
    *,
    is_detection: bool,
    is_multilabel: bool,
    reporter: Any,
    log_fn: Callable[[str], None] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Compute dataset class distribution, metadata, and data quality."""
    train_counter, image_shape, total_train_boxes = count_labels(
        train_loader,
        is_detection=is_detection,
        is_multilabel=is_multilabel,
        capture_shape=True,
    )
    val_counter, _, total_val_boxes = count_labels(
        val_loader,
        is_detection=is_detection,
        is_multilabel=is_multilabel,
    )

    all_classes = sorted(set(train_counter.keys()) | set(val_counter.keys()))
    num_classes = len(all_classes)

    class_distribution = {str(c): train_counter.get(c, 0) for c in all_classes}
    val_class_distribution = {str(c): val_counter.get(c, 0) for c in all_classes}

    total_train = sum(train_counter.values())
    total_val = sum(val_counter.values())

    counts = list(train_counter.values())
    if counts and min(counts) > 0:
        imbalance_ratio = float(max(counts)) / float(min(counts))
    else:
        imbalance_ratio = float("inf")

    if config.detection_class_names:
        class_names = list(config.detection_class_names)
    else:
        class_names = [f"class_{c}" for c in all_classes]

    profile: dict[str, Any] = {
        "num_classes": num_classes,
        "class_distribution": class_distribution,
        "val_class_distribution": val_class_distribution,
        "total_train_samples": total_train,
        "total_val_samples": total_val,
        "imbalance_ratio": round(imbalance_ratio, 2),
        "image_shape": image_shape,
        "class_names": class_names,
    }

    if is_detection:
        profile["task"] = "detection"
        profile["total_train_boxes"] = total_train_boxes
        profile["total_val_boxes"] = total_val_boxes

    try:
        dq_run_dir = getattr(reporter, "run_dir", config.report_dir)
        dq_save_dir = Path(dq_run_dir) / "artifacts" / "data_quality"
        quality_result = run_data_quality_analysis(
            train_loader,
            save_dir=dq_save_dir,
            run_dir=Path(dq_run_dir),
            duplicate_threshold=config.duplicate_hamming_threshold,
        )
        profile.update(quality_result)
        dq = quality_result.get("data_quality", {})
        n_warnings = len(dq.get("warnings", []))
        if log_fn is not None:
            if n_warnings:
                log_fn(
                    f"Data quality: {dq.get('summary', '')} "
                    f"({n_warnings} warning type(s))"
                )
            else:
                log_fn(f"Data quality: {dq.get('summary', 'OK')}")
    except (ValueError, RuntimeError, OSError, TypeError) as exc:
        if logger is not None:
            logger.warning(
                "Data quality analysis failed — skipping: %s",
                exc,
                exc_info=True,
            )

    return profile
