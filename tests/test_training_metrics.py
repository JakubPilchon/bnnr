"""Unit tests for training.metrics."""

from __future__ import annotations

import numpy as np

from bnnr.training.metrics import (
    average_metrics,
    compute_classification_eval_details,
    compute_multilabel_eval_details,
    macro_mean_metric,
)


def test_average_metrics_union_keys() -> None:
    batches = [{"loss": 1.0, "accuracy": 0.5}, {"accuracy": 0.7}]
    out = average_metrics(batches)
    assert out["loss"] == 1.0
    assert abs(out["accuracy"] - 0.6) < 1e-6


def test_compute_classification_eval_details() -> None:
    preds = np.array([0, 1, 1, 0])
    labels = np.array([0, 1, 0, 0])
    per_class, confusion = compute_classification_eval_details(preds, labels)
    assert "0" in per_class
    assert confusion["matrix"]


def test_compute_multilabel_eval_details() -> None:
    preds = np.array([[1, 0], [0, 1]])
    labels = np.array([[1, 0], [1, 1]])
    per_label, confusion = compute_multilabel_eval_details(preds, labels)
    assert "0" in per_label
    assert confusion["type"] == "multilabel_per_label"


def test_macro_mean_metric() -> None:
    per_class = {"0": {"f1": 0.8}, "1": {"f1": 0.6}}
    assert macro_mean_metric(per_class, "f1") == 0.7
