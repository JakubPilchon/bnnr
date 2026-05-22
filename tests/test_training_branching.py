"""Unit tests for training.branching."""

from __future__ import annotations

from bnnr.config_model import BNNRConfig
from bnnr.training.branching import (
    get_current_best_metric,
    select_best_path,
    should_prune_candidate,
    top_k_candidate_names,
)


def test_select_best_path_improves() -> None:
    cfg = BNNRConfig(selection_metric="accuracy", selection_mode="max")
    results = {"aug_a": {"accuracy": 0.9}, "aug_b": {"accuracy": 0.7}}
    baseline = {"accuracy": 0.5}
    assert select_best_path(results, baseline, cfg) == "aug_a"


def test_should_prune_candidate_max_mode() -> None:
    cfg = BNNRConfig(
        candidate_pruning_enabled=True,
        candidate_pruning_relative_threshold=0.9,
        selection_mode="max",
    )
    assert should_prune_candidate({"accuracy": 0.5}, {"accuracy": 0.9}, cfg)


def test_get_current_best_metric() -> None:
    cfg = BNNRConfig(selection_metric="accuracy", selection_mode="max")
    results = {"a": {"accuracy": 0.1}, "b": {"accuracy": 0.9}}
    assert get_current_best_metric(results, cfg) == 0.9


def test_top_k_candidate_names() -> None:
    cfg = BNNRConfig(selection_metric="accuracy", selection_mode="max")
    results = {"a": {"accuracy": 0.1}, "b": {"accuracy": 0.9}, "c": {"accuracy": 0.5}}
    assert top_k_candidate_names(results, cfg, k=2) == ["b", "c"]
