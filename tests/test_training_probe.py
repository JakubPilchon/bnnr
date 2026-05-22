"""Unit tests for training.probe helpers."""

from __future__ import annotations

from pathlib import Path

from bnnr.training.probe import (
    probe_labels_from_tensor,
    probe_sample_ids_from_list,
    to_artifact_reference,
)


def test_probe_labels_from_tensor_none() -> None:
    assert probe_labels_from_tensor(None) == []


def test_probe_sample_ids_from_list() -> None:
    assert probe_sample_ids_from_list(["a", "b"]) == ["a", "b"]


def test_to_artifact_reference_relative() -> None:
    run_dir = Path("/tmp/run")
    path = run_dir / "artifacts" / "samples" / "x.png"
    ref = to_artifact_reference(path, run_dir)
    assert ref.startswith("artifacts/")
