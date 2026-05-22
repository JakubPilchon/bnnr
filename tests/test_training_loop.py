"""Unit tests for training.loop orchestration helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bnnr.training.loop import run_single_iteration


@pytest.fixture()
def mock_trainer() -> MagicMock:
    trainer = MagicMock()
    trainer.config.m_epochs = 2
    trainer.config.selection_metric = "accuracy"
    trainer.config.selection_mode = "max"
    trainer.config.candidate_pruning_warmup_epochs = 1
    trainer._active_augmentations = []
    trainer._clone_state_dict.return_value = {"w": 1}
    trainer.model.state_dict.return_value = {"w": 1}
    trainer._check_pause = MagicMock()
    trainer.train_loader = MagicMock()
    trainer.val_loader = MagicMock()
    return trainer


def test_run_single_iteration_returns_best_epoch(mock_trainer: MagicMock) -> None:
    with (
        patch("bnnr.training.loop.train_epoch") as mock_train,
        patch("bnnr.training.loop.evaluate") as mock_eval,
        patch("bnnr.training.loop._branching.should_prune_candidate", return_value=False),
    ):
        mock_train.return_value = {"loss": 0.1}
        mock_eval.side_effect = [
            {"accuracy": 0.4, "loss": 0.2},
            {"accuracy": 0.9, "loss": 0.1},
        ]
        aug = MagicMock()
        aug.name = "icd"

        metrics, state, best_epoch, pruned = run_single_iteration(
            mock_trainer,
            aug,
            baseline_metrics={"accuracy": 0.95},
        )

    assert metrics["accuracy"] == 0.9
    assert best_epoch == 2
    assert pruned is False
    assert state == {"w": 1}
    assert mock_train.call_count == 2
    assert mock_eval.call_count == 2
