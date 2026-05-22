"""Tests for the quick_run high-level API."""

from __future__ import annotations

import importlib
from unittest.mock import patch

from bnnr.config import default_train_config
from bnnr.core import BNNRConfig

# ``bnnr.quick_run`` on the package is the function (see ``bnnr.__init__``);
# patch targets must use the submodule object, not ``patch("bnnr.quick_run.*")``.
_quick_run_mod = importlib.import_module("bnnr.quick_run")
quick_run = _quick_run_mod.quick_run
_guess_target_layers = _quick_run_mod._guess_target_layers


def test_quick_run_smoke(dummy_model, dummy_dataloader, temp_dir) -> None:
    cfg = BNNRConfig(
        m_epochs=1,
        max_iterations=1,
        xai_enabled=False,
        device="cpu",
        checkpoint_dir=temp_dir / "c",
        report_dir=temp_dir / "r",
    )
    result = quick_run(dummy_model, dummy_dataloader, dummy_dataloader, config=cfg)
    assert result.report_json_path.exists()


def test_quick_run_default_config(dummy_model, dummy_dataloader, temp_dir) -> None:
    cfg = default_train_config(
        checkpoint_dir=temp_dir / "c",
        report_dir=temp_dir / "r",
    )
    with patch.object(_quick_run_mod, "BNNRTrainer") as trainer_cls:
        trainer_cls.return_value.run.return_value.report_json_path = temp_dir / "r" / "x.json"
        quick_run(
            dummy_model,
            dummy_dataloader,
            dummy_dataloader,
            config=cfg,
            m_epochs=1,
            max_iterations=1,
            xai_enabled=False,
            device="cpu",
        )
    passed_cfg = trainer_cls.call_args.kwargs["config"]
    assert passed_cfg.m_epochs == 1
    assert passed_cfg.xai_enabled is False


def test_quick_run_uses_train_defaults_when_config_none(
    dummy_model, dummy_dataloader, temp_dir
) -> None:
    with patch.object(_quick_run_mod, "default_train_config") as mock_defaults:
        mock_defaults.return_value = BNNRConfig(
            m_epochs=3,
            max_iterations=2,
            xai_enabled=True,
            device="auto",
            checkpoint_dir=temp_dir / "c",
            report_dir=temp_dir / "r",
        )
        with patch.object(_quick_run_mod, "BNNRTrainer") as trainer_cls:
            trainer_cls.return_value.run.return_value.report_json_path = temp_dir / "r" / "x.json"
            quick_run(
                dummy_model,
                dummy_dataloader,
                dummy_dataloader,
                xai_enabled=False,
                device="cpu",
                m_epochs=1,
                max_iterations=1,
            )
        mock_defaults.assert_called_once()
        cfg = trainer_cls.call_args.kwargs["config"]
        assert cfg.m_epochs == 1


def test_guess_target_layers_dummy_cnn(dummy_model) -> None:
    layers = _guess_target_layers(dummy_model)
    assert len(layers) == 1
    assert layers[0] is dummy_model.conv1


def test_quick_run_infers_target_layers(dummy_model, dummy_dataloader, temp_dir) -> None:
    cfg = BNNRConfig(
        m_epochs=1,
        max_iterations=1,
        xai_enabled=True,
        device="cpu",
        checkpoint_dir=temp_dir / "c",
        report_dir=temp_dir / "r",
    )
    with patch.object(_quick_run_mod, "BNNRTrainer") as trainer_cls:
        trainer_cls.return_value.run.return_value.report_json_path = temp_dir / "r" / "x.json"
        quick_run(dummy_model, dummy_dataloader, dummy_dataloader, config=cfg)
    adapter = trainer_cls.call_args.kwargs["model"]
    assert adapter.target_layers == [dummy_model.conv1]


def test_quick_run_dashboard_starts_before_run(dummy_model, dummy_dataloader, temp_dir) -> None:
    cfg = BNNRConfig(
        m_epochs=1,
        max_iterations=1,
        xai_enabled=False,
        device="cpu",
        checkpoint_dir=temp_dir / "c",
        report_dir=temp_dir / "r",
    )
    with (
        patch("bnnr.dashboard.start_dashboard") as mock_dash,
        patch.object(_quick_run_mod, "BNNRTrainer") as trainer_cls,
    ):
        trainer_cls.return_value.run.return_value.report_json_path = temp_dir / "r" / "x.json"
        quick_run(
            dummy_model,
            dummy_dataloader,
            dummy_dataloader,
            config=cfg,
            dashboard=True,
        )
    mock_dash.assert_called_once()
    assert mock_dash.call_args.kwargs["run_root"] == cfg.report_dir
