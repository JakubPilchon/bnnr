"""CLI tests for ``bnnr demo``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from bnnr.cli import app
from bnnr.icd import ICD
from bnnr.reporting import BNNRRunResult

runner = CliRunner()


def _mock_run_result(tmp_path: Path) -> BNNRRunResult:
    run_dir = tmp_path / "reports" / "run_demo"
    run_dir.mkdir(parents=True)
    xai_dir = run_dir / "xai"
    xai_dir.mkdir()
    report_json = run_dir / "report.json"
    report_json.write_text("{}", encoding="utf-8")
    from bnnr.config_model import BNNRConfig

    return BNNRRunResult(
        config=BNNRConfig(),
        checkpoints=[],
        best_path="baseline -> icd",
        best_metrics={"accuracy": 0.5},
        selected_augmentations=["icd"],
        total_time=1.0,
        report_json_path=report_json,
        report_html_path=None,
    )


class TestDemoCommand:
    def test_demo_help_lists_command(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "demo" in result.stdout

    def test_demo_invokes_train_path(self, tmp_path: Path) -> None:
        mock_result = _mock_run_result(tmp_path)

        with (
            patch("bnnr.cli._run_train", return_value=mock_result) as mock_run,
            patch("bnnr.cli._print_demo_followup") as mock_followup,
        ):
            result = runner.invoke(app, ["demo"])

        assert result.exit_code == 0
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["dataset"] == "cifar10"
        assert call_kwargs["augmentation_preset"] == "demo"
        assert call_kwargs["max_train_samples"] == 128
        mock_followup.assert_called_once_with(mock_result)

    def test_demo_followup_messages(self, tmp_path: Path) -> None:
        from bnnr.cli import _print_demo_followup

        mock_result = _mock_run_result(tmp_path)
        with patch("bnnr.cli.typer.echo") as echo:
            _print_demo_followup(mock_result)
        combined = " ".join(str(c.args[0]) for c in echo.call_args_list if c.args)
        assert "Your report:" in combined
        assert "XAI heatmaps" in combined


class TestDemoPresetIntegration:
    def test_demo_preset_icd_requires_model(self) -> None:
        import torch.nn as nn

        from bnnr.presets import get_preset

        model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU())
        augs = get_preset("demo", model=model, target_layers=[model[0]])
        assert any(isinstance(a, ICD) for a in augs)
