"""High-level convenience API to execute a minimal BNNR run."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from bnnr.adapter import SimpleTorchAdapter
from bnnr.augmentations import BaseAugmentation
from bnnr.config import default_train_config
from bnnr.core import BNNRConfig, BNNRTrainer
from bnnr.reporting import BNNRRunResult, Reporter


def _guess_target_layers(model: nn.Module) -> list[nn.Module]:
    """Pick XAI target layers when the caller omits ``target_layers``."""
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if convs:
        return [convs[-1]]

    features = getattr(model, "features", None)
    if isinstance(features, nn.Module):
        feat_convs = [m for m in features.modules() if isinstance(m, nn.Conv2d)]
        if feat_convs:
            return [feat_convs[-1]]

    raise ValueError(
        "Could not infer target_layers from the model (no Conv2d found). "
        "Pass target_layers=[...] explicitly for XAI / ICD."
    )


def quick_run(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    augmentations: list[BaseAugmentation] | None = None,
    config: BNNRConfig | None = None,
    criterion: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    target_layers: list[nn.Module] | None = None,
    eval_metrics: list[str] | None = None,
    dashboard: bool | None = None,
    **overrides: object,
) -> BNNRRunResult:
    """Recommended quickstart for image **classification** (PyTorch).

    Runs a short BNNR search with sensible defaults. For detection, multilabel,
    or custom adapters see ``docs/golden_path.md``.

    Parameters
    ----------
    dashboard:
        When ``True``, starts the live dashboard before training (non-blocking
        after ``run()`` returns — notebook/API friendly). ``None``/``False`` skip it.
    **overrides:
        Fields merged into :class:`~bnnr.core.BNNRConfig` (e.g. ``m_epochs=1``).
    """
    cfg = config or default_train_config()
    if overrides:
        cfg = BNNRConfig(**{**cfg.model_dump(), **overrides})

    if target_layers is None and cfg.xai_enabled:
        target_layers = _guess_target_layers(model)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    adapter = SimpleTorchAdapter(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        target_layers=target_layers,
        device=cfg.device,
        eval_metrics=eval_metrics or [m for m in cfg.metrics if m != "loss"],
    )

    if augmentations is None:
        from bnnr.presets import auto_select_augmentations

        augmentations = auto_select_augmentations(random_state=cfg.seed)

    if dashboard:
        from bnnr.dashboard import start_dashboard

        start_dashboard(run_root=cfg.report_dir, auto_open=False)

    reporter = Reporter(cfg.report_dir)

    trainer = BNNRTrainer(
        model=adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        augmentations=augmentations,
        config=cfg,
        reporter=reporter,
    )
    return trainer.run()
