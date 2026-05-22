# Python quickstart (`quick_run`)

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/bnnr?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/bnnr)

Recommended entry point for **single-label image classification** in PyTorch.

## Minimal example

```python
import bnnr

result = bnnr.quick_run(model, train_loader, val_loader)
print(result.best_metrics)
print(result.report_json_path)
```

## Defaults

When `config` is omitted, `quick_run` uses the same defaults as `bnnr train` without `--config` (`default_train_config()`):

- `m_epochs=3`, `max_iterations=2`
- `device="auto"`, `xai_enabled=True`
- Auto-selected augmentations via `auto_select_augmentations()`

Fast smoke test:

```python
bnnr.quick_run(model, train_loader, val_loader, m_epochs=1, max_iterations=1, device="cpu")
```

## Optional arguments

| Argument | Behavior |
|----------|----------|
| `config` | Full `BNNRConfig`; merged with `**overrides` |
| `target_layers` | XAI layers; inferred (last `Conv2d`) when `None` and XAI is on |
| `dashboard=True` | Starts live dashboard before training (does not block after `run()`) |
| `augmentations` | Custom list; `None` → auto-select |

## Beyond classification

Multi-label, object detection, and custom adapters: [golden_path.md](golden_path.md).

CLI zero-flag demo: `python -m bnnr demo`.
