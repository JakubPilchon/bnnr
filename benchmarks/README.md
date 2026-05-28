# CIFAR-10 benchmarks

Reproducible comparison of **three training setups** on the same demo CNN, dataset split, and epoch budget:

| Condition | What it is |
|-----------|------------|
| `no_bnnr` | Crop + flip only — no BNNR augmentations, no branch search |
| `randaugment` | **torchvision RandAugment** — random policy-based augmentations (external baseline) |
| `bnnr_branch_search` | Full **BNNR branch search** over **ICD**, **AICD**, and ChurchNoise |

## What we compare

1. **BNNR vs no augmentation** — does the branching system (with saliency-guided ICD/AICD) beat plain training?
2. **BNNR vs RandAugment** — does targeted, XAI-aware augmentation beat off-the-shelf random augs?
3. **Attention maps** — after each run, **OptiCAM** overlays on the **same 8 validation images** (`config.yaml` → `xai_val_indices`). Compare `runs/*/xai/attention_*.png` to see where each model looks.

Lower **edge ratio** and more focused **coverage** on the object usually indicate less background reliance.

## Layout

```
benchmarks/
  config.yaml      # shared epochs, metrics, RandAugment params, XAI indices
  lib.py           # conditions, training, attention export
  run.py           # CLI (resume-safe)
  summarize.py     # metrics + attention stats table
  results.json     # aggregated results (commit after review)
  runs/            # per-run logs + xai/ overlays (gitignored)
```

## Run

```bash
python benchmarks/run.py --seeds 42 --device cpu
python benchmarks/summarize.py --markdown
```

Three seeds for publication-ready numbers:

```bash
python benchmarks/run.py --seeds 42,43,44 --device cpu
```

List conditions:

```bash
python benchmarks/run.py --list-conditions
```

## BNNR augmentations in this benchmark

| Name | Role |
|------|------|
| **ICD** | Masks *high-saliency* regions — forces the model to look beyond the easiest cue |
| **AICD** | Masks *low-saliency* background — reduces shortcut learning on context |
| **ChurchNoise** | Lightweight noise aug — non-XAI candidate in the branch pool |

The branch search keeps augmentations that improve validation accuracy; the winning path is recorded in `results.json` → `best_path`.

## Shared fairness rules

- Same demo CNN (`_CifarCNN`), Adam lr=1e-3, batch 64, `m_epochs` from `config.yaml`
- Same random seed per condition within a run
- RandAugment uses `num_ops=2`, `magnitude=9` (standard CIFAR-10 defaults in `_benchmark`)
- Attention maps always use the same validation indices across conditions
