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
- RandAugment uses `num_ops=2`, `magnitude=9` (torchvision defaults in `_benchmark`)
- Attention maps always use the same validation indices across conditions

**Protocol note:** `no_bnnr` and `randaugment` train for **5 epochs** only. `bnnr_branch_search` runs baseline (5 ep) plus candidate screening (up to 3×5 ep per iteration × 3 iterations) — **much more compute** and a different curriculum (augmentations added after baseline). Compare numbers as *“full BNNR product vs fixed-epoch baselines”*, not equal-budget ablation.

## Results (2026-05-28, seeds 42–44, CPU)

| Condition | Median val acc | Δ vs no BNNR | Per-seed |
|-----------|----------------|--------------|----------|
| Without BNNR (crop + flip) | **75.3%** | — | 75.3, 75.6, 75.3 |
| RandAugment (torchvision 2,9) | **72.5%** | −2.8 pp | 72.2, 72.5, 73.5 |
| BNNR branch search | **81.4%** | +6.1 pp | 81.4, 81.3, 81.6 |

Within-run BNNR gain vs its own baseline phase: +7.3 to +12.1 pp (`gain_vs_within_run_baseline_pp` in `results.json`). Winning paths varied by seed (e.g. ChurchNoise→ICD, or full ICD+AICD+ChurchNoise stack).

**Takeaways (honest):**

1. **BNNR full pipeline** clearly beats both baselines on this demo setup — stable ~81% across seeds.
2. **RandAugment at 5 epochs** underperforms crop+flip here; regularization likely needs longer training or lower `magnitude` — not a bug in integration.
3. **Attention (OptiCAM):** BNNR shows lower mean coverage (~13.5% vs ~18%); edge ratio mixed — qualitative XAI in `runs/*/xai/`, not a single headline metric.
4. **Do not claim SOTA or “beats RandAugment” without citing protocol** — demo CNN, short baselines, unequal epoch budget.

Raw data: [`results.json`](results.json). Regenerate table: `python benchmarks/summarize.py --markdown`.

### README figure

Side-by-side OptiCAM on the same val image (seed 44):

```bash
python scripts/build_benchmark_xai_readme_asset.py
```

Output: `docs/assets/benchmark-xai-comparison.png` (used in root README).
