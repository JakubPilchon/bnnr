# XAI comparison options (pick one for README)

Generated from benchmark seed **44** (`benchmarks/runs/*_s44/xai/`).  
Same val indices across all three conditions (OptiCAM).

| File | Val idx | Story (short) | BNNR edge vs baselines |
|------|---------|---------------|------------------------|
| [`comparison_val0_attn0.png`](comparison_val0_attn0.png) | 0 | Ship — baselines hot on **bottom-right corner**; BNNR on hull | **0.13** vs 0.45 / 0.44 |
| [`comparison_val127_attn1.png`](comparison_val127_attn1.png) | 127 | Truck — no BNNR scattered; RA diffuse; **BNNR tighter on vehicle** | **0.16** vs 0.46 / 0.03 |
| [`comparison_val255_attn2.png`](comparison_val255_attn2.png) | 255 | All three similar — weak differentiator | 0.14 ≈ 0.18 / 0.13 |
| [`comparison_val512_attn3.png`](comparison_val512_attn3.png) | 512 | Deer — all fairly centered; subtle | 0.09 vs 0.02 / 0.03 |
| [`comparison_val768_attn4.png`](comparison_val768_attn4.png) | 768 | Minimal difference between setups | ~0.07 all |
| [`comparison_val1024_attn5.png`](comparison_val1024_attn5.png) | 1024 | Bird — **no BNNR corner shortcut** again; BNNR more on body | 0.22 vs **0.46** / 0.21 |
| [`comparison_val1536_attn6.png`](comparison_val1536_attn6.png) | 1536 | Similar edge stats; hard to tell visually | ~0.23 all |
| [`comparison_val2047_attn7.png`](comparison_val2047_attn7.png) | 2047 | Horse — **all look at edges**; BNNR slightly less (0.37 vs 0.63) | modest |

Quick glance (BNNR panel only): [`_contact_sheet_bnnr_only.png`](_contact_sheet_bnnr_only.png)

**Regenerate all options:**

```bash
python scripts/build_benchmark_xai_readme_asset.py --all-options
```

**Current README figure:** val **127** (branded 4-panel: original + 3× OptiCAM) → `benchmark-xai-comparison.png`

```bash
python scripts/build_benchmark_xai_readme_asset.py --pick val127
```
