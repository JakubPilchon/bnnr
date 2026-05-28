# Benchmarks

CIFAR-10 study comparing **no augmentation**, **torchvision RandAugment**, and **BNNR branch search** (ICD + AICD + ChurchNoise), with validation metrics and OptiCAM attention overlays.

See [benchmarks/README.md](../benchmarks/README.md).

```bash
python benchmarks/run.py --seeds 42 --device cpu
python benchmarks/summarize.py --markdown
```

Results: [`benchmarks/results.json`](../benchmarks/results.json) (CIFAR-10 medians: no BNNR **75.3%**, BNNR branch search **81.4%**, RandAugment **72.5%**, 3 seeds). Attention maps: `benchmarks/runs/*/xai/`. See protocol caveats in [`benchmarks/README.md`](../benchmarks/README.md#results-2026-05-28-seeds-4244-cpu).
