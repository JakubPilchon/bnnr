# Benchmarks

CIFAR-10 study comparing **no augmentation**, **torchvision RandAugment**, and **BNNR branch search** (ICD + AICD + ChurchNoise), with validation metrics and OptiCAM attention overlays.

See [benchmarks/README.md](../benchmarks/README.md).

```bash
python benchmarks/run.py --seeds 42 --device cpu
python benchmarks/summarize.py --markdown
```

Results: `benchmarks/results.json`. Attention maps: `benchmarks/runs/*/xai/`.
