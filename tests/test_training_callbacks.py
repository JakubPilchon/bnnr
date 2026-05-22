"""Unit tests for training.callbacks."""

from __future__ import annotations

from bnnr.training.callbacks import (
    adapt_icd_thresholds,
    build_xai_summary,
    saliency_recommendations,
    xai_mean_quality,
)


def test_xai_mean_quality() -> None:
    diag = {"0": {"quality_score": 0.8}, "1": {"quality_score": 0.6}}
    assert abs(xai_mean_quality(diag) - 0.7) < 1e-6


def test_saliency_recommendations_narrow_coverage() -> None:
    stats = {"0": [{"coverage": 0.02, "gini": 0.9, "edge_ratio": 0.1, "spatial_coherence": 0.5}]}
    hints = saliency_recommendations(stats)
    assert any("narrow" in h.lower() or "icd" in h.lower() for h in hints)


def test_build_xai_summary_empty() -> None:
    assert build_xai_summary({}, {}) == {}


def test_adapt_icd_thresholds_disabled() -> None:
    class _FakeAug:
        name = "fake_icd"
        threshold_percentile = 75

    aug = _FakeAug()
    adapt_icd_thresholds([aug], {"0": [{"coverage": 0.01, "gini": 0.95}]}, adaptive_enabled=False)
    assert aug.threshold_percentile == 75
