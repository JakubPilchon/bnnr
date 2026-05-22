"""XAI callback helpers — pure-logic functions extracted from BNNRTrainer."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from bnnr.config_model import BNNRConfig


def xai_mean_quality(xai_diagnoses: dict[str, dict[str, Any]]) -> float | None:
    """Compute mean XAI quality score across all diagnosed classes."""
    if not xai_diagnoses:
        return None
    scores = [
        float(d["quality_score"])
        for d in xai_diagnoses.values()
        if isinstance(d, dict) and "quality_score" in d
    ]
    return float(np.mean(scores)) if scores else None


def build_xai_insights(
    baseline_metrics: dict[str, float],
    best_metrics: dict[str, float],
    selected_augmentations: list[str],
    config: BNNRConfig,
) -> list[str]:
    """Build post-run XAI insight strings."""
    metric = config.selection_metric
    baseline_value = baseline_metrics.get(metric)
    best_value = best_metrics.get(metric)
    insights: list[str] = []

    if isinstance(baseline_value, (int, float)) and isinstance(best_value, (int, float)):
        delta = float(best_value) - float(baseline_value)
        direction = "higher" if config.selection_mode == "max" else "lower"
        insights.append(f"Best {metric} is {delta:+.4f} vs baseline ({direction} is better).")

    if any("icd" == name for name in selected_augmentations):
        insights.append("ICD was selected: model benefits from dropping highly salient regions.")
    if any("aicd" == name for name in selected_augmentations):
        insights.append("AICD was selected: model benefits from suppressing low-saliency background context.")
    if config.xai_method.lower() in {"craft", "nmf", "nmf_concepts"}:
        insights.append("NMF concept maps were used for saliency estimation in this run.")
    else:
        insights.append("OptiCAM heatmaps were used for saliency estimation in this run.")

    return insights


def saliency_recommendations(
    batch_stats: dict[str, list[dict[str, float]]],
    xai_diagnoses: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    """Generate augmentation recommendations from saliency statistics."""
    if not batch_stats:
        return []

    all_coverages: list[float] = []
    all_ginis: list[float] = []
    all_edges: list[float] = []
    all_coherences: list[float] = []
    for stats_list in batch_stats.values():
        for s in stats_list:
            all_coverages.append(s.get("coverage", 0.0))
            all_ginis.append(s.get("gini", 0.0))
            all_edges.append(s.get("edge_ratio", 0.0))
            all_coherences.append(s.get("spatial_coherence", 0.0))

    mean_cov = float(np.mean(all_coverages)) if all_coverages else 0.0
    mean_gini = float(np.mean(all_ginis)) if all_ginis else 0.0
    mean_edge = float(np.mean(all_edges)) if all_edges else 0.0
    mean_coh = float(np.mean(all_coherences)) if all_coherences else 0.0

    hints: list[str] = []

    if mean_cov < 0.05 and all_coverages:
        hints.append(
            f"Very narrow attention (coverage={mean_cov:.1%}). "
            "ICD augmentation may force the model to learn broader features."
        )
    elif mean_cov > 0.40 and all_coverages:
        hints.append(
            f"Diffuse attention (coverage={mean_cov:.1%}). "
            "AICD augmentation may help sharpen focus on discriminative regions."
        )

    if mean_gini < 0.3 and all_ginis:
        hints.append(
            f"Low focus (Gini={mean_gini:.2f}). "
            "Model spreads attention uniformly — consider augmentations that "
            "encourage spatial discrimination (e.g. ICD, cutout-style)."
        )
    elif mean_gini > 0.9 and all_ginis:
        hints.append(
            f"Extremely concentrated focus (Gini={mean_gini:.2f}). "
            "Risk of relying on a tiny region. AICD or spatial jitter "
            "may encourage using multiple cues."
        )

    if mean_edge > 0.3 and all_edges:
        hints.append(
            f"High edge-region attention (edge_ratio={mean_edge:.2f}). "
            "Model may be relying on border artifacts — consider random "
            "cropping or padding augmentation."
        )

    if mean_coh < 0.3 and all_coherences:
        hints.append(
            f"Low spatial coherence ({mean_coh:.2f}). "
            "Saliency is fragmented — noise augmentation or smoothing "
            "transforms may help the model consolidate attention."
        )

    if xai_diagnoses:
        critical_classes = [
            cls_id
            for cls_id, diag in xai_diagnoses.items()
            if isinstance(diag, dict) and diag.get("severity") == "critical"
        ]
        if critical_classes:
            cls_list = ", ".join(critical_classes[:5])
            hints.append(
                f"Critical XAI quality for class(es): {cls_list}. "
                "Targeted augmentation or data rebalancing may be needed."
            )

    return hints


def build_xai_summary(
    prev_xai_batch_stats: dict[str, list[dict[str, float]]],
    baseline_xai_stats: dict[str, list[dict[str, float]]],
) -> dict[str, Any]:
    """Build a post-run XAI summary from accumulated batch stats."""
    if not prev_xai_batch_stats:
        return {}

    coverages: list[float] = []
    ginis: list[float] = []
    per_class: dict[str, dict[str, float]] = {}

    for cls_id, stats_list in prev_xai_batch_stats.items():
        cls_covs = [s.get("coverage", 0.0) for s in stats_list]
        cls_ginis = [s.get("gini", 0.0) for s in stats_list]
        cls_coherence = [s.get("spatial_coherence", 0.0) for s in stats_list]
        cls_edge = [s.get("edge_ratio", 0.0) for s in stats_list]

        coverages.extend(cls_covs)
        ginis.extend(cls_ginis)

        per_class[cls_id] = {
            "coverage": round(float(np.mean(cls_covs)), 4) if cls_covs else 0.0,
            "focus": round(float(np.mean(cls_ginis)), 4) if cls_ginis else 0.0,
            "coherence": round(float(np.mean(cls_coherence)), 4) if cls_coherence else 0.0,
            "edge_ratio": round(float(np.mean(cls_edge)), 4) if cls_edge else 0.0,
        }

    mean_cov = float(np.mean(coverages)) if coverages else 0.0
    mean_gini = float(np.mean(ginis)) if ginis else 0.0

    trend = "insufficient_data"
    if baseline_xai_stats and prev_xai_batch_stats:
        bl_entropies = [
            s.get("entropy", 0.0)
            for stats in baseline_xai_stats.values()
            for s in stats
        ]
        cur_entropies = [
            s.get("entropy", 0.0)
            for stats in prev_xai_batch_stats.values()
            for s in stats
        ]
        if bl_entropies and cur_entropies:
            bl_mean = float(np.mean(bl_entropies))
            cur_mean = float(np.mean(cur_entropies))
            delta = cur_mean - bl_mean
            if delta < -0.5:
                trend = "improving"
            elif delta > 0.5:
                trend = "declining"
            else:
                trend = "stable"

    return {
        "mean_quality_coverage": round(mean_cov, 4),
        "mean_quality_focus": round(mean_gini, 4),
        "quality_trend": trend,
        "per_class": per_class,
        "recommendations": saliency_recommendations(prev_xai_batch_stats),
    }


def adapt_icd_thresholds(
    augmentations: list[Any],
    prev_xai_batch_stats: dict[str, list[dict[str, float]]],
    *,
    adaptive_enabled: bool,
    log_fn: Callable[[str], None] | None = None,
) -> None:
    """Adjust ICD/AICD threshold_percentile based on XAI coverage and focus."""
    if not adaptive_enabled:
        return

    from bnnr.icd import AICD, ICD

    coverages: list[float] = []
    ginis: list[float] = []
    for cls_stats in prev_xai_batch_stats.values():
        for s in cls_stats:
            coverages.append(s.get("coverage", 0.0))
            ginis.append(s.get("gini", 0.0))

    if not coverages:
        return

    mean_cov = float(np.mean(coverages))
    mean_gini = float(np.mean(ginis))

    adjustment = 0
    if mean_cov < 0.05 and mean_gini > 0.8:
        adjustment = 5
    elif mean_cov > 0.30 and mean_gini < 0.4:
        adjustment = -5

    if adjustment == 0:
        return

    for aug in augmentations:
        if isinstance(aug, (ICD, AICD)):
            old_tp = getattr(aug, "threshold_percentile", 75)
            new_tp = max(50, min(90, old_tp + adjustment))
            aug.threshold_percentile = new_tp
            if log_fn is not None:
                log_fn(
                    f"Adaptive ICD: {aug.name} threshold_percentile "
                    f"{old_tp} → {new_tp} (coverage={mean_cov:.2f}, gini={mean_gini:.2f})"
                )
