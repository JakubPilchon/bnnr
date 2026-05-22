"""Augmentation candidate selection and pruning logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bnnr.config_model import BNNRConfig


def select_best_path(
    results: dict[str, dict[str, float]],
    baseline_metrics: dict[str, float],
    config: BNNRConfig,
    xai_scores: dict[str, float] | None = None,
) -> str | None:
    """Pick the best augmentation candidate from *results*, or ``None`` if no improvement."""
    metric = config.selection_metric
    mode = config.selection_mode
    w = config.xai_selection_weight

    baseline_value = baseline_metrics.get(metric)

    if w <= 0 or not xai_scores:
        best_name: str | None = None
        best_value = None
        for aug_name, aug_metrics in results.items():
            val = aug_metrics.get(metric)
            if val is None:
                continue
            if best_value is None or (mode == "max" and val > best_value) or (mode == "min" and val < best_value):
                best_name = aug_name
                best_value = val
        if best_name is None or baseline_value is None or best_value is None:
            return None
        improved = (best_value > baseline_value) if mode == "max" else (best_value < baseline_value)
        return best_name if improved else None

    metric_vals = {name: m.get(metric) for name, m in results.items() if m.get(metric) is not None}
    if not metric_vals:
        return None

    all_vals = list(metric_vals.values())
    min_m = min(v for v in all_vals if v is not None)
    max_m = max(v for v in all_vals if v is not None)
    m_range = max_m - min_m if max_m != min_m else 1.0  # type: ignore[operator]

    best_name = None
    best_composite: float | None = None
    for aug_name, val in metric_vals.items():
        if val is None:
            continue
        if mode == "max":
            norm_m = (float(val) - float(min_m)) / float(m_range)  # type: ignore[arg-type]
        else:
            norm_m = (float(max_m) - float(val)) / float(m_range)  # type: ignore[arg-type]
        xai_q = xai_scores.get(aug_name, 0.0)
        composite = (1.0 - w) * norm_m + w * xai_q
        if best_composite is None or composite > best_composite:
            best_composite = composite
            best_name = aug_name

    if best_name is None or baseline_value is None:
        return None

    best_value = results[best_name].get(metric)
    if best_value is None:
        return None
    improved = (best_value > baseline_value) if mode == "max" else (best_value < baseline_value)
    return best_name if improved else None


def should_prune_candidate(
    candidate_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    config: BNNRConfig,
    xai_quality: float | None = None,
) -> bool:
    """Return ``True`` if the candidate should be pruned early."""
    if not config.candidate_pruning_enabled:
        return False
    metric = config.selection_metric
    candidate_value = candidate_metrics.get(metric)
    baseline_value = baseline_metrics.get(metric)
    if candidate_value is None or baseline_value is None:
        return False

    threshold = config.candidate_pruning_relative_threshold
    if config.selection_mode == "max":
        metric_prune = float(candidate_value) < float(baseline_value) * threshold
    else:
        metric_prune = float(candidate_value) > float(baseline_value) * (2.0 - threshold)

    if metric_prune:
        return True

    xai_thresh = config.xai_pruning_threshold
    if xai_thresh > 0 and xai_quality is not None and xai_quality < xai_thresh:
        return True

    return False


def get_current_best_metric(
    results: dict[str, dict[str, float]],
    config: BNNRConfig,
) -> float | None:
    """Return the best selection-metric value seen so far across candidates."""
    metric = config.selection_metric
    values = [v[metric] for v in results.values() if metric in v]
    if not values:
        return None
    return float(max(values) if config.selection_mode == "max" else min(values))


def top_k_candidate_names(
    results: dict[str, dict[str, float]],
    config: BNNRConfig,
    k: int = 3,
) -> list[str]:
    """Return names of up to *k* best candidates, ordered by selection metric."""
    metric = config.selection_metric
    sorted_items = sorted(
        ((name, metrics) for name, metrics in results.items() if metric in metrics),
        key=lambda item: float(item[1][metric]),
        reverse=(config.selection_mode == "max"),
    )
    return [name for name, _ in sorted_items[:k]]
