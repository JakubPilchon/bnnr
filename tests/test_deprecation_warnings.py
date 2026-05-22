"""Tests for deprecated top-level bnnr imports."""

from __future__ import annotations

import warnings

import bnnr


def test_detection_adapter_deprecated() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cls = bnnr.DetectionAdapter
        assert cls is not None
    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert dep
    assert "deprecated" in str(dep[0].message).lower()


def test_events_deprecated() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sink = bnnr.JsonlEventSink
        assert sink is not None
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
