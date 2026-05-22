"""Unit tests for training.dataset_profile."""

from __future__ import annotations

from collections import Counter

import torch
from torch.utils.data import DataLoader, TensorDataset

from bnnr.training.dataset_profile import count_labels


def test_count_labels_classification() -> None:
    ds = TensorDataset(torch.zeros(4, 3, 8, 8), torch.tensor([0, 1, 1, 0]))
    loader = DataLoader(ds, batch_size=2)
    counter, shape, boxes = count_labels(loader, is_detection=False, is_multilabel=False, capture_shape=True)
    assert counter == Counter({0: 2, 1: 2})
    assert shape == [3, 8, 8]
    assert boxes == 0
