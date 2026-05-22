"""Unit tests for training.checkpoint helpers."""

from __future__ import annotations

import torch

from bnnr.training.checkpoint import clone_state_dict, copy_state_dict_inplace


def test_clone_state_dict_tensor() -> None:
    src = {"w": torch.tensor([1.0, 2.0])}
    dst = clone_state_dict(src)
    assert torch.equal(dst["w"], src["w"])
    dst["w"][0] = 99.0
    assert src["w"][0].item() == 1.0


def test_clone_state_dict_nested() -> None:
    src = {"model": {"w": torch.tensor([3.0])}}
    dst = clone_state_dict(src)
    dst["model"]["w"][0] = 0.0
    assert src["model"]["w"][0].item() == 3.0


def test_copy_state_dict_inplace() -> None:
    dst = {"w": torch.zeros(2)}
    src = {"w": torch.tensor([5.0, 6.0])}
    copy_state_dict_inplace(dst, src)
    assert torch.equal(dst["w"], src["w"])
