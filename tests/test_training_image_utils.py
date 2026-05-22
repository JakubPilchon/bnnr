"""Unit tests for training.image_utils."""

from __future__ import annotations

import numpy as np
import torch

from bnnr.training.image_utils import (
    det_uint8_batch_to_float01,
    resize_saliency_batch,
    tensor_batch_to_preview_uint8,
    tensor_to_uint8,
    uint8_to_tensor,
)


def test_tensor_to_uint8_from_01() -> None:
    images = torch.rand(2, 3, 8, 8)
    out = tensor_to_uint8(images)
    assert out.dtype == np.uint8
    assert out.shape == (2, 8, 8, 3)


def test_uint8_to_tensor_roundtrip() -> None:
    ref = torch.rand(2, 3, 4, 4)
    np_u8 = (ref.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    back = uint8_to_tensor(np_u8, ref_batch=ref)
    assert back.shape == ref.shape
    assert back.max() <= 1.05


def test_det_uint8_batch_to_float01() -> None:
    ref = torch.zeros(1, 3, 4, 4)
    np_u8 = np.ones((1, 4, 4, 3), dtype=np.uint8) * 255
    out = det_uint8_batch_to_float01(np_u8, ref_batch=ref)
    assert out.shape == (1, 3, 4, 4)
    assert float(out.max()) <= 1.0


def test_resize_saliency_batch() -> None:
    maps = np.ones((2, 4, 4), dtype=np.float32) * 0.5
    resized = resize_saliency_batch(maps, 8, 8)
    assert resized.shape == (2, 8, 8)


def test_tensor_batch_to_preview_uint8() -> None:
    images = torch.rand(1, 3, 16, 16)
    out = tensor_batch_to_preview_uint8(images)
    assert out.dtype == np.uint8
    assert out.shape == (1, 16, 16, 3)
