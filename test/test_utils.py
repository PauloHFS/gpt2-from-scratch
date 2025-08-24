import os
import sys
import math
import pytest
import torch
from utils import apply_rope, RMSNorm

# Ensure the src directory (tests colocated with utils.py) is on sys.path
sys.path.insert(0, os.path.dirname(__file__))



def test_apply_rope_basic():
    torch.manual_seed(0)
    batch, heads, seq_len, head_dim = 2, 3, 4, 6  # head_dim is even
    x = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float32)

    # Prepare cos and sin with shape (seq_len_total, head_dim)
    # Use a simple pattern so we can verify broadcasting works
    positions = torch.arange(seq_len, dtype=torch.float32)
    dims = torch.arange(head_dim, dtype=torch.float32)
    theta = torch.outer(positions, dims) * 0.01
    cos = torch.cos(theta)
    sin = torch.sin(theta)

    out = apply_rope(x, cos, sin)

    # Manually compute expected result
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)

    cos_b = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin_b = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
    expected = (x * cos_b) + (rotated * sin_b)

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert torch.allclose(out, expected, rtol=1e-5, atol=1e-6)


def test_apply_rope_odd_head_dim_raises():
    x = torch.randn(1, 1, 2, 5)  # head_dim is odd -> 5
    cos = torch.randn(2, 5)
    sin = torch.randn(2, 5)
    with pytest.raises(AssertionError):
        apply_rope(x, cos, sin)


def test_rmsnorm_float16_qwen3_compatible_and_bias():
    torch.manual_seed(1)
    batch, seq_len, emb = 3, 4, 8
    x_fp16 = (torch.randn(batch, seq_len, emb, dtype=torch.float16) * 0.5).clone()

    # Create RMSNorm with bias enabled
    eps = 1e-6
    rn = RMSNorm(emb_dim=emb, eps=eps, bias=True, qwen3_compatible=True)

    # Set deterministic scale and shift for easy expected computation
    with torch.no_grad():
        rn.scale.copy_(torch.linspace(1.0, 2.0, emb))
        rn.shift.copy_(torch.linspace(-0.5, 0.5, emb))

    out = rn(x_fp16)

    # Manual expected computation in float32 then cast back to input dtype
    x32 = x_fp16.to(torch.float32)
    variance = x32.pow(2).mean(dim=-1, keepdim=True)
    expected = x32 * torch.rsqrt(variance + eps)
    expected = expected * rn.scale
    expected = expected + rn.shift
    expected = expected.to(x_fp16.dtype)

    # dtype preserved
    assert out.dtype == x_fp16.dtype
    # values close (float16 tolerances are loose)
    assert torch.allclose(out, expected, rtol=1e-2, atol=1e-2)


def test_rmsnorm_no_bias_shift_none_and_scale_applied():
    torch.manual_seed(2)
    batch, seq_len, emb = 2, 3, 5
    x = torch.randn(batch, seq_len, emb, dtype=torch.float32)

    rn = RMSNorm(emb_dim=emb, bias=False, qwen3_compatible=True)
    # set scale to non-default to verify it's applied
    with torch.no_grad():
        rn.scale.copy_(torch.linspace(0.5, 1.5, emb))

    out = rn(x)

    # shift should be None when bias=False
    assert rn.shift is None

    x32 = x.to(torch.float32)
    variance = x32.pow(2).mean(dim=-1, keepdim=True)
    expected = x32 * torch.rsqrt(variance + rn.eps)
    expected = expected * rn.scale
    expected = expected.to(x.dtype)

    assert out.dtype == x.dtype
    assert torch.allclose(out, expected, rtol=1e-5, atol=1e-6)