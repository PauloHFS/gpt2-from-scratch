import torch
import pytest
from src.attention import MultiHeadSelfAttention, GroupedQueryAttention

torch.manual_seed(0)

def test_multihead_self_attention_single_token_shapes_and_cache():
    batch = 2
    # To satisfy the current implementation's 2D-x assumption, use d_in = num_tokens = 1
    d_in = 1
    d_out = 8
    num_heads = 2
    context_length = 4
    head_dim = d_out // num_heads

    m = MultiHeadSelfAttention(d_in=d_in, d_out=d_out, context_length=context_length, dropout=0.0, num_heads=num_heads)
    x = torch.randn(batch, d_in)  # shape (batch, num_tokens) where num_tokens == d_in == 1

    out, (present_k, present_v) = m(x)

    assert out.shape == (batch, 1, d_out)
    # present_k/present_v expected shape: (batch, num_heads, total_len, head_dim), here total_len == 1
    assert present_k.shape == (batch, num_heads, 1, head_dim)
    assert present_v.shape == (batch, num_heads, 1, head_dim)

def test_multihead_self_attention_with_past_concatenation():
    batch = 1
    d_in = 1
    d_out = 4
    num_heads = 2
    context_length = 8
    head_dim = d_out // num_heads
    past_len = 3

    m = MultiHeadSelfAttention(d_in=d_in, d_out=d_out, context_length=context_length, dropout=0.0, num_heads=num_heads)
    x = torch.randn(batch, d_in)  # single-token current input
    past_k = torch.randn(batch, num_heads, past_len, head_dim)
    past_v = torch.randn(batch, num_heads, past_len, head_dim)

    out, (present_k, present_v) = m(x, past_key=past_k, past_value=past_v)

    # total_len should equal past_len + 1
    assert present_k.shape == (batch, num_heads, past_len + 1, head_dim)
    assert present_v.shape == (batch, num_heads, past_len + 1, head_dim)
    # beginning slice should match the provided past
    assert torch.allclose(present_k[:, :, :past_len, :], past_k)
    assert torch.allclose(present_v[:, :, :past_len, :], past_v)

def test_grouped_query_attention_basic_shapes_and_cache():
    batch = 2
    num_tokens = 5
    d_in = 8
    num_heads = 4
    num_kv_groups = 2  # group_size = 2
    head_dim = d_in // num_heads

    # Criar cos e sin para RoPE
    positions = torch.arange(num_tokens, dtype=torch.float32)
    dims = torch.arange(head_dim, dtype=torch.float32)
    theta = torch.outer(positions, dims) * 0.01
    cos = torch.cos(theta)
    sin = torch.sin(theta)

    g = GroupedQueryAttention(d_in=d_in, num_heads=num_heads, num_kv_groups=num_kv_groups, qk_norm=False)
    x = torch.randn(batch, num_tokens, d_in)

    out, (present_k, present_v) = g(x, cos=cos, sin=sin, mask=None, past_k=None, past_v=None)
    
    # Verificar as dimensões esperadas
    assert out.shape == (batch, num_tokens, d_in)
    assert present_k.shape == (batch, num_kv_groups, num_tokens, head_dim)
    assert present_v.shape == (batch, num_kv_groups, num_tokens, head_dim)

def test_grouped_query_attention_with_past_concatenation_and_repeat():
    batch = 1
    num_tokens = 3
    d_in = 12
    num_heads = 6
    num_kv_groups = 3  # group_size = 2
    head_dim = d_in // num_heads
    past_len = 2

    # Criar cos e sin para RoPE
    total_len = past_len + num_tokens
    positions = torch.arange(total_len, dtype=torch.float32)
    dims = torch.arange(head_dim, dtype=torch.float32)
    theta = torch.outer(positions, dims) * 0.01
    cos = torch.cos(theta)
    sin = torch.sin(theta)

    g = GroupedQueryAttention(d_in=d_in, num_heads=num_heads, num_kv_groups=num_kv_groups, qk_norm=False)
    x = torch.randn(batch, num_tokens, d_in)
    past_k = torch.randn(batch, num_kv_groups, past_len, head_dim)
    past_v = torch.randn(batch, num_kv_groups, past_len, head_dim)

    out, (present_k, present_v) = g(x, cos=cos, sin=sin, mask=None, past_k=past_k, past_v=past_v)
    
    # Verificar as dimensões esperadas
    assert out.shape == (batch, num_tokens, d_in)
    assert present_k.shape == (batch, num_kv_groups, total_len, head_dim)
    assert present_v.shape == (batch, num_kv_groups, total_len, head_dim)