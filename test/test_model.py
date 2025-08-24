import pytest
import torch
import types

import src.model as model


class DummyMHA(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, kv_cache=None):
        # return zeros with same shape and propagate kv_cache as None
        return torch.zeros_like(x), None


class DummyGQA(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, kv_cache=None):
        return torch.zeros_like(x), None


def test_feedforward_preserves_shape():
    batch, seq, emb = 2, 4, 8
    ff = model.FeedForward(emb_dim=emb, hidden_dim=emb * 4, drop_rate=0.1)
    x = torch.randn(batch, seq, emb)
    out = ff(x)
    assert out.shape == x.shape


def test_transformerblock_with_mha_and_gqa(monkeypatch):
    # Replace attention implementations with dummies
    monkeypatch.setattr(model, "MultiHeadSelfAttention", DummyMHA)
    monkeypatch.setattr(model, "GroupedQueryAttention", DummyGQA)

    batch, seq, emb = 2, 5, 16

    # Test MHA path
    cfg_mha = {
        "attention_type": "mha",
        "emb_dim": emb,
        "n_heads": 2,
        "context_length": seq,
        "drop_rate": 0.0,
        "n_layers": 1,
        "qkv_bias": True,
    }
    block_mha = model.TransformerBlock(cfg_mha)
    x = torch.randn(batch, seq, emb)
    out, new_cache = block_mha(x)
    assert out.shape == x.shape
    assert new_cache is None

    # Test GQA path
    cfg_gqa = {
        "attention_type": "gqa",
        "emb_dim": emb,
        "n_heads": 2,
        "num_kv_groups": 1,
        "drop_rate": 0.0,
        "n_layers": 1,
    }
    block_gqa = model.TransformerBlock(cfg_gqa)
    x2 = torch.randn(batch, seq, emb)
    out2, new_cache2 = block_gqa(x2)
    assert out2.shape == x2.shape
    assert new_cache2 is None


def test_gpt_forward(monkeypatch):
    # Ensure attention classes won't block GPT construction
    monkeypatch.setattr(model, "MultiHeadSelfAttention", DummyMHA)
    monkeypatch.setattr(model, "GroupedQueryAttention", DummyGQA)

    cfg = {
        "vocab_size": 100,
        "emb_dim": 12,
        "drop_rate": 0.0,
        "n_layers": 1,
        "n_heads": 2,
        "context_length": 8,
        "dropout": 0.0,
        "qkv_bias": False,
    }
    gpt = model.GPT(cfg)
    idx = torch.randint(0, cfg["vocab_size"], (2, 4), dtype=torch.long)

    logits, cache = gpt(idx)
    assert logits.shape == (2, 4, cfg["vocab_size"])
    assert isinstance(cache, list)
    assert len(cache) == cfg["n_layers"]