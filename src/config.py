GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

GPT_CONFIG_MICRO = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 384,
    "n_heads": 6,
    "n_layers": 6,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "attention_type": "gqa",
    "num_kv_groups": 2
}

GPT_CONFIG_SMALL_COLAB = {
    "vocab_size": 50257,
    "context_length": 512,
    "emb_dim": 512,
    "n_heads": 8,
    "n_layers": 8,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "attention_type": "gqa",
    "num_kv_groups": 4
}