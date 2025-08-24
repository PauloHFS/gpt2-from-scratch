from torch import nn

from .attention import MultiHeadSelfAttention, GroupedQueryAttention

class FeedForward(nn.Module):
    def __init__(self, emb_dim, hidden_dim, drop_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        attention_type = config.get("attention_type", "mha")
        
        if attention_type == "gqa":
            assert "num_kv_groups" in config, "A configuração deve incluir 'num_kv_groups' para GQA."
            self.attention = GroupedQueryAttention(
                d_in=config["emb_dim"],
                num_heads=config["n_heads"],
                num_kv_groups=config["num_kv_groups"]
            )
        else: # "mha"
            self.attention = MultiHeadSelfAttention(
                d_in=config["emb_dim"],
                d_out=config["emb_dim"],
                context_length=config["context_length"],
                dropout=config["drop_rate"],
                num_heads=config["n_heads"],
                qkv_bias=config["qkv_bias"]
            )
        
        hidden_dim = 4 * config["emb_dim"]
        self.ffn = FeedForward(config["emb_dim"], hidden_dim, config["drop_rate"])
        
        self.ln1 = nn.LayerNorm(config["emb_dim"])
        self.ln2 = nn.LayerNorm(config["emb_dim"])
        
    def forward(self, x, kv_cache=None):        
        attn_output, new_kv_cache = self.attention(self.ln1(x), kv_cache=kv_cache)
        
        x = x + attn_output
        
        ffn_output = self.ffn(self.ln2(x))
        x = x + ffn_output
        
        return x, new_kv_cache

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config["vocab_size"], config["emb_dim"])
                
        self.drop_emb = nn.Dropout(config["drop_rate"])
        
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["n_layers"])]
        )
        
        self.final_ln = nn.LayerNorm(config["emb_dim"])
        
        self.lm_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)
        

    def forward(self, idx, kv_cache=None):
        x = self.token_embedding(idx)
        x = self.drop_emb(x)
        
        if kv_cache is None:
            kv_cache = [None] * len(self.transformer_blocks)
        
        for i, block in enumerate(self.transformer_blocks):
            x, new_cache = block(x, kv_cache[i])
            kv_cache[i] = new_cache
        
        x = self.final_ln(x)
        logits = self.lm_head(x)
        
        return logits, kv_cache