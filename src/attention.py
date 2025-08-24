from torch import nn
import torch
from utils import RMSNorm, apply_rope

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads 
        
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x, kv_cache=None):
        b, num_tokens = x.shape[:2]
        past_len = 0

        queries = self.W_query(x)  # (b, num_tokens, d_out)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            past_key, past_value = kv_cache
            past_len = past_key.shape[2]
            keys = torch.cat([past_key, keys], dim=2)
            values = torch.cat([past_value, values], dim=2)

        attn_scores = queries @ keys.transpose(2, 3) / (self.head_dim ** 0.5)
        
        total_len = keys.shape[2]
        mask_bool = self.mask.bool()[:total_len, :total_len]
        mask_q = mask_bool[past_len:past_len+num_tokens, :total_len]
        attn_scores = attn_scores.masked_fill(mask_q.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        new_kv_cache = (keys, values)
        return context_vec, new_kv_cache
    
class GroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, kv_cache=None):

        b, num_tokens = x.shape[:2]
        past_len = 0

        queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)      # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)  # (b, num_tokens, num_kv_groups * head_dim)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            past_key, past_value = kv_cache
            past_len = past_key.shape[2]
            keys = torch.cat([past_key, keys], dim=2)
            values = torch.cat([past_value, values], dim=2)

        keys_expanded = keys.repeat_interleave(self.group_size, dim=1)
        values_expanded = values.repeat_interleave(self.group_size, dim=1)

        if self.q_norm is not None and self.k_norm is not None:
            queries = self.q_norm(queries)
            keys_expanded = self.k_norm(keys_expanded)

        attn_scores = (queries @ keys_expanded.transpose(2, 3)) / (self.head_dim ** 0.5)

        total_len = keys_expanded.shape[2]
        mask = torch.triu(torch.ones((total_len, total_len), device=x.device), diagonal=1).bool()
        mask = mask[past_len:past_len+num_tokens, :total_len]
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)

        context_vec = (attn_weights @ values_expanded).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        new_kv_cache = (keys, values)
        return context_vec, new_kv_cache

    def _slice_rope(self, tensor, start, length):
        if tensor is None:
            return None
        if tensor.ndim == 2:
            return tensor[start:start+length]
        if tensor.ndim == 3:
            return tensor[:, start:start+length, :]
        return tensor