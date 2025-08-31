from torch import Tensor, triu, ones, inf, softmax
from torch.nn import Dropout, Linear, Module


class MultiHeadAttention(Module):

    d_out: int
    num_heads: int
    head_dim: int
    W_query: Linear
    W_key: Linear
    W_value: Linear
    out_proj: Linear
    dropout: Dropout
    mask: Tensor

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = Linear(d_in, d_out)
        self.dropout = Dropout(dropout)

        self.register_buffer(
            "mask", triu(ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        b, num_tokens, _ = x.shape

        keys: Tensor = self.W_key(x)
        queries: Tensor = self.W_query(x)
        values: Tensor = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool: Tensor = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -inf)

        attn_weights = softmax(attn_scores / keys.shape[1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec: Tensor = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec
