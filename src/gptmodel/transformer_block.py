from torch import Tensor
from torch.nn import Module, Dropout

from src.gptmodel.layers.feed_foward import FeedFoward
from src.gptmodel.layers.layer_norm import LayerNorm
from src.gptmodel.multi_head_attention import MultiHeadAttention


class TransformerBlock(Module):

    att: MultiHeadAttention
    ff: FeedFoward
    norm1: LayerNorm
    drop_shortcut: Dropout

    def __init__(
        self,
        emb_dim: int,
        context_length: int,
        n_heads: int,
        drop_rate: float,
        qkv_bias: bool,
    ) -> None:
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=emb_dim,
            d_out=emb_dim,
            context_length=context_length,
            num_heads=n_heads,
            dropout=drop_rate,
            qkv_bias=qkv_bias,
        )
        self.ff = FeedFoward(emb_dim)
        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim)
        self.drop_shortcut = Dropout(drop_rate)

    def foward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
