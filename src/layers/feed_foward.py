from torch import Tensor
from torch.nn import Module, Linear, Sequential

from src.layers.gelu import GELU


class FeedFoward(Module):

    layers: Sequential

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.layers = Sequential(
            Linear(emb_dim, 4 * emb_dim), GELU(), Linear(4 * emb_dim, emb_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
