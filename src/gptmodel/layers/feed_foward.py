import torch
from torch import Tensor
from torch.nn import Module, Linear, Sequential, functional

from src.gptmodel.layers.gelu import GELU


class FeedFoward(Module):

    layers: Sequential

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.layers = Sequential(
            Linear(emb_dim, 4 * emb_dim), GELU(), Linear(4 * emb_dim, emb_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

class SILUFeedForward(Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=torch.bfloat16, bias=False)
        self.fc2 = Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=torch.bfloat16, bias=False)
        self.fc3 = Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=torch.bfloat16, bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = functional.silu(x_fc1) * x_fc2
        return self.fc3(x)