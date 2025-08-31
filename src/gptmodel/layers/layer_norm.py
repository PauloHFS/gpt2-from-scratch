from torch import Tensor, ones, sqrt, zeros
from torch.nn import Module, Parameter


class LayerNorm(Module):

    ep5: float
    scale: Parameter
    shift: Parameter

    def __init__(self, emb_dim: int):
        super().__init__()
        self.ep5 = 1e-5
        self.scale = Parameter(ones(emb_dim))
        self.shift = Parameter(zeros(emb_dim))

    def forward(self, x: Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / sqrt(var + self.ep5)
        return self.scale * norm_x + self.shift
