from torch.nn import Module
from torch import Tensor, pi, pow, tanh, tensor, sqrt


class GELU(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * x * (1 + tanh(sqrt(tensor(2.0 / pi)) * (x + 0.044715 * pow(x, 3))))
