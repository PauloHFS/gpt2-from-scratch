from math import pi

from time import time
from torch import Tensor, cat, ones, zeros, rsqrt, float32
from torch.cuda import is_available, max_memory_allocated
from torch.nn import Module, Parameter


def get_gpu_memory_usage() -> float:
    if is_available():
        return max_memory_allocated() / (1024**3)
    return 0.0


def measure_throughput(batch_size: int, seq_length: int, start_time: float) -> float:
    total_tokens = batch_size * seq_length
    elapsed_time = time() - start_time
    return total_tokens / elapsed_time if elapsed_time > 0 else 0.0


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor):
    _, _, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]

    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    rotated = cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)


class RMSNorm(Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = Parameter(ones(emb_dim))
        self.shift = Parameter(zeros(emb_dim)) if bias else None

    def forward(self, x: Tensor):
        input_dtype = x.dtype

        if self.qwen3_compatible:
            x = x.to(float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)
