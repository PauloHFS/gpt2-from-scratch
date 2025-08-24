from math import pi
import time
import torch
import torch.nn as nn
from typing import Dict, Any

def get_gpu_memory_usage() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0

def measure_throughput(batch_size: int, seq_length: int, start_time: float) -> float:
    total_tokens = batch_size * seq_length
    elapsed_time = time.time() - start_time
    return total_tokens / elapsed_time if elapsed_time > 0 else 0.0

class MetricsTracker:
    def __init__(self):
        self.metrics: Dict[str, list] = {
            'train_loss': [], 'val_loss': [],
            'throughput': [], 'gpu_memory': []
        }
    
    def update(self, metrics_dict: Dict[str, Any]) -> None:
        for key, value in metrics_dict.items():
            if key in self.metrics:
                self.metrics[key].append(float(value))
    
    def get_latest(self, metric_name: str) -> float:
        values = self.metrics.get(metric_name, [])
        return values[-1] if values else 0.0
    
    def get_mean(self, metric_name: str, window_size: int = 100) -> float:
        values = self.metrics.get(metric_name, [])
        if not values:
            return 0.0
        window = values[-window_size:]
        return sum(window) / len(window)

def apply_rope(x, cos, sin):
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2:]

    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype

        if self.qwen3_compatible:
            x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)