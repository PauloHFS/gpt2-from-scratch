import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional

from .model import GPT
from .utils import get_gpu_memory_usage, measure_throughput


def calculate_perplexity(
    model: GPT,
    test_loader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (input_ids, target_ids) in enumerate(test_loader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            logits = model(input_ids)
            
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += target_ids.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    total_time = time.time() - start_time
    tokens_per_sec = total_tokens / total_time
    
    return {
        'perplexity': perplexity,
        'loss': avg_loss,
        'tokens_per_sec': tokens_per_sec,
        'gpu_memory_gb': get_gpu_memory_usage()
    }
