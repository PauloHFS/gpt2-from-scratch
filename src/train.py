import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .model import GPT
from .utils import MetricsTracker, get_gpu_memory_usage, measure_throughput
from .evaluate import calculate_perplexity


def train_model(
    model: GPT,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[GPT, MetricsTracker]:
    lr = config.get('learning_rate', 3e-4)
    n_epochs = config.get('epochs', 10)
    warmup_steps = config.get('warmup_steps', 1000)
    grad_clip = config.get('grad_clip', 1.0)
    val_interval = config.get('val_interval', 200)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    metrics = MetricsTracker()
    
    model = model.to(device)
    model.train()
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
            step_start = time.time()
            
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            
            metrics.update({
                'train_loss': loss.item(),
                'throughput': measure_throughput(
                    input_ids.size(0), input_ids.size(1), step_start
                ),
                'gpu_memory': get_gpu_memory_usage()
            })
            
            if global_step > 0 and global_step % val_interval == 0:
                model.eval()
                val_metrics = validate_model(model, val_loader, device)
                metrics.update({'val_loss': val_metrics['loss']})
                
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    torch.save(model.state_dict(), 'best_model.pt')
                
                model.train()
            
            global_step += 1
            
        scheduler.step()
        
        print(f"Epoch {epoch + 1}/{n_epochs} - "
              f"Train Loss: {metrics.get_mean('train_loss', 100):.4f}, "
              f"Val Loss: {metrics.get_latest('val_loss'):.4f}, "
              f"Throughput: {metrics.get_latest('throughput'):.1f} tokens/sec")
    
    return model, metrics


def validate_model(
    model: GPT,
    val_loader: DataLoader,
    device: str
) -> Dict[str, float]:
    with torch.no_grad():
        val_loss = 0.0
        n_batches = 0
        
        for input_ids, target_ids in val_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1)
            )
            
            val_loss += loss.item()
            n_batches += 1
        
        return {
            'loss': val_loss / n_batches
        }
