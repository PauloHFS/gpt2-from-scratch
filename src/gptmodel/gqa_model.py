from time import time
import torch
from torch import (
    Tensor,
    no_grad,
    tensor,
    argmax,
    arange,
    cat,
    device,
    multinomial,
    save,
    softmax,
    topk,
    where,
    triu,
    ones,
    bool
)
from torch.cuda import is_available
from torch.nn import Dropout, Embedding, Linear, Module, Sequential, ModuleList
from torch.nn.functional import cross_entropy
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.gptmodel.metrics import MetricsTracker
from src.gptmodel.utils import measure_throughput, get_gpu_memory_usage, RMSNorm, compute_rope_params
from src.gptmodel.gqa_transformer_block import GQATransformerBlock

class GQAModel(Module):
    def __init__(self, cfg):
        super().__init__()

        # Main model parameters
        self.tok_emb = Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=torch.bfloat16)

        self.trf_blocks = ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
            [GQATransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=torch.bfloat16)

        # Reusuable utilities
        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg


    def forward(self, in_idx):
        # Forward pass
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        num_tokens = x.shape[1]
        mask = triu(ones(num_tokens, num_tokens, device=x.device, dtype=bool), diagonal=1)

        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)
        x = self.final_norm(x)
        logits = self.out_head(x.to(torch.bfloat16))
        return logits
    
    def _validate_model(
        model: "GQAModel", val_loader: DataLoader[Tensor], device_str: str
    ) -> float:
        with no_grad():
            val_loss = 0.0
            n_batches = 0

            for input_ids, target_ids in val_loader:
                input_ids: Tensor = input_ids.to(device_str)
                target_ids: Tensor = target_ids.to(device_str)

                logits: Tensor = model(input_ids)
                loss = cross_entropy(
                    logits.view(-1, logits.size(-1)), target_ids.view(-1)
                )

                val_loss += loss.item()
                n_batches += 1

            return val_loss / n_batches

    def _train(
        model: "GQAModel",
        train_loader: DataLoader[Tensor],
        val_loader: DataLoader[Tensor],
        device_str: str = "cuda" if is_available() else "cpu",
        learning_rate: float = 3e-4,
        epochs: int = 1,
        grand_clip: float = 1.0,
        val_interval: int = 200,
        path: str = "bestmodel.pt",
    ) -> tuple["GQAModel", MetricsTracker]:

        print(f"Training model on device: {device_str}")

        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
        # T_max deve ser o número total de passos se o scheduler for por passo
        # ou épocas se for por época. A correção abaixo assume atualização por época.
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        metrics = MetricsTracker()

        model = model.to(device_str)

        global_step = 0
        best_val_loss = float("inf")

        for epoch in tqdm(range(epochs)):
            model.train() # Garante que o modelo está em modo de treino no início da época
            for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
                step_start = time()

                input_ids: Tensor = input_ids.to(device_str)
                target_ids: Tensor = target_ids.to(device_str)

                logits: Tensor = model(input_ids)
                loss = cross_entropy(
                    logits.view(-1, logits.size(-1)), target_ids.view(-1)
                )

                optimizer.zero_grad()
                loss.backward()

                if grand_clip > 0.0:
                    clip_grad_norm_(model.parameters(), grand_clip)

                optimizer.step()

                metrics.train_loss.append(loss.item())
                metrics.throughput.append(
                    measure_throughput(input_ids.size(0), input_ids.size(1), step_start)
                )
                metrics.gpu_memory.append(get_gpu_memory_usage())

                if global_step > 0 and global_step % val_interval == 0:
                    model.eval()
                    val_metrics = GQAModel._validate_model(
                        model, val_loader, device_str
                    )
                    metrics.val_loss.append(val_metrics)

                    tqdm.write(
                        f"Epoch {epoch + 1}/{epochs} | Step {global_step} - "
                        f"Train Loss: {sum(metrics.train_loss[-100:])/len(metrics.train_loss[-100:]):.4f}, "
                        f"Val Loss {val_metrics:.4f}, "
                        f"Throughput: {metrics.throughput[-1]:.1f} tokens / sec"
                    )

                    if val_metrics < best_val_loss:
                        best_val_loss = val_metrics
                        save(model.state_dict(), path)
                        tqdm.write(f"New best model saved to {path} with val_loss: {best_val_loss:.4f}")

                    model.train()

                global_step += 1
            
            scheduler.step()

        return model, metrics

    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        context_size: int,
        temperature: float = 0.0,
        top_k: int = 0,
        eos_id: int = 0,
    ) -> Tensor:

        device("cuda" if is_available() else "cpu")

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with no_grad():
                logits: Tensor = self(idx_cond)
            logits = logits[:, -1, :]

            if top_k:
                top_logits, _ = topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = where(
                    logits < min_val, tensor(float("-inf")).to(logits.device), logits
                )

            if temperature > 0.0:
                logits = logits / temperature
                probs = softmax(logits, dim=-1)
                idx_next = multinomial(probs, num_samples=1)
            else:
                idx_next = argmax(logits, dim=-1, keepdim=True)

            if idx_next == eos_id:
                break

            idx = cat((idx, idx_next), dim=1)

        return idx