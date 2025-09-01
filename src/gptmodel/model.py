from time import time
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
)
from torch.cuda import is_available
from torch.nn import Dropout, Embedding, Linear, Module, Sequential
from torch.nn.functional import cross_entropy
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.gptmodel.metrics import MetricsTracker
from src.gptmodel.utils import measure_throughput, get_gpu_memory_usage
from src.gptmodel.transformer_block import TransformerBlock
from src.gptmodel.layers.layer_norm import LayerNorm


class GPTModel(Module):
    tok_emb: Embedding
    pos_emb: Embedding
    drop_emb: Dropout
    trf_blocks: Sequential
    final_norm: LayerNorm
    out_head: Linear

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        context_length: int,
        drop_rate: float,
        n_heads: int,
        n_layers: int,
        qkv_bias: bool,
    ) -> None:
        super().__init__()
        self.tok_emb = Embedding(vocab_size, emb_dim)
        self.pos_emb = Embedding(context_length, emb_dim)
        self.drop_emb = Dropout(drop_rate)
        self.trf_blocks = Sequential(
            *[
                TransformerBlock(emb_dim, context_length, n_heads, drop_rate, qkv_bias)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = LayerNorm(emb_dim)
        self.out_head = Linear(emb_dim, vocab_size, bias=False)

    def forward(self, in_idx: Tensor) -> Tensor:
        _, seq_len = in_idx.shape
        tok_embeds: Tensor = self.tok_emb(in_idx)
        pos_embeds: Tensor = self.pos_emb(arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    def _validate_model(
        model: "GPTModel", val_loader: DataLoader[Tensor], device_str: str
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
        model: "GPTModel",
        train_loader: DataLoader[Tensor],
        val_loader: DataLoader[Tensor],
        device_str: str = "cuda" if is_available() else "cpu",
        learning_rate: float = 3e-4,
        epochs: int = 1,
        grand_clip: float = 1.0,
        val_interval: int = 200,
        path: str = "bestmodel.pt",
    ) -> tuple["GPTModel", MetricsTracker]:

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
                    val_metrics = GPTModel._validate_model(
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
