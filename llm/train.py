"""
train.py — Training loop for the Shimba LLM (CPU-only).

CPU optimisations applied
--------------------------
1. Gradient accumulation — effective batch = batch_size × grad_accum_steps
   without storing large intermediate activations simultaneously.
2. torch.compile() — if PyTorch ≥ 2.0 and Python < 3.12, compile the model
   with the 'reduce-overhead' mode for CPU-friendly graph optimisation.
   Falls back gracefully if unavailable.
3. Cosine learning-rate schedule with linear warmup.
4. Gradient clipping (max_norm=1.0) to stabilise training.
5. Periodic eval on held-out data with @torch.no_grad() for memory savings.
6. Checkpoint saves whenever val loss improves (best-model tracking).
"""

import math
import os
import time
from dataclasses import dataclass
from typing import Optional

import torch

from .model import GPT, GPTConfig
from .data import DataLoader, TextDataset


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Optimisation
    batch_size:               int   = 8
    gradient_accumulation_steps: int = 4       # effective batch = 8×4 = 32
    learning_rate:            float = 3e-4
    max_iters:                int   = 5000
    weight_decay:             float = 0.1
    grad_clip:                float = 1.0

    # Schedule
    warmup_iters:             int   = 200      # linear LR warmup steps
    lr_decay_iters:           int   = 5000     # cosine decay until here
    min_lr:                   float = 3e-5     # 10% of peak LR

    # Evaluation & checkpointing
    eval_interval:            int   = 500
    eval_iters:               int   = 100      # batches to average for eval loss
    out_path:                 str   = "model.pth"

    # Misc
    log_interval:             int   = 50
    seed:                     int   = 42


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

def get_lr(cfg: TrainConfig, it: int) -> float:
    """
    Cosine annealing with linear warmup.

    warmup phase   : 0 → max_lr over warmup_iters steps
    cosine phase   : max_lr → min_lr over lr_decay_iters steps
    flat phase     : min_lr thereafter
    """
    if it < cfg.warmup_iters:
        return cfg.learning_rate * it / max(1, cfg.warmup_iters)
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    progress = (it - cfg.warmup_iters) / max(1, cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Full training loop encapsulated in one class.

    Usage
    -----
    trainer = Trainer(model_cfg, train_cfg, train_dataset, val_dataset)
    trainer.run()
    """

    def __init__(
        self,
        model_cfg:    GPTConfig,
        train_cfg:    TrainConfig,
        train_dataset: TextDataset,
        val_dataset:   TextDataset,
    ):
        torch.manual_seed(train_cfg.seed)
        torch.set_default_device("cpu")   # all tensors on CPU

        self.mcfg = model_cfg
        self.tcfg = train_cfg

        # Data loaders
        self.train_loader = DataLoader(train_dataset, train_cfg.batch_size, shuffle=True)
        self.val_loader   = DataLoader(val_dataset,   train_cfg.batch_size, shuffle=False)

        # Model
        self.model = GPT(model_cfg)
        self.model.train()

        # Optionally compile for speed (PyTorch 2.x, Python < 3.12)
        self._try_compile()

        # Optimiser
        self.optimizer = self.model.configure_optimizer(
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )

        self.best_val_loss: float = float("inf")
        self.iter_num:      int   = 0

    # ------------------------------------------------------------------
    def _try_compile(self) -> None:
        """Attempt torch.compile — silently skip if unavailable."""
        import sys
        if hasattr(torch, "compile") and sys.version_info < (3, 12):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("[train] torch.compile enabled (reduce-overhead)")
            except Exception as e:
                print(f"[train] torch.compile skipped: {e}")
        else:
            print("[train] torch.compile not available — running in eager mode")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _estimate_loss(self) -> dict:
        """
        Evaluate on train and val splits using `eval_iters` random batches each.
        Returns dict with keys 'train' and 'val'.
        """
        self.model.eval()
        losses = {}
        for split, loader in [("train", self.train_loader), ("val", self.val_loader)]:
            total = 0.0
            count = min(self.tcfg.eval_iters, len(loader))
            for _ in range(count):
                x, y = loader.get_batch()
                _, loss = self.model(x, y)
                total += loss.item()
            losses[split] = total / count
        self.model.train()
        return losses

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Main training loop."""
        tcfg = self.tcfg
        t0 = time.time()

        print(f"\n{'='*60}")
        print(f"  Shimba LLM — Training")
        print(f"  max_iters={tcfg.max_iters}  batch={tcfg.batch_size}"
              f"×{tcfg.gradient_accumulation_steps}(accum)  lr={tcfg.learning_rate}")
        print(f"{'='*60}\n")

        # Grab initial batch outside loop to prime the accumulation buffer
        self.optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        while self.iter_num < tcfg.max_iters:
            # ── Learning-rate update ──────────────────────────────────
            lr = get_lr(tcfg, self.iter_num)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            # ── Eval + checkpoint ────────────────────────────────────
            if self.iter_num % tcfg.eval_interval == 0:
                losses = self._estimate_loss()
                elapsed = time.time() - t0
                print(
                    f"  iter {self.iter_num:5d}/{tcfg.max_iters}  "
                    f"train={losses['train']:.4f}  val={losses['val']:.4f}  "
                    f"lr={lr:.2e}  time={elapsed:.0f}s"
                )
                if losses["val"] < self.best_val_loss:
                    self.best_val_loss = losses["val"]
                    self.model.save(tcfg.out_path)
                    print(f"  ✓ checkpoint saved (val_loss={self.best_val_loss:.4f})")

            # ── Gradient accumulation loop ────────────────────────────
            for micro_step in range(tcfg.gradient_accumulation_steps):
                x, y = self.train_loader.get_batch()
                _, loss = self.model(x, y)
                # Scale loss so gradients are averaged over accumulation steps
                loss = loss / tcfg.gradient_accumulation_steps
                loss.backward()
                accum_loss += loss.item()

            # ── Gradient clip + optimiser step ────────────────────────
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), tcfg.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            # ── Logging ──────────────────────────────────────────────
            if self.iter_num % tcfg.log_interval == 0 and self.iter_num > 0:
                print(
                    f"  iter {self.iter_num:5d}  loss={accum_loss:.4f}  lr={lr:.2e}"
                )
            accum_loss = 0.0
            self.iter_num += 1

        # Final checkpoint if never beaten best
        final_path = tcfg.out_path.replace(".pth", "_final.pth")
        self.model.save(final_path)
        total_time = time.time() - t0
        print(f"\n[train] done in {total_time:.0f}s  best val_loss={self.best_val_loss:.4f}")
        print(f"[train] best model → {tcfg.out_path}")
        print(f"[train] final model → {final_path}")
