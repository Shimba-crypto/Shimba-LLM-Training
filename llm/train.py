"""
train.py — Training loop for the Shimba LLM.

Runs on CUDA, MPS, or CPU — the device is resolved at construction time and
printed in the banner, so you always know what you are training on.

Optimisations applied
---------------------
1. Gradient accumulation — effective batch = batch_size × grad_accum_steps
   without storing large intermediate activations simultaneously.
2. Optional AMP on CUDA (bfloat16 when supported, else float16) via
   torch.autocast + GradScaler. Off by default; enable with `amp=True`.
3. Optional torch.compile via the `compile` flag. Off by default: the first
   compile takes minutes on Colab, and it breaks checkpoint portability
   unless handled (it is handled — see llm/checkpoint.py — but it is still
   not worth the latency for short runs).
4. Cosine learning-rate schedule with linear warmup.
5. Gradient clipping (max_norm=1.0) to stabilise training.
6. Periodic eval on held-out data with @torch.no_grad() for memory savings.
7. Checkpoint saves whenever val loss improves (best-model tracking).
"""

import math
import os
import time
from dataclasses import dataclass
from typing import Optional

import torch

from .model import GPT, GPTConfig
from .data import DataLoader, TextDataset
from .checkpoint import save_checkpoint
from .compat import enable_safe_output
from .device import (
    AUTO,
    autocast_enabled,
    describe_device,
    dtype_for,
    resolve_device,
)


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
    # None → follows max_iters, so the cosine always completes.
    lr_decay_iters:           Optional[int] = None
    min_lr:                   float = 3e-5     # 10% of peak LR

    # Evaluation & checkpointing
    eval_interval:            int   = 500
    eval_iters:               int   = 100      # batches to average for eval loss
    out_path:                 str   = "model.pth"

    # Runtime
    device:                   str   = AUTO     # "auto" | "cpu" | "cuda" | "mps"
    amp:                      bool  = False    # mixed precision on CUDA
    compile:                  bool  = False    # torch.compile (slow first run)

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

    When `lr_decay_iters` is None it tracks `max_iters`, so the cosine always
    finishes exactly at the end of the run regardless of its length.
    """
    decay_iters = cfg.lr_decay_iters if cfg.lr_decay_iters is not None else cfg.max_iters

    if it < cfg.warmup_iters:
        return cfg.learning_rate * it / max(1, cfg.warmup_iters)
    if it > decay_iters:
        return cfg.min_lr
    progress = (it - cfg.warmup_iters) / max(1, decay_iters - cfg.warmup_iters)
    progress = min(1.0, max(0.0, progress))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def resume_from(trainer: "Trainer", path: str, max_iters: int | None = None) -> None:
    """
    Restore model + optimiser + iteration counter from a checkpoint.

    Only model/optimizer/iter/best_val_loss are restored. GradScaler and the
    LR schedule are intentionally left at their defaults: the LR is recomputed
    from `iter_num` on the next step anyway, and a stale scaler can silently
    underflow every gradient.

    `max_iters` is treated as the *total* target, so resuming a 5k run at
    iteration 1200 keeps training to 5000 rather than stopping at 1200.
    """
    from .checkpoint import load_checkpoint

    _, checkpoint = load_checkpoint(path, map_location="cpu")

    trainer.model.load_state_dict(checkpoint["state_dict"])
    trainer.model.to(trainer.device)

    if "optimizer" in checkpoint:
        try:
            trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        except ValueError as exc:
            print(f"[train] optimizer state not restored ({exc}). "
                  "Continuing with a fresh optimizer.")
    else:
        print("[train] checkpoint has no optimizer state — fresh optimizer")

    trainer.iter_num = int(checkpoint.get("iter_num", 0))
    trainer.best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))

    if max_iters is not None:
        trainer.tcfg.max_iters = max(max_iters, trainer.iter_num)

    print(f"[train] resumed at iteration {trainer.iter_num} "
          f"(target {trainer.tcfg.max_iters}, "
          f"best_val_loss={trainer.best_val_loss:.4f})")


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

    `self.device` is the resolved torch.device. `self.model` is the *unwrapped*
    module so state_dict keys stay portable; compilation is tracked separately
    in `self._compiled_model`.
    """

    def __init__(
        self,
        model_cfg:    GPTConfig,
        train_cfg:    TrainConfig,
        train_dataset: TextDataset,
        val_dataset:   TextDataset,
    ):
        torch.manual_seed(train_cfg.seed)
        enable_safe_output()

        self.mcfg = model_cfg
        self.tcfg = train_cfg

        self.device = resolve_device(train_cfg.device)
        self.amp_dtype = dtype_for(self.device)
        self.use_amp = bool(train_cfg.amp) and autocast_enabled(self.device)
        # GradScaler only helps float16; bfloat16 has fp32-range gradients.
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp and self.amp_dtype == torch.float16
        )

        # Data loaders
        self.train_loader = DataLoader(train_dataset, train_cfg.batch_size, shuffle=True)
        self.val_loader   = DataLoader(val_dataset,   train_cfg.batch_size, shuffle=False)

        # Model — always the unwrapped module, moved onto the target device
        self.model = GPT(model_cfg).to(self.device)
        self.model.train()

        self._compiled_model = self._try_compile()
        self._compile_broken = False

        # Optimiser
        self.optimizer = self.model.configure_optimizer(
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )

        self.best_val_loss: float = float("inf")
        self.iter_num:      int   = 0

    # ------------------------------------------------------------------
    def _forward(self, x, y):
        """
        Forward pass under autocast when AMP is on.

        torch.compile can fail at *first call*, not at wrap time — e.g. CPU
        `reduce-overhead` needs a C++ compiler that most machines lack. We
        catch it here, fall back to eager permanently, and keep training.
        """
        if self._compile_broken:
            return self._eager_forward(x, y)

        try:
            if self.use_amp:
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    return self._compiled_model(x, y)
            return self._compiled_model(x, y)
        except Exception as exc:
            if self._compiled_model is self.model:
                raise
            self._compile_broken = True
            self._compiled_model = self.model
            print(f"[train] torch.compile failed at runtime ({type(exc).__name__}: "
                  f"{str(exc)[:120]}) — falling back to eager mode")
            return self._eager_forward(x, y)

    def _eager_forward(self, x, y):
        if self.use_amp:
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                return self.model(x, y)
        return self.model(x, y)

    # ------------------------------------------------------------------
    def _try_compile(self) -> torch.nn.Module:
        """
        Compile only when explicitly asked. Returns the module to call.

        Kept separate from self.model so that saving never has to deal with
        _orig_mod-prefixed keys.
        """
        if not self.tcfg.compile:
            return self.model
        if not hasattr(torch, "compile"):
            print("[train] torch.compile unavailable — running in eager mode")
            return self.model
        try:
            # "reduce-overhead" uses CUDA graphs, which is a GPU-only win and
            # pulls in a C++ compiler on CPU. Default mode is the safe pick
            # everywhere.
            mode = "reduce-overhead" if self.device.type == "cuda" else None
            compiled = torch.compile(self.model, mode=mode) if mode \
                else torch.compile(self.model)
            print("[train] torch.compile enabled (first run will be slow)")
            return compiled
        except Exception as exc:
            print(f"[train] torch.compile skipped: {exc}")
            return self.model

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
                x, y = x.to(self.device), y.to(self.device)
                _, loss = self._forward(x, y)
                total += loss.item()
            losses[split] = total / count
        self.model.train()
        return losses

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Main training loop."""
        tcfg = self.tcfg
        t0 = time.time()

        n_params = sum(p.numel() for p in self.model.parameters())

        print(f"\n{'='*60}")
        print(f"  Shimba LLM — Training")
        print(f"  max_iters={tcfg.max_iters}  batch={tcfg.batch_size}"
              f"×{tcfg.gradient_accumulation_steps}(accum)  lr={tcfg.learning_rate}")
        print(f"  device={describe_device(self.device)}  params={n_params:,}"
              f"  amp={'on/' + str(self.amp_dtype) if self.use_amp else 'off'}")
        print(f"{'='*60}\n")

        # Grab initial batch outside loop to prime the accumulation buffer
        self.optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        accum_count = 0

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
                    self._save(tcfg.out_path)
                    print(f"  ✓ checkpoint saved (val_loss={self.best_val_loss:.4f})")

            # ── Gradient accumulation loop ────────────────────────────
            for _ in range(tcfg.gradient_accumulation_steps):
                x, y = self.train_loader.get_batch()
                x, y = x.to(self.device), y.to(self.device)
                _, loss = self._forward(x, y)
                # Scale loss so gradients are averaged over accumulation steps
                loss = loss / tcfg.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
                accum_loss += loss.item()
                accum_count += 1

            # ── Gradient clip + optimiser step ────────────────────────
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), tcfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            # ── Logging ──────────────────────────────────────────────
            if self.iter_num % tcfg.log_interval == 0 and self.iter_num > 0:
                # Average across the window, not a running sum — a sum scales
                # with log_interval and is not comparable between runs.
                print(
                    f"  iter {self.iter_num:5d}  "
                    f"loss={accum_loss / max(1, accum_count):.4f}  lr={lr:.2e}"
                )
            accum_loss = 0.0
            accum_count = 0
            self.iter_num += 1

        # Final checkpoint if never beaten best
        final_path = tcfg.out_path.replace(".pth", "_final.pth")
        self._save(final_path)
        total_time = time.time() - t0
        print(f"\n[train] done in {total_time:.0f}s  best val_loss={self.best_val_loss:.4f}")
        print(f"[train] best model → {tcfg.out_path}")
        print(f"[train] final model → {final_path}")

    # ------------------------------------------------------------------
    def _save(self, path: str, **extra) -> None:
        """
        Save model + optimiser + scheduler state.

        Writes through the checkpoint helper so the tensors land on disk as
        CPU float32 even when training on a GPU — otherwise a Colab checkpoint
        is only loadable on the exact GPU that made it.
        """
        payload = {
            "optimizer": self.optimizer.state_dict(),
            "iter_num": self.iter_num,
            "best_val_loss": self.best_val_loss,
            **extra,
        }
        save_checkpoint(path, self.model, **payload)
