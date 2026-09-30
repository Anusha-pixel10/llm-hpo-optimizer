"""
Built-in adapter for `nanoGPT <https://github.com/karpathy/nanoGPT>`_
(MIT-licensed, separate project by Andrej Karpathy).

All nanoGPT imports are **lazy** -- this module may be imported without
``torch`` or ``numpy`` installed.  An :class:`ImportError` with an
actionable message is raised only when :meth:`NanoGPTAdapter.run_trial`
is actually called.

Requirements
------------
* ``pip install llm-hpo-optimizer[nanogpt]``  (pulls in ``torch>=2.0`` and ``numpy``)
* nanoGPT's ``model.py`` importable on your Python path::

      export PYTHONPATH=/path/to/nanoGPT:$PYTHONPATH
"""
from __future__ import annotations

import os
import time
from typing import Optional

from llm_hpo_optimizer.adapter import TrainingAdapter, TrainingResult
from llm_hpo_optimizer.space import Categorical, LogUniformFloat, SearchSpace, UniformFloat


def _nanogpt_get_batch(dataset_dir, split, batch_size, block_size, device):
    import numpy as np
    import torch

    filename = "train.bin" if split == "train" else "val.bin"
    data = np.memmap(os.path.join(dataset_dir, filename), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix]
    )
    if device.startswith("cuda"):
        return (
            x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True),
        )
    return x.to(device), y.to(device)


def _nanogpt_resolve_vocab_size(dataset_dir: str, fallback: int = 50304) -> int:
    """Read vocab_size from ``meta.pkl`` if present, else return *fallback*."""
    import pickle

    meta_path = os.path.join(dataset_dir, "meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            return pickle.load(f)["vocab_size"]
    return fallback


class NanoGPTAdapter(TrainingAdapter):
    """Adapter that trains a `nanoGPT <https://github.com/karpathy/nanoGPT>`_ model.

    Parameters
    ----------
    dataset_dir:
        Path to a directory containing ``train.bin``, ``val.bin``, and
        optionally ``meta.pkl``.
    max_iters:
        Total training iterations per trial.
    eval_interval:
        Evaluate on the validation set every this many iterations.
    eval_iters:
        Number of validation batches to average per evaluation.
    block_size:
        Context length in tokens (default 256).
    vocab_size:
        Override vocab size.  If ``None``, read from ``meta.pkl`` or
        fall back to 50 304.
    bias:
        Whether to use bias in Linear / LayerNorm layers.
    weight_decay:
        AdamW weight decay.
    beta1, beta2:
        AdamW momentum terms.
    grad_clip:
        Gradient clipping norm (0 disables).
    device:
        Override device string.  Auto-detected when ``None``.
    lr_warmup_iters:
        Number of iterations for linear learning-rate warm-up.  Set to 0 to
        disable warm-up.  Defaults to 100, matching stock nanoGPT behaviour.
    lr_decay_iters:
        Iteration at which the cosine LR decay reaches its minimum.  When
        ``None`` (default), defaults to *max_iters*.  Set to 0 to use a flat
        learning rate throughout (no scheduler).
    min_lr_ratio:
        Minimum LR expressed as a fraction of the peak learning rate
        (default 0.1, i.e. ``min_lr = 0.1 * learning_rate``).

    Raises
    ------
    ImportError
        If ``torch`` or ``numpy`` is not installed.
    ImportError
        If nanoGPT's ``model.py`` cannot be imported.
    """

    def __init__(
        self,
        dataset_dir: str,
        max_iters: int,
        eval_interval: int,
        eval_iters: int,
        block_size: int = 256,
        vocab_size: Optional[int] = None,
        bias: bool = False,
        weight_decay: float = 1e-1,
        beta1: float = 0.9,
        beta2: float = 0.95,
        grad_clip: float = 1.0,
        device: Optional[str] = None,
        lr_warmup_iters: int = 100,
        lr_decay_iters: Optional[int] = None,
        min_lr_ratio: float = 0.1,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.bias = bias
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.grad_clip = grad_clip
        self.device = device
        self.lr_warmup_iters = lr_warmup_iters
        self.lr_decay_iters = lr_decay_iters if lr_decay_iters is not None else max_iters
        self.min_lr_ratio = min_lr_ratio

    def _get_lr(self, it: int, base_lr: float) -> float:
        """Cosine LR schedule with linear warmup, matching stock nanoGPT."""
        import math as _math
        # Flat LR when scheduler is disabled
        if self.lr_decay_iters == 0:
            return base_lr
        # Linear warmup
        if it < self.lr_warmup_iters:
            return base_lr * (it + 1) / max(self.lr_warmup_iters, 1)
        # After decay horizon: clamp to min_lr
        if it >= self.lr_decay_iters:
            return base_lr * self.min_lr_ratio
        # Cosine decay
        decay_ratio = (it - self.lr_warmup_iters) / max(
            self.lr_decay_iters - self.lr_warmup_iters, 1
        )
        coeff = 0.5 * (1.0 + _math.cos(_math.pi * decay_ratio))
        return base_lr * (self.min_lr_ratio + coeff * (1.0 - self.min_lr_ratio))

    def run_trial(self, hyperparams: dict, seed: int) -> TrainingResult:
        """Train a nanoGPT model and return its metrics.

        Parameters
        ----------
        hyperparams:
            Must contain: ``n_layer``, ``n_head``, ``n_embd``, ``dropout``,
            ``learning_rate``, ``batch_size``.
        seed:
            Random seed for ``torch.manual_seed``.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "NanoGPTAdapter requires torch. "
                "Install it with: pip install llm-hpo-optimizer[nanogpt]"
            ) from exc
        try:
            import numpy  # noqa: F401 -- ensure numpy is available for data loading
        except ImportError as exc:
            raise ImportError(
                "NanoGPTAdapter requires numpy. "
                "Install it with: pip install llm-hpo-optimizer[nanogpt]"
            ) from exc
        try:
            from model import GPTConfig, GPT  # nanoGPT -- must be on PYTHONPATH
        except ImportError as exc:
            raise ImportError(
                "NanoGPTAdapter requires nanoGPT's model.py on your Python path. "
                "Clone https://github.com/karpathy/nanoGPT and add it to PYTHONPATH: "
                "  export PYTHONPATH=/path/to/nanoGPT:$PYTHONPATH"
            ) from exc

        import gc
        from contextlib import nullcontext

        vocab_size = self.vocab_size or _nanogpt_resolve_vocab_size(self.dataset_dir)
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        device_type = "cuda" if device.startswith("cuda") else "cpu"
        torch.manual_seed(seed)
        if device_type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        dtype = (
            torch.bfloat16
            if (device_type == "cuda" and torch.cuda.is_bf16_supported())
            else torch.float16
        )
        ctx = (
            nullcontext()
            if device_type == "cpu"
            else torch.amp.autocast(device_type=device_type, dtype=dtype)
        )

        model = optimizer_ = None
        try:
            cfg = GPTConfig(
                block_size=self.block_size,
                vocab_size=vocab_size,
                n_layer=hyperparams["n_layer"],
                n_head=hyperparams["n_head"],
                n_embd=hyperparams["n_embd"],
                dropout=hyperparams["dropout"],
                bias=self.bias,
            )
            model = GPT(cfg).to(device)
            parameter_count = model.get_num_params()
            model_size_mb = parameter_count * 4 / (1024**2)
            optimizer_ = model.configure_optimizers(
                self.weight_decay,
                hyperparams["learning_rate"],
                (self.beta1, self.beta2),
                device_type,
            )
            batch_size = hyperparams["batch_size"]
            best_val_loss, best_val_iter = float("inf"), 0
            train_loss = val_loss = float("nan")  # populated at first eval; nan signals no eval yet
            samples_seen = 0

            t0 = time.time()
            X, Y = _nanogpt_get_batch(
                self.dataset_dir, "train", batch_size, self.block_size, device
            )
            for it in range(self.max_iters):
                if it % self.eval_interval == 0 or it == self.max_iters - 1:
                    model.eval()
                    with torch.no_grad():
                        losses: dict = {}
                        for split in ["train", "val"]:
                            vals = torch.zeros(self.eval_iters)
                            for k in range(self.eval_iters):
                                Xe, Ye = _nanogpt_get_batch(
                                    self.dataset_dir, split, batch_size, self.block_size, device
                                )
                                with ctx:
                                    _, l = model(Xe, Ye)
                                vals[k] = l.item()
                            losses[split] = vals.mean().item()
                    model.train()
                    train_loss, val_loss = losses["train"], losses["val"]
                    if val_loss < best_val_loss:
                        best_val_loss, best_val_iter = val_loss, it
                # Apply LR schedule
                lr = self._get_lr(it, hyperparams["learning_rate"])
                for param_group in optimizer_.param_groups:
                    param_group["lr"] = lr

                with ctx:
                    _, loss = model(X, Y)
                X, Y = _nanogpt_get_batch(
                    self.dataset_dir, "train", batch_size, self.block_size, device
                )
                loss.backward()
                if self.grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                optimizer_.step()
                optimizer_.zero_grad(set_to_none=True)
                samples_seen += batch_size * self.block_size

            training_time = time.time() - t0
            peak_mem = (
                torch.cuda.max_memory_allocated() / (1024**2)
                if device_type == "cuda"
                else 0.0
            )

            return TrainingResult(
                train_loss=train_loss,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                best_val_iter=best_val_iter,
                training_iterations=self.max_iters,
                parameter_count=parameter_count,
                model_size_mb=model_size_mb,
                peak_memory_mb=peak_mem,
                training_time_seconds=training_time,
                samples_seen=samples_seen,
                samples_per_second=samples_seen / training_time if training_time > 0 else 0.0,
                device=device,
                seed=seed,
                failed=False,
            )
        except RuntimeError as e:
            return TrainingResult(
                train_loss=float("nan"),
                val_loss=float("nan"),
                best_val_loss=float("nan"),
                best_val_iter=-1,
                training_iterations=0,
                parameter_count=0,
                model_size_mb=0.0,
                peak_memory_mb=0.0,
                training_time_seconds=0.0,
                samples_seen=0,
                samples_per_second=0.0,
                device=device,
                seed=seed,
                failed=True,
                error_message=str(e),
            )
        finally:
            del model, optimizer_
            gc.collect()
            if device_type == "cuda":
                torch.cuda.empty_cache()


#: Default search space for nanoGPT hyperparameter optimization.
DEFAULT_NANOGPT_SEARCH_SPACE = SearchSpace(
    parameters={
        "learning_rate": LogUniformFloat(1e-4, 5e-4),
        "batch_size":    Categorical([16, 32, 64]),
        "n_layer":       Categorical([4, 6, 8]),
        "n_head":        Categorical([4, 6, 8]),
        "n_embd":        Categorical([256, 384, 512]),
        "dropout":       UniformFloat(0.0, 0.3),
    }
)

#: Fixed baseline configuration used in trial 0.
DEFAULT_NANOGPT_BASELINE: dict = {
    "learning_rate": 3e-4,
    "batch_size": 32,
    "n_layer": 4,
    "n_head": 4,
    "n_embd": 256,
    "dropout": 0.0,
}
