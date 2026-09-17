"""Raw-PyTorch development trainer with exact, manifest-bound recovery.

Run with `python -m training.pretrain --help`. No download or rental logic.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.attention import SharedAttentionRuntime
from models.ced import CEDRuntime
from models.moe import Gate


class TrainingFailure(RuntimeError):
    pass


class NonfiniteError(TrainingFailure):
    pass


@dataclass
class TrainingConfig:
    max_steps: int = 2000
    stage_id: int = 0
    batch_size: int = 1
    accumulation_steps: int = 1
    warmup_steps: int = 200
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = .1
    checkpoint_seconds: float = 1800
    checkpoint_activations: bool = True
    compile_model: bool = False
    router_bias_rate: float = 1e-3
    text_only: bool = True
    router_z_loss: float = 0.0  # C1 fallback "router_z_loss_1e-3"

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.max_steps, self.batch_size, self.accumulation_steps)):
            raise ValueError("Step and batch counts must be positive integers")
        if self.warmup_steps < 0 or self.stage_id < 0 or not 0 < self.checkpoint_seconds <= 1800:
            raise ValueError("Invalid warmup/stage/checkpoint interval")
        if (not 0 < self.min_learning_rate <= self.learning_rate or self.weight_decay < 0
                or self.router_bias_rate < 0 or self.router_z_loss < 0):
            raise ValueError("Invalid optimizer settings")


def seed_everything(stage_id=0):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    seed = 1337 + stage_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def rng_state():
    n = np.random.get_state()
    return {"python": random.getstate(), "numpy": (n[0], n[1].tolist(), n[2], n[3], n[4]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def reset_runtime(model):
    """Independent training sequences must not retain another batch's graph."""
    if hasattr(model, "shared_attn"):
        model.shared_attn = SharedAttentionRuntime()
        model.ced_runtime = CEDRuntime(model.shared_attn)
    for module in model.modules():
        for name in ("window_kv_cache", "compress_kv_cache", "k_cache", "kv_state", "score_state"):
            if hasattr(module, name):
                setattr(module, name, None)
    if getattr(model, "ngram_hash", None) is not None:
        model.ngram_hash.cache = None


def atomic_save(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def optimizer_groups(model):
    no_decay = set()
    for module in model.modules():
        if isinstance(module, (nn.Embedding, Gate)) or "norm" in type(module).__name__.lower():
            no_decay.update(id(p) for p in module.parameters(recurse=False))
    decay, plain = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad or name.startswith("dspark."):
            continue
        (decay if p.ndim >= 2 and id(p) not in no_decay and "gate" not in name else plain).append(p)
    return decay, plain


class Trainer:
    def __init__(self, model, dataset, config, checkpoint_dir, device="cpu"):
        self.config, self.dataset = config, dataset
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Use CPU development or CUDA BF16; MPS determinism is not validated")
        self.model = model.to(device=self.device, dtype=torch.float32)
        self.model.train()
        if hasattr(model, "cfg") and hasattr(dataset, "cfg") and model.cfg.sha256 != dataset.cfg.sha256:
            raise ValueError("Model/data config hash mismatch")
        resolved = asdict(model.cfg) if hasattr(model, "cfg") else {"class": type(model).__name__}
        self.config_hash = hashlib.sha256(json.dumps({"model": resolved, "training": asdict(config)}, sort_keys=True).encode()).hexdigest()
        self.source_config_hash = getattr(getattr(model, "cfg", None), "sha256", None)
        self.checkpoint_dir = Path(checkpoint_dir)
        decay, plain = optimizer_groups(model)
        self.optimizer = torch.optim.AdamW([
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": plain, "weight_decay": 0.}], lr=config.learning_rate, betas=(.9, .95))
        self.step = self.tokens = self.rollbacks = 0
        self.lr_scale = 1.
        self.clipped = deque(maxlen=100)
        self.last_checkpoint = None
        self.last_save = time.monotonic()
        self._collect = False
        self._loads = {}
        self._router_std = 0.
        self._z_logits: list[torch.Tensor] = []
        self._hooks = [m.register_forward_hook(self._route_hook) for m in model.modules() if isinstance(m, Gate)]
        self.forward_model = torch.compile(model) if config.compile_model else model

    def close(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def _route_hook(self, module, args, output):
        if self.config.router_z_loss:
            # stashed in eager AND checkpoint-replay forwards so the penalty
            # gradient survives activation-checkpoint recompute
            self._z_logits.append(F.linear(args[0].float(), module.weight.float()))
        if not self._collect:
            return
        with torch.no_grad():
            indices = output[1]
            mask = args[1] if len(args) > 1 and args[1] is not None else torch.zeros(indices.shape[0], dtype=torch.bool, device=indices.device)
            counts = self._loads.setdefault(module, [torch.zeros_like(module.bias), torch.zeros_like(module.bias)])
            for i, selected in enumerate((~mask, mask)):
                counts[i].add_(torch.bincount(indices[selected].flatten(), minlength=module.bias.numel()))
            logits = F.linear(args[0].float(), module.weight.float())
            self._router_std = max(self._router_std, float(logits.std(unbiased=False)))

    def _batch(self):
        samples = [next(self.dataset) for _ in range(self.config.batch_size)]
        if self.config.text_only and any(s.get("images") or s.get("image_mask", torch.tensor(False)).any() for s in samples):
            raise ValueError("C0–C6 require text-only data; image batch rejected")
        return {"input_ids": torch.stack([s["input_ids"] for s in samples]).to(self.device),
                "labels": torch.stack([s["labels"] for s in samples]).to(self.device),
                "image_mask": torch.stack([s.get("image_mask", torch.zeros_like(s["input_ids"], dtype=torch.bool)) for s in samples]).to(self.device),
                "images": [s.get("images", []) for s in samples]}

    def _loss(self, batch, collect=False):
        def forward(ids):
            reset_runtime(self.model)
            self._collect = collect and not self._recomputing
            self._z_logits = []
            try:
                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    logits, _ = self.forward_model(ids, image_mask=batch["image_mask"], images=batch["images"])
                loss = F.cross_entropy(logits.float().flatten(0, 1), batch["labels"].flatten(),
                                       ignore_index=-100, reduction="sum")
                if self._z_logits:
                    # router z-loss at the summed-CE scale: per-token (logsumexp)^2
                    z = sum(zl.logsumexp(-1).square().sum() for zl in self._z_logits)
                    loss = loss + self.config.router_z_loss * z
                return loss
            finally:
                self._collect = False
        self._recomputing = False
        if self.config.checkpoint_activations and torch.is_grad_enabled():
            # ponytail: checkpoint the entire forward to isolate mutable CED runtime;
            # per-block replay needs a functional runtime before it can save more memory.
            from contextlib import contextmanager, nullcontext

            @contextmanager
            def recompute():
                self._recomputing = True
                try:
                    yield
                finally:
                    self._recomputing = False
            return checkpoint(forward, batch["input_ids"], use_reentrant=False,
                              context_fn=lambda: (nullcontext(), recompute()))
        return forward(batch["input_ids"])

    def _lr(self):
        c = self.config
        if self.step < c.warmup_steps:
            return c.learning_rate * (self.step + 1) / c.warmup_steps * self.lr_scale
        progress = min(1., (self.step - c.warmup_steps) / max(1, c.max_steps - c.warmup_steps - 1))
        return (c.min_learning_rate + .5 * (c.learning_rate - c.min_learning_rate) * (1 + math.cos(math.pi * progress))) * self.lr_scale

    def train_step(self):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self._loads, self._router_std = {}, 0.
        batches = [self._batch() for _ in range(self.config.accumulation_steps)]
        tokens = sum(int((b["labels"] != -100).sum()) for b in batches)
        if tokens == 0:
            raise ValueError("Optimizer step has no supervised tokens")
        total = 0.
        for batch in batches:
            loss = self._loss(batch, collect=True) / tokens
            if not torch.isfinite(loss):
                raise NonfiniteError("Nonfinite loss")
            total += float(loss.detach())
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise NonfiniteError("Nonfinite gradient norm")
        clipped = list(self.clipped)[-99:] + [float(norm) > 1.]
        if len(clipped) == 100 and sum(clipped) > 5:
            raise TrainingFailure("Clipped on more than 5 of the last 100 steps")
        lr = self._lr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        with torch.no_grad():
            for gate, counts in self._loads.items():
                for bias, count in zip((gate.bias, gate.bias_vl), counts):
                    if count.sum() > 0:
                        bias.add_(self.config.router_bias_rate * (count.mean() - count).sign())
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise NonfiniteError("Nonfinite parameter after optimizer update")
        if any(not torch.isfinite(v).all() for state in self.optimizer.state.values()
               for v in state.values() if isinstance(v, torch.Tensor)):
            raise NonfiniteError("Nonfinite optimizer state after update")
        self.step += 1
        self.tokens += tokens
        self.clipped.append(float(norm) > 1.)
        reset_runtime(self.model)
        return {"step": self.step, "loss": total, "grad_norm": float(norm), "lr": lr,
                "tokens": tokens, "router_logit_std": self._router_std,
                "expert_loads": [(a + b).tolist() for a, b in self._loads.values()]}

    def state_dict(self):
        return {"version": 1, "config_hash": self.config_hash, "source_config_hash": self.source_config_hash,
                "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                "rng": rng_state(), "data": self.dataset.state_dict(), "step": self.step,
                "tokens": self.tokens, "lr_scale": self.lr_scale, "rollbacks": self.rollbacks,
                "clipped": list(self.clipped), "determinism": {
                    "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                    "deterministic": torch.are_deterministic_algorithms_enabled(),
                    "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                    "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "torch_version": str(torch.__version__), "device_type": self.device.type}}

    def _restore(self, state):
        if state["version"] != 1 or state["config_hash"] != self.config_hash or state["source_config_hash"] != self.source_config_hash:
            raise ValueError("Checkpoint config mismatch")
        if state["determinism"] != self.state_dict()["determinism"]:
            raise ValueError("Checkpoint determinism/environment mismatch")
        self.dataset.load_state_dict(state["data"])
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        for name in ("step", "tokens", "lr_scale", "rollbacks"):
            setattr(self, name, state[name])
        self.clipped = deque(state["clipped"], maxlen=100)
        self.optimizer.zero_grad(set_to_none=True)
        restore_rng(state["rng"])
        reset_runtime(self.model)

    def _smoke_loss(self):
        data, rng = copy.deepcopy(self.dataset.state_dict()), rng_state()
        try:
            try:
                batch = self._batch()
            except StopIteration:
                return None
            with torch.no_grad():
                count = int((batch["labels"] != -100).sum())
                if count == 0:
                    raise ValueError("Resume smoke batch has no labels")
                loss = float(self._loss(batch) / count)
            if not math.isfinite(loss):
                raise NonfiniteError("Nonfinite resume smoke")
            return loss
        finally:
            self.dataset.load_state_dict(data)
            restore_rng(rng)
            reset_runtime(self.model)

    def save_checkpoint(self, *, gate=False):
        smoke = self._smoke_loss()
        state = self.state_dict()
        state["smoke_loss"] = smoke
        path = self.checkpoint_dir / f"{'gate' if gate else 'step'}-{self.step:09d}.pt"
        atomic_save(state, path)
        self.last_checkpoint, self.last_save = path, time.monotonic()
        for old in sorted(self.checkpoint_dir.glob("step-*.pt"))[:-3]:
            old.unlink()
        return path

    def load_checkpoint(self, path):
        # weights_only avoids arbitrary pickle execution for external checkpoints.
        state = torch.load(path, map_location="cpu", weights_only=True)
        original = copy.deepcopy(self.state_dict())
        try:
            self._restore(state)
            actual, expected = self._smoke_loss(), state["smoke_loss"]
            if (actual is None) != (expected is None) or (actual is not None and abs(actual - expected) > 1e-3):
                raise ValueError("Resume smoke loss continuity exceeds 1e-3")
        except Exception:
            self._restore(original)
            raise
        self.last_checkpoint, self.last_save = Path(path), time.monotonic()

    def guarded_step(self, on_retry=None):
        """One optimizer step with the nonfinite guard: rollback → skip 2
        batches → halve LR, at most 3 retries. This is run()'s loop body,
        extracted so gate collectors can bracket individual attempts
        (training/ladder.py). Returns (record, retried); on_retry fires
        before each post-rollback re-attempt so callers can drop
        per-attempt collection state."""
        retried = False
        while True:
            try:
                return self.train_step(), retried
            except NonfiniteError:
                retries, scale = self.rollbacks + 1, self.lr_scale / 2
                if retries > 3:
                    raise TrainingFailure("Nonfinite rollback retry limit exceeded")
                self.load_checkpoint(self.last_checkpoint)
                self.rollbacks, self.lr_scale = retries, scale
                for _ in range(2):
                    self._batch()
                retried = True
                if on_retry is not None:
                    on_retry()

    def run(self, steps=None):
        target = self.config.max_steps if steps is None else self.step + steps
        if target > self.config.max_steps or target < self.step:
            raise ValueError("Requested steps exceed configured schedule")
        if self.last_checkpoint is None:
            self.save_checkpoint()
        records = []
        while self.step < target:
            record, _ = self.guarded_step()
            records.append(record)
            records = [r for r in records if r["step"] <= self.step]  # drop discarded updates
            if time.monotonic() - self.last_save >= self.config.checkpoint_seconds:
                self.save_checkpoint()
        self.save_checkpoint()
        return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Offline manifest directory")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--training-config", required=True, help="TrainingConfig JSON; batch sizes must be explicit")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--fixture", action="store_true", help="Waive corpus size only, never integrity")
    args = parser.parse_args()
    from data.dataset import PackedDataset
    from models.config import load_config
    from models.transformer import Transformer
    config = TrainingConfig(**json.loads(Path(args.training_config).read_text()))
    seed_everything(config.stage_id)
    cfg = load_config(args.model_config)
    data = PackedDataset(args.data, config=args.model_config, production=not args.fixture)
    trainer = Trainer(Transformer(cfg, max_seq_len=data.sequence_length), data, config, args.checkpoint_dir, args.device)
    try:
        if args.resume:
            trainer.load_checkpoint(args.resume)
        for record in trainer.run():
            print(json.dumps(record))
    finally:
        trainer.close()
        data.close()


if __name__ == "__main__":
    main()
