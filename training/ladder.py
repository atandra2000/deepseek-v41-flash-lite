"""C0–C6 evidence gates, pinned probes, and bounded execution.

Stage callbacks must implement the named architecture and matched-token control;
this module never relabels the full Transformer as the dense C0 baseline.
"""

import hashlib
import json
import math
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import torch


STEPS = {"C0": 2000, "C1": 1000, "C2": 1000, "C3": 2000,
         "C4": 2000, "C5": 1000, "C6": 1000}
DELTAS = {"C1": .05, "C2": .02, "C3": .03, "C4": .05, "C5": .02, "C6": .02}
FALLBACKS = {
    "C0": ("repeat_core",), "C1": ("router_z_loss_1e-3", "top1"),
    "C2": ("window256",), "C3": ("second_producer_map", "three_producers"),
    "C4": ("layout2", "layout3"), "C5": ("identity_residual",), "C6": (),
}


def _hash_value(value, digest):
    if isinstance(value, torch.Tensor):
        t = value.detach().cpu().contiguous()
        digest.update(str((str(t.dtype), tuple(t.shape))).encode())
        digest.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, dict):
        for key in sorted(value):
            digest.update(key.encode())
            _hash_value(value[key], digest)
    elif isinstance(value, (tuple, list)):
        digest.update(str(len(value)).encode())
        for item in value:
            _hash_value(item, digest)
    elif value is None or isinstance(value, (str, int, float, bool)):
        digest.update(json.dumps(value, allow_nan=False).encode())
    else:
        raise TypeError(f"Unsupported probe value: {type(value).__name__}")


def batch_sha256(batch):
    digest = hashlib.sha256()
    _hash_value(batch, digest)
    return digest.hexdigest()


class ProbeSet:
    """Exactly 512 held-out text batches, checked again before every use."""

    def __init__(self, batches, hashes):
        self.batches, self.hashes = tuple(batches), tuple(hashes)
        self.verify()

    @classmethod
    def pin(cls, batches, path):
        batches = tuple(batches)
        obj = cls(batches, [batch_sha256(b) for b in batches])
        with Path(path).open("x") as stream:
            json.dump({"version": 1, "sha256": obj.hashes}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        return obj

    @classmethod
    def load(cls, batches, path):
        record = json.loads(Path(path).read_text())
        if record["version"] != 1:
            raise ValueError("Unknown probe manifest version")
        return cls(batches, record["sha256"])

    def verify(self):
        if len(self.batches) != 512 or len(self.hashes) != 512:
            raise ValueError("The gate probe must contain exactly 512 batches")
        if any(batch_sha256(b) != h for b, h in zip(self.batches, self.hashes)):
            raise ValueError("Probe checksum mismatch")
        return hashlib.sha256("".join(self.hashes).encode()).hexdigest()


def _series(metrics, name, minimum):
    values = metrics.get(name, [])
    if len(values) < minimum or any(not math.isfinite(float(x)) for x in values):
        raise ValueError(f"Missing/short/nonfinite {name}")
    return [float(x) for x in values]


def _ema_ratios(values, start=0):
    ema = values[0]
    ratios = []
    for i, value in enumerate(values):
        if i >= start:
            ratios.append(value / max(ema, 1e-12))
        ema = .99 * ema + .01 * value
    return ratios


def evaluate_gate(stage, metrics):
    """Fail closed on missing evidence. Windows and EMA definitions are explicit.

    Loss comparisons are 512-batch probe means; control_tokens must equal tokens.
    C0 decreasing means each recorded optimizer-step loss decreases strictly.
    C4 reuse_losses are layer-name -> 500+ per-step probe losses.
    """
    if stage not in STEPS:
        raise ValueError(f"Unknown stage {stage}")
    reasons = []
    try:
        n = STEPS[stage]
        losses = _series(metrics, "losses", n)
        grads = _series(metrics, "grad_norms", len(losses))
        if len(grads) != len(losses):
            raise ValueError("Loss/gradient history length mismatch")
        if any(sum(g > 1 for g in grads[i - 99:i + 1]) > 5 for i in range(99, len(grads))):
            reasons.append("clipped on more than 5 of 100 steps")
        if metrics.get("nonfinite_count") != 0:
            reasons.append("nonfinite event or missing nonfinite count")
        probe = float(metrics["probe_loss"])
        if not math.isfinite(probe) or metrics.get("probe_batches") != 512:
            reasons.append("invalid probe measurement")
        if stage == "C0":
            if not all(b < a for a, b in zip(losses, losses[1:])):
                reasons.append("C0 loss is not strictly decreasing")
            if losses[-1] >= 5 or max(_ema_ratios(grads, 500), default=0) >= 5:
                reasons.append("C0 final loss or gradient EMA bound")
        else:
            control = float(metrics["control_loss"])
            if (not math.isfinite(control) or metrics.get("tokens", 0) <= 0
                    or metrics.get("tokens") != metrics.get("control_tokens")):
                reasons.append("missing matched-token control")
            if probe - control > DELTAS[stage] + 1e-12:
                reasons.append("probe loss delta exceeds stage bound")
        if stage == "C1":
            loads = torch.as_tensor(metrics["expert_loads"], dtype=torch.float64)
            if loads.ndim != 3 or loads.shape[0] < 200 or not torch.isfinite(loads).all() or (loads < 0).any():
                raise ValueError("expert_loads must be [steps, layers, experts]")
            total = loads[-200:].sum(0)
            if (total.min(-1).values <= 0).any() or (total.max(-1).values / total.min(-1).values > 4).any():
                reasons.append("expert load imbalance")
            if max(_series(metrics, "router_logit_std", n)) >= 10:
                reasons.append("router logit std >=10")
        if stage == "C3":
            if max(_ema_ratios(_series(metrics, "attention_norms", n))) > 3:
                reasons.append("attention norm exceeds 3x EMA")
        if stage == "C4":
            if not metrics.get("reuse_losses"):
                raise ValueError("Missing Reuse-layer probes")
            for name, values in metrics["reuse_losses"].items():
                y = _series({name: values}, name, 500)[-500:]
                slope = sum((i - 249.5) * v for i, v in enumerate(y))
                if slope > 0:
                    reasons.append(f"positive Reuse loss slope: {name}")
        if stage == "C5":
            flags = metrics["sinkhorn_finite"]
            if len(flags) != len(losses) or not all(v is True for v in flags):
                reasons.append("nonfinite or missing Sinkhorn step")
        if stage == "C6":
            if metrics.get("engram_init_zero") is not True or max(abs(x) for x in _series(metrics, "gate_magnitudes", n)) >= 1:
                reasons.append("Engram initialization/gate magnitude")
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        reasons.append(str(exc))
    return {"passed": not reasons, "reasons": reasons}


def bitwise_repeat(trainer_factory, steps=200):
    """A strict FP32 loss-curve comparison, never an allclose substitute."""
    if steps != 200:
        raise ValueError("The determinism contract requires 200 steps")
    curves = []
    for _ in range(2):
        trainer = trainer_factory()
        curve = [float(trainer.train_step()["loss"]) for _ in range(steps)]
        if hasattr(trainer, "close"):
            trainer.close()
        if not all(math.isfinite(x) for x in curve):
            raise RuntimeError("Nonfinite repeat loss")
        curves.append(b"".join(struct.pack("<f", x) for x in curve))
    if curves[0] != curves[1]:
        raise RuntimeError("200-step loss curves are not bitwise identical")
    return {"steps": steps, "loss_sha256": hashlib.sha256(curves[0]).hexdigest()}


@dataclass(frozen=True)
class StageRequest:
    stage: str
    variant: str
    steps: int
    parent_checkpoint: str | None
    approved_config_sha256: str
    probe_sha256: str


class LadderRunner:
    """Execute approved stage callbacks, retaining the last known-good checkpoint.

    `run_stage(request, probes)` returns metrics, checkpoint and config_sha256.
    Architecture construction/control runs are explicit callbacks because the
    v2 16-layer dense baseline is not the v3 24-layer model. No implicit migration.
    """

    def __init__(self, probes, approvals, output="runs/ladder.jsonl", *, consumed_hours=0,
                 consumed_dollars=0, hourly_rate=5.96):
        self.probes, self.approvals = probes, dict(approvals)
        self.output = Path(output)
        if min(consumed_hours, consumed_dollars, hourly_rate) < 0:
            raise ValueError("Negative budget")
        self.hours, self.dollars, self.rate = consumed_hours, consumed_dollars, hourly_rate
        self.started = time.monotonic()
        self.failures = 0

    def _budget(self):
        elapsed = (time.monotonic() - self.started) / 3600
        if elapsed >= 6 or self.hours + elapsed >= 54 or self.dollars + elapsed * self.rate >= 300:
            raise RuntimeError("Ladder/60%-before-production budget exhausted")

    def _append(self, record):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def run(self, run_stage):
        checkpoint = None
        for stage in STEPS:
            candidates = []
            variants = ("default",) + FALLBACKS[stage]
            for variant in variants:
                self._budget()
                if stage == "C4" and variant == "layout3" and candidates:
                    break
                key = f"{stage}:{variant}"
                if key not in self.approvals:
                    raise ValueError(f"Missing explicit config approval for {key}")
                request = StageRequest(stage, variant, STEPS[stage], checkpoint,
                                       self.approvals[key], self.probes.verify())
                try:
                    result = run_stage(request, self.probes)
                    self._budget()
                    if self.probes.verify() != request.probe_sha256:
                        raise ValueError("Probe set changed during gate")
                except Exception as exc:
                    self._append({**request.__dict__, "passed": False,
                                  "reasons": [f"{type(exc).__name__}: {exc}"],
                                  "checkpoint": checkpoint})
                    raise
                if result.get("config_sha256") != request.approved_config_sha256:
                    raise ValueError("Gate config does not match approved hash")
                decision = evaluate_gate(stage, result["metrics"])
                path = Path(result.get("checkpoint", ""))
                if decision["passed"] and not path.is_file():
                    decision = {"passed": False, "reasons": ["Missing gate checkpoint"]}
                self._append({**request.__dict__, **decision, "checkpoint": str(path),
                              "metrics": result["metrics"],
                              "elapsed_hours": (time.monotonic() - self.started) / 3600})
                self.failures = 0 if decision["passed"] else self.failures + 1
                if self.failures >= 3:
                    raise RuntimeError("Three consecutive gate failures; owner decision required")
                if decision["passed"]:
                    candidates.append(result)
                    if stage != "C4":
                        break
            if not candidates:
                raise RuntimeError(f"{stage} failed; retained checkpoint: {checkpoint}")
            checkpoint = min(candidates, key=lambda r: r["metrics"]["probe_loss"])["checkpoint"]
        return checkpoint
