"""C0–C6 evidence gates, pinned probes, and bounded execution.

Stage callbacks must implement the named architecture and matched-token control;
this module never relabels the full Transformer as the dense C0 baseline.
`GateRunner` is the canonical callback: it builds the stage's variant and its
matched-token control from models/variants.py, trains both on the same batch
stream, evaluates the pinned 512-batch probes, and emits the metrics dict
`evaluate_gate` consumes.
"""

import hashlib
import json
import math
import os
import struct
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F

from models.transformer import Transformer
from models.variants import NAMED, STAGE_CONTROL, STAGE_VARIANT, VariantSpec, derive_config
from training.pretrain import Trainer, TrainingConfig, reset_runtime


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

    def __init__(self, probes, approvals, output="runs/ladder.jsonl", *, consumed_hours=0):
        self.probes, self.approvals = probes, dict(approvals)
        self.output = Path(output)
        if consumed_hours < 0:
            raise ValueError("Negative budget")
        self.hours = consumed_hours
        self.started = time.monotonic()
        self.failures = 0

    def _budget(self):
        elapsed = (time.monotonic() - self.started) / 3600
        if elapsed >= 6 or self.hours + elapsed >= 54:
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
                # One bounded retry per stage attempt (plan: "one retry each");
                # a second consecutive exception is an environment bug — the
                # ladder aborts for local diagnosis. Failed gate *decisions*
                # (not exceptions) walk the pre-approved fallback variants.
                for attempt in (1, 2):
                    request = StageRequest(stage, variant, STEPS[stage], checkpoint,
                                           self.approvals[key], self.probes.verify())
                    try:
                        result = run_stage(request, self.probes)
                        self._budget()
                        if self.probes.verify() != request.probe_sha256:
                            raise ValueError("Probe set changed during gate")
                    except Exception as exc:
                        self._append({**request.__dict__, "attempt": attempt,
                                      "passed": False,
                                      "reasons": [f"{type(exc).__name__}: {exc}"],
                                      "checkpoint": checkpoint})
                        if attempt == 2:
                            raise
                        continue
                    break
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


# ---- executable stage attempts (Task 11 closeout) ----

SEED = 1337


@dataclass(frozen=True)
class Attempt:
    """One named stage attempt: variant spec + data-table/training overrides."""

    spec: VariantSpec
    cfg_overrides: dict | None = None
    train_overrides: dict | None = None
    seed_offset: int = 0


# Pre-approved fallbacks (training/ladder.STEPS FALLBACKS) as executable
# attempts. Data-table overrides are ascending subsets of the production
# tables (the mode/producer maps are data per the contract); overrides that
# do not fit a given cfg are approved as unbuildable by gate_approvals().
FALLBACK_ATTEMPTS = {
    ("C0", "repeat_core"): Attempt(NAMED["dense"], seed_offset=1),
    ("C1", "router_z_loss_1e-3"): Attempt(NAMED["moe"], train_overrides={"router_z_loss": 1e-3}),
    ("C1", "top1"): Attempt(NAMED["moe"], cfg_overrides={"n_activated_experts": 1}),
    ("C2", "window256"): Attempt(NAMED["swa"], cfg_overrides={"window_size": 256}),
    ("C3", "second_producer_map"): Attempt(NAMED["ced"], cfg_overrides={"kv_source_layers": (8,)}),
    ("C3", "three_producers"): Attempt(
        NAMED["ced"], cfg_overrides={"kv_source_layers": (2, 8, 10),
                                     "index_source_layers": (2, 8, 10, 12, 16, 20)}),
    ("C4", "layout2"): Attempt(NAMED["csa2"], cfg_overrides={"index_source_layers": (2, 8, 16, 20),
                                                             "candidate_source_layer": 16}),
    ("C4", "layout3"): Attempt(NAMED["csa2"], cfg_overrides={"index_source_layers": (2, 8, 12, 20),
                                                             "candidate_source_layer": 12}),
    ("C5", "identity_residual"): Attempt(replace(NAMED["mhc"], name="mhc-identity", mhc_frozen=True)),
}


def stage_attempt(stage: str, variant: str = "default") -> tuple[Attempt, str | None]:
    """(primary attempt, control variant name or None) for one stage attempt."""
    if variant == "default":
        primary = Attempt(NAMED[STAGE_VARIANT[stage]])
    else:
        try:
            primary = FALLBACK_ATTEMPTS[(stage, variant)]
        except KeyError:
            raise ValueError(f"No executable variant for {stage}:{variant}") from None
    return primary, STAGE_CONTROL[stage]


def variant_config_sha(cfg) -> str:
    """Gate identity: sha256 over the resolved variant config."""
    from dataclasses import asdict

    return hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True, default=str).encode()).hexdigest()


def gate_approvals(cfg, stages=None) -> dict[str, str]:
    """Explicit config approval for every stage attempt the runner can issue.

    Fallbacks whose tables do not fit this cfg are approved as 'unbuildable'
    — the callback then fails construction honestly (and the JSONL evidence
    records it) if such an attempt is ever executed."""
    if stages is None:
        stages = STEPS
    approvals = {}
    for stage in stages:
        for variant in ("default",) + FALLBACKS[stage]:
            attempt, _ = stage_attempt(stage, variant)
            try:
                derived = derive_config(cfg, attempt.spec, attempt.cfg_overrides)
                approvals[f"{stage}:{variant}"] = variant_config_sha(derived)
            except (AssertionError, ValueError, KeyError, IndexError, TypeError):
                approvals[f"{stage}:{variant}"] = f"unbuildable:{stage}:{variant}"
    return approvals


# ---- GateRunner: the canonical stage callback ----


class _PerStepCollector:
    """Per-step gate evidence hooks, captured only around the guarded
    training forward (smoke/probe forwards run while inactive): mean
    attention-output norm per step (C3), Sinkhorn finiteness (C5), Engram
    gate magnitude (C6)."""

    def __init__(self, model):
        from models.attention import Attention
        from models.engram import Engram
        from models.mhc import HCMixes

        self.active = False
        self._norms: list[torch.Tensor] = []
        self._sink_ok = True
        self._gate = 0.0
        self.attention_norms: list[float] = []
        self.sinkhorn_finite: list[bool] = []
        self.gate_magnitudes: list[float] = []
        self.reuse_records: dict[str, list[float]] = {}
        self._handles = []
        for m in model.modules():
            if isinstance(m, Attention):
                self._handles.append(m.register_forward_hook(self._attn_hook))
            elif isinstance(m, HCMixes):
                self._handles.append(m.register_forward_hook(self._hc_hook))
            elif isinstance(m, Engram):
                self._handles.append(m.register_forward_hook(self._engram_hook))

    def begin_step(self):
        self.active = True
        self._norms, self._sink_ok, self._gate = [], True, 0.0

    def end_step(self):
        self.active = False
        mean = float(torch.stack(self._norms).mean()) if self._norms else 0.0
        self.attention_norms.append(mean)
        self.sinkhorn_finite.append(self._sink_ok)
        self.gate_magnitudes.append(self._gate)

    def _attn_hook(self, module, args, output):
        if self.active:
            self._norms.append(output.detach().float().pow(2).mean().sqrt())

    def _hc_hook(self, module, args, output):
        if self.active and not all(torch.isfinite(t).all() for t in output):
            self._sink_ok = False

    def _engram_hook(self, module, args, output):
        if self.active:
            magnitude = getattr(module, "last_gate_magnitude", None)
            if magnitude is not None:
                self._gate = max(self._gate, float(magnitude))

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


@torch.no_grad()
def probe_eval(model, batches, device="cpu") -> float:
    """Mean per-token CE over the pinned probe batches (probe_loss /
    control_loss; the gate's only loss comparison)."""
    model.eval()
    total_ce, total_tokens = 0.0, 0
    for batch in batches:
        reset_runtime(model)
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits, _ = model(ids)
        ce = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(),
                             ignore_index=-100, reduction="sum")
        total_ce += float(ce)
        total_tokens += int((labels != -100).sum())
    model.train()
    return total_ce / max(total_tokens, 1)


@torch.no_grad()
def _engram_init_identity(model, batch, device="cpu") -> bool:
    """C6 init evidence: every Engram output bitwise-equals its input at
    step 0 (the zero-init gate contributes exactly nothing)."""
    from models.engram import Engram

    seen: list[bool] = []
    handles = [e.register_forward_hook(lambda m, a, o: seen.append(torch.equal(o, a[0])))
               for e in model.modules() if isinstance(e, Engram)]
    try:
        reset_runtime(model)
        model(batch["input_ids"].to(device))
    finally:
        for handle in handles:
            handle.remove()
    return bool(seen) and all(seen)


def _reuse_layer_names(cfg) -> list[str]:
    """Decoder Reuse-mode layers: on the global pathway, not index sources."""
    return [f"dec{l}" for l in cfg.decoder_layers
            if cfg.global_kv_path(l) != "none" and not cfg.is_index_source(l)]


class GateRunner:
    """Builds `run_stage(request, probes)` for LadderRunner.

    Each stage attempt constructs the named variant and its matched-token
    control (the previous stage's architecture) from the same seed, trains
    both for the request's step count on the SAME batch stream (the corpus
    position is snapshotted and restored around each run), evaluates the
    pinned probe set on both, saves the variant's gate checkpoint, and
    returns runner-shaped evidence. After the stage the corpus position is
    restored, so every stage trains from the same stream position.
    """

    def __init__(self, cfg, data, checkpoint_dir, *, device="cpu", batch_size=1,
                 accumulation_steps=1, learning_rate=3e-4, warmup_steps=200,
                 checkpoint_activations=False):
        self.cfg = cfg
        self.data = data
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.checkpoint_activations = checkpoint_activations

    def _train_steps(self, trainer, steps, collector=None, reuse_probe=None):
        records = []
        if trainer.last_checkpoint is None:
            trainer.save_checkpoint()
        for _ in range(steps):
            if collector is not None:
                collector.begin_step()
            record, _ = trainer.guarded_step(on_retry=collector.begin_step if collector else None)
            if collector is not None:
                collector.end_step()
            records.append(record)
            records = [r for r in records if r["step"] <= trainer.step]  # drop discarded updates
            if reuse_probe is not None:
                for name, value in reuse_probe():
                    collector.reuse_records.setdefault(name, []).append(value)
        return records

    def run_stage(self, request, probes):
        attempt, control_name = stage_attempt(request.stage, request.variant)
        steps = request.steps
        if steps < 1:
            raise ValueError("Gate stages require at least one step")
        stage_num = int(request.stage[1:])
        seed = SEED + stage_num + attempt.seed_offset
        train_cfg = TrainingConfig(
            max_steps=steps, batch_size=self.batch_size, accumulation_steps=self.accumulation_steps,
            warmup_steps=min(self.warmup_steps, max(1, steps - 1)), stage_id=stage_num + attempt.seed_offset,
            learning_rate=self.learning_rate, checkpoint_activations=self.checkpoint_activations,
            **(attempt.train_overrides or {}))
        position0 = self.data.state_dict()

        torch.manual_seed(seed)
        model_cfg = derive_config(self.cfg, attempt.spec, attempt.cfg_overrides)
        model = Transformer(model_cfg, variant=attempt.spec)
        trainer = Trainer(model, self.data, train_cfg,
                          self.checkpoint_dir / f"{request.stage}-{request.variant}", self.device)
        collector = _PerStepCollector(model)
        reuse_layers = _reuse_layer_names(model_cfg) if request.stage == "C4" else []

        def reuse_probe():
            loss = probe_eval(model, [probes.batches[0]], self.device)
            return [(name, loss) for name in reuse_layers]

        try:
            if attempt.spec.engram:
                engram_init_zero = _engram_init_identity(model, probes.batches[0], self.device)
            records = self._train_steps(trainer, steps, collector,
                                        reuse_probe if reuse_layers else None)
            probe = probe_eval(model, probes.batches, self.device)
            checkpoint = trainer.save_checkpoint(gate=True)
            nonfinite_count = trainer.rollbacks
            tokens = trainer.tokens
        finally:
            collector.remove()
            trainer.close()

        metrics = {
            "losses": [r["loss"] for r in records],
            "grad_norms": [r["grad_norm"] for r in records],
            "nonfinite_count": nonfinite_count,
            "probe_loss": probe,
            "probe_batches": len(probes.batches),
            "tokens": tokens,
            "expert_loads": [r["expert_loads"] for r in records],
            "router_logit_std": [r["router_logit_std"] for r in records],
            "attention_norms": collector.attention_norms,
            "sinkhorn_finite": collector.sinkhorn_finite,
            "gate_magnitudes": collector.gate_magnitudes,
        }
        if attempt.spec.engram:
            metrics["engram_init_zero"] = engram_init_zero
        if reuse_layers:
            metrics["reuse_losses"] = collector.reuse_records
        return self._run_control(metrics, control_name, request, probes, train_cfg,
                                 position0, seed, model_cfg, str(checkpoint), tokens)

    def _run_control(self, metrics, control_name, request, probes, train_cfg,
                     position0, seed, model_cfg, checkpoint, variant_tokens):
        if control_name is not None:
            self.data.load_state_dict(position0)  # matched tokens: same batches
            control_cfg = derive_config(self.cfg, NAMED[control_name])
            torch.manual_seed(seed)  # same init where shapes match
            control_model = Transformer(control_cfg, variant=NAMED[control_name])
            control = Trainer(control_model, self.data, train_cfg,
                              self.checkpoint_dir / f"{request.stage}-{request.variant}-control",
                              self.device)
            try:
                self._train_steps(control, train_cfg.max_steps)
                control_loss = probe_eval(control_model, probes.batches, self.device)
            finally:
                control.close()
            metrics["control_loss"] = control_loss
            metrics["control_tokens"] = control.tokens
            if control.tokens != variant_tokens:
                raise ValueError("Matched-token control saw a different token count")
        self.data.load_state_dict(position0)
        return {"config_sha256": variant_config_sha(model_cfg),
                "checkpoint": checkpoint, "metrics": metrics}

