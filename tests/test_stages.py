"""CPU end-to-end gate ladder on toy dims (Task 11 closeout).

The ladder runs with REAL training and REAL gate evaluation at test dims:
every stage builds its named variant + matched-token control, trains both on
the same deterministic corpus, and passes/fails on measured probe evidence.
Only the step counts are scaled down (monkeypatched STEPS); the numeric bars,
probe pinning, and fallback protocol are the production ones.
"""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import training.ladder as ladder
from models.transformer import Transformer
from models.variants import NAMED, derive_config
from tests.test_variants import gate_toy_config
from training.ladder import (SEED, STEPS, GateRunner, LadderRunner, ProbeSet, batch_sha256,
                             bitwise_repeat, gate_approvals, probe_eval, stage_attempt)
from training.pretrain import Trainer, TrainingConfig, seed_everything

SEQ, VOCAB = 24, 512
TEST_STEPS = {"C0": 200, "C1": 200, "C2": 110, "C3": 110, "C4": 510, "C5": 110, "C6": 110}


class SyntheticCorpus:
    """PackedDataset-shaped deterministic corpus: rows are `start + arange(seq)`
    mod vocab with the +1 shift as labels — a purely local rule, so the
    windowed/selected stages (C2/C4) are not structurally handicapped and
    4096 distinct rows keep the loss decreasing through the gates."""

    def __init__(self, n=4096, seq=SEQ, vocab=VOCAB, seed=11):
        g = torch.Generator().manual_seed(seed)
        starts = torch.randint(vocab, (n,), generator=g)
        self.rows = [(s + torch.arange(seq)) % vocab for s in starts]
        self.position = 0

    def __next__(self):
        row = self.rows[self.position % len(self.rows)]
        self.position += 1
        return {"input_ids": row, "labels": (row + 1) % VOCAB}

    def state_dict(self):
        return {"manifest_sha256": "fixture", "position": self.position}

    def load_state_dict(self, state):
        if state["manifest_sha256"] != "fixture" or state["position"] < 0:
            raise ValueError("Manifest/position mismatch")
        self.position = state["position"]

    def close(self):
        pass


def probe_set(n=512, seed=23):
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(VOCAB, (n,), generator=g)
    batches = [{"input_ids": ((s + torch.arange(SEQ)) % VOCAB).unsqueeze(0),
                "labels": (((s + torch.arange(SEQ)) % VOCAB + 1) % VOCAB).unsqueeze(0)} for s in starts]
    return ProbeSet(batches, [batch_sha256(b) for b in batches])


def make_gate_runner(cfg, tmp_path, steps_lr=2e-3):
    corpus = SyntheticCorpus()
    # Each optimizer batch covers every token three times. Keep the +1
    # task, but remove between-step sampling noise from C0's strict decrease.
    corpus.rows = [(s + torch.arange(SEQ)) % VOCAB for s in torch.arange(64) * 8]
    return GateRunner(cfg, corpus, tmp_path / "gates", batch_size=64,
                      learning_rate=steps_lr, warmup_steps=24)


def test_stage_attempts_cover_every_runner_attempt():
    for stage in STEPS:
        for variant in ("default",) + ladder.FALLBACKS[stage]:
            attempt, control = stage_attempt(stage, variant)
            assert attempt.spec.name
            assert control == (None if stage == "C0" else ladder.STAGE_CONTROL[stage])
        default, _ = stage_attempt(stage)
        assert default.spec is NAMED[ladder.STAGE_VARIANT[stage]]


def test_gate_approvals_pin_every_attempt(tmp_path):
    cfg = gate_toy_config()
    approvals = gate_approvals(cfg)
    for stage in STEPS:
        for variant in ("default",) + ladder.FALLBACKS[stage]:
            assert f"{stage}:{variant}" in approvals
    d_default = approvals["C0:default"]
    assert d_default == approvals["C0:repeat_core"]  # same architecture, new seed
    # tables that cannot fit the toy cfg are approved unbuildable, not silently mapped
    assert approvals["C3:second_producer_map"].startswith("unbuildable:")


def test_stage_metrics_shapes_and_matched_tokens(tmp_path):
    cfg = gate_toy_config()
    probes = probe_set()
    runner = make_gate_runner(cfg, tmp_path)
    approvals = gate_approvals(cfg)
    request = ladder.StageRequest("C1", "default", 30, None, approvals["C1:default"], probes.verify())
    result = runner.run_stage(request, probes)
    m = result["metrics"]
    assert result["config_sha256"] == approvals["C1:default"]
    assert Path(result["checkpoint"]).is_file()
    assert len(m["losses"]) == len(m["grad_norms"]) == 30
    assert m["probe_batches"] == 512 and m["nonfinite_count"] == 0
    assert m["tokens"] == m["control_tokens"] > 0
    loads = torch.as_tensor(m["expert_loads"], dtype=torch.float64)
    assert loads.ndim == 3 and loads.shape == (30, 3, cfg.n_routed_experts)  # steps x gates x experts
    assert all(v >= 0 for v in m["router_logit_std"])


def test_full_ladder_e2e_on_toy(tmp_path, monkeypatch):
    monkeypatch.setattr(ladder, "STEPS", dict(TEST_STEPS))
    cfg = gate_toy_config()
    probes = probe_set()
    # Production C4 sweeps 24-layer layouts even after a default pass. This
    # 3-layer fixture has only one pool-builder/Reuse layout; do not relabel
    # an impossible production table as a toy architecture.
    production_approvals = gate_approvals(cfg)
    assert all(production_approvals[f"C4:{v}"].startswith("unbuildable:")
               for v in ("layout2", "layout3"))
    monkeypatch.setattr(ladder, "FALLBACKS", {**ladder.FALLBACKS, "C4": ()})
    approvals = gate_approvals(cfg)  # after the STEPS patch: same keys, per-stage shas
    evidence = tmp_path / "runs" / "ladder.jsonl"
    runner = make_gate_runner(cfg, tmp_path)
    final = LadderRunner(probes, approvals, evidence).run(runner.run_stage)
    assert Path(final).is_file()  # the C6 winner's checkpoint
    records = [json.loads(x) for x in evidence.read_text().splitlines()]
    passing = {r["stage"] for r in records if r["passed"]}
    assert passing == set(TEST_STEPS), records
    for stage in TEST_STEPS:  # every stage either passed or walked its fallbacks
        attempts = [r for r in records if r["stage"] == stage]
        assert attempts and attempts[-1]["passed"]


def test_bitwise_repeat_real_trainer_toy(tmp_path):
    cfg = gate_toy_config()
    spec = NAMED["dense"]

    def factory():
        seed_everything(5)
        torch.manual_seed(SEED)
        model = Transformer(derive_config(cfg, spec), variant=spec)
        return Trainer(model, SyntheticCorpus(), TrainingConfig(max_steps=200, warmup_steps=24, learning_rate=2e-3),
                       tmp_path / "repeat", "cpu")

    evidence = bitwise_repeat(factory)
    assert evidence["steps"] == 200 and len(evidence["loss_sha256"]) == 64
