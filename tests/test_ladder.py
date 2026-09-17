import json

import pytest
import torch

from training.ladder import LadderRunner, ProbeSet, STEPS, batch_sha256, bitwise_repeat, evaluate_gate


def evidence(stage):
    n = STEPS[stage]
    return {"losses": [4 - i / 10000 for i in range(n)], "grad_norms": [.1] * n,
            "probe_loss": 3., "probe_batches": 512, "nonfinite_count": 0,
            "control_loss": 3., "tokens": 100, "control_tokens": 100,
            "expert_loads": [[[1, 1]]] * 200, "router_logit_std": [1.] * n,
            "attention_norms": [1.] * n, "reuse_losses": {"layer1": [1.] * 500},
            "sinkhorn_finite": [True] * n, "engram_init_zero": True,
            "gate_magnitudes": [.1] * n}


@pytest.mark.parametrize("stage", list(STEPS))
def test_gate_accepts_complete_evidence_and_rejects_missing(stage):
    assert evaluate_gate(stage, evidence(stage))["passed"]
    assert not evaluate_gate(stage, {})["passed"]


def test_numeric_boundaries():
    m = evidence("C1")
    m["probe_loss"] = 3.05
    assert evaluate_gate("C1", m)["passed"]
    m["router_logit_std"][-1] = 10
    assert not evaluate_gate("C1", m)["passed"]
    m = evidence("C2")
    m["grad_norms"][:5] = [2.] * 5
    assert evaluate_gate("C2", m)["passed"]
    m["grad_norms"][5] = 2.
    assert not evaluate_gate("C2", m)["passed"]
    m = evidence("C6")
    m["gate_magnitudes"][-1] = 1.
    assert not evaluate_gate("C6", m)["passed"]


def test_probes_pinned_and_tamper_rejected(tmp_path):
    batches = [{"input_ids": torch.tensor([i, i + 1])} for i in range(512)]
    path = tmp_path / "probes.json"
    probes = ProbeSet.pin(batches, path)
    assert ProbeSet.load(batches, path).verify() == probes.verify()
    batches[0]["input_ids"][0] = 9
    with pytest.raises(ValueError, match="checksum"):
        probes.verify()
    with pytest.raises(ValueError, match="512"):
        ProbeSet([], [])


def test_bitwise_repeat():
    class Tiny:
        def __init__(self):
            self.step = 0

        def train_step(self):
            self.step += 1
            return {"loss": 1 / self.step}

    assert bitwise_repeat(Tiny)["steps"] == 200
    calls = iter(range(400))

    class Different:
        def train_step(self):
            return {"loss": next(calls)}

    with pytest.raises(RuntimeError, match="bitwise"):
        bitwise_repeat(Different)


def test_stage_exception_retried_once_then_aborts(tmp_path):
    probes = ProbeSet([torch.tensor([i]) for i in range(512)],
                      [batch_sha256(torch.tensor([i])) for i in range(512)])
    approvals = {f"{s}:default": "approved" for s in STEPS}
    approvals["C4:layout2"] = "approved2"
    output = tmp_path / "ladder.jsonl"
    calls = {"n": 0}

    def flaky(request, probes):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient divergence")
        path = tmp_path / f"{request.stage}-{request.variant}.pt"
        path.write_bytes(b"fixture checkpoint")
        return {"config_sha256": request.approved_config_sha256,
                "checkpoint": str(path), "metrics": evidence(request.stage)}

    assert LadderRunner(probes, approvals, output).run(flaky).endswith("C6-default.pt")
    assert calls["n"] == 9  # 8 stage attempts (7 + C4:layout2) + 1 bounded retry

    def broken(request, probes):
        raise RuntimeError("diverged")

    with pytest.raises(RuntimeError, match="diverged"):
        LadderRunner(probes, approvals, output, consumed_hours=0).run(broken)
    records = [json.loads(x) for x in output.read_text().splitlines()]
    assert len(records) == 11  # 9 from the flaky run + 2 retry records from the abort
    assert records[9]["attempt"] == 1 and records[10]["attempt"] == 2


def test_runner_records_actual_callbacks_and_both_layouts(tmp_path):
    probes = ProbeSet([torch.tensor([i]) for i in range(512)],
                      [batch_sha256(torch.tensor([i])) for i in range(512)])
    approvals = {f"{s}:default": "approved" for s in STEPS}
    approvals["C4:layout2"] = "approved2"
    calls = []

    def run(request, probes):
        calls.append(request)
        path = tmp_path / f"{request.stage}-{request.variant}.pt"
        path.write_bytes(b"fixture checkpoint")
        return {"config_sha256": request.approved_config_sha256,
                "checkpoint": str(path), "metrics": evidence(request.stage)}

    output = tmp_path / "ladder.jsonl"
    assert LadderRunner(probes, approvals, output).run(run).endswith("C6-default.pt")
    assert len(calls) == 8
    assert len([json.loads(x) for x in output.read_text().splitlines()]) == 8
    assert calls[4].parent_checkpoint == calls[5].parent_checkpoint
    with pytest.raises(RuntimeError, match="budget"):
        LadderRunner(probes, approvals, output, consumed_hours=54).run(run)
