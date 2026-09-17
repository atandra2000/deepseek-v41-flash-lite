"""CPU tests for the A100 runbook: wiring only — GPU evidence is Task 12.

`runbook` is exercised on fixture data: env pinning, probe pinning from a
real (fixture) val split, the 200-step bitwise-repeat driver, the
approvals-drift abort, and ladder wiring. Fixture corpora and toy dims never
stand in for production or GPU evidence.
"""

import json
import sys
from itertools import islice
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import PackedDataset
from models.transformer import Transformer
from models.variants import NAMED, derive_config
from tests.test_data import corpus as fixture_corpus
from tests.test_stages import SyntheticCorpus
from tests.test_variants import gate_toy_config
from training import runbook
from training.pretrain import seed_everything


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """Fixture corpus with enough val samples to pin 512 probe batches."""
    tmp = tmp_path_factory.mktemp("corpus")
    docs = [{"id": str(i), "text": str(i) + " document " + "x" * 60} for i in range(3000)]
    return fixture_corpus(tmp, docs, shard_tokens=100000)


def pinned_val_batches(root, sequence_length=24, limit=512):
    dataset = PackedDataset(root, split="val", sequence_length=sequence_length, production=False)
    try:
        return [{"input_ids": s["input_ids"], "labels": s["labels"]} for s in islice(dataset, limit)]
    finally:
        dataset.close()


def test_env_pin_records_and_matches(tmp_path):
    seed_everything(0)
    snap = runbook.pin(tmp_path / "env.json", 0)
    assert snap["cublas_workspace_config"] == ":4096:8" and snap["deterministic"]
    assert not snap["matmul_tf32"] and snap["device_type"] == "cpu"
    assert json.loads((tmp_path / "env.json").read_text()) == snap
    assert runbook.env_matches(snap)


def test_pinned_env_aborts_when_not_pinned(monkeypatch):
    # A caller that dirtied the determinism flags (without re-seeding) must
    # be refused before any GPU spend. seed_everything is stubbed so the
    # dirty flags survive to the check.
    torch.backends.cuda.matmul.allow_tf32 = True
    monkeypatch.setattr(runbook, "seed_everything", lambda stage_id=0: None)
    try:
        with pytest.raises(RuntimeError, match="not pinned"):
            runbook.pinned_env()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = False


def test_pin_probes_real_val_split(prepared, tmp_path):
    probes_path = tmp_path / "probes.json"
    sha = runbook.pin_probes(prepared, probes_path, sequence_length=24, fixture=True)
    from training.ladder import ProbeSet

    loaded = ProbeSet.load(pinned_val_batches(prepared), probes_path)
    assert len(loaded.batches) == 512 and loaded.verify() == sha
    tampered = [{"input_ids": b["input_ids"], "labels": (b["labels"] + 1) % 65536}
                for b in loaded.batches]
    with pytest.raises(ValueError, match="checksum"):
        ProbeSet.load(tampered, probes_path)


def test_repeat_driver_uses_real_trainer(tmp_path):
    cfg = gate_toy_config()

    def factory():
        from training.ladder import SEED
        from training.pretrain import Trainer, TrainingConfig

        seed_everything(0)
        torch.manual_seed(SEED)
        model = Transformer(derive_config(cfg, NAMED["dense"]), variant=NAMED["dense"])
        return Trainer(model, SyntheticCorpus(),
                       TrainingConfig(max_steps=200, warmup_steps=24, learning_rate=2e-3),
                       tmp_path / "repeat", "cpu")

    env = tmp_path / "env.json"
    runbook.pin(env)
    evidence = runbook.repeat(factory, tmp_path / "repeat.json", env)
    assert evidence["steps"] == 200 and len(evidence["loss_sha256"]) == 64
    assert evidence["env_sha256"] == runbook.sha256_file(env)
    assert json.loads((tmp_path / "repeat.json").read_text()) == evidence


def test_ladder_approvals_drift_aborts(tmp_path):
    approvals = tmp_path / "approvals.json"
    approvals.write_text(json.dumps({"C0:default": "drifted"}))
    with pytest.raises(ValueError, match="drifted"):
        runbook.ladder_run(tmp_path, tmp_path / "ck", tmp_path / "l.jsonl", tmp_path / "p.json",
                           approvals_path=approvals)


def test_ladder_wiring_on_fixture_corpus(prepared, tmp_path, monkeypatch):
    """Open the real fixture PackedDatasets, load the pinned probes, and wire
    the canonical GateRunner + LadderRunner together — without executing the
    production step counts (the ladder itself is covered in test_stages)."""
    import training.ladder as ladder

    captured = {}

    class FakeRunner:
        def __init__(self, cfg, data, checkpoint_dir, **kwargs):
            captured["cfg"], captured["data"] = cfg, data
            assert isinstance(data, PackedDataset)
            assert data.sequence_length == cfg.context_train_stage1  # stage-1 train stream
            assert data.split == "train"
            captured["kwargs"] = kwargs

        def run_stage(self, request, probes):
            captured["probes_verify"] = probes.verify()
            return {"config_sha256": request.approved_config_sha256,
                    "checkpoint": str(tmp_path / "unused.pt"),
                    "metrics": {"losses": [0.0], "grad_norms": [0.0], "probe_loss": 0.0,
                                "probe_batches": 512, "tokens": 1, "nonfinite_count": 0}}
    class FakeLadder:
        def __init__(self, probes, approvals, output):
            captured["ladder"] = (probes, approvals)

        def run(self, run_stage):
            captured["run_stage"] = run_stage
            return tmp_path / "final.pt"

    sha = runbook.pin_probes(prepared, tmp_path / "probes.json", sequence_length=24, fixture=True)
    monkeypatch.setattr(ladder, "GateRunner", FakeRunner)
    monkeypatch.setattr(ladder, "LadderRunner", FakeLadder)
    checkpoint = runbook.ladder_run(prepared, tmp_path / "ck", tmp_path / "ladder.jsonl",
                                    tmp_path / "probes.json", fixture=True, learning_rate=1e-3,
                                    probe_sequence_length=24)
    assert checkpoint == tmp_path / "final.pt"
    assert captured["kwargs"]["learning_rate"] == 1e-3
    assert captured["ladder"][1]["C0:default"] == ladder.gate_approvals(captured["cfg"])["C0:default"]
    from training.ladder import StageRequest
    probes, approvals = captured["ladder"]
    request = StageRequest("C0", "default", 1, None, approvals["C0:default"], probes.verify())
    captured["run_stage"](request, probes)  # the wired callback runs a real stage request
    assert captured["probes_verify"] == sha
