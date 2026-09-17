import copy

import pytest
import torch

from models.transformer import Transformer
from tests.toy import ced_toy_config
from training.pretrain import Trainer, TrainingConfig, seed_everything


class ToyData:
    def __init__(self):
        self.position = 0

    def __next__(self):
        x = (torch.arange(8) + self.position * 7) % 40
        self.position += 1
        return {"input_ids": x, "labels": (x + 1) % 40}

    def state_dict(self):
        return {"manifest_sha256": "fixture", "position": self.position}

    def load_state_dict(self, state):
        if state["manifest_sha256"] != "fixture" or state["position"] < 0:
            raise ValueError("Manifest/position mismatch")
        self.position = state["position"]


def trainer(path, checkpoint=True):
    seed_everything()
    cfg = ced_toy_config()
    return Trainer(Transformer(cfg, max_seq_len=8), ToyData(),
                   TrainingConfig(max_steps=6, warmup_steps=2, checkpoint_activations=checkpoint), path)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("checkpoint", [False, True])
def test_real_ced_multistep_exact_resume(tmp_path, checkpoint):
    uninterrupted = trainer(tmp_path / "full", checkpoint)
    uninterrupted.run(4)
    expected = copy.deepcopy(uninterrupted.state_dict())
    interrupted = trainer(tmp_path / "part", checkpoint)
    interrupted.run(2)
    resumed = trainer(tmp_path / "resume", checkpoint)
    resumed.load_checkpoint(interrupted.last_checkpoint)
    resumed.run(2)
    equal(expected, resumed.state_dict())
    for t in (uninterrupted, interrupted, resumed):
        t.close()


def test_checkpoint_replay_matches_eager_updates_and_loads(tmp_path):
    eager = trainer(tmp_path / "eager", False)
    replay = trainer(tmp_path / "replay", True)
    for _ in range(3):
        a, b = eager.train_step(), replay.train_step()
        equal(a, b)
        equal(eager.model.state_dict(), replay.model.state_dict())
        equal(eager.optimizer.state_dict(), replay.optimizer.state_dict())
    eager.close()
    replay.close()


def test_bad_checkpoint_rejected_without_mutation(tmp_path):
    t = trainer(tmp_path / "run")
    t.run(1)
    original = copy.deepcopy(t.state_dict())
    saved = torch.load(t.last_checkpoint, weights_only=True)
    for field, value in (("config_hash", "wrong"), ("smoke_loss", 100.)):
        bad = copy.deepcopy(saved)
        bad[field] = value
        path = tmp_path / f"bad-{field}.pt"
        torch.save(bad, path)
        with pytest.raises(ValueError):
            t.load_checkpoint(path)
        equal(original, t.state_dict())
    bad = copy.deepcopy(saved)
    bad["data"]["manifest_sha256"] = "wrong"
    torch.save(bad, tmp_path / "manifest.pt")
    with pytest.raises(ValueError, match="Manifest"):
        t.load_checkpoint(tmp_path / "manifest.pt")
    equal(original, t.state_dict())
    t.close()


def test_nonfinite_rollback_skips_batches_halves_lr_and_bounds_retries(tmp_path):
    from training.pretrain import NonfiniteError, TrainingFailure
    t = trainer(tmp_path / "run")
    real_step = t.train_step
    calls = 0

    def fail_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            next(t.dataset)
            raise NonfiniteError("fixture")
        return real_step()

    t.train_step = fail_once
    t.run(1)
    assert t.rollbacks == 1 and t.lr_scale == .5
    assert t.dataset.position == 3
    t.train_step = lambda: (_ for _ in ()).throw(NonfiniteError("always"))
    with pytest.raises(TrainingFailure, match="retry limit"):
        t.run(1)
    assert t.rollbacks == 3
    t.close()


def test_optimizer_groups_retention_and_clip_guard(tmp_path, monkeypatch):
    from training.pretrain import TrainingFailure
    t = trainer(tmp_path / "run")
    decay_ids = {id(p) for p in t.optimizer.param_groups[0]["params"]}
    assert id(t.model.embed.weight) not in decay_ids
    assert all(id(b.ffn.gate.weight) not in decay_ids for b in t.model.blocks)
    t.save_checkpoint(gate=True)
    for _ in range(4):
        t.train_step()
        t.save_checkpoint()
    assert len(list(t.checkpoint_dir.glob("step-*.pt"))) == 3
    assert len(list(t.checkpoint_dir.glob("gate-*.pt"))) == 1
    t.clipped.extend([False] * 94 + [True] * 5)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *a, **kw: torch.tensor(2.))
    with pytest.raises(TrainingFailure, match="Clipped"):
        t.train_step()
    assert t.step == 4
    t.close()


def test_packed_dataset_resume_fixture(tmp_path):
    import json
    from dataclasses import asdict
    from data.dataset import PackedDataset
    from data.prepare_data import prepare, sha256
    from models.config import load_config
    from tests.test_data import FixtureTokenizer

    cfg = ced_toy_config(vocab_size=65536, image_token_id=65408)
    raw = asdict(cfg)
    raw.pop("sha256")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw))
    cfg = load_config(config_path)
    tokenizer = FixtureTokenizer(tmp_path / "tokenizer.json")
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps({"id": str(i), "text": f"text {i} " * 8}) + "\n" for i in range(30)))
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps({"sources": [{"path": source.name, "name": "fineweb-edu", "revision": "fixture", "sha256": sha256(source)}]}))
    root = tmp_path / "prepared"
    prepare(sources, root, tokenizer, config=config_path)

    def make(path):
        seed_everything()
        data = PackedDataset(root, config=config_path, sequence_length=8, production=False)
        return Trainer(Transformer(cfg, max_seq_len=8), data,
                       TrainingConfig(max_steps=3, warmup_steps=2), path)

    a = make(tmp_path / "a")
    a.run(2)
    expected = copy.deepcopy(a.state_dict())
    b = make(tmp_path / "b")
    b.run(1)
    c = make(tmp_path / "c")
    c.load_checkpoint(b.last_checkpoint)
    c.run(1)
    equal(expected, c.state_dict())
    for t in (a, b, c):
        t.close()
        t.dataset.close()
