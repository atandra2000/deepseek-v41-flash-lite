"""CPU fixtures exercise the pipeline, never stand in for the production corpus."""

import json
import sqlite3

import numpy as np
import pytest
import torch
from PIL import Image

from data.dataset import PackedDataset, preflight
from data.prepare_data import GPT2Tokenizer, prepare, sha256
from models.config import load_config


class FixtureTokenizer:
    def __init__(self, path):
        self.path = path
        path.write_text('{"fixture": true}')
        self.identity = {"kind": "fixture", "sha256": sha256(path), "base_vocab": 50257,
                         "eos_id": 50256, "special_start": 65408,
                         "reserved_specials": 128, "vocab_size": 65536}

    def encode(self, text):
        return list(text.encode())

    def token_map(self):
        return np.arange(65536, dtype="<i4"), 65536


def corpus(tmp_path, docs=None, shard_tokens=40000):
    tokenizer = FixtureTokenizer(tmp_path / "tokenizer.json")
    docs = docs if docs is not None else [{"id": str(i), "text": f"document {i} " + "x" * 50} for i in range(60)]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(doc) + "\n" for doc in docs))
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps({"sources": [{"path": source.name, "name": "fineweb-edu",
                                               "revision": "fixture-v1", "sha256": sha256(source)}]}))
    root = tmp_path / "prepared"
    prepare(sources, root, tokenizer, shard_tokens=shard_tokens)
    return root


def same(left, right):
    assert left["documents"] == right["documents"]
    for key in ("input_ids", "labels", "image_mask", "valid_mask"):
        assert torch.equal(left[key], right[key])
    assert len(left["images"]) == len(right["images"])


def repin(root, name):
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    for artifact in manifest["artifacts"]:
        if artifact["path"] == name:
            artifact["sha256"] = sha256(root / name)
    path.write_text(json.dumps(manifest))


def test_preparation_packing_and_resume(tmp_path):
    root = corpus(tmp_path, shard_tokens=150)
    report = preflight(root, production=False)
    assert report["documents"] == 60 and not report["production_ready"]
    db = sqlite3.connect(root / "documents.sqlite")
    train = {r[0] for r in db.execute("SELECT content FROM docs WHERE split='train'")}
    val = {r[0] for r in db.execute("SELECT content FROM docs WHERE split='val'")}
    assert train and val and train.isdisjoint(val)
    db.close()
    dataset = PackedDataset(root, sequence_length=32, production=False)
    next(dataset)
    next(dataset)
    state = json.loads(json.dumps(dataset.state_dict()))
    resumed = PackedDataset(root, sequence_length=32, production=False)
    resumed.load_state_dict(state)
    for expected in dataset:
        same(expected, next(resumed))
        assert not ((expected["labels"] != -100) & ~expected["valid_mask"]).any()
    with pytest.raises(StopIteration):
        next(resumed)
    with pytest.raises(ValueError, match="mismatch"):
        resumed.load_state_dict({**state, "manifest_sha256": "wrong"})
    with pytest.raises(ValueError, match="offset"):
        resumed.load_state_dict({**state, "offset": 99999})
    dataset.close()
    resumed.close()
    with pytest.raises(ValueError):
        preflight(root)  # synthetic tokenizer never grants production approval


def test_corruption_token_bounds_and_manifest(tmp_path):
    root = corpus(tmp_path)
    shard = next(root.glob("*.bin"))
    with shard.open("r+b") as stream:
        stream.write(np.asarray([60000], dtype="<u2").tobytes())
    with pytest.raises(ValueError, match="checksum"):
        preflight(root, production=False)
    repin(root, shard.name)
    with pytest.raises(ValueError, match="token bounds"):
        preflight(root, production=False)


def test_duplicate_document_rejected(tmp_path):
    with pytest.raises(sqlite3.IntegrityError):
        corpus(tmp_path, [{"id": "a", "text": "duplicate"}, {"id": "b", "text": "duplicate"}])
    assert not (tmp_path / "prepared" / "manifest.json").exists()


def test_stage2_counts_original_documents(tmp_path):
    root = corpus(tmp_path, [{"id": str(i), "text": str(i) + "x" * 16384} for i in range(30)])
    dataset = PackedDataset(root, stage=2, production=False)
    sample = next(dataset)
    assert sample["input_ids"].shape == (16384,)
    dataset.close()
    other = tmp_path / "short"
    other.mkdir()
    short = corpus(other)
    with pytest.raises(ValueError, match="30%"):
        PackedDataset(short, stage=2, production=False)


def test_image_spans_survive_packing(tmp_path):
    Image.new("RGB", (42, 42), (100, 150, 200)).save(tmp_path / "image.png")
    docs = [{"id": str(i), "parts": [{"text": f"caption {i} " + "x" * 100},
                                      {"image": "image.png"}, {"text": "end"}]} for i in range(8)]
    root = corpus(tmp_path, docs)
    dataset = PackedDataset(root, sequence_length=512, production=False)
    seen, padded = 0, False
    for sample in dataset:
        padded |= not bool(sample["valid_mask"].all())
        for span in sample["images"]:
            seen += 1
            end = span.start + len(span.types)
            assert end <= 512
            assert sample["image_mask"][span.start:end].all()
            assert (sample["input_ids"][span.start:end] == 65408).all()
            assert span.patches.shape[1:] == (3, 14, 14)
            assert span.patches.min() >= -1 and span.patches.max() <= 1
        assert not (sample["labels"] == 65408).any()
    assert seen and padded
    dataset.close()


def test_missing_artifact_and_source_tamper(tmp_path):
    root = corpus(tmp_path)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["artifacts"][0]["path"] = "../source.jsonl"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsafe"):
        preflight(root, production=False)
    sources = tmp_path / "sources.json"
    (tmp_path / "source.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        prepare(sources, tmp_path / "second", FixtureTokenizer(tmp_path / "other-tokenizer.json"))


def test_config_and_tokenizer_rejection(tmp_path):
    root = corpus(tmp_path)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["config_sha256"] = "wrong"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="config"):
        preflight(root, production=False)
    with pytest.raises(ValueError, match="GPT-2"):
        GPT2Tokenizer(tmp_path / "tokenizer.json")
    assert load_config().image_token_id == 65408
