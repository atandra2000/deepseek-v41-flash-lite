"""Manifest-verified, bounded-memory packing and exact single-consumer resume."""

import json
import sqlite3
from pathlib import Path

import numpy as np
import torch

from data.prepare_data import BASE_VOCAB, EOS, SPECIAL_START, checked_path, sha256
from models.config import load_config
from models.vit import ImageSpan


def _database(root):
    path = checked_path(root, "documents.sqlite")
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def _image(root, image, cfg):
    path = checked_path(root, image["path"])
    if sha256(path) != image["sha256"]:
        raise ValueError("Image checksum mismatch")
    patches = np.load(path, allow_pickle=False, mmap_mode="r")
    h, w = image["n_vit_h"], image["n_vit_w"]
    if type(h) is not int or type(w) is not int or min(h, w) <= 0:
        raise ValueError("Invalid image grid")
    r = cfg.vision_downsample_ratio
    expected = [0] + ([1] * ((w + r - 1) // r) + [2]) * ((h + r - 1) // r) + [3]
    if (image["types"] != expected or expected.count(1) > cfg.vision_max_n_token
            or patches.shape != (h * w, 3, cfg.vision_patch_size, cfg.vision_patch_size)
            or patches.dtype != np.dtype("<f4") or not np.isfinite(patches).all()
            or (np.abs(patches) > 1).any()):
        raise ValueError("Invalid preprocessed image")
    return patches


def preflight(root, *, production=True, config=None):
    """Full local integrity scan. Production budgets cannot be waived accidentally."""
    root, cfg = Path(root).resolve(), load_config(config)
    manifest = json.loads(checked_path(root, "manifest.json").read_text())
    if manifest["version"] != 1 or manifest["config_sha256"] != cfg.sha256:
        raise ValueError("Manifest version/config mismatch")
    names = set()
    for artifact in manifest["artifacts"]:
        name = artifact["path"]
        if name in names or sha256(checked_path(root, name)) != artifact["sha256"]:
            raise ValueError(f"Duplicate artifact or checksum mismatch: {name}")
        names.add(name)
    if not {"documents.sqlite", "tokenizer.json", "token_map.npy"} <= names:
        raise ValueError("Missing required artifacts")
    identity = manifest["tokenizer"]
    if (identity["sha256"] != sha256(root / "tokenizer.json")
            or identity["base_vocab"] != BASE_VOCAB or identity["eos_id"] != EOS
            or identity["vocab_size"] != cfg.vocab_size
            or identity["special_start"] != cfg.image_token_id
            or identity["reserved_specials"] != cfg.num_reserved_specials):
        raise ValueError("Tokenizer identity mismatch")
    mapping = np.load(root / "token_map.npy", allow_pickle=False)
    if (mapping.shape != (cfg.vocab_size,) or mapping.dtype != np.dtype("<i4")
            or mapping.min() < 0 or mapping.max() + 1 != manifest["engram_compressed_vocab_size"]
            or len(np.unique(mapping)) != manifest["engram_compressed_vocab_size"]):
        raise ValueError("Invalid Engram token map")
    if production:
        from data.prepare_data import GPT2Tokenizer
        tokenizer = GPT2Tokenizer(root / "tokenizer.json")
        expected_map, size = tokenizer.token_map()
        if identity != tokenizer.identity or not np.array_equal(mapping, expected_map):
            raise ValueError("Production tokenizer/map mismatch")
    counts = {"train_text": 0, "val_text": 0, "train_image_text": 0,
              "val_image_text": 0, "documents": 0, "long_documents": 0,
              "stage2_documents": 0, "image_pairs": 0}
    db = _database(root)
    try:
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Invalid document database")
        for column in ("identity", "content", "key"):
            if db.execute(f"SELECT 1 FROM docs GROUP BY {column} HAVING COUNT(*) > 1 LIMIT 1").fetchone():
                raise ValueError("Duplicate documents/split leakage")
        shard_names = {row[0] for row in db.execute("SELECT DISTINCT shard FROM docs")}
        if shard_names != {name for name in names if name.endswith(".bin")}:
            raise ValueError("Shard listing mismatch")
        for name in sorted(shard_names):
            path = checked_path(root, name)
            if path.stat().st_size % 2 or path.stat().st_size == 0:
                raise ValueError("Invalid shard size")
            tokens = np.memmap(path, mode="r", dtype="<u2")
            offset = 0
            for doc in db.execute("SELECT * FROM docs WHERE shard=? ORDER BY offset", (name,)):
                length = doc["length"]
                if doc["offset"] != offset or length < 2 or offset + length > len(tokens):
                    raise ValueError("Invalid document boundary/offset")
                ids = tokens[offset:offset + length]
                if ids[-1] != EOS or not ((ids < BASE_VOCAB) | (ids == SPECIAL_START)).all():
                    raise ValueError("Invalid token bounds/EOS")
                images = json.loads(doc["images"])
                image_mask = np.zeros(length, dtype=bool)
                previous_end = 0
                for image in images:
                    _image(root, image, cfg)
                    start, end = image["start"], image["start"] + len(image["types"])
                    if (start < previous_end or end >= length or end <= start
                            or not (ids[start:end] == SPECIAL_START).all()
                            or len(image["types"]) > cfg.context_train_stage1):
                        raise ValueError("Invalid image span")
                    image_mask[start:end] = True
                    previous_end = end
                if not np.array_equal(ids == SPECIAL_START, image_mask):
                    raise ValueError("Unmapped image tokens")
                if doc["split"] not in ("train", "val") or doc["modality"] != ("image_text" if images else "text"):
                    raise ValueError("Invalid split/modality")
                counts[doc["split"] + "_" + doc["modality"]] += length - 1
                counts["documents"] += 1
                counts["image_pairs"] += len(images)
                if doc["split"] == "train" and not images:
                    counts["stage2_documents"] += 1
                    counts["long_documents"] += length - 1 >= cfg.context_train_stage2
                offset += length
            if offset != len(tokens):
                raise ValueError("Unreferenced shard tokens")
        if counts["documents"] != manifest["documents"]:
            raise ValueError("Document count mismatch")
        if production:
            if (counts["train_text"] < 20_000_000_000 or counts["val_text"] < 2_000_000_000
                    or counts["train_image_text"] < 2_000_000_000):
                raise ValueError(f"Production corpus budgets not met: {counts}")
            if counts["long_documents"] * 10 < counts["stage2_documents"] * 3:
                raise ValueError("Stage 2 requires >=30% original documents >=16K tokens")
    finally:
        db.close()
    return {"production_ready": production, "manifest_sha256": sha256(root / "manifest.json"), **counts}


class PackedDataset:
    """Stateful single-consumer iterator; use directly, not DataLoader workers.

    Text documents can span samples. Images never split. Labels crossing a
    document boundary, image labels, and right padding are ignored (-100).
    Epochs repeat the same seeded preparation order; resume is O(log documents).
    """

    def __init__(self, root, *, stage=1, split="train", sequence_length=None,
                 epoch=0, production=True, config=None):
        self.root, self.cfg = Path(root).resolve(), load_config(config)
        if stage not in (1, 2) or split not in ("train", "val") or type(epoch) is not int or epoch < 0:
            raise ValueError("Invalid stage/split/epoch")
        report = preflight(root, production=production, config=config)
        self.manifest_hash = report["manifest_sha256"]
        self.stage, self.split, self.epoch = stage, split, epoch
        self.sequence_length = sequence_length or (self.cfg.context_train_stage1 if stage == 1 else self.cfg.context_train_stage2)
        if type(self.sequence_length) is not int or self.sequence_length < 2:
            raise ValueError("sequence_length must be >=2")
        if production and self.sequence_length != (self.cfg.context_train_stage1 if stage == 1 else self.cfg.context_train_stage2):
            raise ValueError("Production sequence length mismatch")
        self.db = _database(self.root)
        self.where = "split=?" + (" AND modality='text'" if stage == 2 else "")
        count, long_count = self.db.execute(
            f"SELECT COUNT(*), SUM(length-1>=?) FROM docs WHERE {self.where}",
            (self.cfg.context_train_stage2, split)).fetchone()
        if not count or (stage == 2 and (long_count or 0) * 10 < count * 3):
            self.close()
            raise ValueError("Empty split or stage-2 long-document fraction below 30%")
        self.key, self.offset, self.step = "", 0, 0
        self._shard_name, self._tokens = None, None

    def close(self):
        self.db.close()
        self._tokens = None

    def __iter__(self):
        return self

    def state_dict(self):
        return {"manifest_sha256": self.manifest_hash, "config_sha256": self.cfg.sha256,
                "stage": self.stage, "split": self.split, "sequence_length": self.sequence_length,
                "epoch": self.epoch, "key": self.key, "offset": self.offset, "step": self.step}

    def load_state_dict(self, state):
        expected = self.state_dict()
        if set(state) != set(expected) or any(state[k] != expected[k] for k in expected if k not in ("key", "offset", "step")):
            raise ValueError("Resume manifest/config/order mismatch")
        if any(type(state[k]) is not int or state[k] < 0 for k in ("offset", "step")):
            raise ValueError("Invalid resume position")
        row = self.db.execute(f"SELECT length, images FROM docs WHERE {self.where} AND key=?", (self.split, state["key"])).fetchone()
        if ((state["key"] and row is None) or (not state["key"] and (state["offset"] or state["step"]))
                or (row is not None and state["offset"] >= row["length"])):
            raise ValueError("Resume document/offset missing")
        if row and any(img["start"] < state["offset"] < img["start"] + len(img["types"]) for img in json.loads(row["images"])):
            raise ValueError("Resume offset splits an image")
        self.key, self.offset, self.step = state["key"], state["offset"], state["step"]

    def __next__(self):
        cfg, length = self.cfg, self.sequence_length
        x = torch.full((length,), cfg.engram_pad_token_id, dtype=torch.long)
        y = torch.full((length,), -100, dtype=torch.long)
        mask = torch.zeros(length, dtype=torch.bool)
        valid = torch.zeros(length, dtype=torch.bool)
        spans, documents, filled = [], [], 0
        while filled < length:
            operator = "=" if self.offset else ">"
            doc = self.db.execute(f"SELECT * FROM docs WHERE {self.where} AND key {operator} ? ORDER BY key LIMIT 1",
                                  (self.split, self.key)).fetchone()
            if doc is None:
                break
            start = self.offset
            end = min(doc["length"], start + length - filled)
            images = json.loads(doc["images"])
            for img in images:
                if len(img["types"]) > length:
                    raise ValueError("Image span exceeds sequence length")
                if start <= img["start"] < end < img["start"] + len(img["types"]):
                    end = img["start"]
            if end == start:
                break  # pad this sample, leave the whole image for the next
            if self._shard_name != doc["shard"]:
                self._tokens = np.memmap(checked_path(self.root, doc["shard"]), dtype="<u2", mode="r")
                self._shard_name = doc["shard"]
            ids = self._tokens[doc["offset"]:doc["offset"] + doc["length"]]
            n = end - start
            x[filled:filled + n] = torch.from_numpy(ids[start:end].astype(np.int64))
            targets = min(n, doc["length"] - start - 1)
            y[filled:filled + targets] = torch.from_numpy(ids[start + 1:start + 1 + targets].astype(np.int64))
            valid[filled:filled + n] = True
            for img in images:
                if start <= img["start"] < end:
                    pos = filled + img["start"] - start
                    patches = _image(self.root, img, cfg)
                    spans.append(ImageSpan(pos, torch.from_numpy(patches.copy()), img["n_vit_h"], img["n_vit_w"], torch.tensor(img["types"])))
                    mask[pos:pos + len(img["types"])] = True
            documents.append({"key": doc["key"], "start": filled, "length": n, "document_offset": start})
            filled += n
            self.key = doc["key"]
            self.offset = end if end < doc["length"] else 0
        if not filled:
            raise StopIteration
        y[y == cfg.image_token_id] = -100
        self.step += 1
        return {"input_ids": x, "labels": y, "image_mask": mask, "images": spans,
                "valid_mask": valid, "documents": documents}
