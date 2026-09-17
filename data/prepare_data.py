"""Offline Task-10 adapter over already cleaned local JSONL; never downloads data."""

import argparse
import hashlib
import json
import math
import re
import shutil
import sqlite3
import time
import unicodedata
from pathlib import Path

import numpy as np

from models.config import load_config

EOS = 50256
BASE_VOCAB = 50257
SPECIAL_START = 65408


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_path(root, name):
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError(f"Missing or unsafe artifact: {name}")
    return path


class GPT2Tokenizer:
    """Local GPT-2 BPE; unused embedding rows stay padding, not invented tokens."""

    def __init__(self, path):
        from tokenizers import Tokenizer

        self.path = Path(path)
        raw = json.loads(self.path.read_text())
        model = raw.get("model", {})
        if (model.get("type") != "BPE" or len(model.get("vocab", {})) != BASE_VOCAB
                or raw.get("pre_tokenizer", {}).get("type") != "ByteLevel"
                or raw.get("decoder", {}).get("type") != "ByteLevel"
                or raw.get("normalizer") is not None):
            raise ValueError("Expected a local, unmodified GPT-2 ByteLevel BPE tokenizer.json")
        self.backend = Tokenizer.from_file(str(path))
        self.backend.no_padding()
        self.backend.no_truncation()
        self.backend.encode_special_tokens = True
        if (self.backend.get_vocab_size() != BASE_VOCAB
                or self.backend.token_to_id("<|endoftext|>") != EOS
                or set(self.backend.get_vocab().values()) != set(range(BASE_VOCAB))):
            raise ValueError("GPT-2 vocabulary/IDs mismatch")
        self.identity = {"kind": "gpt2-bpe", "sha256": sha256(path),
                         "base_vocab": BASE_VOCAB, "eos_id": EOS,
                         "special_start": SPECIAL_START, "reserved_specials": 128,
                         "vocab_size": 65536}

    def encode(self, text):
        ids = self.backend.encode(text, add_special_tokens=False).ids
        if any(t >= EOS for t in ids):
            raise ValueError("Ordinary text must not inject special tokens")
        return ids

    def token_map(self):
        lookup, keys = [], {}
        for token_id in range(65536):
            if token_id >= BASE_VOCAB:
                key = ("reserved", token_id)
            else:
                text = self.backend.decode([token_id], skip_special_tokens=False)
                if "�" in text:
                    key = ("raw", self.backend.id_to_token(token_id))
                else:
                    norm = unicodedata.normalize("NFD", unicodedata.normalize("NFKC", text))
                    norm = "".join(c for c in norm if unicodedata.category(c) != "Mn").lower()
                    norm = re.sub(r"[ \t\r\n]+", " ", norm)
                    norm = " " if norm == " " else norm.strip()
                    key = ("text", norm or text)
            lookup.append(keys.setdefault(key, len(keys)))
        return np.asarray(lookup, dtype="<i4"), len(keys)


def prepare_image(path, out, cfg):
    from PIL import Image, ImageOps

    image_hash = sha256(path)
    name = f"images/{image_hash}.npy"
    with Image.open(path) as src:
        image = ImageOps.exif_transpose(src).convert("RGB")
        width, height = image.size
        patch, ratio = cfg.vision_patch_size, cfg.vision_downsample_ratio
        area = max(width * height, cfg.vision_min_pixels)
        scale = math.sqrt(area / (width * height))
        h, w = max(1, math.ceil(height * scale / patch)), max(1, math.ceil(width * scale / patch))
        while math.ceil(h / ratio) * math.ceil(w / ratio) > cfg.vision_max_n_token:
            if h >= w and h > 1:
                h -= 1
            elif w > 1:
                w -= 1
            else:
                break  # ponytail: 1x1 grid is the floor; oversized 14px image would fail later
        pixels = np.asarray(image.resize((w * patch, h * patch), Image.Resampling.BICUBIC), dtype=np.float32)
    patches = (pixels / 127.5 - 1).reshape(h, patch, w, patch, 3).transpose(0, 2, 4, 1, 3)
    target = out / name
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        np.save(target, patches.reshape(h * w, 3, patch, patch).astype("<f4"), allow_pickle=False)
    types = [0] + ([1] * math.ceil(w / ratio) + [2]) * math.ceil(h / ratio) + [3]
    return {"path": name, "sha256": sha256(target), "source_sha256": image_hash,
            "n_vit_h": h, "n_vit_w": w, "types": types}


def prepare(source_manifest, out, tokenizer, *, shard_tokens=8_000_000, seed=1337, config=None):
    """Source manifest lists local JSONL paths, source name, revision, and sha256.

    Records: {id, text}, or {id, parts: [{text: ...}, {image: relative_path}]}.
    Split and order derive from document content, never from input file order.
    """
    started = time.monotonic()
    cfg = load_config(config)
    if (cfg.vocab_size, cfg.num_reserved_specials, cfg.image_token_id) != (65536, 128, SPECIAL_START):
        raise ValueError("Unsupported vocabulary contract")
    if type(shard_tokens) is not int or shard_tokens <= 0:
        raise ValueError("shard_tokens must be positive")
    source_manifest, out = Path(source_manifest).resolve(), Path(out)
    spec = json.loads(source_manifest.read_text())
    if not isinstance(spec.get("sources"), list) or not spec["sources"]:
        raise ValueError("Source manifest must contain nonempty sources")
    out.mkdir(parents=True, exist_ok=False)
    db = sqlite3.connect(out / "documents.sqlite")
    db.execute("CREATE TABLE docs (key TEXT PRIMARY KEY, identity TEXT UNIQUE, content TEXT UNIQUE, "
               "split TEXT, modality TEXT, shard TEXT, offset INTEGER, length INTEGER, images TEXT)")
    db.execute("CREATE INDEX split_order ON docs(split, key)")
    artifacts, provenance = [], []
    shard = None
    used, shard_index, total = 0, 0, 0
    try:
        for source in spec["sources"]:
            if not all(isinstance(source.get(k), str) and source[k] for k in ("path", "name", "revision", "sha256")):
                raise ValueError("Each source needs path, name, revision and sha256")
            path = checked_path(source_manifest.parent, source["path"])
            if sha256(path) != source["sha256"]:
                raise ValueError(f"Source checksum mismatch: {path}")
            provenance.append(dict(source))
            with path.open() as records:
                for line in records:
                    doc = json.loads(line)
                    if not isinstance(doc.get("id"), str) or not doc["id"]:
                        raise ValueError("Every document needs a nonempty string id")
                    if ("text" in doc) == ("parts" in doc):
                        raise ValueError("Supply exactly one of text or parts")
                    parts = [{"text": doc["text"]}] if "text" in doc else doc["parts"]
                    if not isinstance(parts, list) or not parts:
                        raise ValueError("Document parts must be nonempty")
                    tokens, images, identity_parts = [], [], []
                    for part in parts:
                        if set(part) == {"text"} and isinstance(part["text"], str):
                            tokens.extend(tokenizer.encode(part["text"]))
                            identity_parts.append({"text": part["text"]})
                        elif set(part) == {"image"} and isinstance(part["image"], str):
                            img = prepare_image(checked_path(path.parent, part["image"]), out, cfg)
                            img["start"] = len(tokens)
                            images.append(img)
                            tokens.extend([cfg.image_token_id] * len(img["types"]))
                            identity_parts.append({"image": img["source_sha256"]})
                        else:
                            raise ValueError("Unsupported document part; expected text or local image")
                    if not tokens:
                        raise ValueError("Empty tokenized document")
                    if any(type(t) is not int or not (0 <= t < EOS or t == SPECIAL_START) for t in tokens):
                        raise ValueError("Invalid source token ID")
                    tokens.append(EOS)
                    content = hashlib.sha256(json.dumps(identity_parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                    split = "val" if int(hashlib.sha256(f"{seed}:split:{content}".encode()).hexdigest(), 16) % 11 == 0 else "train"
                    key = hashlib.sha256(f"{seed}:order:{content}".encode()).hexdigest()
                    if shard is None or used + len(tokens) > shard_tokens:
                        if shard is not None:
                            shard.close()
                            artifacts.append({"path": name, "sha256": sha256(out / name)})
                        name = f"shard-{shard_index:05d}.bin"
                        shard_index += 1
                        shard = (out / name).open("xb")
                        if len(tokens) > shard_tokens:
                            raise ValueError("Single document exceeds shard budget")
                        used = 0
                    db.execute("INSERT INTO docs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                               (key, source["name"] + ":" + doc["id"], content, split,
                                "image_text" if images else "text", name, used, len(tokens), json.dumps(images)))
                    shard.write(np.asarray(tokens, dtype="<u2").tobytes())
                    used += len(tokens)
                    total += 1
                    if total % 1000 == 0:
                        db.commit()
            if sha256(path) != source["sha256"]:
                raise ValueError("Source changed during preparation")
        if not total:
            raise ValueError("No documents prepared")
        shard.close()
        shard = None
        artifacts.append({"path": name, "sha256": sha256(out / name)})
        db.commit()
        db.close()
        artifacts.append({"path": "documents.sqlite", "sha256": sha256(out / "documents.sqlite")})
        mapping, size = tokenizer.token_map()
        np.save(out / "token_map.npy", mapping, allow_pickle=False)
        artifacts.append({"path": "token_map.npy", "sha256": sha256(out / "token_map.npy")})
        shutil.copyfile(tokenizer.path, out / "tokenizer.json")
        artifacts.append({"path": "tokenizer.json", "sha256": sha256(out / "tokenizer.json")})
        manifest = {"version": 1, "config_sha256": cfg.sha256, "tokenizer": tokenizer.identity,
                    "engram_compressed_vocab_size": size, "seed": seed, "documents": total,
                    "sources": provenance, "artifacts": artifacts,
                    "image_preprocessing": "exif-rgb-bicubic-patch14-normalize-v1",
                    "host_seconds": time.monotonic() - started}
        # Manifest is the completion marker; failed prep trees cannot be loaded.
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest
    finally:
        if shard is not None:
            shard.close()
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--sources", required=True)
    prep.add_argument("--tokenizer", required=True)
    prep.add_argument("--out", required=True)
    prep.add_argument("--shard-tokens", type=int, default=8_000_000)
    prep.add_argument("--seed", type=int, default=1337)
    check = commands.add_parser("preflight")
    check.add_argument("--data", required=True)
    check.add_argument("--fixture", action="store_true", help="Integrity only; never production approval")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.sources, args.out, GPT2Tokenizer(args.tokenizer),
                         shard_tokens=args.shard_tokens, seed=args.seed)
    else:
        from data.dataset import preflight
        result = preflight(args.data, production=not args.fixture)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
