# Phase 3: offline data prep and loaders

Task 10 pipeline: offline preparation from already-cleaned local JSONL, manifests,
preflight, packed training dataset. No downloads happen here (workspace rule);
the production corpus is prepared once on the host before any GPU rental
(candidate 13 §10).

## Layout of a prepared tree (`data/prepare_data.py prepare`)

- `shard-NNNNN.bin` — uint16 token shards with EOS between documents.
- `documents.sqlite` — document identity/content dedup, split, shard/offset,
  image spans; the resume index.
- `token_map.npy` + `tokenizer.json` — Engram compressed-id map (upstream
  `engram.py:build_compressed_token_map`) and the pinned tokenizer.
- `images/<sha>.npy` — preprocessed `[n_vit_h*n_vit_w, 3, 14, 14]` float32
  patches (resize → normalize to [-1,1]; the 3×3 pixel-unshuffle is a reshape
  at `models/vit.py:PatchEmbed`).
- `manifest.json` — written last: config sha256, tokenizer identity, source
  provenance (path/revision/sha256), per-artifact sha256, host prep seconds.
  A failed run leaves no manifest, so partial trees cannot load.

## Usage

```bash
python -m data.prepare_data prepare --sources sources.json \
    --tokenizer /path/to/gpt2-tokenizer.json --out data/prepared-v1
python -m data.prepare_data preflight --data data/prepared-v1 [--fixture]
```

`sources.json` lists local sources with `path/name/revision/sha256`; documents
are `{"id", "text"}` or `{"id", "parts": [{"text"}, {"image": "rel/path"}]}`.
Split (90/10, content-hash, document-disjoint) and iteration order are seeded
by content, never input file order.

## Dataset

`data/dataset.py:PackedDataset` is a stateful single-consumer iterator (not
DataLoader-worker safe): yields `input_ids/labels/image_mask/images/valid_mask/
documents`. Images never split across samples; boundary/image/padding labels
are `-100`. `state_dict`/`load_state_dict` give exact O(log n) resume validated
against the manifest. `preflight` verifies every checksum, token bound, image
shape/range, dedup and split integrity; production mode also enforces the
candidate §10 budgets (20B train + 2B val text, ~2B image-text tokens, ≥30%
stage-2 documents ≥16K tokens). `--fixture` checks integrity only and never
grants production approval.

## Status

- Code + fixture tests: green (`tests/test_data.py`, 7 CPU tests).
- Production tokenizer: no full GPT-2 BPE (50,257-entry) tokenizer.json exists
  on this machine yet — the strict loader rejects partial vocabularies.
- Production preflight: pending real FineWeb-Edu/DataComp sources; fixture
  tests do not satisfy the Task-10 acceptance gate.
