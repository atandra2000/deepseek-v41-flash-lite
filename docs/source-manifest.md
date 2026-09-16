# DS-V4.1-Flash-Lite — Source manifest (Task 1)

> Pinned 2026-09-16. This file records exactly what upstream material the
> architecture contract was extracted from. Every hash below is sha256.

## Upstream repository

| Field | Value |
|---|---|
| Repo | `deepseek-ai/DeepSeek-V4.1-Flash` (HuggingFace model repo) |
| License | MIT (`LICENSE` sha256 `f2c6c602815669d292889e5be8c802f2ed950653b77999b1584e8e6aed25d040`) |
| Commit (revision) | `dba1be0a40aa45a94ad051997016db3960a90277` |
| Pinned locally under | `upstream/` (vendored in this repo, unmodified) |

## Hashes

- `config.json` (root, released HF config): `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`
- Every vendored file's sha256 is recorded in `upstream/SHA256SUMS` (regenerate
  with `shasum -a 256 upstream/config.json upstream/LICENSE upstream/README.md upstream/inference/* upstream/encoding/* | grep -v '/$' > upstream/SHA256SUMS`).

## Vendored files and their role

| File | Size (B) | Role in Task 1 |
|---|---|---|
| `config.json` | 3,311 | Ground truth for all upstream dims (§2.1 of the design). Root config, HF `transformers` view. |
| `LICENSE` | 1,084 | MIT license text (upstream redistribution). |
| `README.md` | 13,110 | Repo overview; pointers to tech report. |
| `inference/model.py` | 61,549 | **Primary authority**: ModelArgs, Attention/Compressor/Indexer/candidate pool, Engram, MoE gate, Block/mHC, DSpark blocks, Transformer wiring. |
| `inference/kernel.py` | 23,790 | Authority for `sparse_attn` (shared-latent KV, attn-sink, online softmax), `hc_split_sinkhorn`, FP8/FP4 quantization block sizes and scale dtypes. |
| `inference/engram.py` | 8,138 | Authority for n-gram hashing: compressed token map, prime bucket layout, per-layer multipliers, rolling-XOR hash, DEAD/pad semantics. |
| `inference/vision.py` | 4,457 | Authority for ViT (patch embed, 2D-RoPE, SwiGLU MLP) and the 2-layer aligner projector. |
| `inference/image_processor.py` | 7,699 | Authority for image→token-span layout (`[IMAGE_START] + ([IMAGE]·w + [NL])·h + [IMAGE_END]`), patch grid planning, pixel-unshuffle feeding `PatchEmbed`. |
| `inference/generate.py` | 8,722 | Prefill/decode loop; image spans prefilled in one chunk. |
| `inference/convert.py` | 9,458 | Checkpoint namespace map (`model.layers.*`, `mtp.*` for DSpark, tied MTP embed/head, `e_score_correction_bias`→router bias). |
| `inference/config.json` | 1,982 | Released-model runtime view (matches root config; `rope_head_dim: 64` etc.). |
| `inference/requirements.txt`, `inference/run.sh`, `inference/README.md` | — | Runtime scaffolding (not contract-relevant). |
| `encoding/encoding.py` | 37,316 | Prompt-format reference (chat/thinking/DSML/VL). Lite does **not** adopt this format (D4: GPT-2 BPE + 128 specials); retained for SFT-format reference only. |
| `encoding/test_encoding.py` | 19,371 | Format tests (reference only). |
| `encoding/README.md` | 12,120 | Format documentation (reference only). |

## What the reference does and does not pin

- **Pinned by reference code** (see `architecture-contract.md`): producer/consumer
  wiring, CSA2 layer roles, candidate-pool construction, indexer key path,
  Engram lookup/gating, DSpark draft/verify forward, mHC coefficient pipeline,
  attention core semantics, quantization placement.
- **Not in the reference** (training-time facts): optimizer choice, ViT two-stage
  training, FP8-KV QAT, data/prompt-format training details. Those are governed
  by the design doc's disclosed deviations (§2.3) and the candidate's §8/§9/§10
  contracts — never by tech-report prose.

## Tech report

`DeepSeek_V41_Tech_Report.pdf` (1.8 MB, LFS) exists in the upstream repo. It is
deliberately **not** vendored: Task 1 pins from code + config only. If a
mechanism question is unanswerable from code, it is escalated to the owner
rather than answered "per the report".
