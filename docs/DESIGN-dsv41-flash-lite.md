# DS-V4.1-Flash-Lite — Design Specification

> **Status:** design for [candidate 13](../../../llm-research/candidates/13-deepseek-v41-flash.md) v3
> (2026-09-16). Phase 0, 1, and 2 complete (Tasks 1–9 green); Phase 3:
> Task 10 code + fixture tests green (production corpus prep open); Task 11
> complete: trainer/recovery, GateRunner, executable variants and matched-token
> controls are CPU-verified (toy C0–C6 and bitwise repeat). A100
> 200-step bitwise repeat and C0–C6 measured evidence remain for Task 12.
> This document pins the architecture contract; the
> [execution plan](EXECUTION-PLAN-dsv41-flash-lite.md) pins the build order.
> All upstream facts were cross-checked against the released `config.json` on
> 2026-09-16.

## 1. Objective and boundaries

Replicate the DeepSeek-V4.1-Flash architecture — CED + CSA2 + hierarchical
sparse indexer + SWA branch + mHC + Engram + MoE + DSpark + ViT — as a
from-scratch PyTorch model, re-sized to **~1.16B total / ~205M active**,
trained on 4× A100 80GB for ≤ $500 (≤ 90 node-hours). The sizing principle:
this architecture saves **KV memory, not per-token FLOPs**, so the design
maximizes active parameters (retrieval-salience capability floor), pins the
production stage at **16K context with ≥30% long documents** (where sparse
attention has something to demonstrate), and consolidates MoE into few wide
experts.

**Boundaries.** Raw PyTorch, no HF Trainer/Lightning; BF16 compute with FP32
master weights; FA2 + `torch.compile` + activation checkpointing per portfolio
standard. Disclosed deviations from upstream (full list §2.3): AdamW
everywhere (Muon + Sinkhorn-momentum are post-hoc ablations), FP8-KV as a
post-production QAT gate (not during from-scratch training), ViT trained
jointly with AR (SigLIP contrastive stage dropped), tied 65,536 vocabulary
(GPT-2 BPE derived), BF16 Engram tables.

## 2. Source contract

### 2.1 Upstream ground truth (config-verified)

| Component | Upstream setting |
|---|---|
| Layers | 40; `compress_ratios`: two 0-entries, eighteen m=2, eighteen m=1, three 0-entries (the last three are DSpark target layers) |
| Attention | 64 h × d512, q_lora 1280, o_lora 1024, o_groups 8; SWA window 128 |
| CED | decoder layers project global KV from selected encoder states; `kv_source_layer_ids: [2, 8, 14, 20]` (4 producers, not one final state); indexer sources `[2, 8, 14, 20, 24, 28, 32, 36]` |
| Indexer | 32 h × 128, top-k 512; candidate pool `candidate_topk_blocks: 2048` × `block: 8`, built by `candidate_source_layer_id: 20` |
| MoE | 384 routed + 1 shared, top-6, d_ff 2304, `noaux_tc`, `sqrtsoftplus`, routed scaling 1.5, aux-loss-free balancing |
| mHC | hc_mult 4, Sinkhorn 20 iters, eps 1e-6 |
| Engram | `engram_layer_ids: [1, 14]`, max n-gram order 4, ~384M-entry tables over a 16M hash-vocab, 8 h × 256 |
| DSpark | 3 blocks, block size 5, Markov rank 256, 128 routed experts top-3, target layers [37, 38, 39] |
| ViT | 32 layers, d1024, 16 h, patch 14, downsample_ratio 3 (9× token reduction), 2D-RoPE |
| Vocab / context | 129,280 untied; 64K → 1M via YaRN ×16 |
| Optimizers | Muon (head-wise) + AdamW + Sinkhorn-balanced momentum (Algorithm 1) |

### 2.2 The one unresolved source item

The reference implementation (`inference/` + `encoding/` in the upstream
repo) is the authority for producer/consumer wiring that `config.json` does
not fully specify: which decoder layers consume which `kv_source_layer_ids`
producers, Reindex-pool construction order, and the CSA2 mode-assignment
table. The execution plan pins these from the reference code in Phase 0
**before** any model code is written; prose from the tech report is not
sufficient.

### 2.3 Deviations (each is a disclosed Lite adaptation, none silent)

| # | Deviation | Why |
|---|---|---|
| D1 | AdamW on the critical path; Muon + Sinkhorn-momentum as post-hoc ablations | optimizer risk is removed from the $500 critical path; §8.2 of the candidate |
| D2 | FP8-E4M1 main-KV as post-production QAT gate + 500-step finetune; BF16 model is the shipped artifact | the indexer must not learn around quantization noise from scratch |
| D3 | ViT: joint AR training, SigLIP contrastive stage dropped | toy-scale contrastive yields unusable embeddings; image-token pathway unchanged |
| D4 | Vocab 65,536 tied (GPT-2 BPE + 128 reserved specials) | keeps the head out of the active budget; one tokenizer across text+image removes cross-stage mismatch |
| D5 | Engram orders 2–3, 1M-entry tables, BF16, zero-init gate | value at this scale is near zero; retained for fidelity, pre-agreed cut axis |
| D6 | A100 has no FP4/FP8 tensor cores: FP8 is a storage-format cast (dequant before attention), compute stays BF16 | hardware |

## 3. Selected Lite configuration

| Dim | Upstream | Lite (v3) |
|---|---|---|
| Layers | 40 | **24 (12 enc / 12 dec, 3 CSA2 groups per side)** |
| d_model | 5120 | **1024** |
| Heads | 64 × d512, q_lora 1280, o_lora 1024, 8 groups | **16 × d64, q_lora 256, o_lora 256, 4 groups** |
| Indexer | 32 h × 128, top-k 512 | **8 h × 64, top-k 256** |
| Candidate pool | 2048 blk × 8 | **1024 blk × 8 = 8192 positions** |
| SWA | 128 | **128 (kept)** |
| MoE | 384 + 1, top-6, d_ff 2304 | **16 routed + 1 shared, top-2, d_ff 768** |
| Engram | ~384M-entry ×2, orders 2–4, FP8 | **1M-entry ×2, orders 2–3, BF16, zero-init gate** |
| ViT | 32 L d1024, two-stage | **12 L d512, joint AR** |
| DSpark | 3 blocks | **2 blocks, dense, Markov rank 64, frozen backbone** |
| mHC | mult 4 | **mult 2, Sinkhorn 10 iters** |
| Context | 64K→1M | **4K stage-1 → 16K stage-2, ≥30% long docs** |
| Tokens | 45T | **20–24B (~100 tok/active; image-text ≈2B of total)** |
| Vocab | 129,280 untied | **65,536 tied** |

### Parameter and memory accounting

- Attention/layer ≈ 1.5M (q/kv low-rank + o_lora); MoE/layer = 16 × 3·1024·768
  + shared ≈ 40M → **backbone ≈ 1.0B**; + tied embedding 67M + indexer
  projections + Engram 30M + DSpark 25M + ViT ~38M → **~1.16B total**.
- Active/layer ≈ 1.5M attn + 4.7M routed (top-2) + 2.4M shared ≈ 8.6M →
  **~205M active** (excl. embedding, upstream convention). The design must
  re-derive this ledger from the actual `nn.Module` tree in Phase 0 and fail
  if it drifts >5% from these figures. *(Phase-0 re-derivation, 2026-09-16:
  the 1.5M attention estimate omitted the grouped output projection `wo_b`
  (d × o_groups·o_lora = 1.05M/layer); re-derived active = **217.7M** (+6.2%).
  Config unchanged; see architecture-contract.md "Task-2 ledger assertions".)*
- Memory: ~10 GB weights+optimizer+grads (AdamW: 2 fp32 states × 1.16B +
  fp32 master ≈ 14 GB — recount in Phase 0; still trivial on 4×80GB);
  activations <2 GB/gpu-shard at 16K with input checkpointing. **The binding
  constraint is wall-clock, not memory.**
- Compute: 6 · 205e6 · 22e9 ≈ 2.7e19 FLOPs → 15–17 h ideal, 22–28 h
  realistic at 1.4× MoE overhead on 4×A100 (35–40% MFU).

## 4. Forward computation and module ownership

### 4.1 CED producer map (Lite)

- Encoder = layers 0–11 (m=2 compression); decoder = layers 12–23 (m=1).
- Global-KV producers (Lite): encoder layers **2 and 8** (2 producers; the
  design keeps the producer map as a data table, swappable for the C3 gate).
- Every decoder layer owns `W^KV_l`, `W^Z_l` and projects its global KV from
  its assigned producer's final hidden state — never shares a projection.
- Prefill contract: encoder layers compute global KV once at O(N·L/2);
  decoder layers never run full self-attention over the global stream.

### 4.2 CSA2 modes and the indexer

- Mode table (initial; swept in gate C4): per decoder group — one Full-mode,
  one Reindex-mode, one Reuse-mode layer. Encoder layers: Full + compress m=2.
- First decoder Full-mode layer builds the candidate pool (1024 blocks × 8
  positions); Reindex layers search only the pool with the 8h×64 indexer,
  top-k 256; Reuse layers consume the previous layer's selected indices.
- SWA branch: every layer, window 128, layer-local KV always — independent
  of the global pathway.
- Cache invariant: global KV budget = 8192 pool positions + top-k 256
  selected per Reindex layer; local SWA KV = 128/layer. The generation
  cache must expose `global_kv_bytes()` returning measured bytes/token —
  this is the headline metric's source of truth.

### 4.3 Module ownership

- `MoE`: 16 routed + 1 shared, top-2, `noaux_tc` bias balancing, router
  bias zero-init; z-loss 1e-3 only if the C1 gate demands it.
- `Engram`: orders 2–3 multi-head hash lookup, gated residual with
  **zero-init gate** (module outputs exactly 0 at init; assert gate < 1.0).
- `mHC`: hc_mult 2 streams, Sinkhorn 10 iters eps 1e-6, per-step finiteness
  assert; identity-init so the mechanism is a no-op at step 0.
- `ViT`: 12 L d512, patch 14, 3×3 pixel-unshuffle (9× reduction), 2D-RoPE,
  2-layer MLP projector → image tokens from the 128 reserved specials;
  separate module file; trains jointly (interleaved image-text in stage 1).
- `DSpark`: 2 dense blocks, SWA-128, Markov rank 64, 5-position parallel
  draft + confidence-scheduled verification; trained after C8 with the
  backbone frozen — structurally cannot destabilize the production model.

## 5. Data, training and recovery

- **Data (offline, never on the rented clock):** FineWeb-Edu 20B + 2B held-out
  (fixed-seed sample, per-shard sha256 manifest, document-disjoint split);
  ~2B image-text tokens (~2M pairs, fixed manifest); SFT ~200M; RL
  GSM8K+MBPP with pinned checkers. Stage-2 packing guarantees ≥30% documents
  ≥16K tokens. Full table: candidate §10.
- **Training:** AdamW (β 0.9/0.95, wd 0.1 matrices-only), LR 3e-4 → 3e-5
  cosine, 2,000-step warmup (C8) / 200-step (gates); 0.5M tok/step; grad
  clip 1.0 with auto-fail if clipped >5% of last 100 steps; nonfinite guard
  with checkpoint-rollback (max 3 retries). Stage 2: seq 16,384, 8K-token
  micro-batch × 8 accum.
- **Determinism:** seeds 1337 + stage_id; bitwise 200-step repeat rule before
  any long run; config-hash gating on resume; fixed 512-batch probe set for
  all gate comparisons; atomic ≤30-min checkpoints. Full contract: candidate
  §9 — that section is normative for the trainer.
- **Recovery:** resume state = (step, epoch, shard_idx, offset) validated
  against the manifest; 1 smoke step asserting loss continuity within 1e-3;
  2 failed interrupt-resumes → switch Community → Secure pricing.

## 6. Experiments and acceptance

Acceptance is **not** a loss number. The primary outcomes:

1. **Headline metric:** measured global-KV bytes/token, FP8-vs-BF16 (Q1 gate),
   and the sparse-vs-full attention quality curve at 16K.
2. **Mechanism proof:** needle-in-haystack / associative-recall probes at 16K
   — sparse top-k vs a full-attention control at matched active params.
   Success bar: sparse ≥ dense − 3% quality at ~an order-of-magnitude KV
   reduction.
3. **Ablations (matched-token):** CED vs layer-local KV (C3), mode-layout
   sweep (C4), mHC on/off, Engram on/off (pre-agreed cut if Δloss ≤ 0.005),
   FP8 QAT delta (Q1).
4. **Effort-control frontier** from P3 (length-penalty GRPO groups on
   GSM8K/MBPP): accuracy vs response-length curve.
5. **Prefill/decode asymmetry:** measured prefill FLOPs/time vs decode
   KV-bytes at 4K and 16K.

Every comparison states its fixed budget (tokens, active params, or total
params) and uses the same tokenizer/corpus-split/loss definition.

## 7. Risks and decisions

| Risk | Decision in this design |
|---|---|
| Producer map misread from prose | Phase 0 pins it from the reference implementation before any model code |
| CED/CSA2 quality loss at 24 layers | C3/C4 gates with numeric Δloss bars; producer map and mode table are data, not code |
| MoE expert starvation | 16 wide experts (~3B routed tokens/expert); C1 gate asserts load balance |
| Engram useless at 1M entries | zero-init gate + pre-agreed cut — fidelity axis, not capability axis |
| FP8 KV destabilizes | Q1 is post-production; BF16 artifact ships regardless |
| $500 overrun | automatic abort criteria (candidate §11.2); every termination path ships a defined artifact |
| Mechanism instability mid-production | ladder fallback order pre-approved (mHC → Engram → RL → DSpark → FP8 → indexer layout → producer map) |

**Implementation boundary:** no code exists yet; when implementation starts,
the execution plan's Phase 0 tasks precede everything, including the
producer-map pinning from the upstream reference code.
