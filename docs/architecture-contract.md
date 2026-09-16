# DS-V4.1-Flash-Lite — Architecture contract (Task 1)

> Status: pinned 2026-09-16 from the upstream reference code at commit
> `dba1be0a40aa45a94ad051997016db3960a90277` (see
> [source-manifest.md](source-manifest.md)). Every entry cites a reference-file
> line or a `config.json` key; no entry says "per the report". Line numbers
> refer to the vendored files under `upstream/`.
>
> Notation: `model.py:L` = `upstream/inference/model.py` line; `kernel.py:L` =
> `upstream/inference/kernel.py` line; `cfg:` = released root `config.json` key.

## T1 — KV-source consumer map (CED wiring)

**The upstream contract is not a per-layer assignment table — it is a
publish/reuse slot swap.** Each KV-source layer owns a compressor and its own
compressed-KV cache; while it runs it repoints the shared slot at its own cache
(`model.py:747-748`, `SharedAttentionRuntime` `model.py:1166-1180`). Layers that
run after it and before the next source therefore read *its* cache. The
effective consumer map:

| Upstream layers | Global KV read from | Evidence |
|---|---|---|
| 0–1 | none (`compress_ratios` 0 → pure SWA) | `cfg:compress_ratios`, `model.py:774-778` |
| 2–7 | layer 2's cache | `cfg:kv_source_layer_ids [2,8,14,20]`, slot swap `model.py:747-748` |
| 8–13 | layer 8's cache | 〃 |
| 14–19 | layer 14's cache | 〃 |
| 20–39 | layer 20's cache | 〃 |
| 40–42 (DSpark/MTP) | none (ratio 0, SWA only) | `cfg:compress_ratios[40:43] = 0` |

Additional pinned facts:

- Every `compress_ratio > 0` layer (encoder *and* decoder) attends the shared
  compressed stream — sources and consumers alike (`model.py:739-763`: non-source
  reads `shared_attn.compress_kv[:bsz, :compress_len]`; `compress_len =
  (start_pos+seqlen)//ratio` of **its own** ratio, `model.py:744`).
- A ratio-0 layer never touches the global stream (`model.py:775`).
- Each decoder layer owns `W^KV_l`/`W^Z_l`: upstream realizes this as every
  layer having its own `wkv`/`kv_norm` for the *window* path (`model.py:643-644`)
  and every **source** owning the compressor (`wkv`+`wgate`,
  `model.py:437-456`); consumers never re-project the shared cache
  (`model.py:763` reads it as-is). The projection is owned, never shared.
- Compressed latents are single `head_dim` vectors shared by all heads
  (`model.py:643`: `wkv: dim → head_dim`; `sparse_attn` kernel takes `kv
  [b,n,d]` broadcast across heads, `kernel.py:330,365-380`). Config:
  `num_key_value_heads: 1` is this fact's declarative shadow.
- Compressor = softmax-gated pooling over `compress_ratio` consecutive tokens,
  in fp32 for m=2 (`kv * score.softmax`), RMSNorm after pooling; m=1 is a plain
  projection with no gate (`model.py:458-485`).
- Latent RoPE: applied **after** the indexer read, at group-first positions
  j·m, using `compress_rope_theta` = 160000 with YaRN (original 65536, factor
  16, β 32/1) (`model.py:751-761`; `cfg:compress_rope_theta, rope_scaling`).
- Indexer needs the **pre-RoPE** latent, so the indexer runs inside
  `_compress_kv` before the RoPE/quantize write (`model.py:749-761`,
  `Compressor` docstring `model.py:434-435`).

**Lite producer map (config data, swappable for gate C3):**
`kv_source_layers = [2, 8]` (design §4.1); enc = layers 0–11 (m=2), dec = 12–23
(m=1). Effective Lite consumer map: layers 2–7 read 2, layers 8–23 read 8,
layers 0–1 SWA-only.

## T2 — CSA2 mode-assignment table

The code has three layer roles, derived in `Indexer.__init__` (`model.py:498-503`)
and used in `_compress_topk_idxs` / `Indexer.forward` (`model.py:722-737`,
`527-580`):

| Code role | Condition (upstream) | Behavior |
|---|---|---|
| index-source ("Reindex") | `layer_id in index_source_layers` | Runs its own indexer; publishes `shared_attn.topk_idxs` (`model.py:735-737`) |
| non-source ("Reuse") | otherwise, ratio > 0 | Consumes `shared_attn.topk_idxs` unchanged (`model.py:725-726`) |
| candidate-source ("Full") | `layer_id == candidate_source_layer` (=20) | Additionally builds the level-1 candidate pool (`model.py:569-572`) |
| pool-user | `candidate_source_layer < layer_id` | Masks its indexer scores to the pool (`model.py:573-575`, `uses_candidates`) |

Released per-layer table (derivable from `cfg`; ratio from
`cfg:compress_ratios`, sources from `cfg:index_source_layer_ids`):

| Layers | m | Indexer | Pool | Notes |
|---|---|---|---|---|
| 0–1 | 0 | — | — | SWA only |
| 2 | 2 | source, owns K | no | first ratio>0 layer |
| 3–7 | 2 | reuse | no | reuse 2's indices |
| 8 | 2 | source, owns K | no | 〃 |
| 9–13 | 2 | reuse | no | |
| 14 | 2 | source, owns K | no | |
| 15–19 | 2 | reuse | no | |
| 20 | 1 | source, owns K, **candidate source** | builds | first decoder layer |
| 21–23 | 1 | reuse | masked | |
| 24 | 1 | source, **no K** | masked | scores layer 20's index keys |
| 25–27 | 1 | reuse | masked | |
| 28 | 1 | source, no K | masked | |
| 29–31 | 1 | reuse | masked | |
| 32 | 1 | source, no K | masked | |
| 33–35 | 1 | reuse | masked | |
| 36 | 1 | source, no K | masked | |
| 37–39 | 1 | reuse | masked | |
| 40–42 | 0 | — | — | DSpark draft blocks (SWA only) |

Note: encoders 2/8/14 index the **entire** compressed prefix (no pool —
`uses_candidates` requires `candidate_source_layer < layer_id`,
`model.py:503`). The design doc's "Full/Reindex/Reuse" naming maps as: Full =
index-source + candidate-source (layer 20); Reindex = index-source + pool-masked
(24, 28, 32, 36); Reuse = non-source. The design's "3 groups per side" is a
Lite layout choice, swept in gate C4 — the layout is config data, not code.

**Lite mode table (config data, swappable for gate C4):**
`index_source_layers = [2, 8, 12, 16, 20]`, `candidate_source_layer = 12`.
Encoders 2, 8: index sources, own K, no pool. Decoder 12: candidate source
(builds 1024-block pool); 16, 20: reindex inside pool; all others reuse.

## T3 — Candidate-pool construction order

`select_candidate_blocks` (`model.py:583-610`), called from the candidate
source's indexer (`model.py:569-572`):

1. Compute per-(query, compressed-position) scores: `q·k` per head, **ReLU**,
   weighted by per-head weights `weights_proj(x)` scaled by
   `softmax_scale·n_heads^-0.5`, summed over heads (`model.py:555-557`).
2. Mask unreachable positions to −inf (`model.py:563-566`; a compressed
   position is visible once the query has passed its last token).
3. Pool over blocks of `candidate_block_size` (8): block score = **amax** of
   member positions (`model.py:598-599`); final partial block padded with −inf.
4. **Pin the newest block**: the block containing the query's latest reachable
   position gets +inf (always kept) (`model.py:602-605`).
5. top-`candidate_topk_blocks` (2048) by block score; −inf selections dropped
   (`model.py:607-609`).
6. Expand mask back to positions (`repeat_interleave`), truncate to width
   (`model.py:610`). Consumed as a plain bool mask — layers never see blocks.

Pool capacity: 2048 blocks × 8 = 16,384 positions (≈ the compressed length at
the 64K original context for m=2). **Lite:** 1024 × 8 = 8,192 (design §4.2).

## T4 — Indexer K-projection path

- Owner (KV-source) layers only: `k = k_norm(wk(latent))` where `wk: head_dim →
  index_head_dim` (bf16), `latent` is the pre-RoPE compressed latent; RoPE on
  the last `rope_head_dim` dims at group positions j·m; fp4-quantized; written
  to the owner's `k_cache [b, max_seq/ratio, index_head_dim]` and published via
  `shared_attn.index_k` (`model.py:537-548`).
- Index head dim: `cfg:index_head_dim 128`, `cfg:index_n_heads 32`,
  `cfg:index_topk 512`. **Lite:** 8 heads × 64, top-k 256 (design §3).
- Queries come from the q-Lora latent `qr` (the same `wq_a`/`q_norm` output as
  main attention), through the indexer's own `wq_b: q_lora_rank →
  index_n_heads·index_head_dim`; RoPE tail at real positions; fp4-quantized
  (`model.py:550-552`).
- Non-owner indexers read the last-published `shared_attn.index_k`
  (`model.py:554`). Upstream: decoders 24–36 all score **layer 20's** index
  keys, each with its own `wq_b`/`weights_proj`.
- Selection: top-k over compressed positions, re-sorted ascending, −1 for
  unreachable, then shifted by `offset` (= window length) so indices address
  the concatenated `[window, compressed]` kv axis (`model.py:578-580`,
  concat at `model.py:774-778`).
- Weight-scaling detail: `weights_proj` output is multiplied by
  `(head_dim^-0.5)·n_heads^-0.5` before ReLU (`model.py:555`).

## T5 — Engram lookup and gating

Hash construction (`engram.py:NgramHashState.forward`, 159-184):

1. Token ids → compressed id space (`token_map`; NFKC/NFD/strip-accents/
   lowercase collapse — `engram.py:17-61`; compressed vocab size asserted =
   `cfg:engram_compressed_vocab_size` 99092). Image-span positions → DEAD.
2. Look-back window of `max_ngram_size` (4) tokens; stops at sequence start
   and at DEAD, replaced by `pad_id` (`engram.py:169-175`).
3. Per-(layer, lookback) odd multipliers from RNG seed `10007·layer_id`
   (`engram.py:64-83`); rolling XOR over lookbacks yields the hash of each
   2-gram…max-gram (`engram.py:179-183`).
4. Bucket: `hash % prime` per (n-gram size, head); primes drawn in order from
   `engram_vocab_size` (16M hash space), ranges disjoint via cumsum offsets
   (`engram.py:103-126`, `engram.py:148-153`).

Lookup and gate (`model.py:Engram`, 328-365):

- Fetched rows (n_hash_cols = (max_ngram−1)·n_heads of `head_dim`) are
  flattened and projected by `wkv: n_hash_cols·head_dim → dim·(hc_mult+1)`,
  split into key (hc copies) + shared value (`model.py:344-353`).
- Gate = normalized dot product of stream vs key per (token, hc copy):
  `sigmoid(copysign(sqrt(|dot|.clamp_min(1e-6)), dot))` — signed sqrt
  (`model.py:356-362`).
- Output: `h + gate·value` per hc copy (`model.py:365`). Image-span positions
  gate to 0 (`token_mask`, `model.py:363-364`).
- Tables are FP8-stored upstream (`model.py:296-325`). **Lite: BF16 tables,
  1M entries, orders 2–3, zero-init gate (module outputs exactly 0 at init)**
  (design §4.3, deviation D5). Engram token-map compressed-vocab size is a
  Lite-derived value (our GPT-2 BPE), pinned at data-prep time.

## T6 — DSpark draft + verify loop

Forward path only in the reference (the training/speculative loop is out of
scope upstream — `model.py:129-131`):

- 3 draft blocks live under the `mtp.*` namespace; their embedding and LM head
  are **tied to the backbone's** (`model.py:1207-1213`; `convert.py:109-110`).
- Target layers `[37,38,39]`: the drafter reads the **attention input** of
  those backbone layers, hc-collapsed by mean over copies
  (`model.py:1264-1266`), concatenated then `main_proj` (dim·3 → dim) + norm in
  stage 0 (`model.py:1111-1115`).
- Draft prefill only seeds the SWA cache from `main_x` (`model.py:1122-1126`,
  `DSparkAttention.forward` 1032-1074); decode runs a block of
  `dspark_block_size` (5) tokens attending over `[window ring, own draft KV]`
  (`model.py:1064-1066`, `get_dspark_topk_idxs` 1020-1029).
- Head: backbone `head` applied to the drafted hidden (full logits) + per-
  position **Markov bias**: `markov_head(token_ids)` = embed→markov_rank,
  head→vocab logits, added into the main logits (`model.py:1137-1156`).
- Confidence: `proj([hidden, markov_embed]) → scalar` per draft position
  (`model.py:1089-1097`). Verification scheduling is consumer-side (not in the
  reference); Lite pins 5-position draft + confidence-scheduled verification
  (design §4.3). **Lite: 2 blocks, dense FFN, Markov rank 64, backbone frozen**
  (design §3).

## T7 — Attention core semantics

- `sparse_attn` (`kernel.py:310-403`): q `[b,m,h,d]` vs **shared** kv latents
  `[b,n,d]` (one K per position broadcast to all heads); gathers `topk_idxs`
  (−1 = skip); online softmax; **attn_sink** per head enters only the softmax
  denominator: `sum_exp += exp(sink[h] − max)` (`kernel.py:382-386`) — a
  learnable "attend to nothing" logit; rows with no valid index output 0.
- Output projection: inverse-RoPE on the rope tail, then **grouped low-rank**
  `wo_a` (block-diagonal over `o_groups`, einsum) → `wo_b` (`model.py:781-789`;
  `cfg:o_lora_rank 1024, o_groups 8`). **Lite:** o_lora 256, 4 groups.
- Q path: `wq_a: dim→q_lora_rank` → RMSNorm(q_lora) → `wq_b: q_lora→h·d` →
  RoPE tail (`model.py:770-772`; `cfg:q_lora_rank 1280`). **Lite:** 256.
- Window branch: per-layer ring cache of `window_size` (128) slots holding
  fp8-quantized post-RoPE KV latents; prefill attends its own chunk and seeds
  the ring; decode attends the ring (`model.py:700-720`). **Lite: window 128
  kept, layer-local KV always.**
- `attn_sink`: fp32 parameter, one scalar per head (`model.py:639`).
- `rope_head_dim`: 64 of head_dim 512 (12.5%; `cfg:qk_rope_head_dim`,
  `inference/config.json:22`). **Lite decision: rope_head_dim = 16 of head_dim
  64** (upstream fraction would give 8; we keep a stronger position signal at
  small head_dim; sweepable).
- Compressed and window KV are concatenated along one axis and attended in one
  `sparse_attn` call (`model.py:774-780`).

## T8 — mHC coefficient pipeline

- Residual stream = `hc_mult` parallel copies `[b,s,hc,d]`; expand at embed,
  collapse once at the end (`model.py:1253-1268`, collapse `model.py:1268`).
- `hc_mixes`: normalize the flattened stream per token, `F.linear(x, hc_fn)` →
  `hc_split_sinkhorn` (`model.py:948-955`). Params per block: `hc_{attn,ffn}_fn
  [(2+hc)·hc, hc·dim]`, `hc_{attn,ffn}_base [(2+hc)·hc]`,
  `hc_{attn,ffn}_scale [3]` (`model.py:938-946`).
- `hc_split_sinkhorn` (`kernel.py:406-462`): `pre = sigmoid(mix·s0 + base) +
  eps`; `post = 2·sigmoid(mix·s1 + base)`; `comb = softmax_rows(mix·s2 + base)
  + eps` then col-normalize + eps, repeated (iters−1) times → doubly
  stochastic.
- Coefficient chaining: a block's attention consumes the **previous block's
  FFN `pre`**; its FFN consumes its own attention's coefficients (`model.py:968-994`).
- First input is one-hot copy 0 (`make_identity_pre_mix`, `model.py:1159-1163`).
  **Lite: hc_mult 2, Sinkhorn 10 iters, identity-init so the mechanism is a
  no-op at step 0** (design §4.3). **Lite decision (identity init):** `hc_*_fn
  = 0`, `hc_*_base` such that pre = [1,0,…] (one-hot), post = 1, comb = I —
  exact spec in the module tests.

## T9 — MoE gate

- Scoring `sqrtsoftplus`: `softplus(scores).sqrt()` (`model.py:817`); `noaux_tc`:
  selection by `(scores + bias).topk`, **weights from raw gathered scores**
  (`model.py:822-823`); `norm_topk_prob` (÷ sum + 1e-20) then `routed_scaling
  1.5` (`model.py:824-826`; `cfg:routed_scaling_factor`). Gate temp 1.0.
- **Modality bias pairs**: `bias_vl` selected for image-span tokens
  (`model.py:818-820`) — text and image tokens route with separate bias
  vectors. Lite keeps the pair (image tokens exist in stage 1).
- Shared expert exactly 1 (`model.py:886-887`); SwiGLU with `swiglu_limit`
  clamps: up clamped both sides, gate clamped above (`model.py:845-848`;
  `cfg:swiglu_limit 10.0`).
- `e_score_correction_bias` (HF name) = router `bias` (`convert.py:114-115`).
  **Lite: 16 routed + 1 shared, top-2, d_ff 768, router bias zero-init**
  (design §4.3); per-expert load counters exposed for the C1 gate.

## T10 — ViT + aligner

- Patch pipeline (`image_processor.py:115-129`): RGB → normalized bf16
  (x−0.5)/0.5 → patches `[n_vit_h·n_vit_w, 3, 14, 14]`; the 3×3 pixel-unshuffle
  happens where patch (3,14,14) tensors are flattened for `PatchEmbed`
  (`vision.py:37-43`: Linear(3·14² → vision_dim) over `x.flatten(1)` — the
  unshuffle is a reshape, projection is a flat linear).
- Token-span layout: `[IMAGE_START] + ([IMAGE]·n_llm_w + [IMAGE_NEW_LINE])·
  n_llm_h + [IMAGE_END]`; every span position carries `image_token_id`; IMAGE
  slots filled with aligner rows in reading order (`image_processor.py:1-10,
  132-137`; `model.py:1228-1239`). Span delimiters = learned embeddings
  (`model.py:1219-1222`).
- ViT: full bidirectional SDPA attention with 2D-RoPE (h/w position pairs,
  `vision.py:8-15`), SwiGLU MLP, pre-norm blocks, final RMSNorm
  (`vision.py:74-103`).
- Aligner (`vision.py:106-119`): grid `n_h×n_w` → pad to multiple of
  `downsample_ratio` (3) → unfold → Linear(vision_dim·9 → dim) → GELU →
  Linear(dim → dim).
- Grid planning: token budget solver (`image_processor.py:39-67,101-112`;
  `cfg:vision_max_n_token 1024, min_pixels 295936`).
- **Lite: 12L d512, 8 heads (head_dim 64 kept), patch 14, ratio 3, joint AR
  training; SigLIP stage dropped** (design §4.3, D3). ViT MLP inter: upstream
  2816 = 2.75×d1024 → Lite 1408 (same ratio). **Lite decision:** reserved
  specials supply image token ids (D4).

## T11 — Quantization map (upstream)

| Tensor | Format | Block | Scale |
|---|---|---|---|
| Linear weights | fp8 e4m3 (fp4 for routed experts) | 32×32 (K/32 for fp4) | e8m0, `ue8m0` fmt (`kernel.py:12-29,98-124,127-204`) |
| Activations into GEMMs | fp8 e4m3 | 32 along K | e8m0 |
| Window KV cache | fp8 (act_quant inplace) | 32 | e8m0 (`model.py:707`) |
| Compressed global KV | **fp4 e2m1** | 16 | fp8-e4m3 scale, min 6·2⁻⁹ (`model.py:760`, `kernel.py:161-163`) |
| Indexer q/k | fp4 e2m1 | 32 | e8m0 (`model.py:546,552`) |
| Engram tables | fp8 e4m3 rows | 32 | e8m0 (`model.py:296-325`) |

**Lite (D6):** training keeps BF16 KV throughout; FP8-E4M1 main-KV is a
post-production QAT gate (Q1). Routed experts stay BF16 (no fp4 tensor cores
on A100). The upstream block/scale conventions remain the reference for the Q1
implementation.

## T12 — Lite config (normative for `configs/lite-v3.json`)

Derived from design §3/§4 and the tables above. **Lite-decision rows are
explicitly marked — they are design-doc choices, not upstream facts.**

| Key | Value | Source |
|---|---|---|
| n_layers | 24 (enc 0–11 m=2, dec 12–23 m=1) | design §3 |
| d_model | 1024 | design §3 |
| n_heads / head_dim | 16 / 64 | design §3 |
| q_lora_rank / o_lora_rank / o_groups | 256 / 256 / 4 | design §3 |
| rope_head_dim | 16 | **Lite decision** (T7) |
| rope_theta | 10000, no YaRN (stage 2 at 16K ≤ 65,536 original) | cfg + **Lite decision** |
| compress_rope_theta | 20000 | **Lite decision**: upstream = base × YaRN factor (10000×16, `cfg`); Lite has no extension so keep the ×m-position scaling visible: base × m_enc. Swept in C3 if needed. |
| window_size | 128 | cfg (kept) |
| kv_source_layers | [2, 8] | design §4.1 |
| index_source_layers | [2, 8, 12, 16, 20] | **Lite decision** (T2 mapping) |
| candidate_source_layer | 12 | **Lite decision** (T2 mapping) |
| candidate_topk_blocks / block | 1024 / 8 | design §3 |
| index_n_heads / index_head_dim / index_topk | 8 / 64 / 256 | design §3 |
| n_routed_experts / n_shared / topk / d_ff | 16 / 1 / 2 / 768 | design §3 |
| score_func / topk_method / routed_scaling | sqrtsoftplus / noaux_tc / 1.5 | cfg |
| swiglu_limit | 10.0 | cfg |
| hc_mult / sinkhorn iters / eps | 2 / 10 / 1e-6 | design §3 |
| engram_layer_ids | [1, 14] | cfg (kept) |
| engram_num_embeddings | [1_000_000, 1_000_000] | design §3 |
| engram_max_ngram_size | 3 (orders 2–3) | design §3 |
| engram_n_heads / head_dim | 8 / 16 | **Lite decision** (T5 sizing note) |
| engram_compressed_vocab_size | derived from our tokenizer at data prep | cfg mechanism + **Lite** (D4) |
| vocab_size | 65,536 tied (GPT-2 BPE + 128 specials) | design D4 |
| num_nextn_predict_layers / dspark blocks | 2 | design §3 |
| dspark_block_size / markov_rank | 5 / 64 | cfg kept / design §3 |
| dspark_target_layer_ids | [21, 22, 23] | **Lite decision** (last 3 backbone layers) |
| dspark_n_routed_experts | 0 (dense FFN) | design §3 |
| vision layers / dim / heads / inter | 12 / 512 / 8 / 1408 | design §3 + T10 |
| vision_patch_size / downsample_ratio | 14 / 3 | cfg kept |
| image_token_id | 65,408 (first reserved special) | **Lite decision** (D4) |
| norm_eps | 1e-20 | cfg |
| dtype (training) | bf16 compute, fp32 master | design §5 |

## T13 — Resolved §2.2 items

All four previously-unresolved items are pinned above from reference code:
decoder→producer consumption (T1 slot swap), CSA2 mode table (T2), candidate
pool order (T3), indexer K path (T4). The design doc's §2.2 concern —
"prose from the tech report is not sufficient" — is discharged: nothing in
this contract cites prose.

## Task-2 ledger assertions derived from this contract

The ledger (Task 2) instantiates the module tree at the T12 dims and asserts:

- total params ∈ 1.16B ±5% (all modules incl. ViT, DSpark, Engram);
- active params ∈ 205M ±5% **excl. embedding** (upstream convention;
  routed-expert contribution counted at top-k=2 + shared);
- AdamW state = 2 fp32 moments × total, + fp32 master weights; bytes reported,
  must fit the 4×80GB envelope with the design §3 margins.
