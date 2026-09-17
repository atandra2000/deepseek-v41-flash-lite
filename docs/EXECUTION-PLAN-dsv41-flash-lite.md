# DS-V4.1-Flash-Lite — Execution Plan

> **Status:** implementation plan for [candidate 13](../../../llm-research/candidates/13-deepseek-v41-flash.md)
> v3 and the [design specification](DESIGN-dsv41-flash-lite.md). Phase 0, 1, 2
> complete (Tasks 1–9 green). Phase 3 (Data & Trainer): Task 10 code + fixture
> tests green — production corpus prep is blocked on external sources (full
> 50,257-entry GPT-2 tokenizer.json + FineWeb-Edu/DataComp); Task 11 complete:
> trainer/recovery, GateRunner, executable variants and matched-token controls
> are CPU-verified.
> A100 200-step bitwise repeat and C0–C6 measured evidence remain for Task 12.
> Sequencing: DiffusionGemma-Lite / HiLS-Attention-Lite training debt and the
> Gemma-4-E2B decision come first (candidate §11.4).

## Execution rules

- Raw PyTorch only (no HF Trainer/Lightning); BF16 compute, FP32 master
  weights; FA2 + `torch.compile` + activation checkpointing.
- Every task ends in a runnable check; a task is done when its check passes,
  not when code exists.
- The candidate's §8 ladder, §9 determinism contract, §10 data plan and §11
  abort criteria are **normative** — this plan operationalizes them, it does
  not override them.
- No code is written before Task 1 pins the producer map and CSA2 mode table
  from the upstream reference implementation.
- Fallback ladder order is pre-approved (design §7); a gate failure never
  triggers an improvised mid-run decision.

## Phase 0 — Establish the contract

### Task 1 — Source manifest and architecture pinning

**Targets:** `docs/source-manifest.md`, `docs/architecture-contract.md`.

- [x] Pin the upstream repo commit (`deepseek-ai/DeepSeek-V4.1-Flash`,
      `inference/` + `encoding/` reference code) and the `config.json` hash.
- [x] Extract from the reference code (not prose): decoder→`kv_source_layer_ids`
      consumer map, CSA2 mode-assignment table, candidate-pool construction
      order, indexer K projection path, Engram lookup/gating, DSpark draft
      + verify loop. Record each as a named data table with upstream line
      references.
- [x] Write the Lite producer map (enc layers 2 and 8; 12 enc / 12 dec;
      3 CSA2 groups per side) as config data, swappable per the C3/C4 gates.

**Verify:** a reviewer can trace every entry of `architecture-contract.md`
to a reference-code line or a config key; no entry says "per the report".

### Task 2 — Project skeleton and analytical ledger

**Targets:** `models/` `training/` `data/` `scripts/` `tests/`;
`scripts/ledger.py`; `configs/lite-v3.json`.

- [x] Config module loads `lite-v3.json` (§3 of the design) and embeds
      sha256(config) in every artifact.
- [x] `ledger.py` instantiates the module tree at real dims and asserts
      total 1.16B ±5%, active 205M ±5% (excl. embedding), and AdamW
      optimizer-state bytes; fails loudly on drift. *(Phase-0 outcome: the
      re-derived active figure is 217.7M — the design's 205M estimate omitted
      the grouped output projection `wo_b`; design doc and
      architecture-contract.md corrected, config unchanged.)*
- [x] Skeleton follows portfolio layout (DeepSeek-v3-Lite as the closest
      sibling); no training code yet.

**Verify:** `python scripts/ledger.py` passes; `pytest tests/ -q` green
(ledger test only).

## Phase 1 — Tiny end-to-end model (toy dims)

### Task 3 — Attention core

**Targets:** `models/attention.py`, `tests/test_attention.py`.

- [x] MLA-style low-rank q/kv (q_lora 256, o_lora 256, 4 groups) + RoPE;
      SWA-128 branch, layer-local KV.
- [x] Toy config (2L, d64) runs forward+backward; causality tested by
      perturbing future tokens; window boundary tests.

**Verify:** `pytest tests/test_attention.py -q` green.

### Task 4 — CED global-KV pathway

**Targets:** `models/ced.py`, `tests/test_ced.py`.

- [x] Encoder layers compute compressed global KV (m=2); decoder layers
      project from the producer map via layer-owned `W^KV_l`, `W^Z_l`.
- [x] Assert prefill never runs decoder full self-attention over the global
      stream (the O(N·L/2) contract).

**Verify:** toy e2e with CED on/off; shapes + causality + producer-map
swap test green.

### Task 5 — CSA2 modes, indexer and candidate pool

**Targets:** `models/csa2.py`, `models/indexer.py`, `tests/test_csa2.py`.

- [x] Full/Reindex/Reuse modes as a per-layer table; first Full-mode decoder
      layer builds the 1024×8 pool; Reindex layers select top-k 256 via the
      8h×64 indexer; Reuse layers consume the previous selection.
- [x] Cross-check selected-index overlap against the upstream reference
      implementation on a fixed small input (the Phase-0 oracle).

**Verify:** `pytest tests/test_csa2.py -q`; oracle comparison report
committed under `tests/golden/`.

### Task 6 — Full forward: MoE, mHC, Engram, head

**Targets:** `models/moe.py`, `models/mhc.py`, `models/engram.py`,
`models/transformer.py`, `tests/test_forward.py`.

- [x] MoE 16+1 top-2 with `noaux_tc` bias balancing, router bias zero-init;
      per-expert load counters exposed for the C1 gate.
- [x] mHC (mult 2, Sinkhorn 10) identity-init; Engram zero-init gate with
      gate-magnitude assert.
- [x] Tied 65,536 head; toy-dim end-to-end forward/backward over all modules;
      init-time invariants asserted (Engram output exactly 0, mHC ≈ identity).

**Verify:** `pytest tests/test_forward.py -q`; ledger.py still passes.

## Phase 2 — Generation, multimodal and drafter

### Task 7 — Generation cache and measured KV bytes

**Targets:** `models/cache.py`, `scripts/gen.py`, `tests/test_generation.py`.

- [x] Cache exposes global-pool (8192) + per-layer SWA (128) with
      `global_kv_bytes()` returning measured bytes/token (BF16 and FP8 paths).
      *(Accounting view over the live buffers — models/cache.py; real dims:
      1920 B/token BF16, 960 FP8.)*
- [x] Greedy generation with EOS/max-token bounds; incremental decode matches
      full-sequence forward on the same model (logit parity within BF16 tol).
      *(Exact parity (3.6e-7) with exhaustive selection; production top-k is
      tie-near-equal only — see tests/golden/generation-parity.md.)*

**Verify:** `pytest tests/test_generation.py -q`; parity report recorded
(`tests/golden/generation-parity.md`).

### Task 8 — ViT pathway and DSpark drafter

**Targets:** `models/vit.py`, `models/dspark.py`, `tests/test_vision_draft.py`.

- [x] ViT 12L d512: patch 14, pixel-unshuffle 9×, 2D-RoPE, projector →
      reserved-special image tokens; interleaved text+image forward parity
      with text-only path when no images present.
- [x] DSpark 2 blocks (dense, Markov 64, SWA-128, 5-position draft +
      confidence verification); backbone-freeze flag asserted (backbone
      params get no grads in DSpark mode).

**Verify:** `pytest tests/test_vision_draft.py -q`.

### Task 9 — Compile, checkpointing and memory profile

**Targets:** `scripts/profile_memory.py`.

- [x] `torch.compile` + activation checkpointing on real dims; measured
      peak memory per gpu-shard at 4K and 16K (<2 GB activations target).

**Verify:** profile report committed; ledger + all tests still green.

## Phase 3 — Data and trainer

### Task 10 — Data adapter and production preflight

**Targets:** `data/prepare_data.py`, `data/dataset.py`, `tests/test_data.py`.

- [x] Offline prep machinery: FineWeb-Edu 20B+2B held-out and ~2B image-text
      budgets enforced by preflight; per-shard sha256 manifests,
      document-disjoint content-hash split, GPT-2 BPE + 128 specials, padded
      vocab 65,536 (`data/prepare_data.py`). *(Production corpus itself not
      yet prepared — requires the full 50,257-entry GPT-2 tokenizer.json and
      local FineWeb-Edu/DataComp sources; the strict loader rejects partial
      vocabularies.)*

  **Blocked on external sources (Task 10 closeout, cannot be CPU-completed):**
  the production corpus cannot be prepared in this environment. Required but
  unavailable: the full 50,257-entry GPT-2 `tokenizer.json` (vendored
  `tokenizers` cannot fabricate merges) and the FineWeb-Edu / DataComp
  downloads themselves. The loader is intentionally strict — it rejects
  partial vocabularies and padded manifests — so no fixture or partial token
  count can stand in for production prep, and none will be fabricated. This
  closeout is therefore a docs commit: code + fixture tests stay green, and
  the blocked-on-sources status above stands until the sources are provided.
- [x] Stage-2 packing with ≥30% documents ≥16K tokens (asserted by the
      loader and preflight); resumable deterministic ordering
      (`data/dataset.py:PackedDataset`); document-disjointness test.
- [x] Production preflight validates manifests/checksums/token bounds before
      any rental hour is spent. *(Ran on fixtures only; full-manifest pass
      pending real sources.)*

**Verify:** `pytest tests/test_data.py -q` green (7 CPU tests; prep, packing,
exact resume, corruption/token-bound rejection, split disjointness, stage-2
fraction, image spans). Full-manifest preflight and host-side prep time:
pending production sources.

### Task 11 — Training loop, determinism and recovery

**Targets:** `training/pretrain.py`, `training/ladder.py`,
`tests/test_training.py`.

- [x] AdamW per design §5; clip 1.0 with auto-fail (>5% of last 100 steps);
      nonfinite guard with checkpoint rollback (max 3).
- [x] Atomic checkpoints (tmp→fsync→rename, ≤30 min) embedding
      model/optimizer/RNG/data-position/config-hash; resume validated
      against the manifest; smoke step asserting loss continuity ≤1e-3.
      *(Verified on CPU: resume == uninterrupted weights+optimizer+RNG+data
      position, activation checkpointing on and off, real CED toy model and
      manifest-bound PackedDataset.)*
- [x] End-to-end C0–C6 ladder: numeric evaluator, fixed probe set (512
      sha-pinned batches), 200-step repeat utility, GateRunner architecture
      variants and matched-token controls are CPU-verified with real training.
      All seven default gates pass at d8 toy dims with balanced +1-shift
      batches; clip guard and numeric bars are unchanged. The three-layer
      fixture cannot instantiate production C4 layout2/layout3 tables and
      explicitly excludes that sweep (fallback orchestration tested separately).
      Only A100 bitwise repeat and measured production C0–C6 evidence remain
      for Task 12; CPU results are not production-quality evidence.

**Verify:** CPU deterministic-resume test (resume == uninterrupted weights
+ optimizer + data position) — green. Two consecutive 200-step GPU runs
bitwise identical on the first A100 session — pending (Task 12).

## Phase 4 — A100 evidence

### Task 12 — Hardware boundary and ladder gates

- [ ] 4×A100 bring-up: env pinning, 200-step bitwise-repeat, C0 smoke.
- [ ] Run gates C0–C6 (≤6 node-hours incl. one retry each); each gate's
      numeric result recorded to `runs/ladder.jsonl`; fallbacks applied
      strictly in the pre-approved order on failure.

**Verify:** `runs/ladder.jsonl` shows a pass (or pre-approved fallback) for
every stage C0–C6; total hours logged.

### Task 13 — Production run and post-training

- [ ] C8: 20–24B tokens, stage 1 (4K, ~70%) → stage 2 (16K, ≥30% long docs);
      resume-on-interrupt; abort criteria live in the runner (60% budget,
      loss >1.1× EMA over 500 steps).
- [ ] Q1 FP8-KV QAT gate + 500-step finetune (BF16 artifact ships on fail).
- [ ] P2 DSpark (frozen backbone), P3 SFT + effort-conditioned GRPO groups.

**Verify:** checkpoint chain complete with config hashes; stage-2 long-doc
recall probe run; all hour logs written.

### Task 14 — Evaluation and delivery

**Targets:** `scripts/eval_headline.py`, `results/`.

- [ ] Headline metrics: measured global-KV bytes/token FP8-vs-BF16;
      sparse-vs-full recall probes at 16K (success bar: ≥ dense − 3% at
      ~10× KV reduction); prefill/decode asymmetry at 4K and 16K.
- [ ] Ablations: CED/mHC/Engram on-off (matched tokens), mode-layout sweep,
      FP8 delta; effort-control frontier from P3.
- [ ] Final report: measured results vs every prediction in the candidate
      doc; every deviation from the design listed with its gate evidence.

**Verify:** `results/` contains the metric bundle + report; README updated.

## Dependency summary and stop conditions

- Task order: 1 → 2 → (3 → 4 → 5 → 6) → (7, 8 → 9) → (10 → 11) → 12 → 13 → 14.
  Task 1 blocks all code. Tasks 7/8 are independent of 3–6 except for the
  model root (6).
- **Stop conditions:** C0 fails twice → environment bug, stop and diagnose
  locally. Any three consecutive gate failures → halt ladder, owner decision.
  Budget: abort per candidate §11.2 (60% without production start; hard stop
  at 90 node-hours). Every stop ships the artifact defined in candidate
  §11.3.
- Prerequisite sequencing (owner-level): DiffusionGemma-Lite /
  HiLS-Attention-Lite training debt, then the Gemma-vs-this decision, precede
  Task 12's rental.
