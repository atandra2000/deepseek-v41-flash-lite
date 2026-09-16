# Generation parity + measured KV bytes (Task 7)

Incremental decode vs full-sequence forward, same seeded model (toy dims:
2L d64, window 8, prompt 10 + 8 teacher-forced steps; `tests/test_generation.py`).

## Logit parity (exhaustive global selection)

With `index_topk` >= every compressed length, both paths attend the identical
window+global set, so logits must match to fp noise:

- swa-toy: max abs diff 3.6e-7 over all 8 continuation positions
- ced-toy: max abs diff 3.6e-7

This pins the decode machinery end to end: window ring wrap, RoPE positions,
compressor partial-group state, decoder `W^Z/W^KV` projection from producer
hiddens, indexer k-cache, ring seeding at prefill.

## Greedy token parity (production top-k)

Real `index_topk=4`: greedy tokens from incremental decode equal greedy tokens
re-derived from repeated full forwards (both toys, 8 steps). Logits alone are
only tie-near-equal: relu-zeroed indexer scores tie exactly and `topk`'s pick
among tied groups is shape-dependent (batched prefill vs 1-row decode), so one
swapped group cascades ~1e-1 through the next layer's selection. Selection
semantics stay pinned by the Task-5 upstream oracle (`csa2-oracle.md`).

## Bugs the parity test caught (fixed)

- Prefill assigned the exact-size latent tensor as `compress_kv_cache` /
  `k_cache`; decode writes then sliced past the end and silently no-op'd.
  Both now allocate the full `max_seq_len//ratio` buffer at prefill
  (models/ced.py, models/indexer.py).

## Measured cache (real dims, 24L, config sha-embedded)

`scripts/gen.py --json`, context 8 tokens:

- global KV: **1920 B/token BF16**, **960 B/token FP8-E4M1** (storage cast, D6)
  = 128 enc (2 sources, m=2) + 1536 dec (12 project caches, m=1) + 256 index
  (2 ratio-2 owner k-caches + 1 ratio-1 candidate-source k-cache; layers 16/20
  are non-owner indexers reading the published keys, no extra storage)
- local SWA: 393,216 B BF16 total = 24 layers x 128 window x 64 x 2 B
- pool budget: 8192 positions (1024 blocks x 8), compressed length at 16K
  context fills it exactly (16384/2)
