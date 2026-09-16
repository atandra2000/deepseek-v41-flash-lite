# Task 9: Memory profile and compile readiness

Config sha256: `01a76be6e850fcf02cb4dc3bc79d44268a34e3813694c665b71360dbf15d5f00`
Device: cpu (analytical estimates are device-independent)
Target: <2 GB activations per GPU shard (4-way split)

## Analytical activation estimates (1 sample/shard, bf16 compute)

| Seq len | Checkpointing | Per-block (MB) | Logits (MB) | Total (MB) | Per shard (MB) | Target met |
|---------|---------------|----------------|-------------|------------|----------------|------------|
|    4096 |            no |          452.0 |      1024.0 |     7532.8 |         1883.2 |          ✓ |
|    4096 |           yes |          452.0 |      1024.0 |     1960.0 |          490.0 |          ✓ |
|   16384 |            no |         1808.0 |      4096.0 |    30131.2 |         7532.8 |          ✗ |
|   16384 |           yes |         1808.0 |      4096.0 |     7840.0 |         1960.0 |          ✓ |

## Empirical (toy dims, CPU profiler)

| Label | Seq | Checkpoint | Peak alloc (MB) | Wall (s) |
|-------|-----|------------|-----------------|----------|
| toy-nocp | 32 | False | 5.2 | 0.67 |
| toy-ckpt | 32 | True | 6.4 | 0.05 |

## torch.compile

Graph trace (toy dims): **pass**

Note: `torch._inductor` emits a warning about complex operators (RoPE uses
`torch.view_as_complex`); performance may be suboptimal for those subgraphs.
Production A100 runs should benchmark eager vs compiled to measure actual speedup.

## Gradient parity

Checkpointed vs normal max gradient diff: **0.00e+00** (pass)

## In-place operation cleanup (torch.compile prerequisite)

`apply_rotary_emb` converted from in-place `x_.copy_()` to functional
(returns new tensor). All callers updated: `attention.py`, `ced.py`,
`indexer.py`, `dspark.py`. KV cache writes use `.detach()` to avoid
version-counter errors in the backward graph. 55/55 tests green.

## Conclusion

Activation memory targets met at both 4K and 16K with checkpointing enabled.
Compute is the binding constraint, not memory (design §3).
