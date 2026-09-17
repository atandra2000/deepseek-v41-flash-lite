# Phase 3: training loop, gates, recovery

- `pretrain.py` — raw-PyTorch trainer: AdamW (β .9/.95, wd .1 matrices-only),
  cosine 3e-4→3e-5 with warmup, grad-clip 1.0 with the >5-of-100-steps
  auto-fail, nonfinite guard (rollback → skip 2 batches → halve LR, ≤3
  retries), aux-loss-free router-bias updates, and atomic checkpoints
  (tmp→fsync→rename, ≤30 min) embedding model/optimizer/RNG/data
  position/config hashes + determinism flags. Resume = manifest check +
  non-destructive smoke step asserting loss continuity ≤1e-3.
- `ladder.py` — C0–C6 numeric gate evaluation (candidate §8), exactly-512
  sha-pinned probe batches, 200-step bitwise-repeat check, and a
  callback-based runner with pre-approved fallback order and JSONL evidence
  (`runs/ladder.jsonl`). A stage callback that *raises* gets exactly one
  bounded retry (the plan's "one retry each"); a second consecutive exception
  aborts the ladder for local diagnosis (stop-condition: "fails twice →
  environment bug"). Failed gate *decisions* — not exceptions — walk the
  pre-approved fallback variants. Stage callbacks must construct the named
  architecture variant (dense C0, +MoE, +SWA, +CED, +CSA2, +mHC, +Engram);
  no variant is claimed on the full 24-layer model until those toggles exist.

Task 11 verification status: CPU tests green (exact resume ==
uninterrupted weights+optimizer+RNG+data position, with activation
checkpointing on and off, on the real tiny CED model and the manifest-bound
PackedDataset); guard/rollback/clip/retention and gate-boundary tests green.
**Task 11 is not fully complete:** architecture variants and matched-token
control-run adapters remain development work, not a hardware check. The
candidate's old 16L dense baseline must not silently become the 24L v3 model.
Pending hardware (Task 12): 200-step bitwise repeat on A100 and measured gates.
Production corpus prep is Task 10 closeout.

The trainer accepts already-shifted `PackedDataset` labels, accumulates summed
cross-entropy normalized by supervised-token count, and rejects image batches
in text-only gate mode. It stops at dataset exhaustion rather than silently
repeating the corpus; per-epoch reshuffling is not implemented. Batch/accumulation
sizes are explicit: the design's stage-2 8K-token microbatch cannot contain a
16K sequence, so no production batch configuration is guessed here.

Activation checkpointing wraps the whole forward with a fresh runtime on replay;
the existing per-block shared-CED replay is not used. This is correctness-tested,
not a claim that the original per-block memory target is met. CUDA BF16,
`torch.compile`, multi-GPU operation, and production memory are not validated.

- `runbook.py` — CPU-tested CLI wiring for the A100 session (Tasks 12/13):
  `pin` records the determinism environment (CUBLAS workspace, deterministic
  algorithms, TF32 off) and refuses to run unpinned; `pin-probes` pins the
  exactly-512 sha-pinned probe batches from a corpus val split; `repeat` runs
  the 200-step bitwise-repeat driver and records the pin it ran under;
  `ladder` executes C0–C6 through the canonical GateRunner + LadderRunner with
  re-derived approvals (drift from the recorded approvals aborts before GPU
  hours are spent). Fixture mode (`--fixture`) waives corpus size only, never
  integrity. The GPU execution itself is Task 12; nothing here claims it.

Local checks (no downloads):

```bash
env -u PYTHONPATH .venv/bin/python -m pytest tests/test_training.py tests/test_ladder.py tests/test_runbook.py -q
env -u PYTHONPATH .venv/bin/python -m training.pretrain --help
env -u PYTHONPATH .venv/bin/python -m training.runbook --help
```

A100 session order (pending, Task 12; every step logged to runs/):

1. `runbook pin --out runs/env.json` (both repeat sessions must match it)
2. `runbook pin-probes --data <corpus> --out runs/probes.json` (stage context)
3. `runbook repeat --data <corpus> --model-config configs/lite-v3.json
   --training-config <json> --checkpoint-dir runs/repeat-a --out runs/repeat-a.json
   --env runs/env.json --device cuda` — twice, compare `loss_sha256`
4. `runbook ladder --data <corpus> --probes runs/probes.json
   --approvals runs/approvals.json --output runs/ladder.jsonl --device cuda
   --batch-size <explicit> --accumulation-steps <explicit>` (batch/accum
   sizes are decided at run time from stage-2 packing; not guessed here)

For an offline corpus, the module CLI requires model JSON, TrainingConfig JSON,
manifest directory, and checkpoint directory; `--fixture` waives corpus-size
requirements only. Never use it to claim production preflight passed.