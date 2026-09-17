# DS-V4.1-Flash-Lite

From-scratch PyTorch replica of the DeepSeek-V4.1-Flash architecture — CED +
CSA2 + hierarchical sparse indexer + SWA + mHC + Engram + MoE + DSpark + ViT —
re-sized to ~1.16B total / ~205M active parameters, targeting 4×A100 80GB for
≤ $500.

**Visual guide**

[Open the interactive architecture guide](docs/diagrams/deepseek_v41_visual_guide.html) for the model, CED/CSA2 attention, training workflow, and data pipeline, plus layer and selection-budget explorers. Download/open the HTML locally for interactive viewing. [Validation receipts and reproduction](docs/diagrams/RECEIPTS.md).

**Normative documents**

- [Design specification](docs/DESIGN-dsv41-flash-lite.md) — architecture contract, Lite sizing, deviations.
- [Execution plan](docs/EXECUTION-PLAN-dsv41-flash-lite.md) — task order, gates, stop conditions.
- [Source manifest](docs/source-manifest.md) — pinned upstream commit and file hashes.
- [Architecture contract](docs/architecture-contract.md) — producer map, CSA2 mode table, per-mechanism semantics, pinned from the upstream reference code (Task 1).

**Upstream reference** (vendored under `upstream/`, MIT): `deepseek-ai/DeepSeek-V4.1-Flash` @ `dba1be0a40aa45a94ad051997016db3960a90277`.

**Layout**

```
upstream/    vendored DeepSeek-V4.1-Flash reference (inference/ + encoding/ + config.json)
docs/        design, execution plan, architecture contract, source manifest
configs/     lite-v3.json and other configs
models/      model code (attention, ced, csa2, moe, mhc, engram, vit, dspark, transformer)
training/    trainer, gates/ladder (Phase 3+)
data/        offline data prep and loaders (Phase 3)
scripts/     ledger, profiling, generation, eval
tests/       pytest suite
```

**Dev environment** (macOS: CPU/MPS; A100 rental is Phase 4):

```bash
uv venv .venv --python 3.12 && uv pip install -r requirements.txt
pytest tests/ -q
```
