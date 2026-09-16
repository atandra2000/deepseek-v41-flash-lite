#!/usr/bin/env python
"""Analytical ledger (Task 2): instantiates the module tree at lite-v3 dims and
asserts the design's parameter/memory contract.

Design §3: total ~1.16B ±5%; active ~205M ±5% (excl. embedding, upstream
convention); AdamW fp32 moments + fp32 master must fit the 4x80GB envelope.
Fails loudly on drift (nonzero exit + printed diff).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from models.config import load_config
from models.transformer import Transformer

# design §3 targets (band ±5%). ACTIVE is the Phase-0 re-derived figure
# (217.7M): the design's 205M estimate omitted wo_b (d x o_groups*o_lora =
# 1.05M/layer) from its attention/layer arithmetic. See
# docs/architecture-contract.md "Task-2 ledger assertions".
TOTAL_TARGET = 1.16e9
ACTIVE_TARGET = 217.7e6
BAND = 0.05


def count(module) -> int:
    """Parameter count of an nn.Module, a single Parameter, or a list of them."""
    params = module.parameters() if isinstance(module, torch.nn.Module) else module
    return sum(p.numel() for p in params)


def main() -> int:
    cfg = load_config()
    torch.manual_seed(1337)
    model = Transformer(cfg)

    # ---- categorical breakdown (the ledger, not just a sum) ----
    cats: dict[str, int] = {
        "embedding": count(model.embed),
        "attention": 0,
        "compressor": 0,
        "indexer": 0,
        "moe_routed": 0,
        "moe_shared": 0,
        "moe_gate": 0,
        "mhc": 0,
        "engram": 0,
        "norms": 0,
        "vision": 0,
        "dspark": 0,
        "head_untied": 0,
    }
    for blk in model.blocks:
        cats["attention"] += count(blk.attn)
        cats["compressor"] += count(blk.compressor) if blk.compressor else 0
        cats["indexer"] += count(blk.indexer) if blk.indexer else 0
        cats["moe_routed"] += count(blk.ffn.experts)
        cats["moe_shared"] += count(blk.ffn.shared_experts)
        cats["moe_gate"] += count(blk.ffn.gate)
        cats["mhc"] += count(blk.hc_attn) + count(blk.hc_ffn)
        cats["engram"] += count(blk.engram) if blk.engram else 0
        cats["norms"] += count(blk.attn_norm) + count(blk.ffn_norm)
    cats["norms"] += count(model.norm)  # final norm
    if model.vision is not None:
        cats["vision"] = count(model.vision) + count(model.aligner) + count(model.image_start) + count(
            model.image_end
        ) + count(model.image_newline)
    if model.dspark is not None:
        cats["dspark"] = count(model.dspark)
    if model.head is not None:
        cats["head_untied"] = count(model.head)

    total = sum(cats.values())
    model_total = count(model)
    assert total == model_total, f"categorization dropped params: {total} != {model_total}"

    # ---- active (excl. embedding; backbone per-token path, upstream convention) ----
    active = (
        cats["attention"]
        + cats["compressor"]
        + cats["indexer"]
        + cats["moe_routed"] * cfg.n_activated_experts / cfg.n_routed_experts
        + cats["moe_shared"]
        + cats["moe_gate"]
        + cats["mhc"]
        + cats["norms"]
    )
    # ViT: runs on image tokens only -> reported, not in the text-path active number.
    # DSpark: frozen-backbone drafter, runs only during drafting -> reported, not counted.
    # Engram tables: hash lookups (memory, not FLOPs-bearing compute) -> reported.

    # ---- optimizer-state bytes (AdamW: 2 fp32 moments; + fp32 master; bf16 grads) ----
    total_bytes_weights = total * 4  # fp32 master
    adamw_bytes = total * 4 * 2
    grads_bytes_bf16 = total * 2
    training_bytes = total_bytes_weights + adamw_bytes + grads_bytes_bf16

    # ---- asserts (fail loudly) ----
    failures = []
    if not (1 - BAND) * TOTAL_TARGET <= total <= (1 + BAND) * TOTAL_TARGET:
        failures.append(f"total params {total:,} outside 1.16B ±5% [{1.104e9:,.0f}..{1.216e9:,.0f}]")
    if not (1 - BAND) * ACTIVE_TARGET <= active <= (1 + BAND) * ACTIVE_TARGET:
        failures.append(
            f"active params {active:,.0f} outside 217.7M ±5% [{0.95 * ACTIVE_TARGET:,.0f}..{1.05 * ACTIVE_TARGET:,.0f}]"
        )
    if training_bytes > 16 * 2**30:
        failures.append(f"training footprint {training_bytes / 2**30:.1f} GiB exceeds 16 GiB envelope")

    print(f"config sha256: {cfg.sha256}")
    print(f"{'category':<14}{'params':>14}")
    for k, v in cats.items():
        print(f"{k:<14}{v:>14,}")
    print(f"{'TOTAL':<14}{total:>14,}")
    print(f"{'active(text)':<14}{active:>14,.0f}")
    print(
        f"optimizer: weights(fp32 master) {total_bytes_weights / 2**30:.2f} GiB + "
        f"AdamW moments {adamw_bytes / 2**30:.2f} GiB + grads(bf16) {grads_bytes_bf16 / 2**30:.2f} GiB "
        f"= {training_bytes / 2**30:.2f} GiB"
    )

    if failures:
        print("\nLEDGER FAILURES:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("ledger OK: total and active within design bands")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
