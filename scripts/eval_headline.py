#!/usr/bin/env python
"""Task 14 headline metrics, CPU-measured at the dims given (default: toy).

Three measurements, all real (no analytical substitutes):

1. global-KV bytes/token — `GenerationCache.global_kv_bytes` over live decode
   buffers after a prefill + decode, at BF16 and FP8 storage widths (the Q1
   ablation is the same tensors measured at 1 B/element).
2. sparse-vs-full recall — the same weights, two selection widths: a control
   config whose `index_topk`/candidate pool covers every reachable compressed
   group is compared against the sparse selection (per-index-source overlap
   and end-to-end logit divergence). Same seed -> identical weights, so any
   difference is the selection, not initialization.
3. prefill/decode asymmetry — wall-clock per-token cost of one full-context
   forward vs single-token decode steps through the same forward.

Production 16K/A100 execution is pending hardware; running this at production
dims on CPU is possible but slow, and never stands in for that evidence.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.cache import GenerationCache
from models.transformer import Transformer
from models.variants import NAMED, derive_config


def _full_control_overrides(cfg) -> dict:
    """Config scalars that widen the sparse selection to cover the stream.

    `index_topk`/`candidate_*` are data-free scalars (no parameter shapes),
    so the control is the same architecture with exhaustive selection.
    """
    ratio = min(r for r in cfg.compress_ratios if r) if any(cfg.compress_ratios) else 1
    reach = cfg.context_train_stage2 // ratio
    block = max(1, reach)
    return {"index_topk": reach, "candidate_block_size": block,
            "candidate_topk_blocks": max(1, -(-cfg.context_train_stage2 // block))}


def _index_overlap(selected: torch.Tensor, full: torch.Tensor) -> tuple[float, int]:
    """Fraction of the control's valid selections also chosen sparsely."""
    valid_s, valid_f = selected >= 0, full >= 0
    total = int(valid_f.sum())
    if total == 0:
        return 1.0, 0
    match = (selected.unsqueeze(-1) == full.unsqueeze(-2)) & valid_s.unsqueeze(-1) & valid_f.unsqueeze(-2)
    covered = int((match.any(-2) & valid_f).sum())
    return covered / total, total


@torch.no_grad()
def measure_global_kv(model, ids, decode_steps=4) -> dict:
    """Prefill + decode, then the GenerationCache accounting at both widths."""
    model.eval()
    model(ids)
    for pos in range(ids.size(1), ids.size(1) + decode_steps):
        model(ids[:, :1], start_pos=pos)
    cache = GenerationCache(model, context_len=ids.size(1) + decode_steps)
    return cache.report()

@torch.no_grad()
def decode_asymmetry(model, seq, steps, device="cpu") -> dict:
    """Wall-clock prefill (one S-token forward) vs `steps` single-token decodes."""
    ids = torch.arange(seq, device=device).remainder(model.cfg.vocab_size).unsqueeze(0)
    model.eval()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    model(ids)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    prefill = time.perf_counter() - t0
    token = ids[:, :1]
    t0 = time.perf_counter()
    for pos in range(seq, seq + steps):
        model(token, start_pos=pos)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    decode = time.perf_counter() - t0
    return {"prefill_tokens": seq, "prefill_s": round(prefill, 6),
            "prefill_ms_per_token": round(prefill / seq * 1000, 4),
            "decode_steps": steps, "decode_s": round(decode, 6),
            "decode_ms_per_token": round(decode / steps * 1000, 4),
            "asymmetry_ratio": round((decode / steps) / max(prefill / seq, 1e-12), 3)}


@torch.no_grad()
def recall_probe(cfg, ids, spec=None) -> dict:
    """Sparse CSA2 selection vs the exhaustive control on identical weights."""
    spec = NAMED["csa2"] if spec is None else spec
    per_layer: dict[int, list] = {}

    def track(model):
        for m in model.modules():
            if type(m).__name__ == "Indexer":
                m.register_forward_hook(
                    lambda mod, args, output, lid=m.layer_id: per_layer.setdefault(lid, []).append(output.detach()))
        return model

    torch.manual_seed(0)
    sparse = track(Transformer(derive_config(cfg, spec), variant=spec))
    torch.manual_seed(0)
    full = track(Transformer(derive_config(cfg, spec, _full_control_overrides(cfg)), variant=spec))
    logits_s, _ = sparse(ids)
    logits_f, _ = full(ids)
    per_layer.clear()
    sparse(ids)
    sparse_selections = {lid: entries.pop() for lid, entries in per_layer.items()}
    per_layer.clear()
    full(ids)
    full_selections = {lid: entries.pop() for lid, entries in per_layer.items()}
    layers = {}
    for lid, sel in sparse_selections.items():
        if lid in full_selections:
            recall, covered = _index_overlap(sel, full_selections[lid])
            layers[f"layer{lid}"] = {"recall": round(recall, 6), "covered_of_full": covered,
                                     "sparse_k": int((sel >= 0).sum(-1).max()),
                                     "full_k": int((full_selections[lid] >= 0).sum(-1).max())}
    return {"layers": layers,
            "logit_mean_abs_diff": float((logits_s - logits_f).abs().mean()),
            "logit_max_abs_diff": float((logits_s - logits_f).abs().max()),
            "identical": bool(torch.equal(logits_s, logits_f))}


def run(cfg, seq=128, decode_steps=4, device="cpu") -> dict:
    model = Transformer(derive_config(cfg, NAMED["csa2"]), max_seq_len=seq + decode_steps,
                        variant=NAMED["csa2"]).to(device)
    ids = torch.arange(seq, device=device).remainder(model.cfg.vocab_size).unsqueeze(0)
    out = {"config_sha256": cfg.sha256, "seq": seq, "decode_steps": decode_steps, "device": device}
    out["global_kv"] = measure_global_kv(model, ids, decode_steps)
    out["decode_asymmetry"] = decode_asymmetry(model, seq, decode_steps, device)
    out["recall"] = recall_probe(cfg, ids)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="config JSON (default configs/lite-v3.json)")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--decode-steps", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/eval-headline.json")
    args = ap.parse_args()
    from models.config import load_config

    cfg = load_config(args.config and Path(args.config))
    results = run(cfg, args.seq, args.decode_steps, args.device)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
