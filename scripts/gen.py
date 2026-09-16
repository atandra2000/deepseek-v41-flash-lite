#!/usr/bin/env python
"""Greedy generation over the decode path (Task 7).

Prefills the prompt, then decodes one token at a time through the same
`Transformer.forward` the tests exercise. EOS / max-token are the only stops.
Random-init model unless --checkpoint is given (none exists before Task 12).
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.cache import GenerationCache
from models.config import load_config
from models.transformer import Transformer


@torch.no_grad()
def generate(model, prompt_ids: torch.Tensor, max_new_tokens: int, eos_id: int | None = None) -> tuple[torch.Tensor, dict]:
    """Greedy decode. Returns (all_ids [B, P+T], cache report dict)."""
    bsz, prompt_len = prompt_ids.shape
    cache = GenerationCache(model)
    logits, _ = model(prompt_ids)
    next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)
    ids = torch.cat([prompt_ids, next_tok], dim=1)
    pos = prompt_len
    cache.context_len = pos
    for _ in range(max_new_tokens - 1):
        if eos_id is not None and (next_tok == eos_id).all():
            break
        logits, _ = model(next_tok, start_pos=pos)
        pos += 1
        cache.context_len = pos
        next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_tok], dim=1)
    return ids, cache.report()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="config JSON (default configs/lite-v3.json)")
    ap.add_argument("--checkpoint", default=None, help="state_dict to load (optional)")
    ap.add_argument("--prompt", required=True, help="prompt text; encoded as raw byte ids (no BPE before Task 10)")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--eos-id", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", action="store_true", help="print the cache report as JSON")
    args = ap.parse_args()

    cfg = load_config(args.config and Path(args.config))
    model = Transformer(cfg).to(args.device).eval()
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True))
    # byte ids keep this runnable before the Task-10 BPE exists; ids must stay < vocab
    prompt_ids = torch.tensor([list(args.prompt.encode("utf-8")[: cfg.context_train_stage2])], device=args.device)
    if prompt_ids.numel() == 0:
        ap.error("empty prompt")

    ids, report = generate(model, prompt_ids, args.max_new_tokens, args.eos_id)
    text = ids[0].tolist()
    new_text = bytes(t for t in text[len(prompt_ids[0]) :] if t < 256).decode("utf-8", errors="replace")
    print(new_text)
    if args.json:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
