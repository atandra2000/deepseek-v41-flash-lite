"""CSA2 mode table and candidate pool (Task 5 implements selection; pure helpers here).

Contract: docs/architecture-contract.md T2/T3. Modes are a per-layer data
table derived from the config maps; the pool is a bool position mask built by
the candidate source and consumed by pool-users.
"""

import torch


def layer_mode(cfg, layer_id: int) -> str:
    """Mode of one layer, per the contract's code-derived roles:
    'swa' (ratio 0), 'full' (candidate source), 'reindex' (index source, pool
    user), 'reuse' (consumes the previous index source's selection)."""
    if cfg.compress_ratios[layer_id] == 0:
        return "swa"
    if layer_id == cfg.candidate_source_layer:
        return "full"
    if cfg.is_index_source(layer_id):
        return "reindex"
    return "reuse"


def select_candidate_blocks(logits: torch.Tensor, compress_lens, topk_blocks: int, block_size: int) -> torch.Tensor:
    """Level-1 top-k over position blocks (contract T3, upstream model.py:583-610).

    logits: [..., n_positions] with unreachable positions at -inf.
    Returns a bool mask shaped like logits."""
    width = logits.size(-1)
    scores = torch.nn.functional.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)

    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)

    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]
