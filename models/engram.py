"""Engram: n-gram hash lookup, gated residual write (contract T5, Lite D5).

Lite deltas vs upstream: BF16 tables (not FP8), orders 2-3, 1M entries, and a
zero-init `gate_scale` so the module outputs exactly 0 at init (the upstream
gate is sigmoid-bounded, never exactly 0; the design requires exact-0 init).
"""

import torch
from torch import nn

from .layers import init_std_


class NgramHash:
    """Rolling-XOR n-gram hashing with per-layer multipliers and prime buckets
    (upstream engram.py). Pure functions; buffers owned by the Transformer."""

    DEAD = -1

    @staticmethod
    def compute_multipliers(layer_ids, max_ngram_size: int, compressed_vocab: int) -> torch.Tensor:
        import numpy as np

        max_long = np.iinfo(np.int64).max
        bound = max(1, (max_long // max(compressed_vocab, 1)) // 2)
        rows = []
        for layer_id in layer_ids:
            gen = np.random.default_rng(10007 * layer_id)
            vals = gen.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64)
            rows.append(torch.tensor(vals * 2 + 1))
        return torch.stack(rows)

    @staticmethod
    def find_next_prime(start: int, seen: set[int]) -> int:
        candidate = start + 1
        while any(candidate % p == 0 for p in range(2, int(candidate**0.5) + 1)) or candidate in seen:
            candidate += 1
        return candidate

    @classmethod
    def build_primes(cls, layer_ids, max_ngram_size: int, n_heads: int, hash_space: int):
        """[layer][n-gram size][head] disjoint prime bucket moduli (upstream EngramLayout)."""
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes, current = [], hash_space - 1
                for _ in range(n_heads):
                    current = cls.find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return tuple(primes)

    @classmethod
    def forward(cls, compressed_ids: torch.Tensor, positions: torch.Tensor, pad_id: int,
                multipliers: torch.Tensor, primes: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """compressed_ids: [B, max_seq] compressed ids (DEAD for spans);
        positions: [B, L] absolute positions -> hash ids [B, L, n_layers, n_hash_cols]."""
        batch = compressed_ids.size(0)
        seqlen = positions.size(1)
        max_ngram = multipliers.size(1)
        tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(max_ngram):
            src = compressed_ids.gather(1, (positions - shift).clamp_min(0))
            blocked = blocked | (positions < shift) | (src == cls.DEAD)
            tokens.append(torch.where(blocked, pad_id, src))
        tokens = torch.stack(tokens, dim=-1)  # [B, L, max_ngram]

        products = tokens.unsqueeze(2) * multipliers  # [B, L, n_layers, max_ngram]
        rolling, hashes = products[..., 0], []
        for i in range(1, max_ngram):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % primes[:, i - 1])  # broadcast over heads
        return torch.cat(hashes, dim=-1) + offsets


class Engram(nn.Module):
    def __init__(self, cfg, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.hash_index = cfg.engram_layer_ids.index(layer_id)
        self.hc_mult = cfg.hc_mult
        self.dim = cfg.d_model
        num_embeddings = cfg.engram_num_embeddings[self.hash_index]
        n_hash_cols = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads

        self.embed = nn.Embedding(num_embeddings, cfg.engram_head_dim)
        self.wkv = nn.Linear(n_hash_cols * cfg.engram_head_dim, self.dim * (self.hc_mult + 1), bias=False)
        # Lite zero-init gate: output = gate_scale * sigmoid_gate * value (exact 0 at init)
        self.gate_scale = nn.Parameter(torch.zeros(self.hc_mult, self.dim))
        self.eps = cfg.norm_eps
        self.q_weight = nn.Parameter(torch.ones(self.hc_mult, self.dim))
        self.k_weight = nn.Parameter(torch.ones(self.hc_mult, self.dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_std_(self.embed.weight)
        init_std_(self.wkv.weight)

    def forward(self, x: torch.Tensor, hash_ids: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B, L, hc, d]; hash_ids: [B, L, n_hash_cols]; token_mask: [B, L]."""
        raise NotImplementedError("Task 6: engram forward (gate zero-init asserted in tests)")
