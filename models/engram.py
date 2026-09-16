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
        self.gate_scale = nn.Parameter(torch.zeros(self.hc_mult))
        self.eps = cfg.norm_eps
        self.q_weight = nn.Parameter(torch.ones(self.hc_mult, self.dim))
        self.k_weight = nn.Parameter(torch.ones(self.hc_mult, self.dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_std_(self.embed.weight)
        init_std_(self.wkv.weight)

    def forward(self, x: torch.Tensor, hash_ids: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B, L, hc, d]; hash_ids: [B, L, n_hash_cols]; token_mask: [B, L]
        (False shuts the gate). Upstream model.py:350-365 + Lite zero-init
        gate_scale (exact 0 output at init)."""
        values = self.embed(hash_ids)  # [B,L,C,head_dim]
        kv = self.wkv(values.flatten(-2))
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        weight = self.q_weight.float() * self.k_weight.float()  # only ever used as a product
        h = x.float()
        # normalized per (token, hc copy) over d, NOT jointly over the copies
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (h * weight * key).sum(-1) * rstd * self.dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        gate = gate * self.gate_scale  # Lite: zero-init, exact-0 output at step 0
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


class NgramHashState(nn.Module):
    """Owns the hash buffers (compressed token map, per-layer multipliers,
    prime buckets, ring cache) and produces hash ids for the Engram lookups."""

    def __init__(self, cfg, max_seq_len: int, token_map: torch.Tensor | None = None):
        super().__init__()
        self.cfg = cfg
        self.max_ngram = cfg.engram_max_ngram_size
        self.pad_id = cfg.engram_pad_token_id
        compressed_vocab = cfg.engram_compressed_vocab_size or cfg.vocab_size
        multipliers = NgramHash.compute_multipliers(cfg.engram_layer_ids, self.max_ngram, compressed_vocab)
        primes = NgramHash.build_primes(cfg.engram_layer_ids, self.max_ngram, cfg.engram_n_heads,
                                        cfg.engram_vocab_size)
        offsets = torch.cumsum(
            torch.tensor(
                [
                    [0] + [p for per_ngram in layer for p in per_ngram][:-1]
                    for layer in primes
                ],
                dtype=torch.int64,
            ),
            dim=1,
        )
        if token_map is None:
            token_map = torch.arange(cfg.vocab_size, dtype=torch.int64)  # identity (Lite D4 default)
        self.register_buffer("multipliers", multipliers, persistent=False)
        self.register_buffer("primes", torch.tensor(primes, dtype=torch.int64), persistent=False)
        self.register_buffer("offsets", offsets, persistent=False)
        self.register_buffer("token_map", token_map, persistent=False)
        self.cache: torch.Tensor | None = None
        # table rows must cover every bucket (upstream: num_embeddings >= sum of primes)
        total = int(self.offsets[:, -1].max()) + int(self.primes[:, -1, -1].max())
        for layer_i, layer_id in enumerate(cfg.engram_layer_ids):
            assert cfg.engram_num_embeddings[layer_i] >= total, (
                f"engram table {layer_i} ({cfg.engram_num_embeddings[layer_i]:,}) too small for "
                f"hash space ({total:,}); raise engram_num_embeddings or lower engram_vocab_size"
            )

    def forward(self, input_ids: torch.Tensor, start_pos: int, token_mask: torch.Tensor | None) -> torch.Tensor:
        """input_ids [B,L] -> hash ids [B, L, n_engram_layers, n_hash_cols]."""
        compressed = self.token_map[input_ids]
        if token_mask is not None:
            compressed = torch.where(token_mask, compressed, torch.tensor(NgramHash.DEAD, dtype=compressed.dtype))
        if self.cache is None:
            self.cache = torch.zeros(input_ids.size(0), self.cfg.context_train_stage2, dtype=torch.int64,
                                     device=input_ids.device)
        self.cache[: input_ids.size(0), start_pos : start_pos + input_ids.size(1)] = compressed
        positions = torch.arange(start_pos, start_pos + input_ids.size(1), device=input_ids.device)
        positions = positions.unsqueeze(0).expand(input_ids.size(0), -1)
        # gather lookbacks from the full cache (decode chunks are 1 token wide)
        return NgramHash.forward(self.cache[: input_ids.size(0)], positions, self.pad_id,
                                 self.multipliers, self.primes, self.offsets)
