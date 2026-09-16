"""Generation cache accounting (Task 7).

The decode-time buffers already live where they are consumed (Attention rings
and compress caches, Indexer k-caches — upstream keeps the same layout); this
module is the single accounting view over them, per design §4.2:

- global KV budget = pool capacity (8192) + top-k selected per reindex layer,
  measured from the live buffers and divided by context length -> the headline
  metric (measured global-KV bytes/token, FP8-vs-BF16, candidate §3/Q1).
- local SWA KV = window_size (128) per layer, layer-local rings.

Byte costs use the *production storage* element width (BF16 = 2 B, FP8-E4M1 =
1 B per element, deviation D6), not the host fp32 layout: buffer element
counts are real, the storage format is the thing being measured (Q1 casts the
same tensors to FP8). FP4 is not reported — Lite ships BF16, ablates FP8.
"""

import torch

from .config import LiteConfig

_DTYPE_BYTES = {"bf16": 2, "fp8": 1}


class GenerationCache:
    """Accounting view over a Transformer's decode buffers.

    `context_len` is maintained by the generation loop (total tokens the
    cache covers). Buffers are classified by owner: global KV = compress
    caches (kv sources + decoder projections) and indexer K caches; local =
    per-layer SWA rings. The candidate-pool mask is exposed separately — it is
    a bool mask over compressed positions, not KV.
    """

    def __init__(self, model, context_len: int = 0):
        self.cfg: LiteConfig = model.cfg
        self.model = model
        self.context_len = context_len

    # ---- buffer classification (re-derived on demand; decode rewrites them) ----

    def _active_entries(self, buffer: torch.Tensor, ratio: int) -> int:
        """Entries actually written for context_len (buffers are sized to max_seq_len;
        compress caches hold complete groups only — a trailing partial group sits in
        the Compressor's kv_state, not here)."""
        return min(buffer.size(1), self.context_len // ratio)  # floor: complete groups

    def _walk(self):
        """Yield (kind, buffer, ratio) for every live decode buffer."""
        for m in self.model.modules():
            buf = getattr(m, "compress_kv_cache", None)
            if buf is not None:
                yield "global", buf, m.compress_ratio
            buf = getattr(m, "k_cache", None)
            if buf is not None:
                yield "global", buf, m.compress_ratio  # indexer: owner-layer ratio
            buf = getattr(m, "window_kv_cache", None)
            if buf is not None:
                yield "local", buf, 0

    # ---- the headline metric ----

    def global_kv_bytes(self, dtype: str = "bf16") -> float:
        """Measured global-KV bytes/token for the live context.

        Sums the active region of every global buffer at the storage width of
        `dtype` ("bf16" shipped path, "fp8" the Q1 storage cast) and divides by
        context_len — the design §4.2 source of truth."""
        assert dtype in _DTYPE_BYTES, f"dtype must be one of {sorted(_DTYPE_BYTES)}"
        assert self.context_len > 0, "no context cached yet (run prefill first)"
        total = 0
        for kind, buf, ratio in self._walk():
            if kind != "global":
                continue
            total += self._active_entries(buf, ratio) * buf[0, 0].numel() * _DTYPE_BYTES[dtype]
        return total / self.context_len

    def local_kv_bytes(self, dtype: str = "bf16") -> float:
        """Total SWA-ring bytes: window_size/layer, independent of context length."""
        assert dtype in _DTYPE_BYTES, f"dtype must be one of {sorted(_DTYPE_BYTES)}"
        total = 0
        for kind, buf, _ in self._walk():
            if kind != "local":
                continue
            total += self.cfg.window_size * buf[0, 0].numel() * _DTYPE_BYTES[dtype]
        return total

    # ---- the pool (budget view, not bytes) ----

    @property
    def pool_capacity(self) -> int:
        """Pool positions the budget reserves (1024 blocks x 8 = 8192)."""
        return self.cfg.pool_capacity

    def compress_len(self, layer_id: int) -> int:
        """Compressed-stream length a layer currently addresses (occupancy vs pool capacity)."""
        return self.context_len // self.cfg.compress_ratios[layer_id] if self.cfg.compress_ratios[layer_id] else 0

    def pool_mask(self) -> torch.Tensor | None:
        """Live candidate-pool bool mask (None when the config runs pool-less)."""
        return self.model.shared_attn.candidates

    def report(self) -> dict:
        """One dict for the run log: headline metric both ways + pool budget."""
        out = {"context_len": self.context_len}
        for dt in _DTYPE_BYTES:
            out[f"global_kv_bytes_per_token_{dt}"] = round(self.global_kv_bytes(dt), 3)
            out[f"local_kv_bytes_{dt}"] = self.local_kv_bytes(dt)
        out["pool_capacity"] = self.pool_capacity
        if self.cfg.compress_ratios and any(self.cfg.compress_ratios):
            src = next(l for l in range(self.cfg.n_layers) if self.cfg.compress_ratios[l])
            out["compress_len"] = self.compress_len(src)
        return out
