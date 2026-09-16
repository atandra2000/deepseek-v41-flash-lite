"""CED global-KV pathway (Task 4).

Contract: docs/architecture-contract.md T1 + design §4.1. Encoder KV sources
compress global KV with softmax-gated pooling (m=2, upstream-faithful); same-
ratio consumers read the source's published cache (upstream slot swap);
decoder layers (m=1 over an m=2 producer) project their own global KV from the
producer's final hidden state via layer-owned W^KV_l, W^Z_l (design-normative
Lite mechanism — decoders never share a projection and never re-read the
producer's cache).
"""

import torch
from torch import nn

from .layers import RMSNorm, apply_rotary_emb, init_std_


class Compressor(nn.Module):
    """Pools `compress_ratio` consecutive tokens into one KV latent with a
    learned softmax gate (upstream model.py:429-485). m=1: plain projection,
    no gate. fp32 pooling for m>1."""

    def __init__(self, cfg, layer_id: int):
        super().__init__()
        ratio = cfg.compress_ratios[layer_id]
        assert ratio > 0, "Compressor belongs to layers with compress_ratio > 0"
        self.compress_ratio = ratio
        self.head_dim = cfg.head_dim
        self.norm_eps = cfg.norm_eps
        self.norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        self.wkv = nn.Linear(cfg.d_model, cfg.head_dim, bias=False)
        self.wgate: nn.Linear | None = nn.Linear(cfg.d_model, cfg.head_dim, bias=False) if ratio > 1 else None
        # decode-time partial-group state (upstream kv_state/score_state)
        self.kv_state: torch.Tensor | None = None
        self.score_state: torch.Tensor | None = None
        self.reset_parameters(ratio)

    def reset_parameters(self, ratio: int) -> None:
        dtype = torch.float32 if ratio > 1 else torch.get_default_dtype()
        self.wkv.weight.data = init_std_(self.wkv.weight.data.float()).to(dtype)
        if self.wgate is not None:
            self.wgate.weight.data = init_std_(self.wgate.weight.data.float()).to(dtype)

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor | None:
        """x: [B, S, d] -> [B, S/ratio, head_dim] latent pre-RoPE; None when the
        current group is still incomplete (decode)."""
        ratio = self.compress_ratio
        if ratio == 1:
            return self.norm(self.wkv(x))

        x = x.float()
        kv, score = self.wkv(x), self.wgate(x)  # type: ignore[operator]
        bsz = x.size(0)
        if start_pos == 0:
            seqlen = x.size(1)
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            if remainder:  # trailing partial group waits in the state
                self._ensure_state(bsz, x)
                self.kv_state[:bsz, :remainder] = kv[:, cutoff:]  # type: ignore[index]
                self.score_state[:bsz, :remainder] = score[:, cutoff:]  # type: ignore[index]
                kv, score = kv[:, :cutoff], score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:  # decode: fill a slot; pool only when the group just completed
            self._ensure_state(bsz, x)
            slot = start_pos % ratio
            self.kv_state[:bsz, slot] = kv[:, 0]  # type: ignore[index]
            self.score_state[:bsz, slot] = score[:, 0]  # type: ignore[index]
            if (start_pos + 1) % ratio != 0:
                return None
            kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)  # type: ignore[index]
        return self.norm(kv.to(x.dtype))

    def _ensure_state(self, bsz: int, x: torch.Tensor) -> None:
        if self.kv_state is None or self.kv_state.size(0) < bsz:
            self.kv_state = torch.zeros(bsz, self.compress_ratio, self.head_dim, dtype=torch.float32, device=x.device)
            self.score_state = torch.full((bsz, self.compress_ratio, self.head_dim), -torch.inf, dtype=torch.float32, device=x.device)


class CedProjection(nn.Module):
    """Decoder CED projection (design §4.1): W^Z_l (d -> z) then W^KV_l
    (z -> head_dim latent), one latent per token (m=1), no gate, norm after."""

    def __init__(self, cfg):
        super().__init__()
        self.wz = nn.Linear(cfg.d_model, cfg.ced_z_rank, bias=False)
        self.wkv = nn.Linear(cfg.ced_z_rank, cfg.head_dim, bias=False)
        self.norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        init_std_(self.wz.weight)
        init_std_(self.wkv.weight)

    def forward(self, producer_hidden: torch.Tensor) -> torch.Tensor:
        """producer_hidden: [B, S, d] -> [B, S, head_dim] latent pre-RoPE."""
        return self.norm(self.wkv(self.wz(producer_hidden)))


class ProducerMap:
    """Data-table view of the KV consumer map (contract T1; swappable per C3)."""

    def __init__(self, cfg):
        self.cfg = cfg

    def producer_of(self, layer_id: int) -> int | None:
        """Producer whose global KV this layer consumes (None for SWA-only)."""
        path = self.cfg.global_kv_path(layer_id)
        if path == "none":
            return None
        if path == "own":
            return layer_id
        producers = [l for l in self.cfg.kv_source_layers if l < layer_id]
        return producers[-1] if producers else None


def rope_latent(latent: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int, seqlen: int,
                ratio: int) -> torch.Tensor:
    """RoPE the latent tail at group-first positions j*ratio (upstream
    model.py:751-761). In-place on the tail; returns the tensor."""
    if start_pos == 0:
        freqs = freqs_cis[: seqlen - seqlen % ratio : ratio]
    else:
        freqs = freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
    tail = latent[..., -2 * freqs.size(-1) :]
    apply_rotary_emb(tail, freqs)
    return latent


class CEDRuntime:
    """Extends SharedAttentionRuntime with the producer-hidden stash the
    decoder projection path reads (design §4.1: producers publish their final
    hidden states; decoders project from them, never from the KV cache)."""

    def __init__(self, shared):
        self.shared = shared
        self.producer_hidden: dict[int, torch.Tensor] = {}
        self.published_ratio: int | None = None


def reachable_idxs(total_compress: int, start_pos: int, seqlen: int, offset: int, device=None) -> torch.Tensor:
    """Task-4 fallback when no indexer is attached: every query attends every
    reachable compressed position (toy pool-less config only). Prefill only;
    rows padded with -1 to the max reachable count."""
    assert start_pos == 0, "fallback is prefill-only; decode requires an indexer"
    # query t reaches groups whose first token j <= t (upstream indexer masking:
    # compress_lens = arange(1, seqlen+1)//ratio, here ratio 1 -> all positions)
    counts = torch.arange(1, seqlen + 1, device=device)
    width = min(int(counts[-1]), total_compress)
    idxs = torch.full((seqlen, width), -1, dtype=torch.int64, device=device)
    for t in range(seqlen):
        n = min(int(counts[t]), width)
        idxs[t, :n] = torch.arange(n, device=device) + offset
    return idxs.unsqueeze(0).expand(1, seqlen, width)


def ced_attention_forward(attn, x: torch.Tensor, start_pos: int, ced: CEDRuntime,
                          indexer=None, compressor=None, ced_projection=None,
                          producer_hidden: torch.Tensor | None = None) -> torch.Tensor:
    """One layer's attention over [window (+ global KV)] (contract T1/T7).

    - ratio 0: window only.
    - 'own' (kv source): compressor -> indexer (pre-RoPE latent) -> RoPE +
      publish cache -> attend [window, compress].
    - 'cache': read the published same-ratio cache; consume published indices.
    - 'project' (decoder): CedProjection from the producer's hidden state ->
      own cache -> attend [window, own latents] (never reads shared cache).
    """
    cfg = attn.cfg
    shared = ced.shared
    seqlen = x.size(1)
    q, qr = attn.q_proj(x, start_pos)
    window_kv, window_idxs = attn._window_kv(x, start_pos)
    window_idxs = window_idxs.expand(x.size(0), -1, -1)
    offset = window_kv.size(1)
    path = cfg.global_kv_path(attn.layer_id)

    if path == "none":
        return attn.attend(q, window_kv, window_idxs, start_pos, seqlen)

    ratio = attn.compress_ratio
    compress_len = (start_pos + seqlen) // ratio

    if path == "project":
        assert producer_hidden is not None, f"layer {attn.layer_id} needs its producer's hidden states"
        assert ced_projection is not None
        latent = ced_projection(producer_hidden)
    else:
        latent = compressor(x, start_pos) if path == "own" else None

    # index selection BEFORE the latent is RoPE'd/cached (upstream order).
    # Production blocks always attach an indexer to index sources; the
    # reachable-idxs fallback exists for pool-less toy configs only.
    if attn.is_index_source and indexer is not None:
        idxs = indexer(x, qr, latent, start_pos, offset, compress_len, shared)
        shared.topk_idxs = idxs
        shared.topk_ratio = ratio
    elif shared.topk_idxs is not None and shared.topk_ratio == ratio:
        idxs = shared.topk_idxs  # reuse (upstream _compress_topk_idxs)
    else:
        idxs = reachable_idxs(compress_len, start_pos, seqlen, offset, x.device)
        shared.topk_idxs = idxs
        shared.topk_ratio = ratio

    # RoPE + publish the compressed KV
    if latent is not None:
        rope_latent(latent, attn.freqs_cis, start_pos, seqlen, ratio)
        if start_pos == 0:
            attn.compress_kv_cache = latent
        else:
            if attn.compress_kv_cache is None:
                attn.compress_kv_cache = torch.zeros(
                    x.size(0), attn.max_seq_len // ratio, attn.head_dim, dtype=latent.dtype, device=latent.device
                )
            attn.compress_kv_cache[: x.size(0), start_pos // ratio : start_pos // ratio + latent.size(1)] = latent
        shared.compress_kv = attn.compress_kv_cache
        ced.published_ratio = ratio

    if path == "project":
        # the decoder's own published cache (never the encoder's)
        compress_kv = attn.compress_kv_cache[: x.size(0), :compress_len]
    else:
        assert shared.compress_kv is not None, "no global KV published yet"
        assert ced.published_ratio == ratio, "same-ratio consumer expected"
        compress_kv = shared.compress_kv[: x.size(0), :compress_len]

    if compress_kv.size(1) == 0:
        return attn.attend(q, window_kv, window_idxs, start_pos, seqlen)
    kv = torch.cat([window_kv, compress_kv], dim=1)
    topk = torch.cat([window_idxs, idxs], dim=-1)
    return attn.attend(q, kv, topk, start_pos, seqlen)
