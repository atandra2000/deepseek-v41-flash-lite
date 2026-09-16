"""Model configuration: loads configs/lite-v3.json and derives layer maps.

Field names mirror the upstream config keys where they exist (see
docs/architecture-contract.md T12). The producer/mode maps are data, swappable
per the C3/C4 gates.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

CONFIG_ROOT = Path(__file__).resolve().parent.parent / "configs"
DEFAULT_CONFIG = CONFIG_ROOT / "lite-v3.json"


def config_sha256(path: Path | None = None) -> str:
    """sha256 of the raw config file — embedded in every artifact (design §5)."""
    path = Path(path) if path is not None else DEFAULT_CONFIG
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class LiteConfig:
    """Resolved model configuration (Task 2 skeleton; training gates read this too)."""

    # identity
    name: str
    contract: str
    design: str
    # text backbone
    n_layers: int
    n_encoder_layers: int
    compress_ratios: tuple[int, ...]
    d_model: int
    n_heads: int
    head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    rope_head_dim: int
    rope_theta: float
    compress_rope_theta: float
    norm_eps: float
    window_size: int
    # CED / CSA2 maps (data, swappable)
    kv_source_layers: tuple[int, ...]
    index_source_layers: tuple[int, ...]
    candidate_source_layer: int
    candidate_topk_blocks: int
    candidate_block_size: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    # MoE
    n_routed_experts: int
    n_shared_experts: int
    n_activated_experts: int
    moe_inter_dim: int
    score_func: str
    topk_method: str
    norm_topk_prob: bool
    routed_scaling_factor: float
    swiglu_limit: float
    # mHC
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    # Engram
    engram_layer_ids: tuple[int, ...]
    engram_num_embeddings: tuple[int, ...]
    engram_max_ngram_size: int
    engram_vocab_size: int
    engram_n_heads: int
    engram_head_dim: int
    engram_pad_token_id: int
    engram_compressed_vocab_size: int
    # DSpark
    num_nextn_predict_layers: int
    dspark_block_size: int
    dspark_noise_token_id: int
    dspark_target_layer_ids: tuple[int, ...]
    dspark_markov_rank: int
    dspark_inter_dim: int
    # vision
    vision_n_layers: int
    vision_dim: int
    vision_n_heads: int
    vision_inter_dim: int
    vision_patch_size: int
    vision_downsample_ratio: int
    vision_max_n_token: int
    vision_min_pixels: int
    # vocab / context
    vocab_size: int
    num_reserved_specials: int
    image_token_id: int
    tie_word_embeddings: bool
    context_train_stage1: int
    context_train_stage2: int
    dtype: str
    sha256: str = ""

    # ---- derived layer maps (data tables, per the contract) ----

    @property
    def encoder_layers(self) -> tuple[int, ...]:
        return tuple(range(self.n_encoder_layers))

    @property
    def decoder_layers(self) -> tuple[int, ...]:
        return tuple(range(self.n_encoder_layers, self.n_layers))

    def is_kv_source(self, layer_id: int) -> bool:
        return layer_id in self.kv_source_layers

    def is_index_source(self, layer_id: int) -> bool:
        return layer_id in self.index_source_layers

    @property
    def uses_candidate_pool(self) -> bool:
        """Whether any layer reads the pool (candidate source < some index source)."""
        return any(self.candidate_source_layer < l for l in self.index_source_layers)

    def compressed_len(self, seq_len: int, layer_id: int) -> int:
        """Global-stream length a layer sees for a full prefill of seq_len tokens."""
        ratio = self.compress_ratios[layer_id]
        return seq_len // ratio if ratio else 0

    @property
    def pool_capacity(self) -> int:
        return self.candidate_topk_blocks * self.candidate_block_size

    def validate(self) -> None:
        """Fail loudly on structural contradictions (called by __post_init__ and tests)."""
        assert len(self.compress_ratios) == self.n_layers + self.num_nextn_predict_layers, (
            "compress_ratios covers backbone + DSpark layers (upstream shape; drafter ratios are 0)"
        )
        assert all(
            self.compress_ratios[l] == 0 for l in range(self.n_layers, len(self.compress_ratios))
        ), "DSpark layers are SWA-only (ratio 0)"
        assert self.d_model % self.n_heads == 0, "d_model must divide into heads"
        assert self.head_dim == self.d_model // self.n_heads
        assert 0 < self.rope_head_dim <= self.head_dim and self.rope_head_dim % 2 == 0
        assert self.o_groups * (self.n_heads * self.head_dim // self.o_groups) == self.n_heads * self.head_dim
        assert self.n_encoder_layers < self.n_layers, "decoder must be non-empty"
        assert all(0 <= r <= 2 for r in self.compress_ratios), "Lite ratios are 0 (skip), 1, or 2 only"
        assert all(self.compress_ratios[l] > 0 for l in self.kv_source_layers), "kv sources must compress"
        assert self.kv_source_layers == tuple(
            sorted(set(self.kv_source_layers))
        ), "kv sources must be ascending and unique"
        assert self.index_source_layers == tuple(sorted(set(self.index_source_layers)))
        assert set(self.kv_source_layers) <= set(self.index_source_layers), "every kv source indexes"
        if self.candidate_source_layer < 0:
            assert self.candidate_source_layer == -1, "candidate_source_layer is -1 (off) or a valid index source"
        else:
            assert self.candidate_source_layer in self.index_source_layers, "pool builder must run an indexer"
        assert self.pool_capacity >= self.index_topk, "pool must cover top-k"
        assert self.n_activated_experts <= self.n_routed_experts
        assert self.tie_word_embeddings, "Lite vocab is tied (D4)"
        assert self.dspark_target_layer_ids[-1] < self.n_layers
        assert set(self.dspark_target_layer_ids) <= set(range(self.n_encoder_layers, self.n_layers)), (
            "DSpark targets decoder layers only (upstream pattern: tail layers)"
        )
        assert self.vocab_size % 2 == 0
        assert 0 <= self.image_token_id < self.vocab_size


def load_config(path: Path | None = None) -> LiteConfig:
    """Load and validate a config file; LiteConfig fields are exactly the JSON keys."""
    path = Path(path) if path is not None else DEFAULT_CONFIG
    raw = json.loads(path.read_text())
    sha = config_sha256(path)
    cfg = LiteConfig(sha256=sha, **{k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items()})
    cfg.validate()
    return cfg
