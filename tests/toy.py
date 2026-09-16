"""Toy configs for Phase-1 tests (Task 3: 2L, d64)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.config import LiteConfig

_BASE = None  # loaded lazily from configs/lite-v3.json


def toy_config(**over) -> LiteConfig:
    """2-layer toy: enc layer 0 (m=2, kv+index source), dec layer 1 (SWA-only).
    All mechanisms present at dims that run in <1s on CPU."""
    global _BASE
    if _BASE is None:
        _BASE = load_base()
    d = dict(_BASE)
    d.update(
        dict(
            n_layers=2,
            n_encoder_layers=1,
            compress_ratios=(2, 0),
            d_model=64,
            n_heads=4,
            head_dim=16,
            q_lora_rank=32,
            o_lora_rank=16,
            o_groups=2,
            rope_head_dim=8,
            window_size=8,
            kv_source_layers=(0,),
            index_source_layers=(0,),
            candidate_source_layer=-1,  # no pool in the SWA toy
            candidate_topk_blocks=4,
            candidate_block_size=2,
            index_n_heads=2,
            index_head_dim=8,
            index_topk=4,
            n_routed_experts=4,
            n_activated_experts=1,
            moe_inter_dim=32,
            hc_mult=2,
            engram_layer_ids=(),
            engram_num_embeddings=(128, 128),
            engram_max_ngram_size=3,
            engram_vocab_size=40,
            engram_n_heads=2,
            engram_head_dim=8,
            vocab_size=512,
            image_token_id=511,
            vision_n_layers=0,
            num_nextn_predict_layers=0,
            dspark_block_size=0,
            dspark_target_layer_ids=(1,),
            context_train_stage2=512,
        )
    )
    d.update(over)
    cfg = LiteConfig(sha256="toy", **d)
    cfg.validate()
    return cfg


def load_base():
    import json

    from models.config import DEFAULT_CONFIG

    raw = json.loads(DEFAULT_CONFIG.read_text())
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items()}


def ced_toy_config(**over) -> LiteConfig:
    """Toy with the full CSA2 path: enc m=2 (layer 0), dec m=1 (layer 1),
    index sources [0, 1], candidate source 1."""
    return toy_config(
        compress_ratios=(2, 1),
        index_source_layers=(0, 1),
        candidate_source_layer=1,
        **over,
    )
