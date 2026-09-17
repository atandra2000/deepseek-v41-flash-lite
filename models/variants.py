"""Executable gate variants and matched-token controls (Task 11 closeout).

The C0–C6 ladder (candidate §8) builds the architecture incrementally:
dense C0 → +MoE (C1) → +SWA (C2) → +CED (C3) → +CSA2 (C4) → +mHC (C5) →
+Engram (C6). Each stage's control is the previous stage's architecture
trained on the same token stream; gates compare 512-batch probe losses at
matched tokens (training/ladder.py:GateRunner).

Variants are derived LiteConfig copies plus a VariantSpec the module tree
reads at construction. configs/lite-v3.json (the production artifact
identity, sha256-pinned) is never modified; the production model is
Transformer(cfg) with no spec (defaults to FULL). Gates evaluate the text
backbone: ViT and DSpark attach post-C8 (P2/P3) and are stripped from every
gate variant (design §6, candidate §8).
"""

from dataclasses import dataclass, replace

from .config import LiteConfig


@dataclass(frozen=True)
class VariantSpec:
    """Which mechanisms a variant's module tree builds (Block/Transformer).

    The derived config (derive_config) carries the data-table side of each
    toggle; csa2 is structural only (the kv/index tables must remain a valid
    producer map, so Block simply skips the Indexer and ced.py's
    reachable-idxs fallback attends every reachable compressed position —
    the dense-global control for C4)."""

    name: str
    moe: bool = False
    swa: bool = False
    ced: bool = False
    csa2: bool = False
    mhc: bool = False
    engram: bool = False
    # C5 fallback: build mHC but freeze it at its identity init.
    mhc_frozen: bool = False


_MECHANISMS = ("moe", "swa", "ced", "csa2", "mhc", "engram")


def _named() -> dict[str, VariantSpec]:
    """Cumulative ladder variants; each stage adds exactly one mechanism."""
    named = {"dense": VariantSpec("dense")}
    enabled: list[str] = []
    for mech in _MECHANISMS:
        enabled.append(mech)
        named[mech] = VariantSpec(mech, **{m: m in enabled for m in _MECHANISMS})
    return named


NAMED = _named()
FULL = NAMED["engram"]

# Primary variant per gate stage; the matched-token control is the previous
# stage's architecture (C0 is the baseline and has no control).
STAGE_VARIANT = {"C0": "dense", "C1": "moe", "C2": "swa",
                 "C3": "ced", "C4": "csa2", "C5": "mhc", "C6": "engram"}
STAGE_CONTROL = {"C0": None, "C1": "dense", "C2": "moe",
                 "C3": "swa", "C4": "ced", "C5": "csa2", "C6": "mhc"}


def derive_config(cfg: LiteConfig, spec: VariantSpec | None, overrides: dict | None = None) -> LiteConfig:
    """A validated cfg copy for a variant (spec None = FULL, the production
    architecture stripped of nothing).

    - ced off: no compression anywhere — ratios 0, empty producer/index
      tables, no candidate pool (plain MLA, layer-local KV only).
    - swa off: the window is the full training context (dense causal
      attention over the layer-local latent KV).
    - mhc off: hc_mult 1 (Block additionally swaps HCMixes for the
      parameter-free PlainResidual).
    - engram off: no Engram layers (NgramHashState is cfg-driven).
    - csa2 off: structural only — the tables stay a valid producer map so
      the compressed pathway remains, unselected (see VariantSpec).
    Every gate variant strips ViT/DSpark (post-C8 modules) and truncates
    compress_ratios to the backbone accordingly.
    """
    if spec is None:
        spec = FULL
    d: dict = {
        "vision_n_layers": 0,
        "num_nextn_predict_layers": 0,
        "dspark_block_size": 0,
        "dspark_target_layer_ids": (),
        "compress_ratios": cfg.compress_ratios[: cfg.n_layers],
    }
    if not spec.ced:
        d["compress_ratios"] = (0,) * cfg.n_layers
        d["kv_source_layers"] = ()
        d["index_source_layers"] = ()
        d["candidate_source_layer"] = -1
    if not spec.swa:
        d["window_size"] = cfg.context_train_stage2
    if not spec.mhc:
        d["hc_mult"] = 1
    if not spec.engram:
        d["engram_layer_ids"] = ()
    if overrides:
        d.update(overrides)
    derived = replace(cfg, **d)
    derived.validate()
    return derived