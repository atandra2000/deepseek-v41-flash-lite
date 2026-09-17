# Diagram receipts

Generated 2026-09-17 with Archify (~/.agents/skills/archify, `doctor` clean). Each JSON specification is the editable source of its HTML. Deliveries atomically froze these exact specification bytes, and `visual-check` then captured automated-browser evidence from the delivered HTML. Run `python3 docs/diagrams/verify.py` to reconfirm all four bindings without rerendering; it writes `verification.json`.

| Artifact | Type | Specification source | Spec SHA-256 | HTML SHA-256 | Checks | Automated browser |
|---|---|---|---|---|---|---|
| architecture-model.html | architecture | model.architecture.json | 50109595655aab09784b37dbce0a07dadecaf371c7a5734834d4806a983952c8 | bc0c9171006fc5771068f5441405016c27e926c4b2d37b8f9194999fe426618d | 9/9 showcase, 0 errors, 0 warnings | pass (receipt JSON + 4 PNG sidecars) |
| architecture-csa2.html | architecture | architecture-csa2.json | 57c3e2dcdfea2ab5b232d2439bb33c485051b8952b9ca95ffa10b4da404e607c | 7fcfab69ebf552057bd1277cda70b1c27ffc40fb3af73e022271353c944e3658 | 9/9 showcase, 0 errors, 0 warnings | pass (receipt JSON + 4 PNG sidecars) |
| workflow-training-loop.html | workflow | workflow-training-loop.json | 9f01d0e8227756c7a2a7dd79894f7ce148c34786ca2393b14e7d280ea7149b2b | c25c480629b48ec3f85354b9b3bf94f535ed56013e0eb67919de133f1aab53a0 | 9/9 showcase, 0 errors, 0 warnings | pass (receipt JSON + 4 PNG sidecars) |
| dataflow-data-pipeline.html | dataflow | dataflow-data-pipeline.json | c17d878535fa66e4f5618f338a0fc8a9f72c6d81715198b67666d8a07ccb7f1a | 6292c73a476cce68eea7b218fcd427381edb8725b1590749293308b38a69ae12 | 9/9 showcase, 0 errors, 0 warnings | pass (receipt JSON + 4 PNG sidecars) |

## Regeneration

From the repository root:

```bash
node ~/.agents/skills/archify/bin/archify.mjs deliver architecture docs/diagrams/model.architecture.json docs/diagrams/architecture-model.html --quality showcase --json
node ~/.agents/skills/archify/bin/archify.mjs deliver architecture docs/diagrams/architecture-csa2.json docs/diagrams/architecture-csa2.html --quality showcase --json
node ~/.agents/skills/archify/bin/archify.mjs deliver workflow docs/diagrams/workflow-training-loop.json docs/diagrams/workflow-training-loop.html --quality showcase --json
node ~/.agents/skills/archify/bin/archify.mjs deliver dataflow docs/diagrams/dataflow-data-pipeline.json docs/diagrams/dataflow-data-pipeline.html --quality showcase --json
node ~/.agents/skills/archify/bin/archify.mjs visual-check docs/diagrams/architecture-model.html --json
node ~/.agents/skills/archify/bin/archify.mjs visual-check docs/diagrams/architecture-csa2.html --json
node ~/.agents/skills/archify/bin/archify.mjs visual-check docs/diagrams/workflow-training-loop.html --json
node ~/.agents/skills/archify/bin/archify.mjs visual-check docs/diagrams/dataflow-data-pipeline.html --json
python3 docs/diagrams/verify.py
```

## Source mapping

Facts were read from the checked-in architecture contract (`docs/architecture-contract.md`, T1–T13), the design specification (`docs/DESIGN-dsv41-flash-lite.md`), `configs/lite-v3.json`, and the current model code: `models/transformer.py` (forward pass, shared runtimes, block wiring), `models/ced.py` (producer map, decoder `CedProjection`, joint window/global attention), `models/indexer.py` (index keys, candidate pool, top-k selection), `models/csa2.py` (block top-k selection), and the training/data code as mapped inside each workflow/dataflow specification.

- The model diagram reflects `Transformer.forward` ordering: embedding, image-span merge, mHC stream expansion, the 24 blocks, final collapse, and tied FP32 logits.
- The CED/CSA2 diagram shows the Lite decoder ownership contract: layers 12–23 project layer 8's final hidden state through their own `CedProjection` and never read the encoder cache; L12 owns the decoder index-key cache; L16/L20 score it inside the L12 pool; newest candidate block is pinned; SWA-128 is layer-local and joins global KV in one attention call.
- The training workflow covers the implemented trainer path (finite-value checks, AdamW, atomic checkpoints, recovery) and does not depict completed full-size GPU results; the dataflow covers the implemented offline prep (checksummed sources, shards, token map, document-disjoint split, integrity manifest) without asserting production corpora.
- DSpark, QAT/FP8, RL, and stage-2 scale targets are goals or separate paths, not depicted as active training facts.

## Scope and review status

- `visual_review: pending` in each automated-browser receipt: these receipts are machine evidence only.
- Perceptual visual review: not independently assessed this session; the local image reader could not return rendered screenshots for review. `docs/diagrams/verify.py` rechecks deterministic artifact checks plus automated browser evidence for the four diagrams.
- The interactive guide `deepseek_v41_visual_guide.html` is hand-authored static HTML with no build step. Its layer and selection-budget explorers were exercised in a real browser over all 24 layers and budget inputs 1, 128, 256, 8192, 16384, 0, 16385, 1.5, and empty; no horizontal overflow at 1440×900 (scrollWidth 1440 = innerWidth). All numbers derive from `configs/lite-v3.json` and the architecture contract; no production-scale claims are made.
- `architecture-model.delivery.json` and `architecture-csa2.delivery.json` were delivered by this session's parent; workflow and dataflow receipts were produced and independently repaired/verified by a delegated authoring agent in the same repository working tree.
