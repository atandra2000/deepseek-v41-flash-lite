# Archify quality-review delivery evidence

Refreshed 2026-09-17. Scope: only deepseek-v41-flash-lite's diagrams and guide. No model, training or data implementation changed. [Guide](deepseek_v41_visual_guide.html).

## Acceptance status

- **4/4 diagrams delivered, each 9/9 showcase checks, zero composition errors or warnings.**
- **Fresh automated browser evidence passed for all 4 exact delivered hashes.** Light-theme measurements cover 1440×900, 1600×1000, 1920×1080 and 2048×1320. Light/dark endpoint screenshots accompany each receipt. This does not establish intermediate dark-theme geometry.
- **Perceptual review: skipped (image input unavailable).** An image read was attempted, but the provider rejected image input. Automated receipts correctly retain `visualReview: pending`. Earlier visual approval does not certify these regenerated artifacts.
- **The proposed 12px essential / 11px secondary target is not met.** The measured minimum contextual text below is smaller. Containment and the Archify 6px gate are not premium readability certification.
- Browser interaction/mobile guide review could not run through the bridge (broken-pipe failure). Guide local links, navigation anchors, unique IDs and source symbols passed static checks. Focus/search/reset, representative export and narrow-screen guide behavior remain unverified for this revision.

## Source and configuration pin

Source revision: `099c72954c82857c83c0c75d1d2abbb3127919b0`. Configuration: `configs/lite-v3.json + training/pretrain.py:TrainingConfig`. These pins identify reviewed model code, not the later documentation commit. Configured dimensions, calculated sizes and target corpus/training budgets are not fresh GPU measurements. No corpus inventory, full training run or GPU benchmark was performed.

## Exact artifact bindings

| HTML | Type | Specification | Spec SHA-256 | HTML SHA-256 | Min node px at 1440×900 | Browser |
|---|---|---|---|---|---:|---|
| [architecture-csa2.html](architecture-csa2.html) | architecture | [architecture-csa2.json](architecture-csa2.json) | `8600d763b39c97d87620bef7db4fa437e9aa599ed65dc61713857cd9d4ffa983` | `24ce765b8b45d615dd6f092ae45b90b26c246c73f6a96da591b87d9ee8c4c823` | 6.35 | pass |
| [dataflow-data-pipeline.html](dataflow-data-pipeline.html) | dataflow | [dataflow-data-pipeline.json](dataflow-data-pipeline.json) | `a512e320fd473aa52529c0bdae9243dfd145b1e41cf3f6579c97094d77cb6797` | `15429e2fb420c4382be002320d872c1cbee3ddc76f977a47c531c0d3dda2812c` | 6.93 | pass |
| [architecture-model.html](architecture-model.html) | architecture | [model.architecture.json](model.architecture.json) | `d33fc1f90ac88dc16e6a658cbacf3a5d2ca1e70a5bf5f97d2cd72ae390493cc4` | `9b2696d16b8f5abaf6fbded5cf8726a9eafe9d5d283dec78564b0c956dd41a6a` | 7.00 | pass |
| [workflow-training-loop.html](workflow-training-loop.html) | workflow | [workflow-training-loop.json](workflow-training-loop.json) | `6fc982b9d0b783862852f2c5d9b61f2671974da3708d215ae6f6fc790d175aef` | `cde3db8dd4629212896a1b45b37bbbee2ef664e307c0f67b78752f8a13b835c3` | 7.07 | pass |

Delivery JSON includes byte counts. Each HTML has a `.delivery.json`, `.visual-check.json`, contact sheet and four PNGs. Validation JSON records the last successful static check.

## Corrections and source mapping

- Model: SWA-only first two blocks, global paths only afterward. Separate DSpark path retained. Sources: `models/config.py:LiteConfig.global_kv_path`, `models/transformer.py:Transformer.forward`.
- CSA2: unmasked reachable-score L12 top-k is a separate branch from later candidate-masked selection. Per-decoder cache ownership and no-eviction caveat retained. Sources: `models/indexer.py:Indexer.forward`, `models/ced.py:ced_attention_forward`.
- Data: duplicate identities/content abort preparation. Policy copy shortened without asserting corpus completion. Source: `data/prepare_data.py:prepare`.
- Training: error edges now originate at forward, gradient norm and update-state checks. CUDA enables BF16. Propagated StopIteration is not normal final-checkpoint completion. Sources: `training/pretrain.py:Trainer._loss`, `training/pretrain.py:Trainer._batch`, `training/pretrain.py:Trainer.guarded_step`, `training/pretrain.py:Trainer.run`.
- Optional extra loop/abort geometry did not converge in two focused repair attempts and was abandoned. The required norm/update error routes pass; the DS chart still does not draw every continuation/exhaustion transition. Those limitations are disclosed in cards and guide.
- Browser correction rounds: 1 for model (2px overflow removed by shortening redundant evidence copy), 0 for the other three. Final browser evidence passed. This is not a perceptual correction count.

## Documentation checks

`python3 docs/diagrams/verify.py` passed all four artifact bindings, static validations and fresh browser checks. This project has no `tests/test_doc_refs.py` or `scripts/check_docs.py`. Static guide check: 19 local/navigation links and 7 distinct symbol citations passed. No new testing framework was added.

## Regeneration

From the repository root, validate and deliver a changed candidate, then run visual-check only if delivery succeeds:

```bash
node ~/.agents/skills/archify/bin/archify.mjs validate architecture docs/diagrams/architecture-csa2.json --quality showcase --json
node ~/.agents/skills/archify/bin/archify.mjs deliver architecture docs/diagrams/architecture-csa2.json docs/diagrams/architecture-csa2.html --quality showcase --json
node ~/.agents/skills/archify/bin/archify.mjs visual-check docs/diagrams/architecture-csa2.html --json
node ~/.agents/skills/archify/bin/archify.mjs validate dataflow docs/diagrams/dataflow-data-pipeline.json --quality showcase --json
node ~/.agents/skills/archify/bin/archify.mjs deliver dataflow docs/diagrams/dataflow-data-pipeline.json docs/diagrams/dataflow-data-pipeline.html --quality showcase --json
node ~/.agents/skills/archify/bin/archify.mjs visual-check docs/diagrams/dataflow-data-pipeline.html --json
node ~/.agents/skills/archify/bin/archify.mjs validate architecture docs/diagrams/model.architecture.json --quality showcase --json
node ~/.agents/skills/archify/bin/archify.mjs deliver architecture docs/diagrams/model.architecture.json docs/diagrams/architecture-model.html --quality showcase --json
node ~/.agents/skills/archify/bin/archify.mjs visual-check docs/diagrams/architecture-model.html --json
node ~/.agents/skills/archify/bin/archify.mjs validate workflow docs/diagrams/workflow-training-loop.json --quality showcase --json
node ~/.agents/skills/archify/bin/archify.mjs deliver workflow docs/diagrams/workflow-training-loop.json docs/diagrams/workflow-training-loop.html --quality showcase --json
node ~/.agents/skills/archify/bin/archify.mjs visual-check docs/diagrams/workflow-training-loop.html --json
```

Redirect successful delivery output to the matching `.delivery.json`, refresh the receipt table and verify hashes. Never retain a prior visual-pass claim after changing a specification.
