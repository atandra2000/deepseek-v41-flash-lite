# CLAUDE.md

From-scratch PyTorch replica of DeepSeek-V4.1-Flash, resized to ~1.16B total / ~205M active parameters. Python 3.12+, PyTorch 2.4+. The full architecture writeup is in `README.md` and `docs/`.

## Commands

```bash
pip install -r requirements.txt
pytest                 # pytest.ini holds the config
```

No lint or format step is configured. Do not add one.

## Layout

| Path | Holds |
|---|---|
| `models/` | Model code |
| `training/` | Training loops |
| `configs/` | Run configs |
| `scripts/` | Entry-point scripts |
| `tests/` | pytest suite |
| `upstream/` | Vendored reference code. Read-only. |
| `data/`, `results/` | Local artefacts, gitignored |

## Notes

- Use `codegraph explore "<query>"` before you grep. The index is in `.codegraph/`.
- Never edit `upstream/`. It is third-party reference code.
- `requirements.txt` lists pytest on purpose so a bare `pip install -r requirements.txt` is enough for CI. Leave it there.
