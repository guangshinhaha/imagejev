# imagejev

Open System 1 decision engine for images: typed `choice` / `score` / `bool` decisions
over an image (plus optional text state) in a single forward pass, with calibrated confidence.

Status: early development. See `docs/superpowers/specs/2026-10-01-imagejev-design.md`.

> Working name. It will be renamed before release (see the Release epic).

## Develop

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
ruff check . && ruff format --check . && pytest
```
