1. Install `uv`: https://docs.astral.sh/uv/getting-started/installation/#installation-methods

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv sync
source .venv/bin/activate
uv pip install -e .
```
