# BSPGEMM

`alpha*(A@B) + beta*C`

`alpha` and `beta` are float32 scalars. `A`, `B`, and `C` are 3 Dimensional matrices, whose first dimension is batch.
`A@B` is the matrix multiplication operation.

The compute in `A@B` happens in posit, but the accumulation is in float32. scalar multiplication is float32, and addition with `C` is float32 as well.

Equivalent to:
```
beta*C + alpha*(
    posit(
        posit(A) @ posit(B)
    )
)
```

where `posit` is a function that rounds float32 to the nearest posit smaller than float32 in absolute magnitude.

Supports posit_n from 4 to 28, and posit_es == 2.

# Installation

1. Clone this directory into your project folder.
2. Activate your Python virtual environment.
3. `cd` into this folder.
4. Install dependencies `pip install ninja torch` (see pyproject.toml).
5. Run `pip install -e .`.

Now you can use cuposit in your environment.

# Development

1. Install `uv`: https://docs.astral.sh/uv/getting-started/installation/#installation-methods

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv sync
source .venv/bin/activate
uv pip install -e .
```

Then go into the examples folder and run any example you'd like.



