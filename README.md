# CuPosit

CuPosit is a Batched Strided Posit GEMM for PyTorch.
To run your Neural Network in Posit, wrap it's forward pass in CuPosit's dispatcher.

```py
from cuposit.dispatcher import MatMulDispatcher
dispatcher = MatMulDispatcher(positnes=(16, 2))

def train(nepochs):
    model.train()
    for epoch in range(nepochs):
        for inputs, labels in train_loader:            
            optimizer.zero_grad()

            with dispatcher:      # <---- here. Dispatching Forward pass only.
                outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
```

This makes torch ops: `mm`, `addmm`, `matmul`, `bmm`, and `convolution` run in Posit for Forward Pass.
The Backward Pass still happens in Float32. Gradients are in Float32 as well. 

The operations are about 8-10 times slower than Float32 (4 TOPS compared to 30-40 TOPS for FP32), so this
library is only expected to be used for QAT-ing a model already trained in Float32.
See **examples/03_dispatch_full.py** for a complete training example with Posit(16, 2) QAT.

Only Posits with `4 <= n <= 28, es == 2` are supported, however, you can modify `cusrc/positclip.h` to support other `es`.

For other implementations of Posit arithmetic, see the Implementations section in https://en.wikipedia.org/wiki/Unum_(number_format)#Unum_III.


# Installation

1. Clone this directory into your project folder.
2. `cd` into this folder.
3. `uv sync`

Now you can use cuposit in your environment. 

If `ninja` and `torch` aren't installed automatically, install with `pip install ninja torch`.
If you're using `uv` and see a build error about Python headers, install Python `uv python install 3.12`.

# Development

Install `uv`: https://docs.astral.sh/uv/getting-started/installation/#installation-methods

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv pin python 3.12
uv sync
uv run <file>.py
```

Then go into the examples folder and run any example you'd like.

# Caveats

These caveats are design decisions based on the afore-mentioned expectation of usage.

While the arithmetic happens in Posit, accumulation happens in Float32. 
You can modify `cutlass/include/cutlass/arch/mma_sm50.h` to perform accumulation in Posit as well.
That runs at around 1 TOPS.

The library performs operation `matmul(A, B)` by rounding the inputs A & B, and the individual row & column products to Posit.
Here's pseudo code to illustrate:
```py
for row in A:
    for column in B:
        accumulate = 0
        for r in row:
            for c in column:
                accumulate += posit(posit(r) * posit(c))
        result[row][column] = accumulate
```
The `posit` function here rounds a 32-bit Float to the nearest posit smaller than it in absolute magnitude.
The other caveat is that exponents are clamped to `((posit_n - posit_es - 2) * 4 - 1)`, so numbers at the edges of the posit's exponent range will be clamped.
If you know none your intermediate results reach these clamps, or if you don't care, you can remove this clamp in `cusrc/positclip.h` and gain another ~1 TOPS.

# Contribution and License

Contributions are very welcome and appreciated. 
You're also welcome to fork this repo and make any changes you'd like.
Please create an Issue on Github if you face any issues.

You're also free to modify and use this library in any way you see fit for research/personal use,
however, it's licensed with AGPLv3, and requires freely published source code for commercial use.
If you use this in your acedemic publication, please cite us.