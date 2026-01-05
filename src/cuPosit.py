from pathlib import Path
import torch.utils.cpp_extension
import torch.nn.functional


Path('build/cuPosit').mkdir(exist_ok=True, parents=True)
cuPosit = torch.utils.cpp_extension.load(
    name='cuPosit',
    sources=['cusrc/bspgemm.cu'],
    extra_include_paths=['cutlass/include', 'cutlass/tools/util/include', 'cutlass/examples/common'],
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3'],
    build_directory='build/cuPosit',
    with_cuda=True,
    verbose=True
)

def pgemm(A, B, C, alpha=1.0, beta=1.0, posit=(0, 0)):
    detach = lambda x: x.detach().contiguous().clone()

    if posit != (0, 0):
        raise NotImplementedError("Posit support not implemented yet")

    return cuPosit.bspgemm(
        detach(A),
        detach(B),
        detach(C),
        alpha, beta
    )

