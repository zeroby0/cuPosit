from pathlib import Path
import torch.utils.cpp_extension
import torch.nn.functional

__all__ = ['bspgemm']

Path('build/bspgemm_cuda').mkdir(exist_ok=True, parents=True)
bspgemm_cuda = torch.utils.cpp_extension.load(
    name='bspgemm',
    sources=['cusrc/bspgemm.cu'],
    extra_include_paths=['cutlass/include', 'cutlass/tools/util/include', 'cutlass/examples/common'],
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3'],
    build_directory='build/bspgemm_cuda',
    with_cuda=True,
    verbose=True
)

def bspgemm(A, B, C, alpha=1.0, beta=1.0, posit=(0, 0)):
    detach = lambda x: x.detach().contiguous().clone()

    if posit != (0, 0):
        raise NotImplementedError("Posit support not implemented yet")

    return bspgemm_cuda.bspgemm(
        detach(A),
        detach(B),
        detach(C),
        alpha, beta
    )

