from pathlib import Path
import torch.utils.cpp_extension

__all__ = ['bspgemm']

try:
    from . import _CUDA
except ImportError:
    raise ImportError(
        "cuposit C++ extension not found. "
        "Please install cuposit properly: pip install -e ."
    )

# Path('build/bspgemm_cuda').mkdir(exist_ok=True, parents=True)
# bspgemm_cuda = torch.utils.cpp_extension.load(
#     name='bspgemm',
#     sources=['cusrc/bspgemm.cu'],
#     extra_include_paths=['cutlass/include', 'cutlass/tools/util/include', 'cutlass/examples/common'],
#     extra_cflags=['-O3'],
#     extra_cuda_cflags=['-O3'],
#     build_directory='build/bspgemm_cuda',
#     with_cuda=True,
#     verbose=True
# )

def bspgemm(A, B, C, alpha=1.0, beta=1.0, posit=(28, 2)):
    detach = lambda x: x.detach().contiguous().clone()

    if (posit[0] >= 4 and posit[1] == 2) or posit == (0, 0):
        _A, _B, _C = detach(A), detach(B), detach(C)
         
        result = _CUDA.bspgemm(
            _A, _B, _C,
            alpha, beta,
            posit[0], posit[1]
        )

        del _A
        del _B

        return result

    raise ValueError(f"Invalid Posit configuration: {posit}. See Usage section of readme.")