from pathlib import Path
import subprocess
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def get_cuda_version():
    try:
        nvcc = subprocess.check_output(['nvcc', '--version']).decode()
        version_line = [l for l in nvcc.split('\n') if 'release' in l][0]
        return version_line.split('release')[1].strip().split(',')[0]
    except:
        return None

root_dir = Path(__file__).parent.resolve()
cutlass_include = [
    root_dir / 'cutlass/include',
    root_dir / 'cutlass/tools/util/include',
    root_dir / 'cutlass/examples/common'
]

ext_modules = [
    CUDAExtension(
        'cuposit._CUDA',
        sources=['cusrc/bspgemm.cu'],
        include_dirs=cutlass_include,
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': [
                '-O3',
                '--use_fast_math',
                '-lineinfo'
            ]
        }
    )
]

setup(
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    install_requires=[
        'torch>=2.0.0',
    ],
    python_requires='>=3.8',
    zip_safe=False,
)