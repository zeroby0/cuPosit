import warnings
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from cuposit import ops as cuposit_ops


class MatMulDispatcher(TorchDispatchMode):
    def __init__(self, positnes=(28, 2)):
        self.positnes = positnes
        self.enabled = True if positnes != (0, 0) else False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        if not self.enabled:
            return func(*args, **kwargs)
        
        if func in (
            torch.ops.aten.mm.default,
        ):
            return cuposit_ops.mm(*args, **kwargs, positnes=self.positnes)

        if func in (
            torch.ops.aten.addmm.default,
        ):
            return cuposit_ops.addmm(*args, **kwargs, positnes=self.positnes)

        if func in (
            torch.ops.aten.convolution.default,
        ):
            return cuposit_ops.convolution(*args, **kwargs, positnes=self.positnes)

        if func in (
            torch.ops.aten.matmul.default,
            torch.ops.aten.bmm.default,
        ):
            warnings.warn(f"Cuposit Dispatcher noticed an uncaught matmul op: {func}")

        return func(*args, **kwargs)