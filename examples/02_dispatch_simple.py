import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

from cuposit.dispatcher import MatMulDispatcher
dispatcher = MatMulDispatcher(positnes=(28, 2))

x = torch.randn(40, 50, requires_grad=True, device='cuda')
w = torch.randn(50, 60, requires_grad=True, device='cuda')

with dispatcher:
    # Forward pass - dispatches to your kernel
    y = x @ w
    
    # Backward pass - disable dispatcher
    dispatcher.enabled = False
    y.sum().backward()
    dispatcher.enabled = True

z = x @ w

# print(torch.max(torch.abs(y - z)))

assert torch.allclose(y.detach(), z.detach(), atol=1e-5, rtol=1e-5)