import torch

A = torch.randint(low=1, high=10, size=(1, 40, 50), requires_grad=False, device='cuda', dtype=torch.float32)
B = torch.randint(low=1, high=10, size=(1, 50, 60), requires_grad=False, device='cuda', dtype=torch.float32)
C = torch.randint(low=1, high=10, size=(1, 40, 60), requires_grad=False, device='cuda', dtype=torch.float32)
A = torch.ones_like(A)
B = torch.ones_like(B)
C = torch.zeros_like(C)

# C[0][0] = 210

import cuposit

D = cuposit.bspgemm(A, B, C, alpha=1.0, beta=1.0)

E = A@B + C

print(D)
print('ref:')
print(E)