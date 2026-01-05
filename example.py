from icecream import ic
import torch
from src.cuPosit import pgemm


seed=42
torch.manual_seed(seed=seed)
torch.cuda.manual_seed_all(seed=seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def printmat(mat, name='mat'):
    print('mat:', name)
    for batch in range(mat.shape[0]):
        print(' batch:', batch)
        for i in range(mat.shape[1]):
            for j in range(mat.shape[2]):
                print(f'  {mat[batch][i][j].item():.2f}', end=' ')
            print()
        print()



A = (torch.rand(3, 3, 3, dtype=torch.float32, device="cuda") * 64) - 32
B = (torch.rand(3, 3, 3, dtype=torch.float32, device="cuda") * 64) - 32
C = torch.zeros_like(A)

D = pgemm(A, B, C, alpha=1.0, beta=1.0)


printmat(A, 'A')
printmat(B, 'B')
printmat(D, 'Posit-not')
printmat(A@B + C, 'Float32')



