import torch
import time
import numpy as np
from typing import Dict, List, Tuple
import cuposit

def benchmark_bspgemm(
    batch_sizes: List[int] = [1, 4, 16, 64],
    M_sizes: List[int] = [128, 256, 512, 1024],
    N_sizes: List[int] = [128, 256, 512, 1024],
    K_sizes: List[int] = [128, 256, 512, 1024],
    posit_n: int = 16,
    posit_es: int = 2,
    warmup_iters: int = 10,
    bench_iters: int = 100,
) -> Dict:
    results = []
    
    for batch in batch_sizes:
        for M in M_sizes:
            for N in N_sizes:
                for K in K_sizes:
                    A = torch.randn(batch, M, K, dtype=torch.float32, device='cuda')
                    B = torch.randn(batch, K, N, dtype=torch.float32, device='cuda')
                    C = torch.zeros(batch, M, N, dtype=torch.float32, device='cuda')
                    
                    for _ in range(warmup_iters):
                        cuposit.bspgemm(A, B, C, 1.0, 0.0, (posit_n, posit_es))
                    
                    torch.cuda.synchronize()
                    
                    start = time.perf_counter()
                    for _ in range(bench_iters):
                        cuposit.bspgemm(A, B, C, 1.0, 0.0, (posit_n, posit_es))
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                    
                    elapsed = (end - start) / bench_iters
                    flops = 2 * batch * M * N * K
                    gflops = (flops / elapsed) / 1e9
                    
                    results.append({
                        'batch': batch,
                        'M': M, 'N': N, 'K': K,
                        'time_ms': elapsed * 1000,
                        'gflops': gflops,
                        'memory_mb': (A.numel() + B.numel() + C.numel()) * 4 / 1e6
                    })
                    
                    print(f"B={batch:3d} M={M:4d} N={N:4d} K={K:4d} | "
                          f"{elapsed*1000:7.3f}ms | {gflops:8.2f} GFLOPS")
    
    return results


def compare_with_torch(
    batch: int = 16,
    M: int = 512,
    N: int = 512,
    K: int = 512,
    posit_n: int = 16,
    posit_es: int = 2,
    iters: int = 100
):    
    A = torch.randn(batch, M, K, device='cuda')
    B = torch.randn(batch, K, N, device='cuda')
    C_posit = torch.zeros(batch, M, N, device='cuda')
    C_torch = torch.zeros(batch, M, N, device='cuda')
    
    for _ in range(10):
        cuposit.bspgemm(A, B, C_posit, 1.0, 0.0, (posit_n, posit_es))
        torch.bmm(A, B, out=C_torch)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        cuposit.bspgemm(A, B, C_posit, 1.0, 0.0, (posit_n, posit_es))
    torch.cuda.synchronize()
    posit_time = time.perf_counter() - start
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        torch.bmm(A, B, out=C_torch)
    torch.cuda.synchronize()
    torch_time = time.perf_counter() - start
    
    flops = 2 * batch * M * N * K * iters
    
    print(f"\n{'='*60}")
    print(f"Comparison: batch={batch}, M={M}, N={N}, K={K}")
    print(f"{'='*60}")
    print(f"Posit GEMM:  {posit_time*1000/iters:.3f}ms | {flops/posit_time/1e9:.2f} GFLOPS")
    print(f"Torch BMM:   {torch_time*1000/iters:.3f}ms | {flops/torch_time/1e9:.2f} GFLOPS")
    print(f"Speedup:     {torch_time/posit_time:.3f}x")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    print("Benchmarking Batched Strided Posit GEMM\n")
    
    # results = benchmark_bspgemm(
    #     batch_sizes=[1, 8, 32],
    #     M_sizes=[256, 512, 1024],
    #     N_sizes=[256, 512, 1024],
    #     K_sizes=[256, 512, 1024],
    #     warmup_iters=5,
    #     bench_iters=50
    # )
    
    compare_with_torch(batch=16, M=1024, N=1024, K=1024, iters=100)
    
    # print("\nResults summary:")
    # results_arr = np.array([(r['batch'], r['M'], r['N'], r['K'], r['gflops']) 
    #                         for r in results])
    # best_idx = np.argmax(results_arr[:, -1])
    # best = results[best_idx]
    # print(f"Best: B={best['batch']} M={best['M']} N={best['N']} K={best['K']} "
    #       f"| {best['gflops']:.2f} GFLOPS")