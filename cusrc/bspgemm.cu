// Batched Strided Posit GEMM
// Expects tensors of form (N, H, W), where N is the batch dimension
// For NCHW, or other shapes, use im2col (https://docs.pytorch.org/docs/stable/generated/torch.nn.Unfold.html),
// unsqueeze/squeeze, etc.

// TODO: Completely get rid of CUPOSIT_ENABLED later.
// If the user needs float arithmetic, they should use built-in
// arithmetic
// It's here right now to enable debugging
// note that it's also in mma_sm50.h

__constant__ unsigned CUPOSIT_ENABLED;
__constant__ unsigned CUPOSIT_EXP_MIN;
__constant__ unsigned CUPOSIT_EXP_MAX;
__constant__ unsigned CUPOSIT_NMANTISSA_MAX;


#include <torch/extension.h>
#include "cutlass/gemm/device/gemm_batched.h"
#include <cutlass/layout/matrix.h>


torch::Tensor bspgemm(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    float alpha,
    float beta
) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32 && C.dtype() == torch::kFloat32,
                "Only float32 supported");
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3 && C.dim() == 3, "Expected 3D tensors");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous(), "Tensors must be contiguous");

    int batch_count = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    TORCH_CHECK(B.size(0) == batch_count && B.size(1) == K, "B dimension mismatch");
    TORCH_CHECK(C.size(0) == batch_count && C.size(1) == M && C.size(2) == N, "C dimension mismatch");

    unsigned const host_cuposit_enabled = 1;
    unsigned const host_cuposit_exp_min = 127 - 64;
    unsigned const host_cuposit_exp_max = 127 + 64;
    unsigned const host_cuposit_nmantissa_max = 15;
    cudaMemcpyToSymbol(CUPOSIT_ENABLED, &host_cuposit_enabled, sizeof(unsigned));
    cudaMemcpyToSymbol(CUPOSIT_EXP_MIN, &host_cuposit_exp_min, sizeof(unsigned));
    cudaMemcpyToSymbol(CUPOSIT_EXP_MAX, &host_cuposit_exp_max, sizeof(unsigned));
    cudaMemcpyToSymbol(CUPOSIT_NMANTISSA_MAX, &host_cuposit_nmantissa_max, sizeof(unsigned));

    using Gemm = cutlass::gemm::device::GemmBatched<
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm89
    >;

    Gemm::Arguments args{
        {M, N, K},
        {A.data_ptr<float>(), K},
        A.stride(0),
        {B.data_ptr<float>(), N},
        B.stride(0),
        {C.data_ptr<float>(), N},
        C.stride(0),
        {C.data_ptr<float>(), N},
        C.stride(0),
        {alpha, beta},
        batch_count
    };

    Gemm gemm_op;
    
    cutlass::Status status = gemm_op.can_implement(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess, 
                "CUTLASS cannot implement this GEMM configuration");

    status = gemm_op.initialize(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "Failed to initialize CUTLASS GEMM");

    status = gemm_op();
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS GEMM kernel failed");

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bspgemm", &bspgemm, "Batched Strided Posit GEMM");
}