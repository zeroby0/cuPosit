#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <cmath>

// Function to validate posit_clip function
// at the moment, it rounds a number to the nearest posit
// smaller in magnitude than the input number
// Also, this can't be used to measure the speed
// of the function because it's extremely memory bound,
// thus speed-ups in the kernel wont be visible.

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \n", \
                __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__constant__ unsigned CUPOSIT_ENABLED;
__constant__ unsigned CUPOSIT_EXP_MIN;
__constant__ unsigned CUPOSIT_EXP_MAX;
__constant__ unsigned CUPOSIT_NMANTISSA_MAX;

__device__ __forceinline__ uint32_t cutlass_lutmap(uint32_t input) {
    // if input(exponent) is >= 0 (127), find difference from 0 (127) and divide by 4
    // if input(exponent) is < 0 (127), find difference from -1 (126) and divide by 4
    // and then subtract the result from the highest mantissa possible
    return CUPOSIT_NMANTISSA_MAX - (__usad(126 + (input >= 127), input, 0) >> 2);
}

__device__ float posit_clip(float number) {
    // uint32_t bitmask = (0x807FFFFF & (0xFFFFFFFF << (23 - nmantissa)));
    // uint32_t xbits = __float_as_uint(x);
    // xbits = (xbits & bitmask) & (((uint32_t) exp) << 23);
    // return __uint_as_float(xbits);

    // overwrites exponent and truncates mantissa
    // to really round mantissa, it should be incremented when the
    // msb of cutoff bits is 1. but we aren't doing that here.

    // posit_* variables are available in constant memory
    uint32_t x_exponent = (__float_as_uint(number) >> 23) & 0xFF; // TODO: see if frexpf is any faster
    x_exponent = min( max(x_exponent, CUPOSIT_EXP_MIN), CUPOSIT_EXP_MAX);

    // unset exponent bits and unneeded mantissa bits, and then copy exponent bits
    return __uint_as_float(
        (__float_as_uint(number) & (0x807FFFFF & (0xFFFFFFFF << (23 - cutlass_lutmap(x_exponent))))) |
        ((x_exponent) << 23)
    );
} 

__global__ void kernel_posit_clip(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = posit_clip(in[idx]);
        // out[idx] = in[idx] > 0 ? in[idx] : 0.0f;
    }
}


int main(int argc, char** argv) {
    const int N = 1024 * 1024 * 4; 
    size_t bytes = N * sizeof(float);

    if(argc != 2) {
        std::cout << "hey!" << std::endl;
        exit(-1);
    }

    float input = std::stof(argv[1]);
    int n;
    float m;
    m = frexpf(input, &n);
    std::cout << "number: \t" << input << std::endl;
    std::cout << "exponent: \t" << n-1 << std::endl;
    std::cout << "mantissa: \t" << m*2 << std::endl << std::endl;


    // std::cout << argc << std::endl;

    std::cout << "Benchmarking ReLU on " << N << " elements (" << (bytes / 1024.0 / 1024.0) << " MB)..." << std::endl;
    unsigned const host_cuposit_enabled = 1; // posit(6, 2)
    unsigned const host_cuposit_exp_min = 127 - 7;
    unsigned const host_cuposit_exp_max = 127 + 7;
    unsigned const host_cuposit_nmantissa_max = 1;
    cudaMemcpyToSymbol(CUPOSIT_ENABLED, &host_cuposit_enabled, sizeof(unsigned));
    cudaMemcpyToSymbol(CUPOSIT_EXP_MIN, &host_cuposit_exp_min, sizeof(unsigned));
    cudaMemcpyToSymbol(CUPOSIT_EXP_MAX, &host_cuposit_exp_max, sizeof(unsigned));
    cudaMemcpyToSymbol(CUPOSIT_NMANTISSA_MAX, &host_cuposit_nmantissa_max, sizeof(unsigned));

    // Allocate Host Memory
    std::vector<float> h_in(N);
    std::vector<float> h_out_ref(N);
    
    // Random data
    // for (int i = 0; i < N; ++i) {
    //     h_in[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f; // Range [-0.5, 0.5]
    // }

    // Constant data
    for (int i = 0; i < N; ++i) {
        // h_in[i] = 23.92f;
        h_in[i] = input;
    }

    // Allocate Device Memory
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    // Copy to Device
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    // Tuning Parameters
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Create Events for Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float milliseconds = 0;
    int iterations = 100;

    // Warmup and copy data to host
    kernel_posit_clip<<<numBlocks, blockSize>>>(d_out, d_in, N);
    CUDA_CHECK(cudaMemcpy(h_out_ref.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        kernel_posit_clip<<<numBlocks, blockSize>>>(d_out, d_in, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    double avg_time_naive = milliseconds / iterations;
    double bw_naive = (2.0 * bytes) / (avg_time_naive / 1000.0) / 1e9; 
    
    std::cout << "Runtime:     " << avg_time_naive << " ms | Bandwidth: " << bw_naive << " GB/s" << std::endl;

    std::cout << h_in[0]      << "\t" << h_in[19]      << "\t" << h_in[34]      << "\t" << h_in[61]      << std::endl;
    std::cout << h_out_ref[0] << "\t" << h_out_ref[19] << "\t" << h_out_ref[34] << "\t" << h_out_ref[61] << std::endl;

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}