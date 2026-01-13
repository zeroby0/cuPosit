__constant__ unsigned CUPOSIT_ENABLED;
__constant__ unsigned CUPOSIT_EXP_MIN;
__constant__ unsigned CUPOSIT_EXP_MAX;
__constant__ unsigned CUPOSIT_NMANTISSA_MAX;

/////////////////////////////////////////////////////////////////////////////////////////////////
// https://maknee.github.io/blog/2025/Maybe-Consider-Putting-Cutlass-In-Your-CUDA-Kernels/
__forceinline__ __device__ unsigned lutmap(unsigned input) {
    // if input(exponent) is >= 0 (127), find difference from 0 (127) and divide by 4
    // if input(exponent) is < 0 (127), find difference from -1 (126) and divide by 4
    // and then subtract the result from the highest mantissa possible
    return CUPOSIT_NMANTISSA_MAX - (__usad(126 + (input >= 127), input, 0) >> 2);
}

__forceinline__ __device__ float posit_clip(float number) {
    // uint32_t bitmask = (0x807FFFFF & (0xFFFFFFFF << (23 - nmantissa)));
    // uint32_t xbits = __float_as_uint(x);
    // xbits = (xbits & bitmask) & (((uint32_t) exp) << 23);
    // return __uint_as_float(xbits);

    // overwrites exponent and truncates mantissa
    // to really round mantissa, it should be incremented when the
    // msb of cutoff bits is 1. but we aren't doing that here.

    // posit_* variables are available in constant memory
    unsigned x_exponent = (__float_as_uint(number) >> 23) & 0xFF; // TODO: see if frexpf is any faster
    x_exponent = min( max(x_exponent, CUPOSIT_EXP_MIN), CUPOSIT_EXP_MAX);

    // unset exponent bits and unneeded mantissa bits, and then copy exponent bits
    return __uint_as_float(
        (__float_as_uint(number) & (0x807FFFFF & (0xFFFFFFFF << (23 - lutmap(x_exponent))))) |
        ((x_exponent) << 23)
    );
}