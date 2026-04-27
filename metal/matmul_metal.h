#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// computes C = A * B
// A is (M x K), B is (K x N), C is (M x N)
void metal_matmul(
    const float* A,
    const float* B,
    float*       C,
    uint32_t     M,
    uint32_t     N,
    uint32_t     K
);

#ifdef __cplusplus
}
#endif
