#ifndef CONTEXT_H
#define CONTEXT_H

#include <cublas_v2.h>

#ifdef __cplusplus
extern "C" {
#endif

extern cublasHandle_t global_cublas_handle;

void init_framework();

void cleanup_framework();


// Helper function for chess sigmoid
#define SIGMOID_K 410.0f
__device__ inline float chess_sigmoid(float score) {
    return 1.0f / (1.0f + expf(-score / SIGMOID_K));
}

#ifdef __cplusplus
}
#endif

#endif