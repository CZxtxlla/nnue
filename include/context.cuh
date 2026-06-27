#ifndef CONTEXT_H
#define CONTEXT_H

#include <cublas_v2.h>

#ifdef __cplusplus
extern "C" {
#endif

extern cublasHandle_t global_cublas_handle;

void init_framework();

void cleanup_framework();

#ifdef __cplusplus
}
#endif

#endif