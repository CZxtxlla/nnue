#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#define CUDA_CHECK_GOTO(call, label)                                   \
do {                                                       \
    cudaError_t err = call;                                \
                                                           \
    if (err != cudaSuccess) {                              \
        fprintf(stderr,                                    \
                "CUDA Error: %s\nFile: %s\nLine: %d\n",    \
                cudaGetErrorString(err),                   \
                __FILE__,                                  \
                __LINE__);                                 \
                                                           \
        goto label;                                      \
    }                                                      \
} while(0)
                                                        
#endif
