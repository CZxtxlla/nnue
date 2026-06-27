#include <cuda_runtime.h>
#include <cublas_v2.h> // for the cublasSgemm
#include "../include/ops.h"
#include "../include/cuda_utils.h"
#include "../include/context.cuh"
#include <cmath>

// define global handle
cublasHandle_t global_cublas_handle;


extern "C" void init_framework() {
    cublasCreate(&global_cublas_handle);
}

extern "C" void cleanup_framework() {
    cublasDestroy(global_cublas_handle);
}

#define TILE_WIDTH 16

// ------------- Kernels ----------------

__global__ void add_kernel(float* a, float* b, float* out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = a[i] + b[i];
    }
}

__global__ void mul_kernel(float* a, float* b, float* out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = a[i] * b[i];
    }
}

__global__ void bias_kernel(float* a, float* bias, float* out, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int index = row * width + col;
        out[index] = a[index] + bias[col];
    }
}

__global__ void relu_kernel(float* a, float* out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = a[i] > 0 ? a[i] : 0.0f;
    }
}

__global__ void mse_forward_kernel(float* pred, float* target, float* out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float diff = pred[i] - target[i];

        atomicAdd(&out[0], (diff * diff) / size);
    }
}

__global__ void cross_entropy_forward_kernel(float* pred, float* target, float* out, int batch_size, int num_classes) {
    int b = blockIdx.x * blockDim.x + threadIdx.x; // index in batch

    if (b < batch_size) {
        int offset = b * num_classes;

        float max_val = pred[offset];
        for (int c = 1; c < num_classes; c++) {
            if (pred[offset + c] > max_val) {
                max_val = pred[offset + c];
            }
        }

        float exp_sum = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            float e = expf(pred[offset + c] - max_val);
            pred[offset + c] = e;
            exp_sum += e;
        }

        float loss = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            float prob = pred[offset + c] / exp_sum;
            pred[offset + c] = prob;
            if (target[offset + c] == 1.0f) {
                loss -= logf(prob + 1e-7f);
            }
        }

        atomicAdd(&out[0], loss / batch_size);
    }
}


// ------------- Helpers --------------- 


void add_gpu_forward(Tensor* a, Tensor* b, Tensor* out) {
    // helper to call the gpu add kernel

    int threads = 256;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid((a->size + threads - 1)/threads, 1, 1);
    add_kernel<<<dimGrid, dimBlock>>>(a->gpu_data, b->gpu_data, out->gpu_data, a->size);
    CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);

    return;

cleanup:
    exit(EXIT_FAILURE);
}

void mul_gpu_forward(Tensor* a, Tensor* b, Tensor* out) {
    // helper to call the gpu mul kernel

    int threads = 256;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid((a->size + threads - 1)/threads, 1, 1);
    mul_kernel<<<dimGrid, dimBlock>>>(a->gpu_data, b->gpu_data, out->gpu_data, a->size);
    CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);

    return;

cleanup:
    exit(EXIT_FAILURE);
}

void bias_gpu_forward(Tensor* a, Tensor* bias, Tensor* out) {
    // helper to call the gpu bias addition kernel

    int width = a->shape[1];
    int height = a->shape[0];
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid((width + 15)/16, (height + 15)/16, 1);
    bias_kernel<<<dimGrid, dimBlock>>>(a->gpu_data, bias->gpu_data, out->gpu_data, width, height);
    CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);

    return;

cleanup:
    exit(EXIT_FAILURE);
}

void matmul_gpu_forward(Tensor* a, Tensor* b, Tensor* out) {
    // helper to call the matmul gpu kernel

    int m = a->shape[0]; // rows of a
    int k = a->shape[1]; // cols of A / rows of B
    int n = b->shape[1]; // cols of B

    // C = alpha * (A * B) + beta * C
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // perform B * A (this is a trick because cublas performs column major matrix multi and the matrices are stored in row major)
    cublasStatus_t status = cublasSgemm(global_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b->gpu_data, n, a->gpu_data, k, &beta, out->gpu_data, n);

    if (status != CUBLAS_STATUS_SUCCESS) {
        // You can integrate this with your custom CUDA_CHECK_GOTO macro
        printf("cuBLAS SGEMM failed\n");
        goto cleanup;
    }

    return;

cleanup:
    exit(EXIT_FAILURE);
}

void relu_gpu_forward(Tensor* a, Tensor* out) {
    // helper to call the relu gpu kernel

    int threads = 256;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid((a->size + threads - 1)/threads, 1, 1);
    relu_kernel<<<dimGrid, dimBlock>>>(a->gpu_data, out->gpu_data, a->size);
    CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);

    return;

cleanup:
    exit(EXIT_FAILURE);
}

void mse_gpu_forward(Tensor* pred, Tensor* target, Tensor* out) {
    cudaMemset(out->gpu_data, 0, sizeof(float));

    int threads = 256;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid((pred->size + threads - 1)/threads, 1, 1);

    mse_forward_kernel<<<dimGrid, dimBlock>>>(pred->gpu_data, target->gpu_data, out->gpu_data, pred->size);
    CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);
    return;

cleanup:
    exit(EXIT_FAILURE);
}

void cross_entropy_gpu_forward(Tensor* pred, Tensor* target, Tensor* out) {
    cudaMemset(out->gpu_data, 0, sizeof(float));

    int batch_size = pred->shape[0];
    int num_classes = pred->shape[1];

    int threads = 256;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid((batch_size + threads - 1)/threads, 1, 1);

    cross_entropy_forward_kernel<<<dimGrid, dimBlock>>>(pred->gpu_data, target->gpu_data, out->gpu_data, batch_size, num_classes);
    CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);

    return;
cleanup:
    exit(EXIT_FAILURE);

}