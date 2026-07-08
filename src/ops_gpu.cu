#include <cuda_runtime.h>
#include <cublas_v2.h> // for the cublasSgemm
#include "../include/ops.h"
#include "../include/cuda_utils.h"
#include "../include/context.cuh"
#include <math.h>

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

__global__ void clipped_leaky_relu_kernel(float* a, float* out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float val = a[i];
        out[i] = val > CLIPPED_RELU_MAX ? CLIPPED_RELU_MAX : (val > 0.0f ? val : 0.01f * val);
    }
}

__global__ void mse_forward_kernel(float* pred, float* target, float* out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float diff = pred[i] - target[i];

        atomicAdd(&out[0], (diff * diff) / size);
    }
}

__global__ void forward_sparse_linear_kernel(const int* d_inputs, const float* d_weights, const float* d_bias, float* d_out, int batch_size, int active_count, int hidden_dim) {
    int h = blockIdx.x * blockDim.x + threadIdx.x; // hidden neuron
    int b = blockIdx.y * blockDim.y + threadIdx.y; // batch

    if (b >= batch_size || h >= hidden_dim) return;

    float acc = d_bias[h];

    int batch_offset = b * active_count; // active count is the columns, get to the first element of the batch

    // loop through active features
    for (int a = 0; a < active_count; a++) {

        int feature_idx = d_inputs[batch_offset + a]; // index in the big boy matrix
        
        if (feature_idx < 0) continue; // -1 means there's no piece here

        // get the specific weight at col h and row feature_idx (dot product with input which is 1)
        acc += d_weights[feature_idx * hidden_dim + h];
    }

    // write to dense output tensor
    d_out[b * hidden_dim + h] = acc;
}

__global__ void forward_perspective_concat_kernel(const float* w_acc, const float* b_acc, const int* stm, float* out, int batch_size, int half_dim) {
    int h = blockIdx.x * blockDim.x + threadIdx.x; // hidden neuron in the concatenated layer
    int b = blockIdx.y * blockDim.y + threadIdx.y; // batch

    if (b >= batch_size || h >= half_dim * 2) return;

    int is_black_turn = stm[b]; // 0 for white, 1 for black
    float val;

    if (is_black_turn) {
        // black to move [black, white] concat
        if (h < half_dim) {
            val = b_acc[b * half_dim + h];
        } else {
            val = w_acc[b * half_dim + (h - half_dim)];
        }
    } else {
        // white to move [white, black] concat
        if (h < half_dim) {
            val = w_acc[b * half_dim + h];
        } else {
            val = b_acc[b * half_dim + (h - half_dim)];
        }
    }

    out[b * (half_dim * 2) + h] = val;
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

void clipped_leaky_relu_gpu_forward(Tensor* a, Tensor* out) {
    // helper to call the clipped leaky relu gpu kernel

    int threads = 256;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid((a->size + threads - 1)/threads, 1, 1);
    clipped_leaky_relu_kernel<<<dimGrid, dimBlock>>>(a->gpu_data, out->gpu_data, a->size);
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

void sparse_linear_gpu_forward(Tensor* inputs, Tensor* weights, Tensor* bias, Tensor* out) {
    if (!inputs->is_int_tensor || weights->is_int_tensor || bias->is_int_tensor) {
        fprintf(stderr, "Error: invalid tensor types.\n");
        exit(EXIT_FAILURE);
    }

    int batch_size = inputs->shape[0];
    int active_count = inputs->shape[1]; // 32 chess pieces
    int hidden_dim = weights->shape[1]; // 512

    int tx = 32; // 32 along hidden dim, 8 along batch_size
    int ty = 8;
    dim3 dimBlock(tx, ty);
    dim3 dimGrid((hidden_dim + tx - 1) / tx, (batch_size + ty - 1) / ty);

    forward_sparse_linear_kernel<<<dimGrid, dimBlock>>>(inputs->device_int_data, weights->gpu_data, bias->gpu_data, out->gpu_data, batch_size, active_count, hidden_dim);

    CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);

    return;
cleanup:
    exit(EXIT_FAILURE);
}

void perspective_concat_gpu_forward(Tensor* white_acc, Tensor* black_acc, Tensor* stm, Tensor* out) {
    int batch_size = white_acc->shape[0];
    int half_dim = white_acc->shape[1];

    int tx = 32; // 32 along hidden_dim, 8 along batch size
    int ty = 8;
    dim3 dimBlock(tx, ty);
    dim3 dimGrid((half_dim * 2 + tx - 1) / tx, (batch_size + ty - 1) / ty);

    forward_perspective_concat_kernel<<<dimGrid, dimBlock>>>(white_acc->gpu_data, black_acc->gpu_data, stm->device_int_data, out->gpu_data, batch_size, half_dim);

    CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);

    return;
cleanup:
    exit(EXIT_FAILURE);
}