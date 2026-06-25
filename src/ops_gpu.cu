#include <cuda_runtime.h>
#include "../include/ops.h"
#include "../include/cuda_utils.h"
#include <cmath>

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

__global__ void matmul_kernel(float* M, float* N, float* P, int j, int k, int l, unsigned int Mds_sz) {

    //allocate space for tiles in shared memory
    extern __shared__ char Mds_Nds[];
    float* Mds = (float*) Mds_Nds;
    float* Nds = (float*) (Mds_Nds + Mds_sz); // Mds_sz is in bytes

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // row and column of the P value we are computing (TILE_WIDTH is also the block width, i.e. blockDim.x)
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    // Loop over tiles required to compute this P element, ph is the phase index
    float Pvalue = 0;
    for (int ph = 0; ph < (k + TILE_WIDTH - 1) /TILE_WIDTH; ++ph) {
        // load this specific tile (each thread in the block loads their respective element)
        if ((row < j) && (ph * TILE_WIDTH + tx) < k) {
            Mds[ty * TILE_WIDTH + tx] = M[row * k + ph * TILE_WIDTH + tx];
        } else{
            Mds[ty * TILE_WIDTH + tx] = 0.0f;
        }
        if ((ph * TILE_WIDTH + ty) < k && col < l) {
            Nds[ty * TILE_WIDTH + tx] = N[(ph * TILE_WIDTH + ty) * l + col];
        } else {
            Nds[ty * TILE_WIDTH + tx] = 0.0f;
        }
        __syncthreads(); // wait until every thread in the block has loaded its own element of the tiled matrix

        for (int i = 0; i < TILE_WIDTH; ++i) {
            Pvalue += Mds[ty * TILE_WIDTH + i] * Nds[i * TILE_WIDTH + tx]; // perform the dot product
        }
        __syncthreads(); // wait until all dot products are done before loading the next tile
    }
    if (row < j && col < l) {
        P[row * l + col] = Pvalue;
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

    int j = a->shape[0];
    int k = a->shape[1];
    int l = b->shape[1];


    unsigned int Mds_sz = TILE_WIDTH * TILE_WIDTH * sizeof(float);
    unsigned int Nds_sz = TILE_WIDTH * TILE_WIDTH * sizeof(float);

    size_t size = Mds_sz + Nds_sz;

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid((l + TILE_WIDTH - 1)/TILE_WIDTH, (j + TILE_WIDTH -1)/TILE_WIDTH, 1);
    matmul_kernel<<<dimGrid, dimBlock, size>>>(a->gpu_data, b->gpu_data, out->gpu_data, j, k, l, Mds_sz);
    CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);

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