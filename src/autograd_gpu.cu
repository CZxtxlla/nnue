#include "../include/autograd.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../include/cuda_utils.h"
#include "../include/context.cuh"


#define TILE_WIDTH 16


// ------------- Helper Kernels ----------------


__global__ void accumulate_kernel(float* target, float* source, int size) {
    // helper
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        atomicAdd(&target[i], source[i]);
    }
}


// ------------- Main Kernels ----------------


__global__ void backward_add_kernel(float* t_grad, float* a_grad, float* b_grad, int size) {
    // takes both grad arrays for the tensor and a parent and passes on the gradient according to addition
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        if (a_grad != NULL) {
            atomicAdd(&a_grad[i], t_grad[i]);
        }
        if (b_grad != NULL) {
            atomicAdd(&b_grad[i], t_grad[i]);
        } 
    }
}

__global__ void backward_mul_kernel(float* t_grad, float* a_data, float* b_data, float* a_grad, float* b_grad, int size) {
    // takes both grad arrays for the tensor and a parent and passes on the gradient according to addition
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        if (a_grad != NULL) {
            atomicAdd(&a_grad[i], t_grad[i] * b_data[i]);
        }
        if (b_grad != NULL) {
            atomicAdd(&b_grad[i], t_grad[i] * a_data[i]);
        } 
    }
}

__global__ void backward_add_bias_kernel(float* t_grad, float* bias_grad,int batch_size, int features) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // chooses the column

    if (i < features) {
        if (bias_grad != NULL) {
            float sum = 0.0f;

            for (int j = 0; j < batch_size; j++) {
                sum += t_grad[j * features + i]; 
            }

            atomicAdd(&bias_grad[i], sum);
        }
    }

}

__global__ void backward_relu_kernel(float* t_grad, float* a_grad, float* a_data, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        if (a_grad != NULL) {
            float grad = a_data[i] > 0.0f ? t_grad[i] : 0.0f;
            atomicAdd(&a_grad[i], grad);
        }
    }
}

__global__ void mse_backward_kernel(float* pred, float* target, float* pred_grad, float* target_grad, float* out_grad, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float diff = pred[i] - target[i];
        float scale = 2.0f / size;
        
        if (pred_grad != NULL) {
            atomicAdd(&pred_grad[i], scale * diff * out_grad[0]);
        }
        if (target_grad != NULL) {
            atomicAdd(&target_grad[i], -scale * diff * out_grad[0]);
        }
    }
}

__global__ void cross_entropy_backward_kernel(float* pred, float* target, float* pred_grad, float* out_grad, int size, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        pred_grad[i] += (pred[i] - target[i]) * (out_grad[0] / batch_size);
    }
}



// -------------- Helpers ---------------

void backward_gpu_add(Tensor* t, Tensor* a, Tensor* b) {
    // calls the kernel to pass the gradient backwards on the gpu
    int threads = 256;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid((a->size + threads - 1)/threads, 1, 1);
    
    backward_add_kernel<<<dimGrid, dimBlock>>>(t->gpu_grad, a->gpu_grad, b->gpu_grad, t->size);
    CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);
    
    return;

cleanup:
    exit(EXIT_FAILURE);
}

void backward_gpu_mul(Tensor* t, Tensor* a, Tensor* b) {
    // calls the kernel to pass the gradient backwards on the gpu
    int threads = 256;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid((a->size + threads - 1)/threads, 1, 1);
    
    backward_mul_kernel<<<dimGrid, dimBlock>>>(t->gpu_grad, a->gpu_data, b->gpu_data, a->gpu_grad, b->gpu_grad, t->size);
    CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);
    
    return;

cleanup:
    exit(EXIT_FAILURE);
}

void backward_gpu_add_bias(Tensor* t, Tensor* a, Tensor* bias) {
    if (a->requires_grad) {
        int threads = 256;
        dim3 dimBlockA(threads, 1, 1);
        dim3 dimGridA((a->size + threads - 1)/threads, 1, 1);

        accumulate_kernel<<<dimGridA, dimBlockA>>>(a->gpu_grad, t->gpu_grad, a->size);
        CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);
    }
    if (bias->requires_grad) {
        int batch_size = a->shape[0];
        int features = a->shape[1];

        int threads = 256;

        dim3 dimBlockBias(threads, 1, 1);
        dim3 dimGridBias((features + threads - 1)/threads, 1, 1);

        backward_add_bias_kernel<<<dimGridBias, dimBlockBias>>>(t->gpu_grad, bias->gpu_grad, batch_size, features);
        CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);
    }

    return;

cleanup:
    exit(EXIT_FAILURE);
}



void backward_gpu_matmul(Tensor* t, Tensor* a, Tensor* b) {
    int j = a->shape[0];
    int k = a->shape[1];
    int l = b->shape[1];

    const float alpha = 1.0f;
    const float beta = 1.0f; // accumulate

    if (a->requires_grad) {

        cublasStatus_t status = cublasSgemm(global_cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, k, j, l, &alpha, b->gpu_data, l, t->gpu_grad, l, &beta, a->gpu_grad, k);
        if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    }
    if (b->requires_grad) {
        cublasStatus_t status = cublasSgemm(global_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, l, k, j, &alpha, t->gpu_grad, l, a->gpu_data, k, &beta, b->gpu_grad, l);
        if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    }
    return;

cleanup: 
    exit(EXIT_FAILURE);
}

void backward_gpu_relu(Tensor* t, Tensor* a) {
    if (a->requires_grad) {

        int threads = 256;
        dim3 dimBlock(threads, 1, 1);
        dim3 dimGrid((t->size + threads - 1)/threads, 1, 1);
        backward_relu_kernel<<<dimGrid, dimBlock>>>(t->gpu_grad, a->gpu_grad, a->gpu_data, t->size);
        CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);
    }

    return;

cleanup:
    exit(EXIT_FAILURE);
}

void backward_gpu_mse(Tensor* t, Tensor* pred, Tensor* target) {
    int threads = 256;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid((pred->size + threads - 1)/threads, 1, 1);

    mse_backward_kernel<<<dimGrid, dimBlock>>>(pred->gpu_data, target->gpu_data, pred->gpu_grad, target->gpu_grad, t->gpu_grad, pred->size);
    CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);
    return;

cleanup:
    exit(EXIT_FAILURE);
}

void backward_gpu_cross_entropy(Tensor* t, Tensor* pred, Tensor* target) {
    int threads = 256;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid((pred->size + threads - 1)/threads, 1, 1);

    cross_entropy_backward_kernel<<<dimGrid, dimBlock>>>(pred->gpu_data, target->gpu_data, pred->gpu_grad, t->gpu_grad, pred->size, pred->shape[0]);
    CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);
    return;

cleanup:
    exit(EXIT_FAILURE);
}