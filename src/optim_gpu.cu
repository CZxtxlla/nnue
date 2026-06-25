#include "../include/optim.h"
#include <cuda_runtime.h>
#include "../include/cuda_utils.h"
#include <math.h>


// -------- kernels ------------

__global__ void sgd_step_kernel(float* data, float* grad, float lr, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        data[i] -= lr * grad[i];
    }
}

__global__ void adam_step_kernel(float* data, const float* grad, float* m, float* v, float lr, float beta1, float beta2, float eps, float c1, float c2, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        float g = grad[i];

        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * (g * g);

        float m_hat = m[i] / c1;
        float v_hat = v[i] / c2;

        data[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}


// ---------- launchers ------------

void sgd_gpu_step(Tensor* parameter, float lr) {
    if (parameter->gpu_grad != NULL) {
        int threads = 256;
        dim3 dimBlock(threads, 1, 1);
        dim3 dimGrid((parameter->size + threads - 1)/threads, 1, 1);

        sgd_step_kernel<<<dimGrid, dimBlock>>>(parameter->gpu_data, parameter->gpu_grad, lr, parameter->size);
        CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);
    }
    return;

cleanup:
    exit(EXIT_FAILURE);
}

void sgd_gpu_zero_grad(Tensor* parameter) {
    if (parameter->gpu_grad != NULL) {
        size_t bytes = parameter->size * sizeof(float);
        CUDA_CHECK_GOTO(cudaMemset(parameter->gpu_grad, 0, bytes), cleanup);
    }
    return;

cleanup:
    exit(EXIT_FAILURE);
}

void adam_gpu_step(Tensor* p, float* m, float* v, float lr, float beta1, float beta2, float eps, int t) {
    float correction1 = 1.0f - powf(beta1, t);
    float correction2 = 1.0f - powf(beta2, t);

    int threads = 256;
    int blocks = (p->size + threads - 1) / threads;

    adam_step_kernel<<<blocks, threads>>>(p->gpu_data, p->gpu_grad, m, v, lr, beta1, beta2, eps, correction1, correction2, p->size);
}

void adam_gpu_zero_grad(Tensor* param) {
    cudaMemset(param->gpu_grad, 0, param->size * sizeof(float));
} 