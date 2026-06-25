#include "../include/autograd.h"
#include <cuda_runtime.h>
#include "../include/cuda_utils.h"

#define TILE_WIDTH 16

__global__ void matmul_kernel(float* M, float* N, float* P, int j, int k, int l, unsigned int Mds_sz);


// ------------- Helper Kernels ----------------


__global__ void transpose_kernel(float* in, float* out, int width, int height) {
    // helper for backwards matmul, transposes matrix using tiling
    __shared__ float tile[TILE_WIDTH][TILE_WIDTH + 1];

    int in_col = blockIdx.x * blockDim.x + threadIdx.x;
    int in_row = blockIdx.y * blockDim.y + threadIdx.y;

    // load data into shared memory
    if (in_col < width && in_row < height) {
        tile[threadIdx.y][threadIdx.x] = in[in_row * width + in_col];
    }
    __syncthreads();

    int out_col = blockIdx.y * blockDim.y + threadIdx.x;
    int out_row = blockIdx.x * blockDim.x + threadIdx.y;

    if (out_col < height && out_row < width) {
        out[out_row * height + out_col] = tile[threadIdx.x][threadIdx.y];
    }
}


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

void backward_gpu_flatten(Tensor* t, Tensor* a) {
    if (!a->requires_grad) {
        return;
    }

    int threads = 256;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid((a->size + threads - 1)/threads, 1, 1);

    accumulate_kernel<<<dimGrid, dimBlock>>>(a->gpu_grad, t->gpu_grad, a->size);
    CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);

    return;

cleanup:
    exit(EXIT_FAILURE);
}

__global__ void backward_conv2d_kernel(float* t_grad, float* input, float* weight, float* in_grad, 
    float* weight_grad, float* bias_grad, int batch_size, int in_c, int in_h, int in_w, int out_c, 
    int f_h, int f_w, int out_h, int out_w, int stride, int padding, int total_elements) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // boundary check
    if (i < total_elements) {
        // unravel the 1d index into the 4d coors
        int ow = i % out_w;
        int oh = (i / out_w) % out_h;
        int oc = (i / (out_w * out_h)) % out_c;
        int b = i / (out_w * out_h * out_c);

        float grad_out = t_grad[i];

        if (bias_grad != NULL) {
            atomicAdd(&bias_grad[oc], grad_out);
        }

        // same as the cpu, just without the four outer loops (since those are done by the parallelism)
        for (int ic = 0; ic < in_c; ic++) {
            for (int fh = 0; fh < f_h; fh++) {
                for (int fw = 0; fw < f_w; fw++) {
                    int ih = oh * stride - padding + fh;
                    int iw = ow * stride - padding + fw;

                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                        // equivalent to [b, ic, ih, iw]
                        int in_idx = b * (in_c * in_h * in_w) + ic * (in_h * in_w) + ih * in_w + iw;
                        // equivalent to [oc, ic, fh, fw]
                        int w_idx = oc * (in_c * f_h * f_w) + ic * (f_h * f_w) + fh * f_w + fw;

                        if (in_grad != NULL) {
                            atomicAdd(&in_grad[in_idx], weight[w_idx] * grad_out);
                        }
                        if (weight_grad != NULL) {
                            atomicAdd(&weight_grad[w_idx], input[in_idx] * grad_out);
                        }
                    }
                }
            }
        }
    }
}

__global__ void backward_maxpool2d_kernel(float* t_grad, int* max_indices, float* input_grad, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        int max_idx = max_indices[i];
        if (max_idx >= 0) {
            atomicAdd(&input_grad[max_idx], t_grad[i]);
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

    unsigned int Mds_sz = TILE_WIDTH * TILE_WIDTH * sizeof(float);
    size_t shared_mem_size = Mds_sz * 2;

    if (a->requires_grad) {

        float *B_T, *gradA_temp;
        CUDA_CHECK_GOTO(cudaMalloc((void**)&B_T, k * l * sizeof(float)), cleanup);
        CUDA_CHECK_GOTO(cudaMalloc((void**)&gradA_temp, j * k * sizeof(float)), cleanup);

        // transpose B
        dim3 dimGridB((l + TILE_WIDTH - 1)/TILE_WIDTH, (k + TILE_WIDTH - 1)/TILE_WIDTH, 1);
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
        transpose_kernel<<<dimGridB, dimBlock>>>(b->gpu_data, B_T, l, k);


        // perform matmul
        dim3 dimGridMatmulA((k + TILE_WIDTH - 1)/TILE_WIDTH, (j + TILE_WIDTH - 1)/TILE_WIDTH, 1);
        matmul_kernel<<<dimGridMatmulA, dimBlock, shared_mem_size>>>(t->gpu_grad, B_T, gradA_temp, j, l, k, Mds_sz);

        // accumulate into the gradient
        int threads = 256;
        dim3 dimGridAccA((a->size + threads - 1)/ threads, 1, 1);
        accumulate_kernel<<<dimGridAccA, threads>>>(a->gpu_grad, gradA_temp, a->size);
        
        cudaFree(B_T);
        cudaFree(gradA_temp);
    }
    if (b->requires_grad) {
        float *A_T, *gradB_temp;
        CUDA_CHECK_GOTO(cudaMalloc((void**)&A_T, j * k * sizeof(float)), cleanup);
        CUDA_CHECK_GOTO(cudaMalloc((void**)&gradB_temp, k * l * sizeof(float)), cleanup);

        // transpose B
        dim3 dimGridA((k + TILE_WIDTH - 1)/TILE_WIDTH, (j + TILE_WIDTH - 1)/TILE_WIDTH, 1);
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
        transpose_kernel<<<dimGridA, dimBlock>>>(a->gpu_data, A_T, k, j);


        // perform matmul
        dim3 dimGridMatmulB((l + TILE_WIDTH - 1)/TILE_WIDTH, (k + TILE_WIDTH - 1)/TILE_WIDTH, 1);
        matmul_kernel<<<dimGridMatmulB, dimBlock, shared_mem_size>>>(A_T, t->gpu_grad, gradB_temp, k, j, l, Mds_sz);

        // accumulate into the gradient
        int threads = 256;
        dim3 dimGridAccB((b->size + threads - 1)/ threads, 1, 1);
        accumulate_kernel<<<dimGridAccB, threads>>>(b->gpu_grad, gradB_temp, b->size);
        
        cudaFree(A_T);
        cudaFree(gradB_temp);
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

void backward_gpu_conv2d(Tensor* t, Tensor* input, Tensor* weight, Tensor* bias) {

    // standard parameters
    int batch_size = input->shape[0];
    int in_c = input->shape[1];
    int in_h = input->shape[2];
    int in_w = input->shape[3];

    int out_c = weight->shape[0];
    int f_h = weight->shape[2];
    int f_w = weight->shape[3];

    int out_h = t->shape[2];
    int out_w = t->shape[3];

    int total_elements = t->size;

    // used to check if grad is required
    float* in_grad = input->requires_grad ? input->gpu_grad : NULL;
    float* w_grad = weight->requires_grad ? weight->gpu_grad : NULL;
    float* b_grad = bias->requires_grad ? bias->gpu_grad : NULL;

    int threads = 256;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid((t->size + threads - 1) / threads, 1, 1);

    backward_conv2d_kernel<<<dimGrid, dimBlock>>>(t->gpu_grad, input->gpu_data, 
        weight->gpu_data, in_grad, w_grad, b_grad, batch_size, in_c, in_h, in_w, 
        out_c, f_h, f_w, out_h, out_w, t->stride, t->padding, total_elements);

    CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);
    return;

cleanup:
    exit(EXIT_FAILURE);
}

void backward_gpu_maxpool2d(Tensor* t, Tensor* input) {
    if (!input ->requires_grad) {
        return;
    }
    int threads = 256;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid((t->size + threads - 1) / threads, 1, 1);

    backward_maxpool2d_kernel<<<dimGrid, dimBlock>>>(t->gpu_grad, t->gpu_max_indices, input->gpu_grad, t->size);

    CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);
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