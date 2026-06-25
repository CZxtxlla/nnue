#include <cuda_runtime.h>
#include "../include/ops.h"
#include "../include/cuda_utils.h"
#include <cmath>

#define TILE_WIDTH 16

// ------- Helper Kernel -----------

__global__ void im2col_gpu_kernel(float* data_im, int channels, int height, 
    int width, int f_h, int f_w, int padding, int stride, int out_h, 
    int out_w, float* data_col, int total_elements) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < total_elements) {

        // unravel the 1D index into the im2col coordinates
        int w = i % out_w;
        int h = (i / out_w) % out_h;
        int c = (i / (out_w * out_h));

        
        int w_offset = c % f_w;
        int h_offset = c % f_h;
        int c_im = c / f_h / f_w;

        int im_row = h_offset + h * stride - padding;
        int im_col = w_offset + w * stride - padding;

        // boundary check
        if (im_row >= 0 && im_row < height && im_col >= 0 && im_col < width) {
            data_col[i] = data_im[(c_im * height + im_row) * width + im_col];
        } else {
            data_col[i] = 0.0f;
        }
    }
}

__global__ void special_matmul_kernel(float* A, float* B, float* bias, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = bias[row];

        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

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
/*
__global__ void conv2d_forward_kernel(
    float* input, float* weight, float* bias, float* out, 
    int batch_size, int in_c, int in_h, int in_w, 
    int out_c, int f_h, int f_w, int out_h, int out_w, 
    int stride, int padding, int total_elements) {

        int i = blockIdx.x * blockDim.x + threadIdx.x;

        // boundary check
        if (i < total_elements) {
            // unravel the 1d index into the 4d coors
            int ow = i % out_w;
            int oh = (i / out_w) % out_h;
            int oc = (i / (out_w * out_h)) % out_c;
            int b = i / (out_w * out_h * out_c);

            float val = bias[oc];

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

                            val += input[in_idx] * weight[w_idx];
                        }
                    }
                }
            }
            out[i] = val;
        }
} 
*/

__global__ void maxpool2d_forward_kernel(float* input, float* out, int* max_indices, int batch_size, int channels, int in_h, int in_w, int out_h, int out_w, int filter_size, int stride, int padding, int total_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // boundary check
    if (i < total_elements) {
        // unravel the 1d index into the 4d coors
        int ow = i % out_w;
        int oh = (i / out_w) % out_h;
        int c = (i / (out_w * out_h)) % channels;
        int b = i / (out_w * out_h * channels);

        float max_val = -1e20f;
        int max_idx = -1;

        // same as the cpu, just without the four outer loops (since those are done by the parallelism)
        for (int fh = 0; fh < filter_size; fh++) {
            for (int fw = 0; fw < filter_size; fw++) {
                int ih = oh * stride - padding + fh;
                int iw = ow * stride - padding + fw;

                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    // equivalent to [b, ic, ih, iw]
                    int in_idx = b * (channels * in_h * in_w) + c * (in_h * in_w) + ih * in_w + iw;

                    float val = input[in_idx];

                    if (val > max_val) {
                        max_val = val;
                        max_idx = in_idx;
                    }
                }
            }
        }
        out[i] = max_val;
        max_indices[i] = max_idx;
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
/*
void conv2d_gpu_forward(Tensor* input, Tensor* weight, Tensor* bias, Tensor* out, int stride, int padding) {
    int batch_size = input->shape[0];
    int in_c = input->shape[1]; // number of input channels (same as weight->shape[1])
    int in_h = input->shape[2];
    int in_w = input->shape[3]; // used for boundary checks when convoluting

    int out_c = weight->shape[0]; // number of outpit channels (filters)
    int f_h = weight->shape[2]; // height of filter
    int f_w = weight->shape[3]; // width of filter

    int out_h = out->shape[2]; // output image height
    int out_w = out->shape[3]; // output image width

    int total_elements = out->size;

    int threads = 256;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid((total_elements + threads - 1)/threads, 1, 1);

    conv2d_forward_kernel<<<dimGrid, dimBlock>>>(input->gpu_data, weight->gpu_data, bias->gpu_data, out->gpu_data, batch_size, in_c, in_h, in_w, out_c, f_h, f_w, out_h, out_w, stride, padding, total_elements);
    CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);
    return;

cleanup:
    exit(EXIT_FAILURE);
}

*/

void conv2d_gpu_forward(Tensor* input, Tensor* weight, Tensor* bias, Tensor* out, int stride, int padding) {
    int batch_size = input->shape[0];
    int in_c = input->shape[1]; // number of input channels (same as weight->shape[1])
    int in_h = input->shape[2];
    int in_w = input->shape[3]; // used for boundary checks when convoluting

    int out_c = weight->shape[0]; // number of outpit channels (filters)
    int f_h = weight->shape[2]; // height of filter
    int f_w = weight->shape[3]; // width of filter

    int out_h = out->shape[2]; // output image height
    int out_w = out->shape[3]; // output image width

    int M = out_c;
    int K = in_c * f_h * f_w;
    int N = out_h * out_w;

    float* d_im2col_space;
    cudaMalloc((void**)&d_im2col_space, K * N * sizeof(float));

    int im2col_elements = K * N;
    
    int threads = 256;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid((im2col_elements + threads - 1)/threads, 1, 1);

    dim3 dimBlockMatMul(16, 16, 1);
    dim3 dimGridMatMul((N + dimBlockMatMul.x - 1) / dimBlockMatMul.x, (M + dimBlockMatMul.y - 1) / dimBlockMatMul.y, 1);

    for (int b = 0; b < batch_size; b++) {
        float* current_input = input->gpu_data + b * (in_c * in_h * in_w);
        float* current_output = out->gpu_data + b * (out_c * out_h * out_w);

        im2col_gpu_kernel<<<dimGrid, dimBlock>>>(current_input, in_c, in_h, in_w, f_h, f_w, padding, stride, out_h, out_w, d_im2col_space, im2col_elements);

        special_matmul_kernel<<<dimGridMatMul, dimBlockMatMul>>>(weight->gpu_data, d_im2col_space, bias->gpu_data, current_output, M, K, N);
    }
    cudaFree(d_im2col_space);
    CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);
    return;

cleanup:
    exit(EXIT_FAILURE);
}

void maxpool2d_gpu_forward(Tensor* input, Tensor* out, int filter_size, int stride, int padding) {
    int batch_size = input->shape[0];
    int channels = input->shape[1]; // number of input channels (same as weight->shape[1])
    int in_h = input->shape[2];
    int in_w = input->shape[3]; // used for boundary checks when convoluting

    int out_h = out->shape[2]; // output image height
    int out_w = out->shape[3]; // output image width

    int total_elements = out->size;

    int threads = 256;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid((total_elements + threads - 1)/threads, 1, 1);

    maxpool2d_forward_kernel<<<dimGrid, dimBlock>>>(input->gpu_data, out->gpu_data, out->gpu_max_indices, batch_size, channels, in_h, in_w, out_h, out_w, filter_size, stride, padding, total_elements);
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