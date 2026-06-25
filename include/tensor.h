#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    DEVICE_CPU,
    DEVICE_GPU
} DeviceType;

typedef enum {
    OP_NONE,
    OP_ADD,
    OP_MUL,
    OP_MATMUL,
    OP_RELU,
    OP_FLATTEN,
    OP_ADDBIAS,
    OP_CONV2D,
    OP_MAXPOOL2D,
    OP_MSE,
    OP_CROSS_ENTROPY
} opType;

typedef struct Tensor {
    // CPU memory pointers
    float* cpu_data;
    float* cpu_grad;

    // GPU memory pointers 
    float* gpu_data;
    float* gpu_grad;

    DeviceType device; // either DEVICE_CPU or DEVICE_GPU

    int visited_pass_id; // needed for the topological sort

    bool is_view; // true if it doesn't own its data, it is looking at another tensor's data

    int* shape; // shape of the tensor, i.e. [64, 128] for 2D matrix with 64 elements in column, 128 in row
    int ndims; // Number of dimensions
    size_t size; // total number of elements

    // Autograd graph data
    bool requires_grad; 
    struct Tensor** parents; // list of pointers to parent tensors
    int num_parents; // greater than or equal to 1
    opType op; // type of operation used to create this tensor from the parent tensor(s)

    // needed for backward pass through convolutional layer
    int stride;
    int padding;

    int* max_indices; // CPU max-pool argmax cache
    int* gpu_max_indices; // GPU max-pool argmax cache
} Tensor;


Tensor* create_tensor(int* shape, int ndims, DeviceType device, bool requires_grad);
void free_tensor(Tensor* t);

void tensor_to_device(Tensor* t, DeviceType device); // sends tensor to be stored on specified device

float tensor_scalar_value(Tensor* t); // get the value of the first element of the tensor (used to get value of the loss)
void seed_loss_grad(Tensor* loss); // set the loss gradient to 1

Tensor* tensor_slice_view(Tensor* master, int start_row, int num_rows); // generates view tensor of portion of master tensor

void tensor_download_data(Tensor* t, float* dest); // helper for training

#ifdef __cplusplus
}
#endif

#endif