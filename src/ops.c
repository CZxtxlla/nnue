#include "../include/ops.h"
#include <stdio.h>
#include <stdlib.h>

typedef int cudaError_t;
#define cudaSuccess 0
extern cudaError_t cudaMallocManaged(void** devPtr, size_t size);


// The following code acts as a dispatcher. It handles the safety checks as well as the
// stuff for autograd, but the actual calculations are sent off to either the cpu or gpu.
// Those calculations are contained in ops_cpu and ops_gpu respectively.

Tensor* tensor_add(Tensor* a, Tensor* b) {
    if (a->device != b->device) {
        fprintf(stderr, "Error: Cannot add tensors on different devices.\n");
        return NULL;
    }

    if (a->ndims != b->ndims) {
        fprintf(stderr, "Error: Tensors must have the same number of dimensions for addition.\n");
        return NULL;
    }
    for (int i = 0; i < a->ndims; i++) {
        if (a->shape[i] != b->shape[i]) {
            fprintf(stderr, "Error: Tensors must have same shape for addition.\n");
            return NULL;
        }
    }

    // Create output tensor
    Tensor* out = create_tensor(a->shape, a->ndims, a->device, a->requires_grad || b->requires_grad, 0);

    // Perform calculation on proper device
    if (a->device == DEVICE_CPU) {
        fprintf(stderr, "Error: tensors must be on the gpu.\n");
        return NULL;
    } else if (a->device == DEVICE_GPU) {
        add_gpu_forward(a, b, out);
    }

    // autograd stuff, store parents etc...

    if (out->requires_grad) {
        out->parents = (Tensor**)malloc(2 * sizeof(Tensor));
        if (out->parents == NULL) {
            fprintf(stderr, "Error: Failed to allocate memory for list of parents in addition tensor.\n");
            free_tensor(out);
            return NULL;
        }

        out->parents[0] = a;
        out->parents[1] = b;
        out->num_parents = 2;
        out->op = OP_ADD;
    }
    return out;
}

Tensor* tensor_mul(Tensor* a, Tensor* b) {
    if (a->device != b->device) {
        fprintf(stderr, "Error: Cannot multiply tensors on different devices.\n");
        return NULL;
    }

    if (a->ndims != b->ndims) {
        fprintf(stderr, "Error: Tensors must have the same number of dimensions for multiplication.\n");
        return NULL;
    }
    for (int i = 0; i < a->ndims; i++) {
        if (a->shape[i] != b->shape[i]) {
            fprintf(stderr, "Error: Tensors must have same shape for multiplication.\n");
            return NULL;
        }
    }

    // Create output tensor
    Tensor* out = create_tensor(a->shape, a->ndims, a->device, a->requires_grad || b->requires_grad, 0);

    // Perform calculation on proper device
    if (a->device == DEVICE_CPU) {
        fprintf(stderr, "Error: tensors must be on the gpu.\n");
        return NULL;
    } else if (a->device == DEVICE_GPU) {
        mul_gpu_forward(a, b, out);
    }

    // autograd stuff, store parents etc...

    if (out->requires_grad) {
        out->parents = (Tensor**)malloc(2 * sizeof(Tensor));
        if (out->parents == NULL) {
            fprintf(stderr, "Error: Failed to allocate memory for list of parents in multiplication tensor.\n");
            free_tensor(out);
            return NULL;
        }

        out->parents[0] = a;
        out->parents[1] = b;
        out->num_parents = 2;
        out->op = OP_MUL;
    }
    return out;
}

Tensor* tensor_add_bias(Tensor* a, Tensor * bias) {
    if (a->device != bias->device) {
        fprintf(stderr, "Error: Cannot add tensor and bias on different devices.\n");
        return NULL;
    }

    // a should be 2D [Batch, Features] and bias should be either [Features] or [1, Features]
    if (a->ndims != 2 || (bias->ndims != 1 && bias->ndims != 2)) {
        fprintf(stderr, "Error: Bias and Tensor columns do not match for bias addition.\n");
        return NULL;
    }
    if ((bias->ndims == 1 && bias->shape[0] != a->shape[1]) ||
        (bias->ndims == 2 && (bias->shape[0] != 1 || bias->shape[1] != a->shape[1]))) {
        fprintf(stderr, "Error: Bias and Tensor columns do not match for bias addition.\n");
        return NULL;
    }

    Tensor* out = create_tensor(a->shape, a->ndims, a->device, a->requires_grad || bias->requires_grad, 0);

    // Perform calculation on proper device
    if (a->device == DEVICE_CPU) {
        fprintf(stderr, "Error: tensors must be on the gpu.\n");
        return NULL;
    } else if (a->device == DEVICE_GPU) {
        bias_gpu_forward(a, bias, out);
    }

    if (out->requires_grad) {
        out->parents = (Tensor**)malloc(2 * sizeof(Tensor));
        if (out->parents == NULL) {
            fprintf(stderr, "Error: Failed to allocate memory for list of parents in bias addiiton tensor.\n");
            free_tensor(out);
            return NULL;
        }

        out->parents[0] = a;
        out->parents[1] = bias;
        out->num_parents = 2;
        out->op = OP_ADDBIAS;
    }
    return out;
}

Tensor* tensor_matmul(Tensor* a, Tensor* b) {
    if (a->device != b->device) {
        fprintf(stderr, "Error: Cannot matrix multiply two tensors on different devices.\n");
        return NULL;
    }

    if (a->ndims != 2 || b->ndims != 2) {
        fprintf(stderr, "Error: Both tensors must be 2D for matrix multiplication.\n");
        return NULL;
    }

    if (a->shape[1] != b->shape[0]) {
        fprintf(stderr, "Error: Inner dimensions must match for matrix multiplication.\n");
    }

    int out_shape[] = {a->shape[0], b->shape[1]};
    Tensor* out = create_tensor(out_shape, 2, a->device, a->requires_grad || b->requires_grad, 0);

    if (a->device == DEVICE_CPU) {
        fprintf(stderr, "Error: tensors must be on the gpu.\n");
        return NULL;
    } else if (a->device == DEVICE_GPU) {
        matmul_gpu_forward(a, b, out);
    }

    if (out->requires_grad) {
        out->parents = (Tensor**)malloc(2 * sizeof(Tensor));
        if (out->parents == NULL) {
            fprintf(stderr, "Error: Failed to allocate memory for list of parents in matmul tensor.\n");
            free_tensor(out);
            return NULL;
        }

        out->parents[0] = a;
        out->parents[1] = b;
        out->num_parents = 2;
        out->op = OP_MATMUL;
    }
    return out;
}

Tensor* tensor_relu(Tensor* a) {
    
    Tensor* out = create_tensor(a->shape, a->ndims, a->device, a->requires_grad, 0);

    if (a->device == DEVICE_CPU) {
        fprintf(stderr, "Error: tensors must be on the gpu.\n");
        return NULL;
    } else if (a->device == DEVICE_GPU) {
        relu_gpu_forward(a, out);
    }

    if (out->requires_grad) {
        out->parents = (Tensor**)malloc(sizeof(Tensor));
        if (out->parents == NULL) {
            fprintf(stderr, "Error: Failed to allocate memory for list of parents in relu tensor.\n");
            free_tensor(out);
            return NULL;
        }
        
        out->parents[0] = a;
        out->num_parents = 1;
        out->op = OP_RELU;   
    }

    return out;
}


Tensor* tensor_clipped_leaky_relu(Tensor* a) {
    
    Tensor* out = create_tensor(a->shape, a->ndims, a->device, a->requires_grad, 0);

    if (a->device == DEVICE_CPU) {
        fprintf(stderr, "Error: tensors must be on the gpu.\n");
        return NULL;
    } else if (a->device == DEVICE_GPU) {
        clipped_leaky_relu_gpu_forward(a, out);
    }

    if (out->requires_grad) {
        out->parents = (Tensor**)malloc(sizeof(Tensor));
        if (out->parents == NULL) {
            fprintf(stderr, "Error: Failed to allocate memory for list of parents in clipped leaky relu tensor.\n");
            free_tensor(out);
            return NULL;
        }
        
        out->parents[0] = a;
        out->num_parents = 1;
        out->op = OP_CLIPPED_LEAKY_RELU;   
    }

    return out;
}

Tensor* tensor_mse(Tensor* pred, Tensor* target) {
    int shape[] = {1};
    Tensor* out = create_tensor(shape, 1, pred->device, true, 0);
    out->op = OP_MSE;
    out->num_parents = 2;
    out->parents = (Tensor**)malloc(2 * sizeof(Tensor*));
    out->parents[0] = pred;
    out->parents[1] = target;

    if (pred->device == DEVICE_CPU) {
        fprintf(stderr, "Error: tensors must be on the gpu.\n");
        return NULL;
    } else if (pred->device == DEVICE_GPU) {
        mse_gpu_forward(pred, target, out);
    }
    return out;
}

Tensor* tensor_sparse_linear_forward(Tensor* inputs, Tensor* weights, Tensor* bias) {
    if (!inputs->is_int_tensor || weights->is_int_tensor || bias->is_int_tensor) {
        fprintf(stderr, "Error: invalid tensor types.\n");
        exit(EXIT_FAILURE);
    }

    int batch_size = inputs->shape[0];
    int active_count = inputs->shape[1]; // 32 chess pieces
    int hidden_dim = weights->shape[1]; // 512

    int out_shape[] = {batch_size, hidden_dim};
    Tensor* out = create_tensor(out_shape, 2, DEVICE_GPU, weights->requires_grad || bias->requires_grad, 0);

    out->op = OP_SPARSE_LINEAR;
    out->num_parents = 3;
    out->parents = (Tensor**)malloc(3 * sizeof(Tensor*));
    out->parents[0] = inputs;
    out->parents[1] = weights;
    out->parents[2] = bias;

    if (inputs->device == DEVICE_CPU) {
        fprintf(stderr, "Error: tensors must be on the gpu.\n");
        return NULL;
    } else if (inputs->device == DEVICE_GPU) {
        sparse_linear_gpu_forward(inputs, weights, bias, out);
    }
    return out;
}

Tensor* tensor_perspective_concat_forward(Tensor* white_acc, Tensor* black_acc, Tensor* stm) {
    if (!stm->is_int_tensor) {
        fprintf(stderr, "Error: stm must be int tensor.\n");
        return NULL;
    }

    int batch_size = white_acc->shape[0];
    int half_dim = white_acc->shape[1];
    int shape[] = {batch_size, half_dim * 2};

    Tensor* out = create_tensor(shape, 2, white_acc->device, white_acc->requires_grad || black_acc->requires_grad, 0);
    out->op = OP_PERSPECTIVE_CONCAT;
    out->num_parents = 3;
    out->parents = (Tensor**)malloc(3 * sizeof(Tensor*));
    out->parents[0] = white_acc;
    out->parents[1] = black_acc;
    out->parents[2] = stm;

    perspective_concat_gpu_forward(white_acc, black_acc, stm, out);

    return out;
}