#include "../include/tensor.h"
#include <cuda_runtime.h>
#include "../include/cuda_utils.h"




Tensor* create_tensor(int* shape, int ndims, DeviceType device, bool requires_grad) {
    // Allocate and initialize one tensor given the input parameters

    Tensor* t = (Tensor*) calloc(1, sizeof(Tensor)); // creates memory for 1 tensor device, all intialized to 0
    if (t == NULL) {
        fprintf(stderr, "Error: failed to allocate memory for tensor.\n");
        return NULL;
    }
    t->stride = 0;
    t->padding = 0;
    t->is_view = false;
    t->visited_pass_id = 0;
    t->ndims = ndims;
    t->requires_grad = requires_grad;
    t->op = OP_NONE;
    t->parents = NULL;
    t->num_parents = 0;
    t->max_indices = NULL;
    t->gpu_max_indices = NULL;

    t->size = 1; // initialize to accumulate dimensions via multipication
    t->shape = (int*)malloc(ndims * sizeof(int));
    if (t->shape == NULL) {
        fprintf(stderr, "Error: failed to allocate memory for shape.\n");
        goto cleanup;
    }

    for (int i = 0; i < ndims; i++) {
        t->shape[i] = shape[i];
        t->size *= shape[i];
    }

    t->device = device;
    if (device == DEVICE_CPU) {
        t->gpu_data = NULL;
        t->gpu_grad = NULL;
        t->cpu_data = (float*) calloc(t->size, sizeof(float)); //everything set to 0
        if (t->cpu_data == NULL) {
            fprintf(stderr, "Error: failed to allocate memory for cpu data.\n");
            goto cleanup;
        }

        if (requires_grad) {
            t->cpu_grad = (float*) calloc(t->size, sizeof(float));
            if (t->cpu_grad == NULL) {
                fprintf(stderr, "Error: failed to allocate memory for cpu tensor grad.\n");
                goto cleanup;
            }
        } else {
            t->cpu_grad = NULL;
        }

    } else if (device == DEVICE_GPU) {
        t->cpu_data = NULL;
        t->cpu_grad = NULL;


        CUDA_CHECK_GOTO(cudaMalloc((void**) &t->gpu_data, t->size * sizeof(float)), cleanup); 
        CUDA_CHECK_GOTO(cudaMemset(t->gpu_data, 0, t->size * sizeof(float)), cleanup);

        
        if (requires_grad) {
            CUDA_CHECK_GOTO(cudaMalloc((void**) &t->gpu_grad, t->size * sizeof(float)), cleanup);
            CUDA_CHECK_GOTO(cudaMemset(t->gpu_grad, 0, t->size * sizeof(float)), cleanup);
        } else {
            t->gpu_grad = NULL;
        }
    } else {
        fprintf(stderr, "Error: Not a valid device.\n");
        goto cleanup;
    }

    return t;

cleanup: 
    free_tensor(t);
    return NULL;
}

void free_tensor(Tensor* t) {
    // Free all allocated memory corresponding to a tensor
    
    if (t == NULL) {
        return;
    }
    if (!t->is_view) {
        // Free cpu data and gpu data only for owning tensors.
        free(t->cpu_data);
        cudaFree(t->gpu_data);
    }

    // Gradient buffers are owned by the tensor itself, even for views.
    free(t->cpu_grad);
    cudaFree(t->gpu_grad);

    if (t->max_indices != NULL) free(t->max_indices);
    if (t->gpu_max_indices != NULL) cudaFree(t->gpu_max_indices);
    // Free rest
    free(t->shape);
    free(t->parents);
    // Finally, free tensor
    free(t);
}

// ------ Helpers for managing data across CPU and GPU ---------


void tensor_to_device(Tensor* t, DeviceType device) {
    // transfers tensor data and gradient to the specified device
    if (t == NULL || device == t->device) {
        return;
    } 

    size_t bytes = t->size * sizeof(float);

    if (device == DEVICE_GPU) {
        CUDA_CHECK_GOTO(cudaMalloc((void**) &t->gpu_data, bytes), cleanup); 
        CUDA_CHECK_GOTO(cudaMemcpy(t->gpu_data, t->cpu_data, bytes, cudaMemcpyHostToDevice), cleanup);

        free(t->cpu_data);
        t->cpu_data = NULL;

        if (t->requires_grad) {
            CUDA_CHECK_GOTO(cudaMalloc((void**)&t->gpu_grad, bytes), cleanup);

            if (t->cpu_grad != NULL) {
                CUDA_CHECK_GOTO(cudaMemcpy(t->gpu_grad, t->cpu_grad, bytes, cudaMemcpyHostToDevice), cleanup);
                free(t->cpu_grad);
                t->cpu_grad = NULL;
            }
        }
        t->device = DEVICE_GPU;

    } else if (device == DEVICE_CPU) {
        t->cpu_data = (float*)malloc(bytes);
        if (t->cpu_data == NULL) {
            fprintf(stderr, "Error: failed to allocate memory during memory transfer to cpu.\n");
            exit(EXIT_FAILURE);
        }

        CUDA_CHECK_GOTO(cudaMemcpy(t->cpu_data, t->gpu_data, bytes, cudaMemcpyDeviceToHost), cleanup);

        cudaFree(t->gpu_data);
        t->gpu_data = NULL;

        if (t->requires_grad) {
            t->cpu_grad = (float*)malloc(bytes);
            if (t->cpu_grad == NULL) {
                fprintf(stderr, "Error: failed to allocate memory during memory transfer to cpu.\n");
                exit(EXIT_FAILURE);
            }
            if (t->gpu_grad != NULL) {
                CUDA_CHECK_GOTO(cudaMemcpy(t->cpu_grad, t->gpu_grad, bytes, cudaMemcpyDeviceToHost), cleanup);
                cudaFree(t->gpu_grad);
                t->gpu_grad = NULL;
            }
        }
        t->device = DEVICE_CPU;
    }

    return;

cleanup:
    exit(EXIT_FAILURE);
}

float tensor_scalar_value(Tensor* t) {
    // return first element of the tensor
    if (t->device == DEVICE_CPU){
        return t->cpu_data[0];
    } else {
        float val;
        CUDA_CHECK_GOTO(cudaMemcpy(&val, t->gpu_data, sizeof(float), cudaMemcpyDeviceToHost), cleanup);
        return val;
    }

cleanup:
    exit(EXIT_FAILURE);
}

void seed_loss_grad(Tensor* loss) {
    // set the gradient of the loss (so only the first element) to 1
    if (loss->device == DEVICE_CPU) {
        loss->cpu_grad[0] = 1.0f;
    } else {
        float one = 1.0f;

        CUDA_CHECK_GOTO(cudaMemcpy(loss->gpu_grad, &one, sizeof(float), cudaMemcpyHostToDevice), cleanup);
    }

    return;

cleanup:
    exit(EXIT_FAILURE);
}

Tensor* tensor_slice_view(Tensor* master, int start_row, int num_rows) {
    int features = master->shape[1];
    int shape[] = {num_rows, features};

    Tensor* view = create_tensor(shape, 2, master->device, false);

    // make sure view tensor data is empty
    if (view->device == DEVICE_CPU && view->cpu_data != NULL) {
        free(view->cpu_data);
        view->cpu_data = NULL;
    } else if (view->device == DEVICE_GPU && view->gpu_data != NULL) {
        cudaFree(view->gpu_data);
        view->gpu_data = NULL;
    }

    // mark as view tensor
    view->is_view = true;

    int offset = start_row * features;

    if (master->device == DEVICE_CPU) {
        view->cpu_data = master->cpu_data + offset; // jump offset * sizeof(float) bytes in memory
    } else if (master->device == DEVICE_GPU) {
        view->gpu_data = master->gpu_data + offset;
    }

    return view;

}

void tensor_download_data(Tensor* t, float* dest) {
    if (t == NULL || dest == NULL) return;
    
    size_t bytes = t->size * sizeof(float);
    
    if (t->device == DEVICE_CPU) {
        memcpy(dest, t->cpu_data, bytes);
    } else if (t->device == DEVICE_GPU) {
        cudaMemcpy(dest, t->gpu_data, bytes, cudaMemcpyDeviceToHost);
    }
}