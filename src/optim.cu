#include "../include/optim.h"
#include <cuda_runtime.h>
#include "../include/cuda_utils.h"

// ------- SGD stuff ---------

SGD* sgd_create(Tensor** parameters, int num_parameters, float lr) {
    // Allocates the optimizer and stores the pointers to the learnable parameters
    SGD* optim = (SGD*)malloc(sizeof(SGD));
    if (optim == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for SGD optimizer.\n");
        return NULL;
    }
    optim->parameters = parameters;
    optim->num_parameters = num_parameters;
    optim->lr = lr;

    return optim;
}

void sgd_step(SGD* optim) {
    for (int i = 0; i < optim->num_parameters; i++) {
        Tensor* param = optim->parameters[i];
        if (param->requires_grad && param->device == DEVICE_CPU) {
            sgd_cpu_step(param, optim->lr);
        } else if (param->requires_grad && param->device == DEVICE_GPU) {
            sgd_gpu_step(param, optim->lr);
        }
    }
}

void sgd_zero_grad(SGD* optim) {
    if (optim == NULL) {
        return;
    }

    for (int i = 0; i < optim->num_parameters; i++) {
        Tensor* param = optim->parameters[i];

        if (param->requires_grad && param->device == DEVICE_CPU) {
            sgd_cpu_zero_grad(param);
        } else if (param->requires_grad && param->device == DEVICE_GPU) {
            sgd_gpu_zero_grad(param);
        }
    }
}

void sgd_free(SGD* optim) {
    free(optim);
}

// ------- ADAM stuff ---------

Adam* adam_create(Tensor** parameters, int num_parameters, float lr) {
    Adam* optim = (Adam*)malloc(sizeof(Adam));
    if (optim == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for ADAM optimizer.\n");
        return NULL;
    }
    optim->parameters = parameters;
    optim->num_parameters = num_parameters;
    optim->lr = lr;

    optim->beta_1 = 0.9f;
    optim->beta_2 = 0.999f;
    optim->epsilon = 1e-8f;
    optim->t = 0; // start at time 0

    optim->m = (float**)malloc(num_parameters * sizeof(float*));
    optim->v = (float**)malloc(num_parameters * sizeof(float*));

    if (optim->m == NULL || optim->v == NULL) {
        fprintf(stderr, "Error: failed to allocate memory for the momentum arrays in ADAM optimizer.\n");
        return NULL;
    }

    for (int i = 0; i < num_parameters; i++) {
        // set all momentum arrays to 0
        int size = parameters[i]->size;

        if (parameters[i]->device == DEVICE_CPU) {
            //allocate on cpu
            optim->m[i] = (float*)calloc(size, sizeof(float)); 
            optim->v[i] = (float*)calloc(size, sizeof(float));
        } else {
            // allocate on gpu
            CUDA_CHECK_GOTO(cudaMalloc((void**)&optim->m[i], size * sizeof(float)), cleanup);
            CUDA_CHECK_GOTO(cudaMalloc((void**)&optim->v[i], size * sizeof(float)), cleanup);
            cudaMemset(optim->m[i], 0, size * sizeof(float));
            cudaMemset(optim->v[i], 0, size * sizeof(float));
        }
    }

    return optim;

cleanup:
    exit(EXIT_FAILURE);
}



void adam_step(Adam* optim) {
    optim->t += 1;
    for (int i = 0; i < optim->num_parameters; i++) {
        Tensor* param = optim->parameters[i];
        if (param->requires_grad && param->device == DEVICE_CPU) {
            adam_cpu_step(param, optim->m[i], optim->v[i], optim->lr, optim->beta_1, optim->beta_2, optim->epsilon, optim->t);
        } else if (param->requires_grad && param->device == DEVICE_GPU) {
            adam_gpu_step(param, optim->m[i], optim->v[i], optim->lr, optim->beta_1, optim->beta_2, optim->epsilon, optim->t);
        }
    }
}

void adam_zero_grad(Adam* optim) {
    if (optim == NULL) {
        return;
    }

    for (int i = 0; i < optim->num_parameters; i++) {
        Tensor* param = optim->parameters[i];

        if (param->requires_grad && param->device == DEVICE_CPU) {
            adam_cpu_zero_grad(param);
        } else if (param->requires_grad && param->device == DEVICE_GPU) {
            adam_gpu_zero_grad(param);
        }
    }
}


void adam_free(Adam* optim) {
    if (optim != NULL) {
        for (int i = 0; i < optim->num_parameters; i++) {
            if (optim->parameters[i]->device == DEVICE_CPU) {
                free(optim->m[i]);
                free(optim->v[i]);
            } else {
                cudaFree(optim->m[i]);
                cudaFree(optim->v[i]);
            }
        }
        free(optim->m);
        free(optim->v);
        free(optim);
    }
}