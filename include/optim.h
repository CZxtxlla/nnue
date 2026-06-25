#ifndef OPTIM_H
#define OPTIM_H

#include "tensor.h"
#include "ops.h"

#ifdef __cplusplus
extern "C" {
#endif

// --------- Stochastic gradient descent optimizer ---------

typedef struct {
    Tensor** parameters; // pointer to array of tensors we want to update (weights, biases)
    int num_parameters; // number of tensors in the array
    float lr; // learning rate
} SGD;

SGD* sgd_create(Tensor** parameters, int num_parameters, float lr); // creates optimizer

void sgd_step(SGD* optim); // applies learning rule to all parameters, (data = data - (lr * grad))
void sgd_gpu_step(Tensor* parameter, float lr);
void sgd_cpu_step(Tensor* parameter, float lr);


void sgd_zero_grad(SGD* optim); // resets all gradients to 0 before next forward pass
void sgd_cpu_zero_grad(Tensor* parameter);
void sgd_gpu_zero_grad(Tensor* parameter);

void sgd_free(SGD* optim); // free optimizer, but not tensors it points to

// --------- ADAM optimizer -----------

typedef struct {
    Tensor** parameters;
    int num_parameters;
    float lr; // learning rate (set around 0.001)

    float beta_1; // exp decay rate for first moment estimate (0.9)
    float beta_2; // exp decay rate for second moment estimate (0.999)

    float epsilon;

    int t; // time, epoch/batch counter

    float** m; // first moment
    float** v; // second moment
} Adam;

Adam* adam_create(Tensor** parameters, int num_parameters, float lr); // create optimizer

void adam_step(Adam* optim); // apply learning rule to parameters
void adam_gpu_step(Tensor* p, float* m, float* v, float lr, float beta1, float beta2, float eps, int t);
void adam_cpu_step(Tensor* p, float* m, float* v, float lr, float beta1, float beta2, float eps, int t);

void adam_zero_grad(Adam* optim); // reset all gradients to 0
void adam_cpu_zero_grad(Tensor* parameter);
void adam_gpu_zero_grad(Tensor* parameter);

void adam_free(Adam* optim); // free optimizer, but not parameters



#ifdef __cplusplus
}
#endif


#endif