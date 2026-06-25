#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "ops.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    Tensor** array;
    int size;
    int capacity;
} TensorArray;

void backward_add(Tensor* t);
void backward_cpu_add(Tensor* t, Tensor* a, Tensor* b);
void backward_gpu_add(Tensor* t, Tensor* a, Tensor* b);

void backward_mul(Tensor* t);
void backward_cpu_mul(Tensor* t, Tensor* a, Tensor* b);
void backward_gpu_mul(Tensor* t, Tensor* a, Tensor* b);

void backward_add_bias(Tensor* t);
void backward_cpu_add_bias(Tensor* t, Tensor* a, Tensor* bias);
void backward_gpu_add_bias(Tensor* t, Tensor* a, Tensor* bias);

void backward_matmul(Tensor* t);
void backward_cpu_matmul(Tensor* t, Tensor* a, Tensor* b);
void backward_gpu_matmul(Tensor* t, Tensor* a, Tensor* b);

void backward_relu(Tensor* t);
void backward_cpu_relu(Tensor* t, Tensor* a);
void backward_gpu_relu(Tensor* t, Tensor* a);

void backward_flatten(Tensor* t);
void backward_cpu_flatten(Tensor* t, Tensor* a);
void backward_gpu_flatten(Tensor* t, Tensor* a);

void backward_conv2d(Tensor* t);
void backward_cpu_conv2d(Tensor* t, Tensor* input, Tensor* weight, Tensor* bias);
void backward_gpu_conv2d(Tensor* t, Tensor* input, Tensor* weight, Tensor* bias);

void backward_maxpool2d(Tensor* t);
void backward_cpu_maxpool2d(Tensor* t, Tensor* input);
void backward_gpu_maxpool2d(Tensor* t, Tensor* input);

void backward_mse(Tensor* t);
void backward_cpu_mse(Tensor* t, Tensor* pred, Tensor* target);
void backward_gpu_mse(Tensor* t, Tensor* pred, Tensor* target);

void backward_cross_entropy(Tensor* t);
void backward_cpu_cross_entropy(Tensor* t, Tensor* pred, Tensor* target);
void backward_gpu_cross_entropy(Tensor* t, Tensor* pred, Tensor* target);


void build_topo(Tensor* u, TensorArray* topo);
void free_graph(Tensor* root);
void backward(Tensor* t);

#ifdef __cplusplus
}
#endif

#endif