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
void backward_gpu_add(Tensor* t, Tensor* a, Tensor* b);

void backward_mul(Tensor* t);
void backward_gpu_mul(Tensor* t, Tensor* a, Tensor* b);

void backward_add_bias(Tensor* t);
void backward_gpu_add_bias(Tensor* t, Tensor* a, Tensor* bias);

void backward_matmul(Tensor* t);
void backward_gpu_matmul(Tensor* t, Tensor* a, Tensor* b);

void backward_relu(Tensor* t);
void backward_gpu_relu(Tensor* t, Tensor* a);

void backward_clipped_leaky_relu(Tensor* t);
void backward_gpu_clipped_leaky_relu(Tensor* t, Tensor* a);

void backward_mse(Tensor* t);
void backward_gpu_mse(Tensor* t, Tensor* pred, Tensor* target);

void backward_sparse_linear(Tensor* t);
void backward_gpu_sparse_linear(Tensor* t, Tensor* inputs, Tensor* weights, Tensor* bias);

void backward_perspective_concat(Tensor* t);
void backward_gpu_perspective_concat(Tensor* t, Tensor* w_acc, Tensor* b_acc, Tensor* stm);


void build_topo(Tensor* u, TensorArray* topo);
void free_graph(Tensor* root);
void backward(Tensor* t);

#ifdef __cplusplus
}
#endif

#endif