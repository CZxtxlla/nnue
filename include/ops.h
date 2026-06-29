#ifndef OPS_H
#define OPS_H

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward operations on tensors
Tensor* tensor_add(Tensor* a, Tensor* b);
void add_gpu_forward(Tensor* a, Tensor* b, Tensor* out);

Tensor* tensor_mul(Tensor* a, Tensor* b);
void mul_gpu_forward(Tensor* a, Tensor* b, Tensor* out);

Tensor* tensor_add_bias(Tensor* a, Tensor* bias);
void bias_gpu_forward(Tensor* a, Tensor* bias, Tensor* out);

Tensor* tensor_matmul(Tensor* a, Tensor* b);
void matmul_gpu_forward(Tensor* a, Tensor* b, Tensor* out);

Tensor* tensor_relu(Tensor* a);
void relu_gpu_forward(Tensor* a, Tensor* out);

Tensor* tensor_clipped_relu(Tensor* a);
void clipped_relu_gpu_forward(Tensor* a, Tensor* out);

Tensor* tensor_mse(Tensor* pred, Tensor* target);
void mse_gpu_forward(Tensor* pred, Tensor* target, Tensor* out);

Tensor* tensor_cross_entropy(Tensor* pred, Tensor* target);
void cross_entropy_gpu_forward(Tensor* pred, Tensor* target, Tensor* out);

Tensor* tensor_sparse_linear_forward(Tensor* inputs, Tensor* weights, Tensor* bias);
void sparse_linear_gpu_forward(Tensor* inputs, Tensor* weights, Tensor* bias, Tensor* out);

Tensor* tensor_blended_loss_forward(Tensor* pred, Tensor* teacher_probs, Tensor* outcomes, float lambda);
void blended_loss_gpu_forward(Tensor* pred, Tensor* teacher_probs, Tensor* outcomes, float lambda, Tensor* out);

#ifdef __cplusplus
}
#endif

#endif