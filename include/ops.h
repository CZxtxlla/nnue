#ifndef OPS_H
#define OPS_H

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward operations on tensors
Tensor* tensor_add(Tensor* a, Tensor* b);
void add_cpu_forward(Tensor* a, Tensor* b, Tensor* out);
void add_gpu_forward(Tensor* a, Tensor* b, Tensor* out);

Tensor* tensor_mul(Tensor* a, Tensor* b);
void mul_cpu_forward(Tensor* a, Tensor* b, Tensor* out);
void mul_gpu_forward(Tensor* a, Tensor* b, Tensor* out);

Tensor* tensor_add_bias(Tensor* a, Tensor* bias);
void bias_cpu_forward(Tensor* a, Tensor* bias, Tensor* out);
void bias_gpu_forward(Tensor* a, Tensor* bias, Tensor* out);

Tensor* tensor_matmul(Tensor* a, Tensor* b);
void matmul_cpu_forward(Tensor* a, Tensor* b, Tensor* out);
void matmul_gpu_forward(Tensor* a, Tensor* b, Tensor* out);

Tensor* tensor_relu(Tensor* a);
void relu_cpu_forward(Tensor* a, Tensor* out);
void relu_gpu_forward(Tensor* a, Tensor* out);

Tensor* tensor_conv2d(Tensor* input, Tensor* weight, Tensor* bias, int stride, int padding);
void conv2d_cpu_forward(Tensor* input, Tensor* weight, Tensor* bias, Tensor* out, int stride, int padding);
void conv2d_gpu_forward(Tensor* input, Tensor* weight, Tensor* bias, Tensor* out, int stride, int padding);

Tensor* maxpool2d_forward(Tensor* input, int filter_size, int stride, int padding);
void maxpool2d_cpu_forward(Tensor* input, Tensor* out, int filter_size, int stride, int padding);
void maxpool2d_gpu_forward(Tensor* input, Tensor* out, int filter_size, int stride, int padding);

Tensor* tensor_mse(Tensor* pred, Tensor* target);
void mse_cpu_forward(Tensor* pred, Tensor* target, Tensor* out);
void mse_gpu_forward(Tensor* pred, Tensor* target, Tensor* out);

Tensor* tensor_cross_entropy(Tensor* pred, Tensor* target);
void cross_entropy_cpu_forward(Tensor* pred, Tensor* target, Tensor* out);
void cross_entropy_gpu_forward(Tensor* pred, Tensor* target, Tensor* out);

#ifdef __cplusplus
}
#endif

#endif