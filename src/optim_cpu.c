#include "../include/optim.h"
#include <string.h>
#include <math.h>

void sgd_cpu_step(Tensor* parameter, float lr) {
    if (parameter->cpu_grad != NULL) {
        for (int i = 0; i < parameter->size; i++) {
            parameter->cpu_data[i] -= lr * parameter->cpu_grad[i];
        }
    }
}

void sgd_cpu_zero_grad(Tensor* parameter) {
    if (parameter->cpu_grad != NULL) {
        size_t bytes = parameter->size * sizeof(float);
        memset(parameter->cpu_grad, 0, bytes);
    }
}

void adam_cpu_step(Tensor* p, float* m, float* v, float lr, float beta1, float beta2, float eps, int t) {
    float c1 = 1.0f - powf(beta1, t);
    float c2 = 1.0f - powf(beta2, t);

    for (int i = 0; i < p->size; i++) {
        float grad = p->cpu_grad[i];

        // update moment estimates
        m[i] = beta1 * m[i] + (1.0f - beta1) * grad;
        v[i] = beta2 * v[i] + (1.0f - beta2) * (grad * grad);

        float m_hat = m[i] / c1;
        float v_hat = v[i] / c2;

        p->cpu_data[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

void adam_cpu_zero_grad(Tensor* param) {
    memset(param->cpu_grad, 0, param->size * sizeof(float));
}