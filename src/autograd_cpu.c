#include "../include/autograd.h"


void backward_cpu_add(Tensor* t, Tensor* a, Tensor* b) {
    // performs the backwards transmission of gradient on cpu for addition
    if (a->requires_grad) {
        for (int i = 0; i < a->size; i++) {
            a->cpu_grad[i] += t->cpu_grad[i]; 
        }
    }
    if (b->requires_grad) {
        for (int i = 0; i < b->size; i++) {
            b->cpu_grad[i] += t->cpu_grad[i]; 
        }
    }
}

void backward_cpu_mul(Tensor* t, Tensor* a, Tensor* b) {
    // performs the backwards transmission of gradient on cpu for multiplication
    if (a->requires_grad) {
        for (int i = 0; i < a -> size; i++) {
            a->cpu_grad[i] += b->cpu_data[i] * t->cpu_grad[i];
        }
    }
    if (b->requires_grad) {
        for (int i = 0; i < a -> size; i++) {
            b->cpu_grad[i] += a->cpu_data[i] * t->cpu_grad[i];
        }
    }
}

void backward_cpu_add_bias(Tensor* t, Tensor* a, Tensor* bias) {
    // performs the backwards transmission of gradient on cpu for bias addition
    int batch_size = a->shape[0]; // rows
    int features = a->shape[1]; // columns

    if (a->requires_grad) {
        for (int i = 0; i < a->size; i++) {
            a->cpu_grad[i] += t->cpu_grad[i];
        }
    }
    if (bias->requires_grad) {
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < features; j++) {
                bias->cpu_grad[j] += t->cpu_grad[i * features + j];
            }
        }
    }

}

void backward_cpu_matmul(Tensor* t, Tensor* a, Tensor* b) {
    // performs the backwards transmission of gradient on cpu for matrix multiplication
    int j = a->shape[0];
    int k = a->shape[1];
    int l = b->shape[1];

    if (a->requires_grad) {
        for (int m = 0; m < j; m++) {
            for (int n = 0; n < k; n++) {
                float grad_a_mn = 0.0f;
                for (int p = 0; p < l; p++) {
                    grad_a_mn += t->cpu_grad[m * l + p] * b->cpu_data[n * l + p]; // dot product of mth row of gradT and nth row of B (b/c transpose)
                }
                a->cpu_grad[m * k + n] += grad_a_mn;
            }
        }
    }

    if (b->requires_grad) {
        for (int m = 0; m < k; m++) {
            for (int n = 0; n < l; n++) {
                float grad_b_mn= 0.0f;
                for (int p = 0; p < j; p++) {
                    grad_b_mn += a->cpu_data[p * k + m] * t->cpu_grad[p * l + n];
                }
                b->cpu_grad[m * l + n] += grad_b_mn;
            }
        }
    }
}

void backward_cpu_relu(Tensor* t, Tensor* a) {
    // performs the backwards transmission of gradient on cpu for relu
    if (a->requires_grad) {
        for (int i = 0; i < t->size; i++) {
            a->cpu_grad[i] += a->cpu_data[i] > 0 ? t ->cpu_grad[i] : 0.0f; // grad input = grad output if input > 0
        }
    }
}

void backward_cpu_flatten(Tensor* t, Tensor* a) {
    // reshape gradients back onto the original 4D tensor without changing values
    if (!a->requires_grad) {
        return;
    }

    for (int i = 0; i < t->size; i++) {
        a->cpu_grad[i] += t->cpu_grad[i];
    }
}

void backward_cpu_conv2d(Tensor* t, Tensor* input, Tensor* weight, Tensor* bias) {
    // performs the backward transmission of gradient on the cpu for conv2d operation.
    // very similar to the forward just different insides to pass gradient
    int batch_size = input->shape[0];
    int in_c = input->shape[1];
    int in_h = input->shape[2];
    int in_w = input->shape[3];

    int out_c = weight->shape[0];
    int f_h = weight->shape[2];
    int f_w = weight->shape[3];

    int out_h = t->shape[2];
    int out_w = t->shape[3];

    int stride = t->stride;
    int padding = t->padding;

    for (int b = 0; b < batch_size; b++) {
        for (int oc = 0; oc < out_c; oc++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    int out_idx = b * (out_c * out_h * out_w) + oc * (out_h * out_w) + oh * (out_w) + ow;
                    float grad_out = t->cpu_grad[out_idx];
                    
                    if (bias->requires_grad) {
                        bias->cpu_grad[oc] += grad_out; // gradient gets directly passed back since bias is from addition
                    }

                    for (int ic = 0; ic < in_c; ic++) {
                        for (int fh = 0; fh < f_h; fh++) {
                            for (int fw = 0; fw < f_w; fw++) {
                                int ih = oh * stride - padding + fh;
                                int iw = ow * stride - padding + fw;

                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    int in_idx = b * (in_c * in_h * in_w) + ic * (in_h * in_w) + ih * in_w + iw;
                                    int w_idx = oc * (in_c * f_h * f_w) + ic * (f_h * f_w) + fh * f_w + fw;

                                    if (input->requires_grad) {
                                        input->cpu_grad[in_idx] += weight->cpu_data[w_idx] * grad_out;
                                    }
                                    if (weight->requires_grad) {
                                        weight->cpu_grad[w_idx] += input->cpu_data[in_idx] * grad_out;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void backward_cpu_maxpool2d(Tensor* t, Tensor* input) {
    // perform backpropogation for maxpool operation, simple since max was stored
    if (!input->requires_grad) {
        return;
    }
    for (int i = 0; i < t->size; i++) {
        int max_idx = t->max_indices[i];

        if (max_idx >= 0 && max_idx < input->size) {
            input->cpu_grad[max_idx] += t->cpu_grad[i];
        }
    }
}

void backward_cpu_mse(Tensor* t, Tensor* pred, Tensor* target) {
    // performs backwards propogation of gradient on cpu for mse
    float scale = 2.0f / pred->size;

    if (pred->requires_grad) {
        for (int i = 0; i < pred->size; i++) {
            pred->cpu_grad[i] += scale * (pred->cpu_data[i] - target->cpu_data[i]) * t->cpu_grad[0];
        }
    } 
    if (target->requires_grad) {
        for (int i = 0; i < target->size; i++) {
            target->cpu_grad[i] += -scale * (pred->cpu_data[i] - target->cpu_data[i]) * t->cpu_grad[0];
        }
    }

}

void backward_cpu_cross_entropy(Tensor* t, Tensor* pred, Tensor* target) {
    int batch_size = pred->shape[0];
    float scale = t->cpu_grad[0] / batch_size;

    if (pred->requires_grad) {
        for (int i = 0; i < pred->size; i++) {
            // combined derivative
            pred->cpu_grad[i] += (pred->cpu_data[i] - target->cpu_data[i]) * scale;
        }
    }
}