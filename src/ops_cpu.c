#include "../include/ops.h"
#include <math.h>

// helper for convolution with im2col speedup

void im2col_cpu(float* data_im, int channels, int height, int width, int f_h, int f_w, int padding, int stride, float* data_col) {

    // out coordinates
    int out_h = (height + 2 * padding - f_h) / stride + 1;
    int out_w = (width + 2 * padding - f_w) / stride + 1;
    int channels_col = channels * f_h * f_w;

    // resulting matrix has dims [out_h * out_w, channels * f_h * f_w]

    for (int c = 0; c < channels_col; c++) {
        int w_offset = c % f_w;
        int h_offset = c % f_h;
        int c_im = c / f_h / f_w;

        for (int h = 0; h < out_h; h++) {
            for (int w = 0; w < out_w; w++) {
                int im_row = h_offset + h * stride - padding;
                int im_col = w_offset + w * stride - padding;

                int col_index = (c * out_h + h) * out_w + w;

                // boundary check
                if (im_row >= 0 && im_row < height && im_col >= 0 && im_col < width) {
                    data_col[col_index] = data_im[(c_im * height + im_row) * width + im_col];
                } else {
                    data_col[col_index] = 0.0f;
                }
            }
        }
    }
}



// ---------- forward operations ----------

void add_cpu_forward(Tensor* a, Tensor* b, Tensor* out) {
    for (int i = 0; i < a->size; i++) {
        out->cpu_data[i] = a->cpu_data[i] + b->cpu_data[i];
    }
}

void mul_cpu_forward(Tensor* a, Tensor* b, Tensor* out) {
    for (int i = 0; i < a->size; i++) {
        out->cpu_data[i] = a->cpu_data[i] * b->cpu_data[i];
    }
}

void bias_cpu_forward(Tensor* a, Tensor* bias, Tensor* out) {
    for (int i = 0; i < a->shape[0]; i++) {
        for (int j = 0; j < a->shape[1]; j++) {
            out->cpu_data[i * a->shape[1] + j] = a->cpu_data[i * a->shape[1] + j] + bias->cpu_data[j];
        }
    }
}

void matmul_cpu_forward(Tensor* a, Tensor* b, Tensor* out) {
    for (int i = 0; i < a->shape[0]; i++) {
        for (int j = 0; j < b->shape[1]; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a -> shape[1]; k++) {
                sum += a->cpu_data[i * a->shape[1] + k]* b->cpu_data[k * b->shape[1] + j];
            }
            out->cpu_data[i * b->shape[1] + j] = sum;
        }
    }
}

void relu_cpu_forward(Tensor* a, Tensor* out) {
    for (int i = 0; i < a->size; i++) {
        out->cpu_data[i] = a->cpu_data[i] > 0 ? a->cpu_data[i] : 0.0f;
    }
}

/*
void conv2d_cpu_forward(Tensor* input, Tensor* weight, Tensor* bias, Tensor* out, int stride, int padding) {
    int batch_size = input->shape[0];
    int in_c = input->shape[1]; // number of input channels (same as weight->shape[1])
    int in_h = input->shape[2];
    int in_w = input->shape[3]; // used for boundary checks when convoluting

    int out_c = weight->shape[0]; // number of outpit channels (filters)
    int f_h = weight->shape[2]; // height of filter
    int f_w = weight->shape[3]; // width of filter

    int out_h = out->shape[2]; // output image height
    int out_w = out->shape[3]; // output image width

    for (int b = 0; b < batch_size; b++) {
        // select a single image out of the batch
        for (int oc = 0; oc < out_c; oc++) {
            // select a single filter to slide across the image
            for (int oh = 0; oh < out_h; oh++) {
                // select the row (y coord) of the output image pixel to calculate
                for (int ow = 0; ow < out_w; ow++) {
                    // select the col (x coord) of the output image pixel

                    float val = bias->cpu_data[oc]; // bias the same across a single filter

                    for (int ic = 0; ic < in_c; ic++) {
                        // choose the input channel
                        for (int fh = 0; fh < f_h; fh++) {
                            // choose the filter pixel row
                            for (int fw = 0; fw < f_w; fw++) {
                                // choose the filter pixel col

                                // map output pixel to the corresponding input pixel
                                int ih = oh * stride - padding + fh;
                                int iw = ow * stride - padding + fw;

                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    // boundary checks

                                    // equivalent to [b, ic, ih, iw]
                                    int in_idx = b * (in_c * in_h * in_w) + ic * (in_h * in_w) + ih * in_w + iw;
                                    // equivalent to [oc, ic, fh, fw]
                                    int w_idx = oc * (in_c * f_h * f_w) + ic * (f_h * f_w) + fh * f_w + fw;

                                    val += input->cpu_data[in_idx] * weight->cpu_data[w_idx];
                                }
                            }
                        }
                    }

                    int out_idx = b * (out_c * out_h * out_w) + oc * (out_h * out_w) + oh * out_w + ow;
                    out->cpu_data[out_idx] = val;
                }
            }
        }
    }
}
*/

void conv2d_cpu_forward(Tensor* input, Tensor* weight, Tensor* bias, Tensor* out, int stride, int padding) {
    int batch_size = input->shape[0];
    int in_c = input->shape[1]; // number of input channels (same as weight->shape[1])
    int in_h = input->shape[2];
    int in_w = input->shape[3]; // used for boundary checks when convoluting

    int out_c = weight->shape[0]; // number of outpit channels (filters)
    int f_h = weight->shape[2]; // height of filter
    int f_w = weight->shape[3]; // width of filter

    int out_h = out->shape[2]; // output image height
    int out_w = out->shape[3]; // output image width

    int M = out_c;
    int K = in_c * f_h * f_w;
    int N = out_h * out_w;

    float* im2col_space = (float*)malloc(K * N * sizeof(float)); // temp workspace for im2col matrix
    if (im2col_space == NULL) {
        fprintf(stderr, "Error: problem allocating memory for temp im2col space.\n");
        return;
    }

    for (int b = 0; b < batch_size; b++) {

        float* current_input = input->cpu_data + b * (in_c * in_h * in_w);
        float* current_output = out->cpu_data + b * (out_c * out_h * out_w);

        im2col_cpu(current_input, in_c, in_h, in_w, f_h, f_w, padding, stride, im2col_space);

        // output = weight @ im2col_space + bias
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float sum = bias->cpu_data[m];

                for (int k = 0; k < K; k++) {
                    sum += weight->cpu_data[m * K + k] * im2col_space[k * N + n];
                }

                current_output[m * N + n] = sum;
            }
        }

    }  
    free(im2col_space);
}

void maxpool2d_cpu_forward(Tensor* input, Tensor* out, int filter_size, int stride, int padding) {
    int batch_size = input->shape[0];
    int channels = input->shape[1]; // number of channels
    int in_h = input->shape[2];
    int in_w = input->shape[3]; // used for boundary checks when convoluting


    int out_h = out->shape[2]; // output image height
    int out_w = out->shape[3]; // output image width

    for (int b = 0; b < batch_size; b++) {
        // select a single image out of the batch
        for (int c = 0; c < channels; c++) {
            // select a single filter to slide across the image
            for (int oh = 0; oh < out_h; oh++) {
                // select the row (y coord) of the output image pixel to calculate
                for (int ow = 0; ow < out_w; ow++) {
                    // select the col (x coord) of the output image pixel

                    float max_val = -INFINITY;
                    int max_idx = -1;

                    
                    for (int fh = 0; fh < filter_size; fh++) {
                        // choose the filter pixel row
                        for (int fw = 0; fw < filter_size; fw++) {
                            // choose the filter pixel col

                            // map output pixel to the corresponding input pixel
                            int ih = oh * stride - padding + fh;
                            int iw = ow * stride - padding + fw;

                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                // boundary checks

                                // equivalent to [b, ic, ih, iw]
                                int in_idx = b * (channels * in_h * in_w) + c * (in_h * in_w) + ih * in_w + iw;
                                

                                float val = input->cpu_data[in_idx];

                                if (val > max_val) {
                                    max_val = val;
                                    max_idx = in_idx;
                                }
                            }
                        }
                    }

                    int out_idx = b * (channels * out_h * out_w) + c * (out_h * out_w) + oh * out_w + ow;
                    out->cpu_data[out_idx] = max_val;
                    out->max_indices[out_idx] = max_idx;
                }
            }
        }
    }
}

void mse_cpu_forward(Tensor* pred, Tensor* target, Tensor* out) {
    float sum = 0.0f;
    for (int i = 0; i < pred->size; i++) {
        float diff = pred->cpu_data[i] - target->cpu_data[i];
        sum += diff * diff;
    }
    out->cpu_data[0] = sum/pred->size;
}

void cross_entropy_cpu_forward(Tensor* pred, Tensor* target, Tensor* out) {
    int batch_size = pred->shape[0];
    int num_classes = pred->shape[1];

    float total_loss = 0.0f;

    for (int b = 0; b < batch_size; b++) {
        int offset = b * num_classes;

        // compute max
        float max_val = pred->cpu_data[offset];
        for (int c = 1; c < num_classes; c++) {
            if (pred->cpu_data[offset + c] > max_val) {
                max_val = pred->cpu_data[offset + c];
            }
        }
        // compute exponentials
        float exp_sum = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            float e = expf(pred->cpu_data[offset + c] - max_val);
            pred->cpu_data[offset + c] = e;
            exp_sum +=e;
        }

        // normalize to probabilities and compute loss, -log(p) of the probability of the true class
        for (int c = 0; c < num_classes; c++) {
            float prob = pred->cpu_data[offset + c] / exp_sum;
            pred->cpu_data[offset + c] = prob;

            if (target->cpu_data[offset + c] == 1.0f) {
                total_loss -= logf(prob + 1e-7f);
            }
        }
    }
    out->cpu_data[0] = total_loss/batch_size;
}