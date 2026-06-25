# Introduction

This document contains explanation for select operations that I deemed confusing / warranted a deeper look. This is not exhaustive and does not explain everything in the code, but it's more to provide a bit of mathematical reasoning behind some of the more confusing parts. Note all the code I will be demonstrating is the CPU variant, since it is easier to digest.

## Some Backward Functions
-----
### backward_add_bias

The forward operation of tensor_add_bias operates with a $2 \times 2$ matrix as follows. We have a matrix and a bias vector,

$$
A = \begin{bmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22} \\
\end{bmatrix} 
\quad b = \begin{bmatrix}
 b_1 & b_2\\
\end{bmatrix}
$$

These are compatible because they have the same number of columns. We extend $b$ such that it is compatible with matrix addition, i.e.

$$
B = \begin{bmatrix}
        b_1 & b_2 \\
        b_1 & b_2 \\
    \end{bmatrix} \quad \Rightarrow \quad A + B = 
    \begin{bmatrix}
        a_{11} + b_1 & a_{12} + b_2 \\
        a_{21} + b_1 & a_{22} + b_2 \\
    \end{bmatrix}
$$

To understand how the gradient backpropogates through the operation, call $T = A + B$. We know the overall gradient of the loss $L$ with respect to $T$,

$$
\nabla_T = \begin{bmatrix}
        \frac{\partial L}{\partial t_{11}} & \frac{\partial L}{\partial t_{12}} \\
        \frac{\partial L}{\partial t_{21}} & \frac{\partial L}{\partial t_{22}} \\
        \end{bmatrix} \quad \text{ where } t_{ij} = a_{ij} + b_j 
$$

Then by the multivariable chain rule,

$$
\frac{\partial L}{\partial b_j} = \frac{\partial L}{\partial t_{1j}} \cdot \frac{\partial t_{1j}}{\partial b_j} + \frac{\partial L}{\partial t_{2j}} \cdot \frac{\partial t_{2j}}{\partial b_j}
$$

Note now, $\frac{\partial t_{ij}}{\partial b_j} =1$ always, so we end up just summing the overall gradients with respect to each $t_{ij}$, i.e.

$$
\frac{\partial L}{\partial b_1} = \frac{\partial L}{\partial t_{11}} + \frac{\partial L}{\partial t_{21}} \quad \text{ and } \quad \frac{\partial L}{\partial b_2} = \frac{\partial L}{\partial t_{12}} + \frac{\partial L}{\partial t_{22}}
$$

More generally, for an arbitrary size matrix such that the number of columns matches the length of the bias vector, letting k be the number of rows, i.e. the batch size, we get the general formula

$$
\frac{\partial L}{\partial b_j} = \sum_{i = 1}^k \frac{\partial L}{\partial t_{ij}}
$$

So 

$$
\nabla b = \left( \frac{\partial L}{\partial b_1}, \frac{\partial L}{\partial b_2}, \dots,
    \frac{\partial L}{\partial b_n} \right) = \left(\sum_{i = 1}^k \frac{\partial L}{\partial t_{i1}}, \sum_{i = 1}^k \frac{\partial L}{\partial t_{i2}}, \dots, \sum_{i = 1}^k \frac{\partial L}{\partial t_{in}}\right) = \sum_{i =1}^k \left(\frac{\partial L}{\partial t_{i1}}, \frac{\partial L}{\partial t_{i2}}, \dots, \frac{\partial L}{\partial t_{in}}\right)
$$

In code (for the cpu):

```c
if (bias->requires_grad) {
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < features; j++) {
            bias->cpu_grad[j] += t->cpu_grad[i * features + j];
        }
    }
}
```

It is easy to see how the gradient can be backpropogated through the matrix $A$ too. 

$$
\frac{\partial L}{\partial a_{ij}} = \frac{\partial L}{\partial t_{ij}} \cdot \frac{\partial t_{ij}}{\partial a_{ij}} = \frac{\partial L}{\partial t_{ij}} \quad \text{ since } \quad  \frac{\partial t_{ij}}{\partial a_{ij}} = 1
$$

Thus in code (for the cpu):

```c
if (a->requires_grad) {
    for (int i = 0; i < a->size; i++) {
        a->cpu_grad[i] += t->cpu_grad[i];
    }
}
```

### backward_matmul

The forward operation of tensor_matmul operates with $2 \times 2$ matrices as follows.

$$
A = 
\begin{bmatrix}
    a_{11} & a_{12} \\
    a_{21} & a_{22} \\
\end{bmatrix} 
\quad B = 
\begin{bmatrix}
    b_{11} & b_{12} \\
    b_{21} & b_{22} \\
\end{bmatrix} \quad \Rightarrow AB = 
\begin{bmatrix}
    a_{11}b_{11} + a_{12}b_{21} & a_{11}b_{12} + a_{12}b_{22} \\
    a_{21}b_{11} + a_{22}b_{21} & a_{21}b_{12} + a_{22}b_{22} \\
\end{bmatrix} 
$$

Now to understand how the gradient backpropogates through the operation, call $T = AB$. We know the overall gradient with respect to $T$,

$$
\nabla_T = \begin{bmatrix}
    \frac{\partial L}{\partial t_{11}} & \frac{\partial L}{\partial t_{12}} \\
    \frac{\partial L}{\partial t_{21}} & \frac{\partial L}{\partial t_{22}} \\
    \end{bmatrix} \quad \text{ where } t_{ij} = \sum_{k = 1}^n a_{ik}b_{kj}
$$

Then by the multivariable chain rule,

$$
\frac{\partial L}{\partial a_{11}} = \frac{\partial L}{\partial t_{11}} \cdot \frac{\partial t_{11}}{\partial a_{11}} + \frac{\partial L}{\partial t_{12}} \cdot \frac{\partial t_{12}}{\partial a_{11}} \quad \text{ and } \quad
\frac{\partial L}{\partial a_{12}} = \frac{\partial L}{\partial t_{11}} \cdot \frac{\partial t_{11}}{\partial a_{12}} + \frac{\partial L}{\partial t_{12}} \cdot \frac{\partial t_{12}}{\partial a_{12}}
$$
and
$$
\frac{\partial L}{\partial a_{21}} = \frac{\partial L}{\partial t_{21}} \cdot \frac{\partial t_{21}}{\partial a_{21}} + \frac{\partial L}{\partial t_{21}} \cdot \frac{\partial t_{21}}{\partial a_{21}} \quad \text{ and } \quad
\frac{\partial L}{\partial a_{22}} = \frac{\partial L}{\partial t_{21}} \cdot \frac{\partial t_{21}}{\partial a_{22}} + \frac{\partial L}{\partial t_{22}} \cdot \frac{\partial t_{22}}{\partial a_{22}}
$$

Noticing the pattern,

$$
\frac{\partial L}{\partial a_{ij}} = \sum_{n = 1}^N \left(\frac{\partial L}{\partial t_{in}}\right) \left(\frac{\partial t_{in}}{\partial a_{ij}}\right) = \sum_{n = 1}^N \nabla T_{in} \cdot b_{jn}
$$

Now notice that $b_{jn} = (B^T)_{nj}$. Thus, substituting this in

$$
\frac{\partial L}{\partial a_{ij}} = \sum_{n = 1}^N \nabla T_{in} \cdot b_{jn} = 
\sum_{n = 1}^N \nabla T_{in} \cdot (B^T)_{nj} \quad \Rightarrow \quad \nabla_A= \nabla_T  B^T
$$

With similar reasoning $\nabla_B = A^T \nabla_T$.

Thus in code (for the cpu):

```c
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
```

### backward_cross_entropy

We really need to understand the forward direction first. Let $x = [x_1, x_2, \dots, x_n]$ be the array of of raw output logits from our prediction, and $y = [y_1, y_2, \dots, y_n]$ the target array (one hot encoded i.e. one of the values is 1.0 and the rest are 0). First, the *softmax* function is applied to convert the raw logits from the prediction into a probability distribution:

$$
p_i = \frac{e^{x_i}}{\sum_{j = 1}^n e^{x_j}}
$$

Next we calculate the *cross entropy loss* which measures the difference between the predicted probability distribution and the target distribution:

$$
L = - \sum_{i = 1}^n y_i \log(p_i) = - y_t \log(p_t) = -\log(p_t)
$$

where $t$ is the true target class (since all the other $y_i = 0$ and $y_t = 1$).

Now we want to find the gradient of the loss with respect to the original raw predicition logits, i.e. $\frac{\partial L}{\partial x_k}$. By the multivariable chain rule, 

$$
\frac{\partial L}{\partial x_k} = \sum_{i = 1}^n \frac{\partial L}{\partial p_i} \cdot \frac{\partial p_i}{\partial x_k}
$$

The derivative of the loss with respect to a single probability $p_i$ is straightforward, it is just

$$
\frac{\partial L}{\partial p_i} = \frac{\partial}{\partial p_i} \left( - \sum_{i = 1}^n y_i \log(p_i) \right) =  \frac{\partial}{\partial p_i} (-y_i \log(p_i)) = -\frac{y_i}{p_i}
$$

The second part of the derivatve is more complicated. We will split into two cases, when $i = k$ and when $i \neq k$. For the first case, assume $i = k$. We will apply quotient rule to take the derivative:

$$
\frac{\partial p_k}{\partial x_k} = \frac{e^{x_k} \sum_{j = 1}^n e^{x_j} - e^{x_k}e^{x_k}}{\left( \sum_{j = 1}^n e^{x_j} \right)^2} = \frac{e^{x_k}}{\sum e^{x_j}} - \left(\frac{e^{x_k}}{\sum e^{x_j}}\right)^2 = p_k - p_k^2 = p_k(1 - p_k)
$$

Now for the second case assume $i \neq k$:

$$
\frac{\partial p_i}{\partial x_k} = \frac{0 \cdot \sum_{j = 1}^n e^{x_j} - e^{x_i} e^{x_k}}{\left({\sum_{j = 1}^n e^{x_j}}\right)^2} = - \left(\frac{e^{x_i}}{\sum e^{x_j}}\right) \cdot \left(\frac{e^{x_k}}{\sum e^{x_j}}\right) = -p_i p_k
$$

Putting this together, 

$$
\frac{\partial L}{\partial x_k} = \sum_{i = 1}^n \frac{\partial L}{\partial p_i} \cdot \frac{\partial p_i}{\partial x_k} = \left(\sum_{i \neq k} \frac{\partial L}{\partial p_i} \cdot \frac{\partial p_i}{\partial x_k}\right) + \left(\frac{\partial L}{\partial p_k} \cdot \frac{\partial p_k}{\partial x_k}\right) = \sum_{i \neq k} \left(-\frac{y_i}{p_i}\right) \cdot -p_ip_k + \left(-\frac{y_k}{p_k}\right) \cdot p_k(1 - p_k)
$$

With some simplification,

$$
\frac{\partial L}{\partial x_k} = \sum_{i \neq k} y_i p_k - y_k(1 - p_k) = \sum_{i \neq k} y_i p_k - y_k + y_kp_k = -y_k + p_k \left(y_k + \sum_{i \neq k} y_i\right) = p_k - y_k
$$

Thus we are left with the very simple formula $\frac{\partial L}{\partial x_k} = p_k - y_k$. 

Thus in code (for the cpu):

```c
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
```

Note here we are dividing by batch size because we are taking the average loss over all the images. Additionally, `pred->cpu_data[i]` was updated during the forward pass to contain $p_i$.

## backward

Backward as a whole just pieces together the operation specific backwards to slide the gradient on backwards. It's important that the gradient in a later operation is passed back before an earlier operation, and to ensure this a topological graph of the computations is made. Then, proceeding backwards through the topological graph, starting with the loss the gradient is passed back using the individual backpropogate functions for each operation until it has fully backpropogated to all learnable parameters. 

## Convolutional Layer

The forward operation `tensor_conv2d(Tensor* input, Tensor* weight, Tensor* bias, int stride, int padding)` works as follows. Instead of just a big matrix multiplication like for the linear layer, we take a small filter (i.e. $3 \times 3$) and slide it across the image one step at a time, performing the convolution operation each time. Mathematically, if $I$ is the input image and $W$ is the weights (filter), then to calculate a single pixel of the output $O$, with coordinates $(x, y)$, we use the following formula:

$$
O(x, y) = \sum_{m = 0}^2 \sum_{n = 0}^2 I(x + m, y + n) \cdot W(m, n) 
$$

The following is the corresponding CPU code, yes it looks a bit complicated but the only thing that really matters is the inner computation after getting the bias. The innermost two loops perform the convolution.


```c
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
```

Now to understand how the backwards pass works. We can essentially reuse the forward pass structure, and use the same seven loop setup to recreate what pieces would've gone into computing a single element. Instead of computing the element, we accumulate the gradient instead. The following is this in CPU code. Note that it is almost identical to the forward direction just instead of computing the bias the gradient is accumulated backward, and instead of computing the product of the weight and the input the gradient is accumulated backward.

```c
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
```

In the future I would like to speed this up because I think this code is a large bottleneck.



# Results
---
The goal of this project was to learn about the mathematical foundations behind neural networks, as well as practically applying the CUDA skills I have learned, but what good is all that if the model doesn't perform well. The main target was a good accuracy on the MNIST dataset of handwritten letters. I will be testing the two models I have implemented, first just a standard fully connected MLP, and second a simple CNN.