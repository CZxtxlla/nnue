### Sparse Layer Forward and Backward

Performing a dense linear forward with an input of 40000+ features is very computational, but luckily in this case only a maximum of 32 of the features will actually be active since there are only 32 chess pieces. We can save a lot of computation by using a sparse matrix representation as well as sparse matrix multiplication.
Since the input only has 32 active features, it is much more efficient, instead of storing the whole huge input tensor, store an array of indices where the feature is active. Then our array only needs to be 32 long instead of 41024. 

First lets observe how a dense linear forward works. We have an input vector $X$ of dimension [batch_size, $D_{\text{in}}$], weight matrix $W$ of dimension [$D_{\text{in}}$, hidden_neurons] and bias vector $B$ of dimension [1, hidden_neurons]. In the standard linear forward, the input is a series of vectors $x_b$ (where $b$ is the batch index) of size $1 \times D_{\text{in}}$ where 

$$
x_{b, i} =
\begin{cases}
    1 \text{ if feature } i \text{ is active} \\
    0 \text{ if feature } i \text{ is inactive}
\end{cases}
$$

To optimize this, Instead of storing a whole bunch of zeros, we simply store the indices of the active features in a tensor

$$
\mathcal{A}_b = \{i \in \{0, 1, \dots, D_{in} - 1\} : x_{b, i} = 1 \}
$$

In the dense forward pass, each element of our output matrix is calculated using the following standard formula

$$
y_{b, h} = \left( \sum_{i = 0}^{D_{in} - 1} x_{b, i} \cdot W_{i, h} \right) + B_h
$$

Since a lot of these multiplications are zeros, we can simplify this equation greatly using our set of active feature indices

$$
y_{b, h} = \left( \sum_{i \in \mathcal{A}_b} W_{i, h} \right) + B_h
$$

This turns the forward operation from a large matrix multiplication with time complexity O(batch_size $\cdot D_{in} \cdot$ hidden_neurons) into just a few dot products with time complexity O(batch_size $\cdot |\mathcal{A}_b| \cdot$ hidden_neurons).

Now to understand the backwards pass. We have the element $\frac{\partial \mathcal{L}}{\partial y_{b, h}}$ and we want to compute the gradients with respect to the biases and with respect to the weights.

First for the biases. Let $B$ be the batch size. Then to calculate the gradient of a single element of the bias $B_h$, we take the partial derivative

$$
\frac{\partial \mathcal{L}}{\partial B_h} = \sum_{b = 0}^{B - 1} \frac{\partial \mathcal{L}}{\partial y_{b, h}} \cdot \frac{\partial y_{b, h}}{\partial B_h}
$$

We know that $\frac{\partial y_{b, h}}{\partial B_h} = 1$ thus the formula simplifies to

$$
\frac{\partial \mathcal{L}}{\partial B_h} = \sum_{b = 0}^{B - 1} \frac{\partial \mathcal{L}}{\partial y_{b, h}}
$$

Now for the weights. Again letting $B$ be the batch size, we get the following equation for the gradient with respect to the weight

$$
\frac{\partial \mathcal{L}}{\partial W_{i, h}} = \sum_{b = 0}^{B - 1} \frac{\partial \mathcal{L}}{\partial y_{b, h}} \cdot \frac{\partial y_{b, h}}{\partial W_{i, h}}
$$

From the equation for the forward pass, we know

$$
\frac{\partial y_{b, h}}{\partial W_{i, h}} = \begin{cases}
1 \text{ if } i \in \mathcal{A}_b \\
0 \text{ if } i \notin \mathcal{A}_b
\end{cases}
$$

Thus we get

$$
\frac{\partial \mathcal{L}}{\partial W_{i, h}} = \sum_{b |i \in \mathcal{A}_b}\frac{\partial \mathcal{L}}{\partial y_{b, h}}
$$