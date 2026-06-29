### Perpsective Concatenation

After the sparse forward pass through the first layer, we end up with two accumulators, vectors of size 256, one for black and one for white. In order for the rest of the network to function properly, we must concatenate the white perspective accumulator and black perspective accumulator such that the active player's accumulator is first. Let's examine the forward pass and eventually observe what this operation means for the backpropogation of the gradients.

First the forward pass, we have two accumulators $\mathcal{A}_w$ and $\mathcal{A}_b$ both of size 256. We also have a side to move $S \in \{0, 1\}$ where 0 means it's white's turn to move and 1 means it's black's turn to move. The output vector is a vector of length 2 * 256 = 512, defined as follows

$$
Y = \begin{cases}
    \begin{bmatrix}
    \mathcal{A}_w \\
    \mathcal{A}_b
\end{bmatrix} & \text{ if } S = 0 \\
\begin{bmatrix}
    \mathcal{A}_b \\
    \mathcal{A}_w
\end{bmatrix} & \text{ if } S = 1
\end{cases}
$$

Letting $N = 256$ be the half_dim length, we can get an elementwise definition as well

$$
y_i = \begin{cases}
    \mathcal{A}_{w, i} & \text{ if } S = 0 \text{ and } i < N \\
    \mathcal{A}_{b, i - N} & \text{ if } S = 0 \text{ and } i \geq N \\
    \mathcal{A}_{b, i} & \text{ if } S = 1 \text{ and } i < N \\
    \mathcal{A}_{w, i - N} & \text{ if } S = 1 \text{ and } i \geq N
\end{cases}
$$

Now to understand the backwards pass. We have the incoming gradient $\frac{\partial \mathcal{L}}{\partial Y}$ and we want the derivatives with respect to each accumulator. Since the values aren't actually changing the gradient just gets directly passed back. Splitting the gradient vector into two equal 256 length halfs $\frac{\mathcal{L}}{\partial \mathcal{Y}_{top}}$ and $\frac{\mathcal{L}}{\partial \mathcal{Y}_{bot}}$, we get the following.

$$
\frac{\partial \mathcal{L}}{\partial \mathcal{A}_w} = \begin{cases}
    \frac{\partial \mathcal{L}}{\partial \mathcal{Y}_{top}} & \text{ if } S = 0 \\
    \frac{\partial \mathcal{L}}{\partial \mathcal{Y}_{bot}} & \text{ if } S = 1
\end{cases}
$$

A symmetric definition applies to the black accumulator.