# NNUE Trainer

---

This project servers as a complementary to another of mine, the [Josiah](https://github.com/CZxtxlla/Josiah-Chess) chess engine. It provides a 768 feature nnue implementation as well as training loop and quantization written purely in Cuda C, based off another one of my projects [here](https://github.com/CZxtxlla/smallGrad). Two python scripts for data processing are also provided, labeling fens from pgn files using [stockfish](https://github.com/official-stockfish/stockfish). 

This is not really meant for other people to use and is probably inefficient but it was really more about personal learning and a cool project to work on. Below I am attaching some of my notes for implementing some of the things required, it doesn't cover everything but it was useful for outlining the math for the gradients.

---

### Sparse Layer Forward and Backward

Performing a dense linear forward with an input of 768 features is very computational, but luckily in this case only a maximum of 32 of the features will actually be active since there are only 32 chess pieces. We can save a lot of computation by using a sparse matrix representation as well as sparse matrix multiplication.
Since the input only has 32 active features, it is much more efficient, instead of storing the whole huge input tensor, store an array of indices where the feature is active. Then our array only needs to be 32 long instead of 768. 

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

---

### Perpsective Concatenation

After the sparse forward pass through the first layer, we end up with two accumulators, vectors of size 1024, one for black and one for white. In order for the rest of the network to function properly, we must concatenate the white perspective accumulator and black perspective accumulator such that the active player's accumulator is first. Let's examine the forward pass and eventually observe what this operation means for the backpropogation of the gradients.

First the forward pass, we have two accumulators $\mathcal{A}_w$ and $\mathcal{A}_b$ both of size 1024. We also have a side to move $S \in \{0, 1\}$ where 0 means it's white's turn to move and 1 means it's black's turn to move. The output vector is a vector of length 2 * 1024 = 2048, defined as follows

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

Letting $N = 1024$ be the half_dim length, we can get an elementwise definition as well

$$
y_i = \begin{cases}
    \mathcal{A}_{w, i} & \text{ if } S = 0 \text{ and } i < N \\
    \mathcal{A}_{b, i - N} & \text{ if } S = 0 \text{ and } i \geq N \\
    \mathcal{A}_{b, i} & \text{ if } S = 1 \text{ and } i < N \\
    \mathcal{A}_{w, i - N} & \text{ if } S = 1 \text{ and } i \geq N
\end{cases}
$$

Now to understand the backwards pass. We have the incoming gradient $\frac{\partial \mathcal{L}}{\partial Y}$ and we want the derivatives with respect to each accumulator. Since the values aren't actually changing the gradient just gets directly passed back. Splitting the gradient vector into two equal 1024 length halfs $\frac{\mathcal{L}}{\partial \mathcal{Y}_{top}}$ and $\frac{\mathcal{L}}{\partial \mathcal{Y}_{bot}}$, we get the following.

$$
\frac{\partial \mathcal{L}}{\partial \mathcal{A}_w} = \begin{cases}
    \frac{\partial \mathcal{L}}{\partial \mathcal{Y}_{top}} & \text{ if } S = 0 \\
    \frac{\partial \mathcal{L}}{\partial \mathcal{Y}_{bot}} & \text{ if } S = 1
\end{cases}
$$

A symmetric definition applies to the black accumulator.

---

### Blended Loss Function

A loss function is needed that takes into account both the centipawn score assigned by the teacher engine as well as the actual game outcome (0 for loss, 0.5 for draw, 1 for win). To do this we will define the following loss function.

Let $\hat{y}$ be the output score of the neural network, $q$ the evaluation score assigned by the teacher engine, $z$ the game outcome, and $\lambda$ the blending factor between the teacher engine and the true outcome. To get a probability between 0 and 1 we will use a sigmoid function. Define 

$$
\hat{p} = \sigma(\hat{y}) = \frac{1}{1 + e^{-\hat{y}/K}} \quad p_{teacher} = \sigma(q) = \frac{1}{1 + e^{-q / K}}
$$

Where $K$ is some scaling constant, [stockfish uses 410](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html#loss) (I use 400 in the code) but it depends on the engine/data. 

Then we can apply a weighted mean squared error to achieve the final loss

$$
\mathcal{L} = \lambda(\hat{p} - p_{teacher})^2 + (1 - \lambda)(\hat{p} - z)^2
$$

Now to calculate the backpropogation of the gradient. We want $\frac{\partial \mathcal{L}}{\partial \hat{y}}$. Using chain rule,

$$
\frac{\partial \mathcal{L}}{\partial \hat{y}} = \frac{\partial \mathcal{L}}{\partial \hat{p}} \cdot \frac{\partial \hat{p}}{\partial \hat{y}}
$$

Then

$$
\frac{\partial \hat{p}}{\partial \hat{y}} = \frac{1}{K} \cdot \hat{p}(1 - \hat{p})
$$

and 

$$
\frac{\partial \mathcal{L}}{\partial \hat{p}} = 2 \lambda (\hat{p} - p_{teacher}) + 2(1- \lambda)(\hat{p} - z)
$$

Combining this,

$$
\frac{\partial \mathcal{L}}{\partial \hat{y}} = \frac{2}{K} \cdot \hat{p}(1 - \hat{p}) \cdot \left[ \lambda(\hat{p} - p_{teacher}) + (1 - \lambda) (\hat{p} - z) \right]
$$

You might notice that this loss function is not actually implemented as a standalone operation in the code. That's because an optimization is implemented. Instead of computing that loss, the loss is instead computed as

$$
\mathcal{L} = (\hat{p} - (\lambda z + (1 - \lambda)p_{teacher}))^2
$$

Now notice that we can reconfigure the previous gradient $\frac{\partial \mathcal{L}}{\partial \hat{p}}$ equation as the following:

$$
\frac{\partial \mathcal{L}}{\partial \hat{p}} = 2(\hat{p} - (\lambda p_{teacher} + (1 - \lambda)z))
$$

and this is exactly the gradient of the new loss being computed, except instead of having to compute two MSE losses we only have to compute one. Additionally this allows the reuse of the MSE forward and backward from my previous project.