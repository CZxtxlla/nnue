### Blended Loss Function

A loss function is needed that takes into account both the centipawn score assigned by the teacher engine as well as the actual game outcome (0 for loss, 0.5 for draw, 1 for win). To do this we will define the following loss function.

Let $\hat{y}$ be the output score of the neural network, $q$ the evaluation score assigned by the teacher engine, $z$ the game outcome, and $\lambda$ the blending factor between the teacher engine and the true outcome. To get a probability between 0 and 1 we will use a sigmoid function. Define 

$$
\hat{p} = \sigma(\hat{y}) = \frac{1}{1 + e^{-\hat{y}/K}} \quad p_{teacher} = \sigma(q) = \frac{1}{1 + e^{-q / K}}
$$

Where $K$ is some scaling constant, [stockfish](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html#loss) uses 410 but it depends on the engine/data. 

Then we canv apply a weighted mean squared error to achieve the final loss

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