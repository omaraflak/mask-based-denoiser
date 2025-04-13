# Mask-based Denoiser

Here we explore if we can train a neural network to detect the useful parts of the input for a given prediction task. We do this by training a neural network to mask the input such that the original prediction task can still be learnt. More specifically, we train a 'focuser' that learns to output a set of weights which will be multiplied by predefined masks that can be applied on the input of the original problem.

Let's call:

- `(x, y)`: the original pair of input/output that we want to learn with a neural network.
- `f`: the model that learns to predict `y` from `x`.
- `m_i`: a set of predefined masks that are the same shape as `x`, and which are applied to `x` by computing element-wise multiplication with `x`. Each mask constitutes ta part of the input to **keep**, therefore masks are mostly 0s, and contain 1s on areas of the input to keep.
- `p`: the model that learns to output a set of scalar weights given `x`: one weight per mask.

We optimize a loss function:

```
L = gamma * J(y, f(x)) + J(y, f(x * sum(m * p(x)))) + epsilon * ||p(x)||^2

where J(y, y*) is a cost function for a single output pair.
```

Thus, we train the neural network `f` on the original task to predict `y` from `x`, but we also train the neural network `p` to learn how to mask the input such that `f` can still do predictions. To avoid the case where `p` learns to output 1s everywhere (thus not masking any of the input), we add a regularizer that tries to minimise the weights predicted by `p`.

We experiment with this idea with a toy MNIST example.

![image](https://raw.githubusercontent.com/omaraflak/mask-based-denoiser/refs/heads/main/evaluations.png)
