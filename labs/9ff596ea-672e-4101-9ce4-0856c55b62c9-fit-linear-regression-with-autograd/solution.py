# --- cell 1 ---
import torch
import numpy as np


def fit_linear_regression(X, y, lr=0.1, steps=500):
    """Fit y ~= X @ w + b with full-batch GD using only autograd.

    Args:
        X: Float tensor (N, D)
        y: Float tensor (N,) or (N, 1)
        lr: learning rate
        steps: number of gradient descent iterations

    Returns:
        w: Float tensor (D,) learned weights (no grad)
        b: Float tensor scalar learned bias (no grad)
    """
    # TODO: ensure y is shape (N,)
    y = y.reshape(-1)
    
    # TODO: initialize w (D,) and b with requires_grad=True
    D = X.shape[1]
    w = (torch.rand(D) * 0.01).requires_grad_(True)
    b = torch.zeros((), requires_grad= True)
    
    # TODO: for each step:
    for _ in range(steps):
    #   - predict, compute MSE loss
        L = ((X @ w + b - y)**2).mean()
    #   - loss.backward()
        L.backward()
    #   - manual GD update under torch.no_grad()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            
    #   - zero gradients
        w.grad.zero_()
        b.grad.zero_()
    # TODO: return detached w, b
    
    return w.detach(), b.detach()
    raise NotImplementedError