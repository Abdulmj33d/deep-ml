import torch
import torch.nn as nn
from collections import OrderedDict


def two_layer_mlp_forward(x, w1, b1, w2, b2):
    """Build a 2-layer MLP, set fixed weights, return scalar output.

    Args:
        x (torch.Tensor): Input of shape (1, 2).
        w1 (torch.Tensor): First Linear weight, shape (2, 2).
        b1 (torch.Tensor): First Linear bias, shape (2,).
        w2 (torch.Tensor): Second Linear weight, shape (1, 2).
        b2 (torch.Tensor): Second Linear bias, shape (1,).

    Returns:
        float: Scalar network output.
    """
    layers = OrderedDict([
        ('linear1',nn.Linear(2,2)),
        ('relu', nn.ReLU()),
        ('linear2', nn.Linear(2,1))
    ])

    model = nn.Sequential(layers)

    with torch.no_grad():
        model.linear1.weight.copy_(w1)
        model.linear1.bias.copy_(b1)

        model.linear2.weight.copy_(w2)
        model.linear2.bias.copy_(b2)
        
        fwd_pass = model(x)
        return fwd_pass.item()
