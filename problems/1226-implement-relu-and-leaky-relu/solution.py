import torch

def relu(t):
    """Element-wise ReLU: max(0, t).

    Args:
        t (torch.Tensor): input tensor

    Returns:
        torch.Tensor: activated tensor
    """
    # TODO: implement with pure torch ops
    t_copy = t.clone()
    return torch.clamp(t_copy, min= 0)

def leaky_relu(t, slope=0.01):
    """Element-wise Leaky ReLU with given negative slope.

    Args:
        t (torch.Tensor): input tensor
        slope (float): slope for negative values

    Returns:
        torch.Tensor: activated tensor
    """
    # TODO: implement with pure torch ops
    t_copy = t.clone()
    return torch.where(t_copy > 0, t_copy, t_copy * slope)
