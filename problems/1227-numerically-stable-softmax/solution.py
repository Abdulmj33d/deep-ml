import torch

def softmax(t, dim):
    """Numerically stable softmax along dim.

    Args:
        t (torch.Tensor): input tensor
        dim (int): dimension along which to apply softmax

    Returns:
        torch.Tensor: tensor of same shape as t; slices along dim sum to 1
    """
    # TODO: subtract max along dim, exp, then normalize
    c, _ = torch.max(t, dim, keepdims= True)
    stable_e = torch.exp(t - c)
    softmax= stable_e / torch.sum(stable_e, dim= dim, keepdim= True)

    
    # max_t, _ = torch.max(t, dim=-1, keepdim=True)
    # e_t = torch.exp(t - max_t)
    # softmax = e_t / torch.sum(e_t, dim= dim,  keepdim= True)


    return softmax
