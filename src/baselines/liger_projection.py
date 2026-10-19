"""Equivalent matrix backend for LIGER's full-width residual convolutions."""
from torch import nn
import torch.nn.functional as F


class FullWidthLinear(nn.Module):
    """Keep Conv1d parameter identity, names and shape; change only its kernel.

    For x[N,1,D], W[O,1,D], stride=D and zero padding, Conv1d produces
    y[N,O,1] = (x[:,0,:] @ W[:,0,:].T + bias)[...,None]. No parameters,
    dropout draws, activations or residual connections are removed.
    """
    def __init__(self, convolution):
        super().__init__()
        if (not isinstance(convolution, nn.Conv1d) or convolution.in_channels != 1
                or convolution.groups != 1 or convolution.padding != (0,)
                or convolution.dilation != (1,) or convolution.padding_mode != 'zeros'
                or convolution.stride != convolution.kernel_size):
            raise ValueError('Expected single-channel, unpadded full-width residual convolution')
        self.width = convolution.kernel_size[0]
        self.weight = convolution.weight
        self.bias = convolution.bias
        self.train(convolution.training)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (1, self.width):
            raise ValueError('Residual input must be N x 1 x full kernel width')
        return F.linear(x[:, 0, :], self.weight[:, 0, :], self.bias).unsqueeze(-1)


def use_linear_residuals(projection):
    """In-place execution adaptation; state_dict and optimizer references survive."""
    if not isinstance(getattr(projection, 'residuals', None), nn.ModuleList) or not projection.residuals:
        raise ValueError('Expected official MLP residual ModuleList')
    replacements = [FullWidthLinear(layer) for layer in projection.residuals]
    for index, replacement in enumerate(replacements):
        projection.residuals[index] = replacement
    return len(replacements)
