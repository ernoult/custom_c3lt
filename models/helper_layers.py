import torch
import torch.nn as nn


class ReshapeLayer(nn.Module):
    """
    Implements a reshape layer for usage in Sequential
    """
    def __init__(self, out_shape: tuple = (4, 4, 4)):
        """
        Parameters
        ----------
        out_shape: Shape of the output excluding the batch dimension
        """
        super(ReshapeLayer, self).__init__()
        self._out_shape = out_shape

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs: A torch tensor of shape (?, multiply(out_shape))

        Returns
        -------
        output: A torch tensor of shape (?, out_shape)
        """
        return torch.reshape(inputs, [-1] + [x for x in self._out_shape])
