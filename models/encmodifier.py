import abc
import torch
import torch.nn as nn


class Encmodifier(nn.Module):
    """
    Implements an abstract encoder modifier which takes an embedding z as an input 
    and produces another embedding as an output
    """

    def __init__(self,
                 input_dim: int = 32,
                 n_steps: int = 1) -> None:
        """

        Parameters
        ----------
        input_dim: dimension of the encoder output
        """

        super(Encmodifier, self).__init__()

        self._config = dict()
        self._config['input_dim'] = input_dim
        self._config['n_steps'] = n_steps

        self._input_dim = input_dim
        self._n_steps = n_steps
    
    @abc.abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Unimplemented forward function for the neural network.

        Parameters
        ----------
        encodings: A (?, input_dim) torch tensor

        Returns
        -------
        output: A (?, input_dim) torch tensor
        """

        raise NotImplementedError('_model in Encmodifier not implemented')



    @property
    def input_dim(self):
        return self._input_dim

    @property
    def n_steps(self):
        return self._n_steps

    @property
    def config(self):
        return self._config
