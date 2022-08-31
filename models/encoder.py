import abc
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Implements an abstract encoder which takes an image as input and produces a vector of features as output
    """

    def __init__(self,
                 image_dim: int = 32,
                 image_channels: int = 3,
                 feature_dim: int = 32) -> None:
        """

        Parameters
        ----------
        image_dim: Width and height of the square input image
        image_channels: Number of channels in the input image
        feature_dim: Number of elements in the output vector
        """

        super(Encoder, self).__init__()

        self._config = dict()
        self._config['image_dim'] = image_dim
        self._config['image_channels'] = image_channels
        self._config['feature_dim'] = feature_dim

        self._image_dim = image_dim
        self._image_channels = image_channels
        self._feature_dim = feature_dim

    @abc.abstractmethod
    def forward(self,
                inputs: torch.Tensor) -> torch.Tensor:
        """
        Unimplemented forward function for the neural network.

        Parameters
        ----------
        inputs: A (?, image_channels, image_dim, image_dim) torch tensor

        Returns
        -------
        output: A (?, feature_dim) torch tensor
        """
        raise NotImplementedError('Function forward in Encoder not implemented')

    @property
    def image_dim(self):
        return self._image_dim

    @property
    def feature_dim(self):
        return self._feature_dim

    @property
    def image_channels(self):
        return self._image_channels

    @property
    def config(self):
        return self._config
