import abc
import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Implements an abstract decoder which takes an image and a vector as input and produces an image as output
    """

    def __init__(self,
                 image_dim: int = 32,
                 image_channels: int = 3,
                 feature_dim: int = 32) -> None:
        """

        Parameters
        ----------
        image_dim: Number of channels in the output image (same channels are expected of input image)
        image_channels: Width and height of the square output image (same dimensions are expected of input image)
        feature_dim: Dimension of the input vector
        """

        super(Decoder, self).__init__()

        self._config = dict()
        self._config['image_dim'] = image_dim
        self._config['image_channels'] = image_channels
        self._config['feature_dim'] = feature_dim

        self._image_dim = image_dim
        self._image_channels = image_channels
        self._feature_dim = feature_dim

    @abc.abstractmethod
    def forward(self,
                encodings: torch.Tensor) -> torch.Tensor:
        """
        Unimplemented forward function for the neural network.

        Parameters
        ----------
        encodings: A (?, feature_dim) torch tensor

        Returns
        -------
        output: A (?, image_channels, image_dim, image_dim) torch tensor
        """
        raise NotImplementedError('Function forward in Decoder not implemented')

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
