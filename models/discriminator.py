import abc
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    Implements an abstract discriminator which takes an image as input and classifies it as real/fake
    """

    def __init__(self,
                 image_dim: int = 32,
                 image_channels: int = 3) -> None:
        """

        Parameters
        ----------
        image_dim: Width and height of the square input image
        image_channels: Number of channels in the input image
        """

        super(Discriminator, self).__init__()

        self._config = dict()
        self._config['image_dim'] = image_dim
        self._config['image_channels'] = image_channels

        self._image_dim = image_dim
        self._image_channels = image_channels

    @abc.abstractmethod
    def forward(self,
                images: torch.Tensor) -> torch.Tensor:
        """
        Unimplemented forward function for the neural network.

        Parameters
        ----------
        images: A (?, image_channels, image_dim, image_dim) torch tensor

        Returns
        -------
        output: A (?, 1) torch tensor containing logit values of image being real
        """
        raise NotImplementedError('Function forward in Discriminator not implemented')

    @property
    def image_dim(self):
        return self._image_dim

    @property
    def image_channels(self):
        return self._image_channels

    @property
    def feature_dim(self):
        return self._feature_dim

    @property
    def config(self):
        return self._config
