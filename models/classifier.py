import abc
import torch
import torch.nn as nn


class Classifier(nn.Module):
    """
    Implements an abstract (encoder, classifier) stack which takes a vector as input and produces a 1-dimensional output
    """

    def __init__(self,
                image_dim: int = 28,
                image_channels: int = 3,            
                feature_dim: int = 32,
                num_classes: int = 10) -> None:
        """

        Parameters
        ----------
        feature_dim: Dimension of the input vector
        num_classes: Number of classes in the dataset
        """

        super(Classifier, self).__init__()

        self._config = dict()

        self._config['image_dim'] = image_dim
        self._config['image_channels'] = image_channels
        self._config['feature_dim'] = feature_dim
        self._config['num_classes'] = num_classes

        self._image_dim = image_dim
        self._image_channels = image_channels
        self._feature_dim = feature_dim
        self._num_classes = num_classes

    @abc.abstractmethod
    def forward(self,
                inputs: torch.Tensor,
                return_feat: bool) -> torch.Tensor:
        """
        Unimplemented forward function for the neural network.

        Parameters
        ----------
        inputs: A (?, feature_dim) torch tensor

        Returns
        -------
        output: A (?, num_classes) log-softmax torch tensor
        """
        raise NotImplementedError('Function forward in Classifier not implemented')

    @property
    def feature_dim(self):
        return self._feature_dim

    @property
    def image_channels(self):
        return self._image_channels

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def config(self):
        return self._config
