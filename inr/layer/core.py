"""
Module containing the base layer class.

Classes
-------
BaseLayer
    Base class for all INR layers.
"""

import torch


class BaseLayer(torch.nn.Module):
    """
    Base class for all INR layers.

    Attributes
    ----------
    in_features : int
        Number of input features.

    out_features : int
        Number of output features.

    linear : torch.nn.Linear
        The core linear layer.

    Methods
    -------
    forward(x)
        Forward pass of the layer.
    """
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            **kwargs
        ):
        """
        Base class for all INR layers.

        Parameters
        ----------
        in_features : int
            Number of input features.

        out_features : int
            Number of output features.

        bias : bool, optional
            Whether to use bias in the linear layer, by default True.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.linear = torch.nn.Linear(
            in_features,
            out_features,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """

        return self.linear(x)
