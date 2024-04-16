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

    def forward(
            self,
            x: torch.Tensor,
            weight: torch.Tensor = None,
            bias: torch.Tensor = None
        ) -> torch.Tensor:
        """
        Forward pass of the layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        weight : torch.Tensor, optional
            Weight tensor, by default None.
            When provided, uses the given weight tensor instead of the learned weights.

        bias : torch.Tensor, optional
            Bias tensor, by default None.
            When provided, uses the given bias tensor instead of the learned bias.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """

        if weight is None:
            return self.linear(x)

        x = torch.matmul(x, weight.T)
        if bias is None:
            return x
        return x + bias
