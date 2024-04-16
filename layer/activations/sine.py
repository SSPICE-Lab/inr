"""
Module containing the `SineLayer` class.

Classes
-------
SineLayer
    Sine activation layer.
"""

import numpy as np
import torch

from inr.layer.core import BaseLayer


class SineLayer(BaseLayer):
    """
    Sine activation layer.
    Based on the paper: https://arxiv.org/abs/2006.09661

    Attributes
    ----------
    in_features : int
        Number of input features.

    out_features : int
        Number of output features.

    linear : torch.nn.Linear
        The core linear layer.

    is_first : bool
        Whether the layer is the first layer in the network.

    scale_factor : float
        Scaling parameter in the sine activation function.

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
        Sine activation layer.

        Parameters
        ----------
        in_features : int
            Number of input features.

        out_features : int
            Number of output features.

        bias : bool, optional
            Whether to use bias in the linear layer, by default True.

        Keyword Arguments
        -----------------
        is_first : bool, optional
            Whether the layer is the first layer in the network, by default False.

        scale_factor : float, optional
            Scaling parameter in the sine activation function, by default 30.
        """
        super().__init__(in_features, out_features, bias=bias, **kwargs)

        self.is_first = kwargs.get('is_first', False)
        self.scale_factor = kwargs.get('scale_factor', 30)

        self._init_weights()

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
            When provided, the layer uses the given weight tensor instead of the learned weights.

        bias : torch.Tensor, optional
            Bias tensor, by default None.
            When provided, the layer uses the given bias tensor instead of the learned biases.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """

        if weight is None:
            return torch.sin(self.scale_factor * self.linear(x))

        x = torch.matmul(x, weight.T)
        if bias is not None:
            x += bias
        return torch.sin(self.scale_factor * x)

    def _init_weights(self) -> None:
        """
        Initialize the weights of the layer.
        """

        if self.is_first:
            torch.nn.init.uniform_(
                self.linear.weight,
                -1 / self.in_features,
                1 / self.in_features
            )
        else:
            torch.nn.init.uniform_(
                self.linear.weight,
                -np.sqrt(6 / self.in_features) / self.scale_factor,
                np.sqrt(6 / self.in_features) / self.scale_factor
            )
