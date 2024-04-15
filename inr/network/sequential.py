"""
Module containing sequential network.

Classes
-------
Sequential
    Sequential network.
"""

from typing import List

import torch

import inr.layer

from .core import BaseNetwork


class Sequential(BaseNetwork):
    """
    Sequential network.

    Attributes
    ----------
    layers : torch.nn.ModuleList
        List of layers in the network.

    Methods
    -------
    forward(x)
        Forward pass of the network.
    """
    def __init__(
            self,
            input_features: int,
            output_features: int,
            hidden_features: List[int],
            **kwargs
        ):
        """
        Sequential network.

        Parameters
        ----------
        input_features : int
            Number of input features.

        output_features : int
            Number of output features.

        hidden_features : list
            List of hidden features.

        Keyword Arguments
        -----------------
        activation : inr.layer.Activation, optional
            Activation function to use, by default inr.layer.Activation.SINE.

        outermost_activation : str, optional
            Activation function to use in the last layer, by default None.

        first_scale_factor : float, optional
            Scaling factor to use in the first layer, by default 30.
            Relevant only if activation is 'sine'.

        hidden_scale_factor : float, optional
            Scaling factor to use in the hidden layers, by default 30.
            Relevant only if activation is 'sine'.

        bias : bool, optional
            Whether to use bias in the linear layers, by default True.
        """
        super().__init__(**kwargs)

        outermost_activation = kwargs.get('outermost_activation', None)

        self.layers = torch.nn.ModuleList()

        if len(hidden_features) == 0 or hidden_features is None:
            kwargs['activation'] = outermost_activation
            self.layers.append(self._get_first_layer(
                input_features,
                output_features,
                **kwargs
            ))
            return

        self.layers.append(self._get_first_layer(
            input_features,
            hidden_features[0],
            **kwargs
        ))

        for i in range(len(hidden_features) - 1):
            self.layers.append(self._get_layer(
                hidden_features[i],
                hidden_features[i + 1],
                **kwargs
            ))

        kwargs['activation'] = outermost_activation
        self.layers.append(self._get_layer(
            hidden_features[-1],
            output_features,
            **kwargs
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """

        for layer in self.layers:
            x = layer(x)

        return x

    def _get_first_layer(
            self,
            input_features: int,
            output_features: int,
            **kwargs
        ) -> torch.nn.Module:
        """
        Get the first layer in the network.

        Parameters
        ----------
        input_features : int
            Number of input features.

        output_features : int
            Number of output features.

        Keyword Arguments
        -----------------
        activation : str, optional
            Activation function to use, by default 'sine'.

        first_scale_factor : float, optional
            Scaling factor to use in the first layer, by default 30.
            Relevant only if activation is 'sine'.

        bias : bool, optional
            Whether to use bias in the linear layers, by default True.

        Returns
        -------
        torch.nn.Module
            The first layer in the network.
        """

        activation = kwargs.get('activation', inr.layer.Activation.SINE)
        first_scale_factor = kwargs.get('first_scale_factor', 30)
        bias = kwargs.get('bias', True)

        if activation == inr.layer.Activation.LINEAR:
            return inr.layer.BaseLayer(
                input_features,
                output_features,
                bias=bias
            )

        if activation == inr.layer.Activation.SINE:
            return inr.layer.SineLayer(
                input_features,
                output_features,
                is_first=True,
                scale_factor=first_scale_factor,
                bias=bias
            )

        raise ValueError(f'Unknown activation function {activation}')

    def _get_layer(
            self,
            input_features: int,
            output_features: int,
            **kwargs
        ) -> torch.nn.Module:
        """
        Get a hidden layer in the network.

        Parameters
        ----------
        input_features : int
            Number of input features.

        output_features : int
            Number of output features.

        Keyword Arguments
        -----------------
        activation : str, optional
            Activation function to use, by default 'sine'.

        hidden_scale_factor : float, optional
            Scaling factor to use in the hidden layers, by default 30.
            Relevant only if activation is 'sine'.

        bias : bool, optional
            Whether to use bias in the linear layers, by default True.

        Returns
        -------
        torch.nn.Module
            A hidden layer in the network.
        """

        activation = kwargs.get('activation', inr.layer.Activation.SINE)
        hidden_scale_factor = kwargs.get('hidden_scale_factor', 30)
        bias = kwargs.get('bias', True)

        if activation == inr.layer.Activation.LINEAR:
            return inr.layer.BaseLayer(
                input_features,
                output_features,
                bias=bias
            )

        if activation == inr.layer.Activation.SINE:
            return inr.layer.SineLayer(
                input_features,
                output_features,
                scale_factor=hidden_scale_factor,
                bias=bias
            )

        raise ValueError(f'Unknown activation function {activation}')
