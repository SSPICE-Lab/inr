"""
Module containing list of activation functions.

Classes
-------
Activation
    Activation functions.
"""

from enum import Enum


class Activation(Enum):
    """
    Activation functions.

    Attributes
    ----------
    LINEAR : None
        Linear activation function.

    SINE : str
        Sine activation function.
    """
    LINEAR = None
    SINE = 'sine'
