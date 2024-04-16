"""
Module containing dataset for a coordinate grid.

Classes
-------
CoordGrid
    Dataset for a coordinate grid.
"""

from typing import Optional, Tuple

import torch

from .utils import get_grid


class CoordGrid(torch.utils.data.Dataset):
    """
    Dataset for a coordinate grid.

    Attributes
    ----------
    coord_grid : torch.Tensor
        Coordinates spanning the grid.
        Shape of the tensor is (num_points, dim).
    """
    def __init__(
            self,
            dim: int,
            shape: Tuple[int, ...],
            ranges: Optional[Tuple[Tuple[float, float], ...]] = None
        ):
        """
        Dataset for a coordinate grid.

        Parameters
        ----------
        dim : int
            Dimension of the grid.

        shape : Tuple[int, ...]
            Shape of the grid.

        ranges : Optional[Tuple[Tuple[float, float], ...]], optional
            Ranges of each dimension
            Defaults to [-1, 1] for each dimension.
        """
        super().__init__()

        self.coord_grid = get_grid(dim, shape, ranges)

    def __len__(self) -> int:
        """
        Returns the number of pixels in the image.
        """

        return len(self.coord_grid)

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Returns the coordinates.
        """

        return self.coord_grid[index]
