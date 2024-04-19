"""
Module containing utility functions for data processing.

Functions
---------
get_grid
    Returns coordinates spanning the grid of given shape and dimension.

get_image_extensions
    Returns a list of supported image extensions.
"""

from typing import List, Optional, Tuple

import numpy as np
import PIL.Image
import torch


def get_grid(
        dim : int,
        shape : Tuple[int, ...],
        ranges : Optional[Tuple[Tuple[float, float], ...]] = None
    ) -> torch.Tensor:
    """
    Returns coordinates spanning the grid of given shape and dimension.

    Parameters
    ----------
    dim : int
        Dimension of the grid.

    shape : Tuple[int, ...]
        Shape of the grid.

    ranges : Optional[Tuple[Tuple[float, float], ...]], optional
        Ranges of each dimension
        Defaults to [-1, 1] for each dimension.

    Returns
    -------
    torch.Tensor
        Coordinates spanning the grid of given shape and dimension.
        Shape of the returned tensor is (shape[0] * ... * shape[dim-1], dim).

    Raises
    ------
    ValueError
        If shape and ranges do not have the same length as dimension.

    Examples
    --------
    >>> get_grid(2, (2, 2))
    tensor([[-1., -1.],
            [-1.,  1.],
            [ 1., -1.],
            [ 1.,  1.]])

    >>> get_grid(2, (2, 2), ((0, 1), (0, 1)))
    tensor([[0., 0.],
            [0., 1.],
            [1., 0.],
            [1., 1.]])

    Notes
    -----
    The entire grid is returned as a single tensor. This might be problematic
    for large grids. In such cases, consider breaking the grid into smaller
    chunks and processing them separately.
    """

    if len(shape) != dim:
        raise ValueError("Shape must have the same length as dimension.")
    if ranges is None:
        ranges = ((-1, 1),) * dim
    if len(ranges) != dim:
        raise ValueError("Ranges must have the same length as dimension.")

    grid = []
    for i, (start, end) in enumerate(ranges):
        grid.append(torch.linspace(start, end, shape[i]))
    grid = torch.stack([x.flatten() for x in torch.meshgrid(*grid, indexing='ij')], dim=-1)
    return grid

def get_image_extensions() -> List[str]:
    """
    Returns a list of supported image extensions.

    Returns
    -------
    List[str]
        List of supported image extensions.
    """

    exts = PIL.Image.registered_extensions()
    return [ext for ext, fmt in exts.items() if fmt in PIL.Image.OPEN]

def save_image(
        pixels: torch.Tensor,
        file_path: str,
        image_size: Optional[Tuple[int, int]] = None
    ) -> None:
    """
    Save the pixels as an image.

    Parameters
    ----------
    pixels : torch.Tensor
        Pixels to be saved as an image.
        Shape of the tensor is (num_pixels, num_channels).
    file_path : str
        Path to the image file.
    image_size : Optional[Tuple[int, int]], optional
        Size of the image, by default None.
    """

    if image_size is None:
        image_size = (int(pixels.shape[0] ** 0.5),) * 2
    elif isinstance(image_size, int):
        image_size = (image_size,) * 2

    img = pixels.reshape(image_size[0], image_size[1], -1).cpu().detach().numpy()
    img = img * 0.5 + 0.5
    img = img * 255
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)

    img = PIL.Image.fromarray(np.squeeze(img))
    img.save(file_path)
