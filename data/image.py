"""
Module containing the `ImageData` class.

Classes
-------
ImageData
    Dataset for an image.
"""

import os
from typing import Optional, Tuple

import PIL.Image
import torch
import torchvision

from .utils import get_grid, get_image_extensions

DEFAULT_TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(0.5, 0.5)
])

SUPPORTED_FORMATS = get_image_extensions()


def _is_image(file_path : str) -> bool:
    """
    Checks if the file is supported image.

    Parameters
    ----------
    file_path : str
        Path to the file.

    Returns
    -------
    bool
        True if the file is a supported image, False otherwise.
    """

    for fmt in SUPPORTED_FORMATS:
        if file_path.lower().endswith(fmt):
            return True
    return False

class ImageData(torch.utils.data.Dataset):
    """
    Dataset for an image.
    Derived from `torch.utils.data.Dataset`.

    Attributes
    ----------
    pixels : torch.Tensor
        Pixels of the image.
        Shape of the tensor is (num_pixels, num_channels).

    height : int
        Height of the image (in pixels).

    width : int
        Width of the image (in pixels).

    num_channels : int
        Number of channels in the image.

    coord_grid : torch.Tensor
        Coordinates of the pixels.
        Shape of the tensor is (num_pixels, 2).

    Methods
    -------
    __len__()
        Returns the number of pixels in the image.

    __getitem__(index)
        Returns the coordinates and corresponding pixel value.

    get_image_size()
        Returns the size of the image.

    get_channels()
        Returns the number of channels in the image.
    """
    def __init__(
            self,
            file_path : str,
            image_size : Optional[Tuple[int, int]] = None,
            transform : torchvision.transforms.transforms.Compose = DEFAULT_TRANSFORM,
            ranges : Optional[Tuple[Tuple[float, float], ...]] = None,
            **kwargs
        ):
        """
        Dataset for an image.

        Parameters
        ----------
        file_path : str
            Path to the image file. Supports any format supported by PIL.

        image_size : Tuple[int, int], optional
            Size of the image.
            Resizes the image if specified, otherwise uses the original size.
            The format is (height, width).

        transform : torchvision.transforms.transforms.Compose, optional
            Transform to apply to the image.
            Defaults to the default transform of converting to tensor
            and normalizing to [-1, 1].

        ranges : Tuple[Tuple[float, float], ...], optional
            Ranges of each dimension.
            Defaults to [-1, 1] for each dimension.

        Keyword Arguments
        -----------------
        coords_noise : float, optional
            Noise to add to the coordinates.
            Noise is sampled from a uniform distribution in the range
            [`-coords_noise`, `coords_noise`].
            Defaults to 0.0.

        coords_noise_prob : float, optional
            Probability of adding noise to the coordinates.
            Ignored if `coords_noise` is 0.
            Defaults to 0.1.

            Note: Noise is added to x and y coordinates independently.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.

        ValueError
            If the file is not a valid image type.
            Valid image types are specified in `SUPPORTED_FORMATS`.

        ValueError
            If `image_size` is not a tuple of length 2.
        """
        super().__init__()

        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")

        # Check if the file is an image
        if not _is_image(file_path):
            raise ValueError(f"File {file_path} is not a valid image type."
                             f"Supported types are {SUPPORTED_FORMATS}.")

        # Load the image
        img = PIL.Image.open(file_path)

        # Resize the image if specified
        if image_size is not None:
            if isinstance(image_size, int):
                image_size = (image_size, image_size)
            if len(image_size) != 2:
                raise ValueError(f"image_size must be a tuple of length 2, got {len(image_size)}.")
            height, width = image_size
            img = img.resize((width, height))

        # Get the height and width of the image
        self.width, self.height = img.size

        # Apply the transform
        img = transform(img)

        # Shape of img is (C, H, W)
        self.pixels = img.reshape(img.shape[0], -1).T
        self.num_channels = self.pixels.shape[1]

        # Get the grid
        self.coord_grid = get_grid(2, (self.height, self.width), ranges=ranges)

        coords_noise = kwargs.get('coords_noise', 0.0)
        if coords_noise > 0:
            coords_noise_prob = kwargs.get('coords_noise_prob', 0.1)

            # Generate Bernoulli RV with probability coords_noise_prob
            mask = torch.rand_like(self.coord_grid) < coords_noise_prob

            # Generate noise
            # Uniform distribution in the range [-coords_noise, coords_noise]
            noise = 2 * coords_noise * torch.rand_like(self.coord_grid) - coords_noise

            # Apply noise to the coordinates
            self.coord_grid[mask] += noise[mask]


    def __len__(self) -> int:
        """
        Returns the number of pixels in the image.

        Returns
        -------
        int
            Number of pixels in the image.
        """

        return self.pixels.shape[0]

    def __getitem__(self, index : int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the coordinates and corresponding pixel value.

        Parameters
        ----------
        index : int
            Index of the coordinate.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Coordinate and corresponding pixel value.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        """

        if index >= len(self) or index < 0:
            raise IndexError

        return self.coord_grid[index], self.pixels[index]

    def get_image_size(self) -> Tuple[int, int]:
        """
        Returns the size of the image.

        Returns
        -------
        Tuple[int, int]
            Size of the image in the format (height, width).
        """

        return (self.height, self.width)

    def get_channels(self) -> int:
        """
        Returns the number of channels in the image.

        Returns
        -------
        int
            Number of channels in the image.
        """

        return self.num_channels
