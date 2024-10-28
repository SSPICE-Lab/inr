# Implicit Neural Representation Package

This package provides PyTorch-based modules for implicit neural representations. The package is based on the paper [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661) by Vincent Sitzmann et al. This package is used as a submodule in internal INR-based projects.

## Pre-requisites

- Python 3.6+
- PyTorch 1.6+
- NumPy
- PIL

## Package Structure

The package is structured as follows:

- `inr/`: The main package directory.
  - `data`: Module for handling data.
    - `image.py`: Module for handling image pixel data (target of INR).
    - `coord.py`: Module for handling coordinate data (input of INR).
  - `network`: Module for easy construction of INR networks.
    - `core.py`: Base network module for training and evaluation.
    - `sequential.py`: Sequential network module for easy construction of standard INR networks.
  - `layer`: Module for INR layers.
    - `core.py`: Base layer module for INR layers.
    - `sine.py`: SIREN layer.
  - `README.md`: The package README file.
  - `LICENSE`: The package license file.

## Usage

The package provides modules for handling data, constructing networks, and defining layers. The following is an example of constructing a simple INR network to generate an image from a 2D coordinate input:

```python
import torch
import inr

# Construct a simple INR network
net = inr.network.Sequential(
    input_features=2,
    output_features=3,
    hidden_features=[256, 256],
    activation=inr.layer.Activation.SINE,
    outermost_activation=inr.layer.Activation.LINEAR
)

# Load image data
image_data = inr.data.image.ImageData('path/to/image.png')
dataloader = torch.utils.data.DataLoader(
    image_data,
    batch_size=100000
)

# Train the network
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
net.fit(dataloader, optimizer, epochs=1000)

# Generate a 256x256 image
coord = inr.data.CoordGrid(2, (256, 256))
test_loader = torch.utils.data.DataLoader(
    coord,
    batch_size=100000
)
test_image = net.generate(test_loader)

# Save the generated image
inr.data.utils.save_image(
    test_image,
    'path/to/generated_image.png',
    image_size=(256, 256)
)
```

## License

This package is licensed under the Apache License 2.0. See the `LICENSE` file for more details.

## Contact

For any questions or issues, please contact the author at syerragu@buffalo.edu.
