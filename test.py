import torch

import inr


if __name__ == "__main__":
    torch.manual_seed(0)

    # Load the image and create a dataloader
    data = inr.data.image.ImageData("test_images/cameraman.jpeg")
    dataloader = torch.utils.data.DataLoader(data, batch_size=100000, shuffle=True)

    # Create the neural network
    net = inr.network.Sequential(
        input_features=data.coord_grid.shape[-1],
        output_features=data.get_channels(),
        hidden_features=[1024, 1024, 1024],
        activation=inr.layer.Activation.SINE,
        outermost_activation=inr.layer.Activation.LINEAR,
        first_scale_factor=30,
        hidden_scale_factor=30,
        bias=True
    )

    # Create the optimizer and the learning rate scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Train the network
    net.fit(
        dataloader,
        epochs=1000,
        optimizer=optimizer,
        verbose=True,
        save_path="test_models/cameraman.pt"
    )

    # Load best model
    net.load_state_dict(torch.load("test_models/cameraman.pt"))

    # Generate the test coordinates
    test_coords = inr.data.CoordGrid(2, (1024, 1024))
    test_dataloader = torch.utils.data.DataLoader(test_coords, batch_size=100000, shuffle=False)

    # Generate the test images
    test_images = net.generate(test_dataloader)

    # Save the generated images
    inr.data.utils.save_image(
        test_images,
        "test_images/cameraman_generated.png",
        image_size=(1024, 1024))
