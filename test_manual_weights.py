import torch

import inr


if __name__ == "__main__":
    torch.manual_seed(0)

    # Set device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Create the neural network
    net = inr.network.Sequential(
        input_features=2,
        output_features=1,
        hidden_features=[1024, 1024, 1024],
        activation=inr.layer.Activation.SINE,
        outermost_activation=inr.layer.Activation.LINEAR,
        first_scale_factor=30,
        hidden_scale_factor=30,
        bias=True
    ).to(device)

    net_weights = net.state_dict()
    key_list = list(net_weights.keys())

    layer0_weights = net_weights[key_list[0]]
    layer0_bias = net_weights[key_list[1]]
    layer1_weights = net_weights[key_list[2]]
    layer1_bias = net_weights[key_list[3]]
    layer2_weights = net_weights[key_list[4]]
    layer2_bias = net_weights[key_list[5]]
    layer3_weights = net_weights[key_list[6]]
    layer3_bias = net_weights[key_list[7]]

    random_input = torch.rand(100000, 2).to(device)

    layer0_output = net.layers[0].forward(random_input, weight=layer0_weights, bias=layer0_bias)
    layer1_output = net.layers[1].forward(layer0_output, weight=layer1_weights, bias=layer1_bias)
    layer2_output = net.layers[2].forward(layer1_output, weight=layer2_weights, bias=layer2_bias)
    layer3_output = net.layers[3].forward(layer2_output, weight=layer3_weights, bias=layer3_bias)

    net_output = net(random_input)

    print(torch.allclose(layer3_output, net_output))
