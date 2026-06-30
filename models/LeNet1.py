import torch
import torch.nn as nn

def conv_block(in_channels, out_channels, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2)
    )


class model(nn.Module):
    def __init__(self, name, num_layers, input_channels, hidden_channels, kernel_sizes):
        super(model, self).__init__()
        layers = []
        self.name = name
        self.num_layers = num_layers
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_sizes = kernel_sizes
        
        for i in range(0, self.num_layers-1):
            layers.append(conv_block(input_channels[i], hidden_channels[i], kernel_sizes[i]))
        
        layers.extend([nn.Conv2d(in_channels = self.input_channels[-1], out_channels = self.hidden_channels[-1], kernel_size=self.kernel_sizes[-1]),
            nn.Tanh()])
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return x
        