import torch
import torch.nn as nn

def conv_block(in_channels, out_channels, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2)
    )


def linear_block(in_features, out_features, activation=True):
    layers = [nn.Linear(in_features, out_features)]
    if activation:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)



class model(nn.Module):
    def __init__(self, name, num_layers, input_channels, hidden_channels, kernel_sizes, linear_layers):
        super(model, self).__init__()
        layers = []
        classifier_layers = []
        self.name = name
        self.num_layers = num_layers
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_sizes = kernel_sizes
        self.linear_layers = linear_layers
        
        for i in range(0, self.num_layers):
            layers.append(conv_block(input_channels[i], hidden_channels[i], kernel_sizes[i]))
        
        self.features = nn.Sequential(*layers)
        classifier_layers.extend([nn.Flatten(),linear_block(linear_layers[0], linear_layers[1]), linear_block(linear_layers[1], linear_layers[2]), linear_block(linear_layers[2], linear_layers[3], activation=False)])
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
        