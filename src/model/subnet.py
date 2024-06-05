import torch
import torch.nn as nn


class SubNet(nn.Module):
    def __init__(self, channels=1, num_of_layers=17, features=192, **kwargs):
        super(SubNet, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=channels,
                out_channels=features,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        )
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(
                    in_channels=features,
                    out_channels=features,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                )
            )
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(
                in_channels=features,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        )
        self.subnet = nn.Sequential(*layers)

    def forward(self, x):
        out = self.subnet(x)
        return out
