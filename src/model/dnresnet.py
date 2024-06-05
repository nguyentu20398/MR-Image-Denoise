import torch.nn as nn
import torch.nn.init as init
from loguru import logger


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True
    ):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(in_channels, eps=0.0001, momentum=0.95)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )
        self.bn2 = nn.BatchNorm2d(in_channels, eps=0.0001, momentum=0.95)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = input + x
        x = self.relu2(x)
        return x


class DnResnet(nn.Module):
    def __init__(self, depth, input_channels=1, n_channels=64, kernel_size=3, out_channels = 1):
        super(DnResnet, self).__init__()
        padding = 1
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)
        residual_blocks = []
        for _ in range((depth - 2) // 2):
            residual_blocks.append(
                ResidualBlock(in_channels=n_channels, out_channels=n_channels)
            )

        self.residual_blocks = nn.Sequential(*residual_blocks)
        self._initialize_weights()

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(x)
        x = self.residual_blocks(x)
        x = self.conv2(x)
        # out = input - x
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                logger.info("init weight")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
