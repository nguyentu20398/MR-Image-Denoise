import torch
import torch.nn as nn

class HybridNet(nn.Module):
    def __init__(self, denoise_net: nn.Module, noisemap_net: nn.Module):
        super(HybridNet, self).__init__()
        self.denoise_net = denoise_net
        self.noisemap_net = noisemap_net

    def forward(self, x):
        noise_level = self.noisemap_net(x)
        concat_img = torch.cat([x, noise_level], dim=1)
        out = self.denoise_net(concat_img)
        return noise_level, out






