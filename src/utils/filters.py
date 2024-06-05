import torch
import torch.nn.functional as F
from torch import Tensor

def get_sobel_kernel_3x3() -> Tensor:
    """Utility function that returns a sobel kernel of 3x3."""
    return torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])

def get_sobel_kernel2d() -> Tensor:
    kernel_x = get_sobel_kernel_3x3()
    kernel_y = kernel_x.transpose(0, 1)
    return torch.stack([kernel_x, kernel_y])

def spatial_gradient(input: Tensor) -> Tensor:
    r"""
    Args:
        input: input image tensor with shape :math:`(B, C, H, W)`.
        mode: derivatives modality, can be: `sobel` or `diff`.
        order: the order of the derivatives.
        normalized: whether the output is normalized.

    Return:
        the derivatives of the input feature map. with shape :math:`(B, C, 2, H, W)`
    """

    kernel = get_sobel_kernel2d().to(input.device)

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...]
    # print(tmp_kernel)
    # Pad with "replicate for spatial dims, but with zeros for channel
    spatial_pad = [kernel.size(1) // 2, kernel.size(1) // 2, kernel.size(2) // 2, kernel.size(2) // 2]
    # print(spatial_pad)
    out_channels = 2
    # print(input.reshape(b * c, 1, h, w))
    padded_inp = F.pad(input.reshape(b * c, 1, h, w), spatial_pad, 'replicate')
    # print(padded_inp)
    out = F.conv2d(padded_inp, tmp_kernel, groups=1, padding=0, stride=1)
    return out.reshape(b, c, out_channels, h, w)

def sobel(input: Tensor, eps: float = 1e-6) -> Tensor:
    r"""

    Args:
        input: the input image with shape :math:`(B,C,H,W)`.
        normalized: if True, L1 norm of the kernel is set to 1.
        eps: regularization number to avoid NaN during backprop.

    Return:
        the sobel edge gradient magnitudes map with shape :math:`(B,C,H,W)`.
    """
    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")

    # comput the x/y gradients
    edges = spatial_gradient(input)

    # unpack the edges
    gx = edges[:, :, 0]
    gy = edges[:, :, 1]

    # compute gradient maginitude
    magnitude = torch.sqrt(gx * gx + gy * gy + eps)

    return magnitude