import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from loguru import logger

from src.utils.filters import sobel


def get_criterion(name: str, **kwargs):
    if name == 'L1':
        criterion = Mean_Absolute_Error()
    elif name == 'L2':
        try:
            logger.debug(f"Param loss: {kwargs}")
        except:
            logger.error("Can get alpha loss from")
        criterion = Mean_Square_Error(alpha=kwargs.get('alpha_loss', 1), beta=kwargs.get('beta_loss', 0.5))
    elif name == 'L1+L2':
        criterion = L1AddL2(alpha=kwargs.get('alpha', 0.5), beta=kwargs.get('beta', 0.5))
    elif name == 'hybrid':
        criterion = Hybrid(alpha=kwargs.get('alpha', 0.5), beta=kwargs.get('beta', 0.05))
    else:
        logger.error(f"We don't support Loss {name}")
        sys.exit()
    return criterion


class Mean_Absolute_Error(_Loss):  # PyTorch 0.4.1

    def __init__(self, size_average=None, reduce=None, reduction="mean"):
        super(Mean_Absolute_Error, self).__init__(size_average, reduce, reduction)
        self.reduction = reduction
        self.loss = torch.nn.L1Loss(reduction="mean")

    def forward(self, input, target):
        output = self.loss(input, target).div_(2)
        return output


class Mean_Square_Error(_Loss):  # PyTorch 0.4.1

    def __init__(self, size_average=None, reduce=None, reduction="mean", alpha=1, **kwargs):
        super(Mean_Square_Error, self).__init__(size_average, reduce, reduction)
        self.reduction = reduction
        self.alpha = alpha
        self.beta = kwargs.get('beta', 0.1)

        self.n = 0

    def forward(self, predict, label):
        l2_loss = F.mse_loss(predict, label, reduction=self.reduction).div_(2)
        # l2_loss = torch.mean(torch.abs(0.3 - torch.lt(label, predict).float()) * torch.pow(predict - label, 2))

        count_h = self._tensor_size(predict)
        count_w = self._tensor_size(predict)
        h_tv = torch.pow((torch.roll(predict, shifts=1, dims=2) - predict), 2).sum()
        w_tv = torch.pow((torch.roll(predict, shifts=1, dims=3) - predict), 2).sum()
        tvloss = h_tv / count_h + w_tv / count_w

        if self.n > 1000:
            logger.info(f"l2: {round(l2_loss.item(), 4)} - tvloss: {round(tvloss.item(), 4)}")
            self.n = 0
        self.n += 1

        loss = self.alpha * l2_loss + self.beta * tvloss
        return loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class Hybrid(_Loss):  # PyTorch 0.4.1

    def __init__(self, size_average=None, reduce=None, reduction="mean", alpha=1, **kwargs):
        super(Hybrid, self).__init__(size_average, reduce, reduction)
        self.reduction = reduction
        self.alpha = alpha
        self.beta = kwargs.get('beta', 0.1)
        self.main_loss = torch.nn.L1Loss(reduction=reduction)

        self.n = 0

    def forward(self, predict_noise, predict_image, gt_noise, gt_image):
        l1_loss = self.main_loss(predict_image, gt_image)
        l2_loss = torch.mean(
            torch.abs(0.3 - torch.lt(gt_noise, predict_noise).float()) * torch.pow(predict_noise - gt_noise, 2))

        count_h = self._tensor_size(predict_noise)
        count_w = self._tensor_size(predict_noise)
        h_tv = torch.pow((torch.roll(predict_noise, shifts=1, dims=2) - predict_noise), 2).sum()
        w_tv = torch.pow((torch.roll(predict_noise, shifts=1, dims=3) - predict_noise), 2).sum()
        tvloss = h_tv / count_h + w_tv / count_w

        if self.n > 1000:
            logger.info(f"l2: {round(l2_loss.item(), 4)} - tvloss: {round(tvloss.item(), 4)}")
            self.n = 0
        self.n += 1

        loss = l1_loss + self.alpha * l2_loss + self.beta * tvloss
        return loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class L1AddL2(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', alpha=0.5, beta=0.5) -> None:
        super().__init__(size_average, reduce, reduction)
        self.reduction = reduction
        self.alpha = alpha
        self.beta = beta
        self.l1 = torch.nn.L1Loss(reduction="mean")
        self.l2 = torch.nn.MSELoss(reduction="mean")

    def forward(self, input, target):
        output = self.alpha * self.l1(input, target) + self.beta * self.l2(input, target)
        return output


class fixed_loss(_Loss):
    def __init__(self, alpha=1, beta=0, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, est_noise, gt_noise, if_asym):
        asym_loss = torch.mean(
            if_asym * torch.abs(0.3 - torch.lt(gt_noise, est_noise).float()) * torch.pow(est_noise - gt_noise, 2))

        h_x = est_noise.size()[2]
        w_x = est_noise.size()[3]
        count_h = self._tensor_size(est_noise[:, :, 1:, :])
        count_w = self._tensor_size(est_noise[:, :, :, 1:])
        h_tv = torch.pow((est_noise[:, :, 1:, :] - est_noise[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((est_noise[:, :, :, 1:] - est_noise[:, :, :, :w_x - 1]), 2).sum()
        tvloss = h_tv / count_h + w_tv / count_w

        loss = self.alpha * asym_loss + self.beta * tvloss

        return loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]
