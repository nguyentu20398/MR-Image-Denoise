import glob
import json
import os
import random

import cv2
import numpy as np
import torch
from loguru import logger
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.dataset import data_aug, datagenerator, read_npy
from src.utils import write_log


class NoiseMapDataset(Dataset):
    def __init__(
            self,
            data_dir="data",
            phase="train",
            stride=9,
            patch_size=21,
            sigma=5,
            normalize=False,
            norm_noise=80,
            clip=False, **kwargs
    ):
        super(NoiseMapDataset, self).__init__()
        self.phase = phase
        self.sigma = sigma

        if self.phase == "test":
            self.data = glob.glob(f"{data_dir}/{phase}/sigma_{sigma}/*.npy")
        else:
            self.data = datagenerator(
                data_dir, phase=phase, stride=stride, patch_size=patch_size
            )
        self.transform = transforms.ToTensor()
        self.normalize = normalize
        self.norm_noise = norm_noise
        self.clip = clip
        self.noise_mode = kwargs.get('noise_mode', 'rice')
        logger.success(f"{phase} subnet with - sigma: {sigma} - noise_mode: {self.noise_mode}")

    def __getitem__(self, index):
        # Origin Image
        if self.phase == "test":
            y = read_npy(self.data[index], normalize=False)

            x = read_npy(self.data[index].replace(f"/sigma_{self.sigma}", "/origin"))
            if self.clip:
                y = y.clip(0, 255)
        else:
            x = self.data[index]
            x = data_aug(x, random.randint(0, 7))

            # Noise
            noise1 = np.random.randn(x.shape[0], x.shape[1]) * self.sigma
            noise2 = np.random.randn(x.shape[0], x.shape[1]) * self.sigma

            if self.noise_mode == 'rice':
                # Noisy Image
                y = np.sqrt((x + noise1) * (x + noise1) + noise2 * noise2)

            elif self.noise_mode == 'gausse':
                y = x + noise1

        x = x * 0 + self.sigma
        # Noise map
        x = self.transform(x.astype(np.int32)).float()
        # Noised image
        y = self.transform(y.astype(np.int32)).float()

        if self.clip:
            x = x.clamp(0, 255)
            y = y.clamp(0, 255)

        if self.normalize:
            x = x.div(self.norm_noise)
            y = y.div(255)

        return y, x

    def __len__(self):
        return len(self.data)


class NoiseMapTestPhase:
    def __init__(
            self,
            range_sigma: list = [5, 80],
            step: int = 5,
            data_dir: str = "data",
            n_workers: int = 8,
            batch_size: int = 1,
            log_dir="result/test",
            normalize=False,
            norm_noise=80,
            clip=False,
            device=None, **kwargs
    ) -> None:
        self.dataloader = {}
        self.range_sigma = range_sigma
        self.step = step
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.normalize = normalize
        self.clip = clip
        self.range_sigma = range_sigma
        self.norm_noise = norm_noise
        if device is not None:
            self.device = device
        else:
            self.device = "cpu"
        for sigma in range_sigma:
            dataset = NoiseMapDataset(
                data_dir=data_dir,
                phase="test",
                sigma=sigma,
                normalize=normalize,
                clip=clip,
                noise_mode=kwargs.get('noise_mode', 'rice'),
                norm_noise=self.norm_noise
            )
            self.dataloader[sigma] = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=n_workers,
            )

    @torch.no_grad()
    def run(self, model, epoch):
        torch.set_grad_enabled(False)
        model.train(False)
        model.eval()
        model.to(self.device)
        epoch_mse = {}

        for sigma in tqdm(
                self.range_sigma,
                desc=f"Phase: TEST",
        ):
            epoch_mse[sigma] = 0
            for i, data in enumerate(self.dataloader[sigma]):
                noisy_imgs, noise_map = data[0].to(self.device), data[1].to(self.device)

                predicted_noise_map = model(noisy_imgs)

                if self.normalize:
                    predicted_noise_map = predicted_noise_map * self.norm_noise
                    noise_map = noise_map * self.norm_noise
                if self.clip:
                    predicted_noise_map = predicted_noise_map.clamp(0, 255)
                    noise_map = noise_map.clamp(0, 255)

                epoch_mse[sigma] += torch.mean(
                    torch.square(noise_map - predicted_noise_map).cpu().detach()
                ).item()

                epoch_mse[sigma] = round(epoch_mse[sigma] / len(self.dataloader[sigma]), 4)
        logger.info("MSE:\n" + json.dumps(epoch_mse, sort_keys=True, indent=4))
        write_log(data=epoch_mse, epoch=epoch, log_file=self.log_dir + "_mse")


if __name__ == "__main__":
    dataset12 = NoiseMapDataset(
        phase="test",
        sigma=50,
        normalize=True,
        clip=False,
        noise_mode='gausse'
    )
    DLoader = torch.utils.data.DataLoader(
        dataset=dataset12,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
    )
    for n_count, batch_yx in enumerate(DLoader):
        batch_x, batch_y = batch_yx[0], batch_yx[1] * 255

        # batch_x = batch_x.view(batch_x.shape[2], batch_x.shape[3])

        logger.info(
            f"Size of noisy image: {batch_x.shape} - Max: {torch.max(batch_x)} - Min: {torch.min(batch_x)}"
        )
        logger.info(
            f"Size of noise map: {batch_y.shape} - Max: {torch.max(batch_y)} - Min: {torch.min(batch_y)}"
        )
        if n_count == 50:
            break
