import glob
import json
import os
import random

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from src.dataset import data_aug, datagenerator, read_npy
from src.utils import write_log


class HybridDataset(Dataset):
    def __init__(
            self,
            data_dir="data",
            phase="train",
            stride=9,
            patch_size=21,
            sigma=5,
            normalize=False,
            clip=False,
            output_mode="image",
            with_map=False,
            **kwargs
    ):
        super(HybridDataset, self).__init__()
        self.sigma = sigma
        self.phase = phase
        if self.phase == "test":
            self.data = glob.glob(f"{data_dir}/{phase}/sigma_{sigma}/*.npy")
        else:
            self.data = datagenerator(
                data_dir, phase=phase, stride=stride, patch_size=patch_size
            )
        self.transform = transforms.ToTensor()
        self.normalize = normalize
        self.clip = clip
        self.with_map = with_map
        self.norm_noise = kwargs.get('norm_noise', 55)
        self.return_path = kwargs.get("return_path", False)

        assert output_mode in [
            "image",
            "noise",
        ], f"output_mode must be in ['image', 'noise'], output_mode is {output_mode}"
        self.output_mode = 0 if output_mode == "image" else 1
        logger.success(
            f"{phase} with mode: {output_mode} - sigma: {sigma} - clip: {clip} - normalize: {normalize} - with_map: {with_map}")

    def __getitem__(self, index):
        # Origin Image
        if self.phase == "test":
            noisy_img = read_npy(self.data[index], normalize=False)
            output = read_npy(self.data[index].replace(f"/sigma_{self.sigma}", "/origin"))
            if self.clip:
                noisy_img = noisy_img.clip(0, 255)
        else:
            output = self.data[index]
            output = data_aug(output, random.randint(0, 7))
            # Noise
            noise1 = np.random.randn(output.shape[0], output.shape[1]) * self.sigma
            noise2 = np.random.randn(output.shape[0], output.shape[1]) * self.sigma

            # Noisy Image
            noisy_img = np.sqrt((output + noise1) * (output + noise1) + noise2 * noise2)

        if self.output_mode:
            # Noisy
            output = self.transform((noisy_img - output).astype(np.int32)).float()
        else:
            # CLean Image
            output = self.transform(output.astype(np.int32)).float()

        noisy_img = self.transform(noisy_img.astype(np.int32)).float()

        if self.with_map:
            noise_map = torch.ones_like(noisy_img) * self.sigma
            # noisy_img = torch.cat((noisy_img, map), dim=0)

        if self.clip:
            noise_map = noise_map.clamp(0, 255)
            output = output.clamp(0, 255)
            noisy_img = noisy_img.clamp(0, 255)

        if self.normalize:
            noise_map = noise_map.div(self.norm_noise)
            output = output.div(255)
            noisy_img = noisy_img.div(255)

        if self.return_path:
            return noisy_img, noise_map, output, self.data[index]
        else:
            return noisy_img, noise_map, output


    def __len__(self):
        return len(self.data)


class HybridTestPhase:
    def __init__(
            self,
            range_sigma: list = [5, 80],
            step: int = 5,
            data_dir: str = "data",
            n_workers: int = 8,
            batch_size: int = 1,
            log_dir="result/test",
            normalize=False,
            clip=False,
            device=None,
            output_mode="image",
            with_map=False,
            **kwargs
    ) -> None:
        self.dataloader = {}
        self.range_sigma = range_sigma
        self.step = step
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.normalize = normalize
        self.clip = clip
        self.with_map = with_map
        self.norm_noise = max(range_sigma)
        self.return_path = kwargs.get("return_path", False)


        assert output_mode in [
            "image",
            "noise",
        ], f"output_mode must be in ['image', 'noise'], output_mode is {output_mode}"
        self.output_mode = 0 if output_mode == "image" else 1
        self.range_sigma = range_sigma
        if device is not None:
            self.device = device
        else:
            self.device = "cpu"
        for sigma in range_sigma:
            logger.info(f"Test at Sigma: {sigma}")
            dataset = HybridDataset(
                data_dir=data_dir,
                phase="test",
                sigma=sigma,
                normalize=normalize,
                clip=clip,
                output_mode=output_mode,
                with_map=self.with_map,
                norm_noise=self.norm_noise,
                return_path=self.return_path
            )
            self.dataloader[sigma] = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=n_workers,
            )

    @torch.no_grad()
    def run(self, model, epoch):
        torch.set_grad_enabled(False)
        model.train(False)
        model.eval()
        model.to(self.device)
        epoch_psnr = {}
        epoch_ssim = {}
        epoch_mse = {}

        for sigma in tqdm(
                self.range_sigma,
                desc=f"Phase: TEST",
        ):
            epoch_psnr[sigma] = 0
            epoch_ssim[sigma] = 0
            epoch_mse[sigma] = 0
            for i, data in enumerate(self.dataloader[sigma]):
                if self.return_path:
                    noisy_img, noise_map, output, image_path = data[0].to(self.device), data[1].to(self.device), data[2].to(
                        self.device), data[3]
                else:
                    noisy_img, noise_map, output = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)

                predicted_noise_level, predicted_output = model(noisy_img)
                if self.output_mode:
                    clean_imgs = noisy_img - output
                    denoise_imgs = noisy_img - predicted_output
                else:
                    clean_imgs = output
                    denoise_imgs = predicted_output
                if self.normalize:
                    clean_imgs = clean_imgs * 255
                    denoise_imgs = denoise_imgs * 255
                    predicted_noise_level = predicted_noise_level * self.norm_noise
                    noise_map = noise_map * self.norm_noise

                clean_imgs = clean_imgs.clamp(0, 255)
                denoise_imgs = denoise_imgs.clamp(0, 255)
                predicted_noise_level = predicted_noise_level.clamp(0, 255)
                noise_map = noise_map.clamp(0, 255)

                clean_imgs = clean_imgs.cpu().detach().numpy().astype(np.uint8).squeeze(0).squeeze(0)
                denoise_imgs = denoise_imgs.cpu().detach().numpy().astype(np.uint8).squeeze(0).squeeze(0)

                epoch_mse[sigma] += torch.mean(
                    torch.square(noise_map - predicted_noise_level).cpu().detach()
                ).item()
                epoch_psnr[sigma] += psnr(clean_imgs, denoise_imgs)
                epoch_ssim[sigma] += ssim(clean_imgs, denoise_imgs)
            epoch_psnr[sigma] = round(epoch_psnr[sigma] / len(self.dataloader[sigma]), 2)
            epoch_ssim[sigma] = round(epoch_ssim[sigma] / len(self.dataloader[sigma]), 4)
            epoch_mse[sigma] = round(epoch_mse[sigma] / len(self.dataloader[sigma]), 4)

        logger.info("PSNR:\n" + json.dumps(epoch_psnr, sort_keys=True, indent=4))
        logger.info("SSIM:\n" + json.dumps(epoch_ssim, sort_keys=True, indent=4))
        logger.info("MSE:\n" + json.dumps(epoch_mse, sort_keys=True, indent=4))

        write_log(data=epoch_psnr, epoch=epoch, log_file=self.log_dir + "_psnr")
        write_log(data=epoch_ssim, epoch=epoch, log_file=self.log_dir + "_ssim")
        write_log(data=epoch_mse, epoch=epoch, log_file=self.log_dir + "_mse")


if __name__ == "__main__":
    dataset12 = HybridDataset(
        phase="val",
        sigma=[5, 10, 15],
        normalize=False,
        clip=False,
        output_mode='image'
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
        batch_output, batch_y = batch_yx[0], batch_yx[1]

        logger.info(
            f"Size of noisy image: {batch_output.shape} - Max: {torch.max(batch_output)} - Min: {torch.min(batch_output)}"
        )
        logger.info(
            f"Size of GT: {batch_y.shape} - Max: {torch.max(batch_y)} - Min: {torch.min(batch_y)}"
        )
        if n_count == 50:
            break
