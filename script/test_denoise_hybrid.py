import os
import warnings
import json

warnings.filterwarnings("ignore")

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from src.model.utils import get_model
from src.model.hybrid import HybridNet
from src.dataset.hybrid import HybridTestPhase
from src.dataset.denoise import DenoiseTestPhase

torch.manual_seed(1234)
np.random.seed(1234)

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Device : {device}")

def show_im(im, figsize=(20, 10)):
    plt.figure(figsize=figsize)
    if isinstance(im, list):
        plt.imshow(np.hstack([i for i in im]), cmap="gray")
    else:
        plt.imshow(im, cmap="gray")
    plt.show()


def show_multi_im(imgs: list, ratio=(5, 12), figsize=(20, 10)):
    index = 0
    f, axarr = plt.subplots(ratio[0], ratio[1], figsize=figsize)
    for r in range(ratio[0]):
        for c in range(ratio[1]):

            axarr[r, c].imshow(imgs[index], cmap="gray")
            index += 1
    plt.show()

def get_infor(data, lib="torch"):
    lib = torch if lib == "torch" else np
    return f"Max: {lib.max(data)} - Min: {lib.min(data)}"

def make_dir(config, layer):
    condition = f"{config['clip']}clip_{config['normalize']}normalize_sigrange{config['sigma_range']}_{config['output_mode']}_{config['with_map']}WithMap"
    save_dir = f"report/images/LuanVan/layers_{config['model_name']}{layer}_{condition}"
    os.makedirs(os.path.join("../report", "images"), exist_ok=True)
    os.makedirs(os.path.join("../report", "images", "LuanVan"), exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

config = {
    "layers": [10],
    "channels": 2,
    "out_channels": 1,
    "features": 192,
    "n_workers": 1,
    "scheduler_mode": min,
    "scheduler_patience": 1,
    "model_name": "hybrid",
    "loss_function": "L1",
    "alpha_loss": [0.001],
    "sigma_range": [10,25,35,45,55],
    "weight": r"result/hybrid_dnresnet_fcn_Falseclip_Truenormalize_sigrange10_25_image_withMap/model_003.pth",
    "data_dir": "data",
    "clip": False,
    "normalize": True,
    "output_mode": 'image',
    "with_map": True
}
denoise_config = {"model_name": 'dnresnet',
    "layer": 10,
    "input_channel": 2,
    "out_channels": 1,
    "features": 192}

noisemap_config = {"model_name": 'fcn',
    "layer": 5,
    "input_channel": 1,
    "features": 32}

if __name__=="__main__":
    for layer in config["layers"]:
        save_dir = make_dir(config, layer)

        test_phase = HybridTestPhase(
            range_sigma=config["sigma_range"],
            step=5,
            data_dir=config["data_dir"],
            n_workers=1,
            batch_size=1,
            log_dir=f"{save_dir}_test",
            normalize=config["normalize"],
            clip=config["clip"],
            device=config.get("device", device),
            output_mode=config["output_mode"],
            with_map=config["with_map"]
        )

        epoch_psnr = {}
        epoch_ssim = {}

        denoise_model = get_model(
            name=denoise_config["model_name"],
            layers=denoise_config["layer"],
            input_channels=denoise_config["input_channel"],
            feature=denoise_config["features"],
            out_channels=denoise_config.get("out_channels", 1)
        )

        noisemap_model = get_model(
            name=noisemap_config["model_name"],
            layers=noisemap_config["layer"],
            input_channels=noisemap_config["input_channel"],
            feature=noisemap_config["features"],
        )

        model = HybridNet(denoise_net=denoise_model, noisemap_net=noisemap_model)
        model.eval()
        check_point = torch.load(config["weight"])
        # model = torch.load(os.path.join(save_dir, "model_%03d.pth" % initial_epoch))
        model.load_state_dict(check_point)
        model.cuda()

        for sigma in config["sigma_range"]:
            sigma_savedir = os.path.join(save_dir,f"sigma{sigma}")
            os.makedirs(sigma_savedir, exist_ok=True)
            epoch_psnr[sigma] = 0
            epoch_ssim[sigma] = 0
            for i, data in tqdm(enumerate(test_phase.dataloader[sigma]), desc=f"sigma {sigma}"):

                noisy_img, noise_map, output = data[0].to(test_phase.device), data[1].to(test_phase.device), data[2].to(test_phase.device)

                predicted_noise_level, predicted_output = model(noisy_img)
                if test_phase.output_mode:
                    clean_imgs = noisy_img - output
                    denoise_imgs = noisy_img - predicted_output
                else:
                    clean_imgs = output
                    denoise_imgs = predicted_output
                if test_phase.normalize:
                    clean_imgs = clean_imgs * 255
                    denoise_imgs = denoise_imgs * 255
                    predicted_noise_level = predicted_noise_level * test_phase.norm_noise
                    noise_map = noise_map * test_phase.norm_noise

                clean_imgs = clean_imgs.clamp(0, 255)
                denoise_imgs = denoise_imgs.clamp(0, 255)

                clean_imgs = clean_imgs.cpu().detach().numpy().astype(np.uint8).squeeze(0).squeeze(0)
                denoise_imgs = denoise_imgs.cpu().detach().numpy().astype(np.uint8).squeeze(0).squeeze(0)
                if noisy_img.shape[1] == 2:
                    noisy_img = noisy_img[:,0,:,:]
                    noisy_img = (noisy_img*255).clamp(0,255).cpu().detach().numpy().astype(np.uint8).squeeze(0)
                else:
                    noisy_img = (noisy_img*255).clamp(0,255).cpu().detach().numpy().astype(np.uint8).squeeze(0).squeeze(0)
                # show_im([clean_imgs, noisy_imgs, denoise_imgs])
                # cv2.imwrite(os.path.join(save_dir, f"{i}.png"), np.hstack([clean_imgs, (noisy_imgs*255).clamp(0,255).cpu().detach().numpy().astype(np.uint8).squeeze(0).squeeze(0), denoise_imgs]))
                # cv2.imwrite(os.path.join(sigma_savedir, f"{i}.png"), denoise_imgs)


                # print(f"PSNR: {psnr(clean_imgs, denoise_imgs)} - SSIM: {ssim(clean_imgs, denoise_imgs)}")
                epoch_psnr[sigma] += psnr(clean_imgs, denoise_imgs)
                epoch_ssim[sigma] += ssim(clean_imgs, denoise_imgs)

            epoch_psnr[sigma] = round(epoch_psnr[sigma] / len(test_phase.dataloader[sigma]), 4)
            epoch_ssim[sigma] = round(epoch_ssim[sigma] / len(test_phase.dataloader[sigma]), 8)
        logger.info("PSNR:\n" + json.dumps(epoch_psnr, sort_keys=True, indent=4))
        logger.info("SSIM:\n" + json.dumps(epoch_ssim, sort_keys=True, indent=4))