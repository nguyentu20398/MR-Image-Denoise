import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import yaml
import torch
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from loguru import logger

from src.model.hybrid import HybridNet
from src.dataset.hybrid import HybridTestPhase

from src.logger.logger import configure_logging
from src.dataset.utils import get_concate_dataset
from src.model.utils import get_model
from src.utils import findLastCheckPoint, make_dir, write_log, batch_ssim, batch_psnr
from src.utils.loss import get_criterion

torch.manual_seed(1234)
np.random.seed(1234)

logger = configure_logging()


def train_engine(config):
    denoise_config = config['model']['denoise']
    noisemap_config = config['model']['noisemap']

    n_epoch = config["n_epoch"]
    batch_size = config["batch_size"]
    n_workers = config["n_workers"]
    scheduler_mode = config["scheduler_mode"]
    scheduler_patience = config["scheduler_patience"]

    make_dir("../result")
    make_dir("../report")
    condition = f"{config['clip']}clip_{config['normalize']}normalize_sigrange{config['sigma_range'][0]}_{config['sigma_range'][-1]}_{config['output_mode']}"
    if config["with_map"]:
        condition += "_withMap"
    save_dir = f"result/hybrid_{denoise_config['model_name']}_{noisemap_config['model_name']}_{condition}"
    save_report = f"report/hybrid_{denoise_config['model_name']}_{noisemap_config['model_name']}_{condition}"

    make_dir(save_dir)
    # make_dir(save_report)

    logger.info(f"Save direct: {save_dir}")

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device : {device}")

    logger.info(f"===> Building hybrid model")

    denoise_model = get_model(
        name=denoise_config["model_name"],
        layers=denoise_config["layer"],
        input_channels=denoise_config["input_channel"],
        feature=denoise_config["features"],
        out_channels=denoise_config.get("out_channels", 1)
    )

    if denoise_config.get("pretrained", None) is not None:
        denoise_model.eval()
        check_point = torch.load(denoise_config["pretrained"])
        denoise_model.load_state_dict(check_point)

    noisemap_model = get_model(
        name=noisemap_config["model_name"],
        layers=noisemap_config["layer"],
        input_channels=noisemap_config["input_channel"],
        feature=noisemap_config["features"],
    )

    model = HybridNet(denoise_net=denoise_model, noisemap_net=noisemap_model)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["alpha_adam"],
        betas=(config["beta1_adam"], config["beta2_adam"]),
        eps=config["epsilon_adam"],
    )
    initial_epoch = findLastCheckPoint(save_dir)
    logger.info(f"Initial epoch : {initial_epoch}")

    if cuda:
        model = model.cuda()

    if initial_epoch > 0:
        logger.info(f"Resume by loading epoch {initial_epoch}")
        check_point = torch.load(os.path.join(save_dir, "last.pt"))
        # model = torch.load(os.path.join(save_dir, "model_%03d.pth" % initial_epoch))
        model.load_state_dict(check_point["model"])
        optimizer.load_state_dict(check_point["optimizer"])
        config["alpha_adam"] = check_point["lr"]
        initial_epoch = check_point["epoch"]


    criterion = get_criterion(config["loss_function"], alpha=config["alpha_loss"], beta=config["beta_loss"])

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[19], gamma=0.1)

    test_phase = HybridTestPhase(
        range_sigma=config["sigma_range"],
        step=5,
        data_dir=config["data_dir"],
        n_workers=n_workers,
        batch_size=1,
        log_dir=f"{save_report}_test",
        normalize=config["normalize"],
        clip=config["clip"],
        device=config.get("device", device),
        output_mode=config["output_mode"],
        with_map=config["with_map"]
    )
    dataloader = get_concate_dataset(config, mode='hybrid')

    prev_grad_state = torch.is_grad_enabled()

    for epoch in range(initial_epoch, n_epoch):
        logger.info(f"Epoch: {epoch}".center(50, "="))
        epoch_loss = {"train": 0.0}
        epoch_psnr = {"val": 0.0}
        epoch_ssim = {"val": 0.0}
        epoch_mse = {"val": 0.0}
        data_log = {}

        for phase in ["train", "val", "test"]:
            logger.info(f"Phase: {phase.upper()}")
            if phase == "train":
                model.train(True)
                for i, data in enumerate(
                        tqdm(dataloader[phase], desc=f"{phase.upper()}")
                ):
                    if i == len(dataloader[phase]) - 1:
                        break
                    noisy_img, noise_map, output = data[0].to(device), data[1].to(device), data[2].to(device)
                    optimizer.zero_grad()

                    predicted_noise_level, predicted_output = model(noisy_img)

                    loss = criterion(predicted_noise_level, predicted_output, noise_map, output)

                    epoch_loss[phase] += loss.item()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    if i % len(dataloader[phase]) == 3:
                        logger.info(
                            "%4d %4d / %4d loss = %2.4f"
                            % (
                                epoch,
                                i,
                                noisy_img.size(0) // batch_size,
                                loss.item() / batch_size,
                            )
                        )
                logger.info(
                    f"epoch = {epoch} , loss = {round(epoch_loss[phase] / i, 4)}"
                )
                data_log["train_loss"] = round(epoch_loss[phase] / i, 4)
                epoch_loss[phase] = epoch_loss[phase] / i

            if phase == "val":
                # if scheduler:
                #     scheduler.step(epoch_loss["train"])
                data_log["lr"] = scheduler.optimizer.param_groups[0]["lr"]
                    # print(f"-Reduce learning rate {data_log['lr']}")

                prev_grad_state = torch.is_grad_enabled()
                torch.set_grad_enabled(False)
                model.train(False)
                model.eval()
                for i, data in enumerate(
                        tqdm(dataloader[phase], desc=f"{phase.upper()}")
                ):
                    if i == len(dataloader[phase]) - 1:
                        break
                    noisy_img, noise_map, output = data[0].to(device), data[1].to(device), data[2].to(device)
                    optimizer.zero_grad()

                    predicted_noise_level, predicted_output = model(noisy_img)

                    if config["output_mode"] == "noise":
                        clean_imgs = noisy_img - output
                        denoise_imgs = noisy_img - predicted_output
                    else:
                        clean_imgs = output
                        denoise_imgs = predicted_output

                    if config["normalize"]:
                        clean_imgs = clean_imgs * 255
                        denoise_imgs = denoise_imgs * 255
                        predicted_noise_level = predicted_noise_level * max(config["sigma_range"])
                        noise_map = noise_map * max(config["sigma_range"])

                    clean_imgs = clean_imgs.clamp(0, 255)
                    denoise_imgs = denoise_imgs.clamp(0, 255)
                    predicted_noise_level = predicted_noise_level.clamp(0, 255)
                    noise_map = noise_map.clamp(0, 255)

                    clean_imgs = clean_imgs.cpu().detach().numpy().astype(np.int32)
                    denoise_imgs = denoise_imgs.cpu().detach().numpy().astype(np.int32)

                    epoch_psnr[phase] += batch_psnr(
                        clean_imgs, denoise_imgs, batch_size
                    )
                    epoch_ssim[phase] += batch_ssim(
                        clean_imgs, denoise_imgs, batch_size
                    )
                    epoch_mse[phase] += torch.mean(
                        torch.square(noise_map - predicted_noise_level).cpu().detach()
                    ).item()
                data_log[f"{phase}_psnr"] = round(epoch_psnr[phase] / i, 2)
                data_log[f"{phase}_ssim"] = round(epoch_ssim[phase] / i, 4)
                data_log[f"{phase}_mse"] = round(epoch_mse[phase] / i, 4)


                logger.info(
                    f"epoch = {epoch}, phase: {phase}, "
                    + f"psnr = {data_log[f'{phase}_psnr']}, "
                    + f"ssim = {data_log[f'{phase}_ssim']}"
                    + f"mse = {data_log[f'{phase}_mse']}, "
                )

                write_log(data_log, epoch, f"{save_report}_train")
                torch.save(
                    model.state_dict(), os.path.join(save_dir, "model_%03d.pth" % (epoch))
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr": scheduler.optimizer.param_groups[0]["lr"],
                    },
                    os.path.join(save_dir, "last.pt"),
                )
            if phase == "test":
                test_phase.run(model, epoch)
            torch.set_grad_enabled(prev_grad_state)


if __name__ == "__main__":
    with open(f"../config/train_hybrid_config.yml") as f:
        config = yaml.safe_load(f)
    train_engine(config)
