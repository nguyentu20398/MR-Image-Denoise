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
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from src.logger.logger import configure_logging
from src.dataset.denoise import DenoiseTestPhase
from src.dataset.utils import get_concate_dataset
from src.model.utils import get_model
from src.utils import findLastCheckPoint, make_dir, write_log, batch_ssim, batch_psnr
from src.utils.loss import get_criterion

torch.manual_seed(1234)
np.random.seed(1234)

logger = configure_logging()


def train_engine(layers, config):
    n_epoch = config["n_epoch"]
    batch_size = config["batch_size"]
    channels = config["channels"]
    n_workers = config["n_workers"]
    scheduler_mode = config["scheduler_mode"]
    scheduler_patience = config["scheduler_patience"]

    dataset_stride = config["dataset_stride"]

    make_dir("../result")
    make_dir("../report")
    condition = f"{config['clip']}clip_{config['normalize']}normalize_sigrange{config['sigma_range'][0]}_{config['sigma_range'][0]}_{config['output_mode']}"
    if config["with_map"]:
        condition += "_withMap"
    save_dir = f"result/layers_{config['model_name']}{layers}_{condition}"
    save_report = f"report/layers_{config['model_name']}{layers}_{condition}"

    make_dir(save_dir)
    # make_dir(save_report)

    logger.info(f"Save direct: {save_dir}")

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device : {device}")

    logger.info(f"===> Building model {config['model_name']} {layers}")
    model = get_model(
        name=config["model_name"],
        layers=layers,
        input_channels=channels,
        feature=config["features"],
        out_channels=config.get("out_channels", 1)
    )

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

    criterion = get_criterion(config["loss_function"])

    # scheduler = MultiStepLR(optimizer, milestones=[30, 50, 70], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, scheduler_mode, patience=scheduler_patience
    )
    test_phase = DenoiseTestPhase(
        range_sigma=config["sigma_range"],
        step=5,
        data_dir="../data",
        n_workers=n_workers,
        batch_size=1,
        log_dir=f"{save_report}_test",
        normalize=config["normalize"],
        clip=config["clip"],
        device=config.get("device", device),
        output_mode=config["output_mode"],
        with_map=config["with_map"]
    )
    dataloader = get_concate_dataset(config, mode='image')

    prev_grad_state = torch.is_grad_enabled()

    for epoch in range(initial_epoch, n_epoch):
        logger.info(f"Epoch: {epoch}".center(50, "="))
        epoch_loss = {"train": 0.0}
        epoch_psnr = {"val": 0.0}
        epoch_ssim = {"val": 0.0}
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
                    noisy_imgs, GT_noise = data[0].to(device), data[1].to(device)
                    optimizer.zero_grad()

                    predicted_noise = model(noisy_imgs)

                    loss = criterion(predicted_noise, GT_noise)

                    epoch_loss[phase] += loss.item()
                    loss.backward()
                    optimizer.step()
                    if i % len(dataloader[phase]) == 3:
                        logger.info(
                            "%4d %4d / %4d loss = %2.4f"
                            % (
                                epoch,
                                i,
                                noisy_imgs.size(0) // batch_size,
                                loss.item() / batch_size,
                            )
                        )
                logger.info(
                    f"epoch = {epoch} , loss = {round(epoch_loss[phase] / i, 4)}"
                )
                data_log["train_loss"] = round(epoch_loss[phase] / i, 4)
                epoch_loss[phase] = epoch_loss[phase] / i

            if phase == "val":
                if scheduler:
                    scheduler.step(epoch_loss["train"])
                    data_log["lr"] = scheduler.optimizer.param_groups[0]["lr"]
                    print(f"-Reduce learning rate {data_log['lr']}")

                prev_grad_state = torch.is_grad_enabled()
                torch.set_grad_enabled(False)
                model.train(False)
                model.eval()
                for i, data in enumerate(
                    tqdm(dataloader[phase], desc=f"{phase.upper()}")
                ):
                    if i == len(dataloader[phase]) - 1:
                        break
                    noisy_imgs, GT = data[0].to(device), data[1].to(device)
                    optimizer.zero_grad()

                    predicted = model(noisy_imgs)

                    if config["output_mode"] == "noise":
                        clean_imgs = noisy_imgs - GT
                        denoise_imgs = noisy_imgs - predicted
                    else:
                        clean_imgs = GT
                        denoise_imgs = predicted

                    if config["normalize"]:
                        clean_imgs = clean_imgs * 255
                        denoise_imgs = denoise_imgs * 255

                    clean_imgs = clean_imgs.clamp(0, 255)
                    denoise_imgs = denoise_imgs.clamp(0, 255)

                    clean_imgs = clean_imgs.cpu().detach().numpy().astype(np.int32)
                    denoise_imgs = denoise_imgs.cpu().detach().numpy().astype(np.int32)

                    epoch_psnr[phase] += batch_psnr(
                        clean_imgs, denoise_imgs, batch_size
                    )
                    epoch_ssim[phase] += batch_ssim(
                        clean_imgs, denoise_imgs, batch_size
                    )
                data_log[f"{phase}_psnr"] = round(epoch_psnr[phase] / i, 4)
                data_log[f"{phase}_ssim"] = round(epoch_ssim[phase] / i, 4)

                logger.info(
                    f"epoch = {epoch}, phase: {phase}, "
                    + f"psnr = {data_log[f'{phase}_psnr']}, "
                    + f"ssim = {data_log[f'{phase}_ssim']}"
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


def train_dncnn(config):
    list_layers = config["layers"]
    for layers in list_layers:
        try:
            train_engine(layers, config)
        except Exception as e:
            logger.exception(e)


if __name__ == "__main__":
    with open(f"../config/train_blind_config.yml") as f:
        config = yaml.safe_load(f)
    train_dncnn(config)
