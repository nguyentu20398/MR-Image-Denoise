import os
import warnings

warnings.filterwarnings("ignore")

import yaml
import torch
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from loguru import logger

from src.logger.logger import configure_logging
from src.dataset.noisemap import NoiseMapTestPhase
from src.dataset.utils import get_concate_dataset
from src.model.utils import get_model
from src.utils import findLastCheckPoint, make_dir, write_log
from src.utils.loss import get_criterion

torch.manual_seed(1234)
np.random.seed(1234)

logger = configure_logging()


def train_engine(alpha_loss, beta_loss, config):
    layers = config["layers"]
    n_epoch = config["n_epoch"]
    batch_size = config["batch_size"]
    channels = config["channels"]
    n_workers = config["n_workers"]
    scheduler_mode = config["scheduler_mode"]
    scheduler_patience = config["scheduler_patience"]

    dataset_stride = config["dataset_stride"]

    make_dir("../result")
    make_dir("../report")
    condition = f"{config['clip']}clip_{config['normalize']}normalize_sigrange{config['sigma_range'][0]}_{config['sigma_range'][-1]}_{config['output_mode']}"
    save_dir = f"result/layers_{config['model_name']}{layers}_{condition}_{alpha_loss}_{beta_loss}"
    save_report = (
        f"report/layers_{config['model_name']}{layers}_{condition}_{alpha_loss}_{beta_loss}"
    )
    make_dir(save_dir)

    logger.info(f"Save direct: {save_dir}")

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device : {device}")

    logger.info(f"===> Building model {config['model_name']} {layers}")
    # Model
    model = get_model(
        name=config["model_name"],
        layers=layers,
        input_channels=channels,
        feature=config["features"],
    )

    initial_epoch = findLastCheckPoint(save_dir)
    logger.info(f"Initial epoch : {initial_epoch}")

    # Optimizer
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
    criterion = get_criterion(config["loss_function"], alpha_loss=alpha_loss, beta_loss=beta_loss)

    if cuda:
        model = model.cuda()
    # scheduler = MultiStepLR(optimizer, milestones=[30, 50, 70], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, scheduler_mode, patience=scheduler_patience
    )
    # Test phase
    test_phase = NoiseMapTestPhase(
        data_dir=config.get('data_dir', 'data'),
        range_sigma=config["sigma_range"],
        step=5,
        n_workers=n_workers,
        batch_size=1,
        log_dir=f"{save_report}_test",
        normalize=config["normalize"],
        clip=config["clip"],
        device=device,
        noise_mode=config.get('noise_mode', 'rice'),
        norm_noise=max(config["sigma_range"])
    )

    # Train and evaluate dataset
    dataloader = get_concate_dataset(config=config, mode='noise')
    prev_grad_state = torch.is_grad_enabled()

    for epoch in range(initial_epoch, n_epoch):
        logger.info(f"Epoch: {epoch}".center(50, "="))
        epoch_loss = {"train": 0.0}
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
                    noisy_imgs, map_noise = data[0].to(device), data[1].to(device)
                    optimizer.zero_grad()

                    predicted_map_noise = model(noisy_imgs)

                    loss = criterion(predicted_map_noise, map_noise)

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

                    noisy_imgs, map_noise = data[0].to(device), data[1].to(device)
                    optimizer.zero_grad()

                    # predicted_map_noise = (
                    #     model(noisy_imgs)
                    #     .cpu()
                    #     .detach()
                    #     .numpy()
                    #     .clip(0, 255)
                    #     .astype(np.uint8)
                    # )
                    predicted_map_noise = model(noisy_imgs)
                    # map_noise = map_noise.clamp(0, 255)
                    if config["normalize"]:
                        predicted_map_noise = predicted_map_noise * 255
                        map_noise = map_noise * 255
                    if config["clip"]:
                        predicted_map_noise = predicted_map_noise.clamp(0, 255)
                        map_noise = map_noise.clamp(0, 255)
                    epoch_mse[phase] += torch.mean(
                        torch.square(map_noise - predicted_map_noise).cpu().detach()
                    ).item()

                data_log[f"{phase}_mse"] = round(epoch_mse[phase] / i, 4)

                logger.info(
                    f"epoch = {epoch}, phase: {phase}, "
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


def train_subnet(config):
    list_alpha = config["alpha_loss"]
    list_beta = config["beta_loss"]

    for i, alpha in enumerate(list_alpha):
        try:
            train_engine(alpha,list_beta[i], config)
        except Exception as e:
            logger.exception(e)


if __name__ == "__main__":
    with open(f"../config/train_subnet_config.yml") as f:
        config = yaml.safe_load(f)
    train_subnet(config)
