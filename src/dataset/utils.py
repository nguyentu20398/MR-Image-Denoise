import torch
from tqdm import tqdm
from loguru import logger
from torch.utils.data import DataLoader

from src.dataset.denoise import DenoisingDataset
from src.dataset.noisemap import NoiseMapDataset
from src.dataset.hybrid import HybridDataset



def get_concate_dataset(config: dict, mode="image"):
    if mode == "image":
        Dataset = DenoisingDataset
    elif mode == "noise":
        Dataset = NoiseMapDataset
    elif mode == "hybrid":
        Dataset = HybridDataset
    datasets = {'train': [], 'val': []}
    for sigma in config["sigma_range"]:
        datasets['train'].append(Dataset(data_dir=config.get('data_dir', 'data'),
                                         phase=config["trainset"],
                                         stride=config["dataset_stride"],
                                         normalize=config["normalize"],
                                         clip=config["clip"],
                                         sigma=sigma,
                                         output_mode=config['output_mode'],
                                         with_map=config.get("with_map", False),
                                         noise_mode=config.get("noise_mode", "rice"),
                                         norm_noise=max(config["sigma_range"]))
                                 )
        datasets['val'].append(Dataset(data_dir=config.get('data_dir', 'data'),
                                       phase=config["valset"],
                                       stride=config["dataset_stride"],
                                       normalize=config["normalize"],
                                       clip=config["clip"],
                                       sigma=sigma,
                                       output_mode=config['output_mode'],
                                       with_map=config.get("with_map", False),
                                       noise_mode=config.get("noise_mode", "rice"),
                                       norm_noise=max(config["sigma_range"])))

    datasets['train'] = torch.utils.data.ConcatDataset(datasets['train'])
    datasets['val'] = torch.utils.data.ConcatDataset(datasets['val'])
    dataloader = {
        "train": DataLoader(
            dataset=datasets["train"],
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["n_workers"],
        ),
        "val": DataLoader(
            dataset=datasets["val"],
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["n_workers"],
        ),
    }
    return dataloader


if __name__ == "__main__":
    # test_data_generator()
    import os
    import yaml
    import cv2
    import numpy as np
    training_saved_samples_folder = "report/LV_ThS_TuPhuongNguyen/paper/tranining_samples"
    os.makedirs(f"{training_saved_samples_folder}", exist_ok=True)

    with open(f"config/train_hybrid_config.yml") as f:
        config = yaml.safe_load(f)
    dataloader = get_concate_dataset(config, mode='hybrid')
    for i, data in enumerate(
            tqdm(dataloader['train'], desc=f"{'train'.upper()}")
    ):
        if i == len(dataloader['train']) - 1:
            break
        noisy_img, noise_map, output = data[0], data[1], data[2]
        logger.info(
            f"Size of noisy image: {noisy_img.shape} - Max: {torch.max(noisy_img)} - Min: {torch.min(noisy_img)}"
        )
        logger.info(
            f"Size of GT: {output.shape} - Max: {torch.max(output)} - Min: {torch.min(output)}"
        )
        logger.info(
            f"Size of Noise_map: {noise_map.shape} - Max: {torch.max(noise_map)} - Min: {torch.min(noise_map)}"
        )
        np_noisy_im = noisy_img.cpu().squeeze(0).squeeze(0).detach().numpy()
        np_noisy_im = (np_noisy_im * 255.0).clip(0, 255).astype(np.uint8)

        np_output = output.cpu().squeeze(0).squeeze(0).detach().numpy()
        np_output = (np_output * 255.0).clip(0, 255).astype(np.uint8)
        cv2.imwrite(f"{training_saved_samples_folder}/{i}_noisy.png", np_noisy_im)
        cv2.imwrite(f"{training_saved_samples_folder}/{i}_origin.png", np_output)

        if i == 200:
            break
