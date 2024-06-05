import glob
import os
import random

import cv2
import torch
import numpy as np
from tqdm import tqdm
from loguru import logger

aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation="nearest", cmap="gray")
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def read_npy(path: str, normalize=True):
    with open(path, "rb") as f:
        a = np.load(f)
        if normalize:
            a = (a / np.max(a)) * 255
    return a


def data_aug(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def gen_patches(file_name, stride=9, patch_size=21):
    # get multiscale patches from a single image
    # img = cv2.imread(file_name, 0)  # gray scale
    img = read_npy(file_name, normalize=True).astype(np.uint8)
    h, w = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h * s), int(w * s)
        img_scaled = cv2.resize(
            img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC
        )
        # extract patches
        for i in range(0, h_scaled - patch_size + 1, stride):
            for j in range(0, w_scaled - patch_size + 1, stride):
                x = img_scaled[i: i + patch_size, j: j + patch_size]
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))
                    patches.append(x_aug)
    return patches


def datagenerator(
        data_dir="data", verbose=False, phase="train", stride=9, patch_size=21
):
    # generate clean patches from a dataset

    if phase == "train" or phase == "val":
        data = []
        file_list = glob.glob(f"{data_dir}/{phase}/*.npy")
        # generate patches

        for i in tqdm(range(len(file_list)), desc="Data gen patches"):
            patches = gen_patches(file_list[i], stride=stride, patch_size=patch_size)
            for patch in patches:
                h, w = patch.shape
                if h == patch_size and w == patch_size:
                    data.append(patch)
            if verbose:
                logger.info(str(i + 1) + "/" + str(len(file_list)) + " is done ^_^")
        random.shuffle(data)
        logger.info(f"Số ảnh {phase}:{len(data)}")

    return data


def test_data_generator(data_dir="data/test", noise_mode='rice', min=1, max=80, step=5):
    data = glob.glob(f"{data_dir}/origin/*.npy")
    sigmas = [i for i in range(min, max+1, step)]
    for sigma in tqdm(sigmas, desc="Generate Test Data"):
        os.makedirs(f"{data_dir}/sigma_{sigma}", exist_ok=True)
        for n, image_path in enumerate(data):
            image = read_npy(image_path)
            # np.random.seed(n)
            if noise_mode == 'rice':
                noise1 = np.random.randn(image.shape[0], image.shape[1]) * sigma
                noise2 = np.random.randn(image.shape[0], image.shape[1]) * sigma
                y = np.sqrt((image + noise1) * (image + noise1) + noise2 * noise2)
            elif noise_mode == 'gausse':
                noise1 = np.random.randn(image.shape[0], image.shape[1]) * sigma
                y = image + noise1
            with open(
                    os.path.join(data_dir, f"sigma_{sigma}",
                                 f"{os.path.split(image_path)[-1].replace('.png', '.npy')}"),
                    "wb",
            ) as f:
                np.save(f, y.astype(np.int32))


if __name__ == "__main__":
    test_data_generator(step=1)
