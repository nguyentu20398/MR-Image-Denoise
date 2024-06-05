import datetime
import os

import pandas as pd
from loguru import logger
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def findLastCheckPoint(save_dir):
    epochs = []
    for file in os.listdir(save_dir):
        # print(f'File in model: {file}')
        try:
            epochs.append(int(file.split("_")[-1].split(".")[0]))
        except:
            logger.error(file)
    # print(f'Epochs : {epochs}')
    if len(epochs) > 0:
        return max(epochs)
    else:
        return 0


def batch_ssim(clean_imgs, denoise_imgs, batch_size):
    if len(clean_imgs.shape) == 4:
        clean_imgs = clean_imgs.squeeze(1)
    if len(denoise_imgs.shape) == 4:
        denoise_imgs = denoise_imgs.squeeze(1)
        
    b_ssim = 0
    for b in range(batch_size):
        clean_img = clean_imgs[b]
        denoise_img = denoise_imgs[b]
        b_ssim += ssim(
            clean_img,
            denoise_img,
            data_range=denoise_img.max() - denoise_img.min(),
        )
    return b_ssim / batch_size

def batch_psnr(clean_imgs, denoise_imgs, batch_size):
    if len(clean_imgs.shape) == 4:
        clean_imgs = clean_imgs.squeeze(1)
    if len(denoise_imgs.shape) == 4:
        denoise_imgs = denoise_imgs.squeeze(1)
    b_psnr = 0
    for b in range(batch_size):
        clean_img = clean_imgs[b]
        denoise_img = denoise_imgs[b]
        b_psnr += psnr(
            clean_img,
            denoise_img,
        )
    return b_psnr / batch_size


def make_dir(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)


def log(*args, **kwargs):
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"))


def write_log(data, epoch, log_file):
    if epoch > 1:
        df = pd.read_excel(f"{log_file}.xlsx")
        df = pd.DataFrame(df, columns=data.keys())
        new_row = data
        df = df.append(new_row, ignore_index=True)
        df.to_excel(f"{log_file}.xlsx")
    elif epoch == 1:
        df = pd.DataFrame([data.values()], columns=data.keys())
        df.to_excel(f"{log_file}.xlsx")
