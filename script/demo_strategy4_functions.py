import glob
import os

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

if "script" in os.getcwd():
    os.chdir("..")

from script.demo_inference_functions import initialize_model, create_data
from script.demo_inference_configs import strategy_configs

del strategy_configs['1']
del strategy_configs['2']
del strategy_configs['3']
del strategy_configs['5']


class Inference:
    def __init__(self, configs=strategy_configs, strategy='4'):
        self.models = initialize_model(configs[strategy]['model'], configs[strategy]['model_configs'])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.configs = configs[strategy]

    def preprocess(self, noisy_img, noise_map, output):
        self.noise_image = (noisy_img * 255).clamp(0, 255).cpu().detach().numpy().astype(np.uint8).squeeze(0).squeeze(0)
        self.origin_image = (output * 255).clamp(0, 255).cpu().detach().numpy().astype(np.uint8).squeeze(0).squeeze(0)

        noisy_img, noise_map, output = noisy_img.to(self.device), noise_map.to(self.device), output.to(self.device)
        return torch.cat((noisy_img, noise_map), dim=1), output

    def run(self, inputs: list, noise_map_sigmas=[]):
        input_tensor = torch.cat(inputs, dim=0)
        denoise_imgs = {}
        with torch.no_grad():
            predicted_outputs = self.models(input_tensor)
        for i, noise_map_sigma in enumerate(noise_map_sigmas):
            denoise_imgs[noise_map_sigma] = predicted_outputs[i, :, :, :]
        return denoise_imgs

    def post_process(self, denoise_imgs, noisy_imgs, clean_img, noise_map_sigmas=[]):
        img_results = {}
        psnrs = {}
        ssims = {}
        if self.configs['normalize']:
            clean_img = clean_img * 255
        clean_img = clean_img.clamp(0, 255)
        clean_img = clean_img.cpu().detach().numpy().astype(np.uint8).squeeze(0).squeeze(0)

        img_results['Original'] = clean_img

        for i, noise_map_sigma in enumerate(noise_map_sigmas):
            if self.configs['output_mode'] != 'image':
                denoise_img = noisy_imgs[noise_map_sigma] - denoise_imgs[noise_map_sigma]
            else:
                denoise_img = denoise_imgs[noise_map_sigma]

            if self.configs['normalize']:
                denoise_img = denoise_img * 255

            denoise_img = denoise_img.clamp(0, 255)
            denoise_img = denoise_img.cpu().detach().numpy().astype(np.uint8).squeeze(0)
            img_results[noise_map_sigma] = denoise_img
            psnrs[noise_map_sigma] = round(psnr(clean_img, denoise_img), 2)
            ssims[noise_map_sigma] = round(ssim(clean_img, denoise_img), 4)
        return img_results, psnrs, ssims

    def inference(self, origin_image_path, noise_sigma, noise_map_sigmas=[]):
        noisy_imgs = {}
        input_tensors = {}

        for noise_map_sigma in noise_map_sigmas:
            noisy_img, noise_map, output = create_data(
                npy_path=origin_image_path,
                sigma=noise_sigma,
                return_path=False,
                config=self.configs,
                create_image=False,
                create_noise_map=True,
                noise_map_sigma=noise_map_sigma
            )
            noisy_imgs[noise_map_sigma] = noisy_img
            input_tensor, clean_img = self.preprocess(noisy_img, noise_map, output)
            input_tensors[noise_map_sigma] = input_tensor

        with torch.no_grad():
            denoise_imgs = self.run(list(input_tensors.values()), noise_map_sigmas)

        img_results, psnrs, ssims = self.post_process(denoise_imgs, noisy_imgs, output, noise_map_sigmas)
        img_results["Noise"] = (noisy_img * 255).clamp(0, 255).cpu().detach().numpy().astype(np.uint8).squeeze(0).squeeze(0)
        return img_results, psnrs, ssims

if __name__ == '__main__':
    inference = Inference()
    origin_image_path = "data/test/origin/1synBrainPd.npy"
    noise_sigma = 35
    noise_map_sigmas = [i*5 for i in range(1, 15)]
    img_results, psnrs, ssims = inference.inference(origin_image_path, noise_sigma, noise_map_sigmas)
    print(ssims)

