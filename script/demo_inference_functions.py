import glob
import sys

import numpy as np
import torch
from loguru import logger
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

logger.remove()
logger.add(sys.stderr, level="SUCCESS")

from src.dataset import read_npy
from src.model.utils import get_model
from src.model.hybrid import HybridNet
from demo_inference_configs import strategy_configs, sigmas

transform = transforms.ToTensor()


def create_data(npy_path, sigma, return_path, config,
                create_image=True,
                create_noise_map=False,
                noise_map_sigma=10
                ):
    output = read_npy(npy_path)
    if create_image:
        # Noise
        noise1 = np.random.randn(output.shape[0], output.shape[1]) * sigma
        noise2 = np.random.randn(output.shape[0], output.shape[1]) * sigma

        # Noisy Image
        noisy_img = np.sqrt((output + noise1) * (output + noise1) + noise2 * noise2)
    else:
        noisy_path = npy_path.replace(r"origin/", f"sigma_{sigma}/")
        noisy_img = read_npy(noisy_path, normalize=False)

    if config["output_mode"] != 'image':
        # Noisy
        output = transform((noisy_img - output).astype(np.int32)).float()
    else:
        # CLean Image
        output = transform(output.astype(np.int32)).float()
    # logger.success(get_infor(noisy_img, 'numpy'))
    noisy_img = transform(noisy_img.astype(np.int32)).float()
    # logger.success(get_infor(noisy_img))
    if not create_noise_map:
        noise_map = torch.ones_like(noisy_img) * sigma
    else:
        noise_map = torch.ones_like(noisy_img) * noise_map_sigma

    if config['clip']:
        noise_map = noise_map.clamp(0, 255)
        output = output.clamp(0, 255)
        noisy_img = noisy_img.clamp(0, 255)

    if config['normalize']:
        noise_map = noise_map.div(255)
        output = output.div(255)
        noisy_img = noisy_img.div(255)
    # logger.success(get_infor(noisy_img))

    # if config['with_map']:
    #     noisy_img = torch.cat((noisy_img, noise_map), dim=0)

    if return_path:
        return noisy_img.unsqueeze(0), noise_map.unsqueeze(0), output.unsqueeze(0), npy_path
    else:
        return noisy_img.unsqueeze(0), noise_map.unsqueeze(0), output.unsqueeze(0)


def initialize_model(model_names, model_configs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if len(model_names) == 1:
        model = get_model(name=model_names[0], **model_configs)

    else:
        denoise_model = get_model(
            name=model_names[0], **model_configs["denoise"]
        )
        print(model_names[1])
        noisemap_model = get_model(name=model_names[1], **model_configs["noisemap"])

        model = HybridNet(denoise_net=denoise_model, noisemap_net=noisemap_model)

    check_point = torch.load(model_configs["weigth_path"])
    model.load_state_dict(check_point)
    model.eval()
    # model = torch.load(os.path.join(save_dir, "model_%03d.pth" % initial_epoch))
    model.to(device)
    return model


class DemoUI:
    def __init__(self, configs=strategy_configs):
        self.models = {}
        for key, value in configs.items():
            self.models[key] = initialize_model(value['model'], value['model_configs'])
        self.data_paths = glob.glob(f"data/test/origin/*.npy")
        self.configs = configs
        self.run_flag = False

        self.sigma = None
        self.data_path = None
        self.origin_image = None
        self.noise_image = None
        self.denoise_images = {
            "1": None,
            "2": None,
            "3": None,
            "4": None,
            "5": None,
        }
        self.psnr = {
            "1": None,
            "2": None,
            "3": None,
            "4": None,
            "5": None,
        }
        self.ssim = {
            "1": None,
            "2": None,
            "3": None,
            "4": None,
            "5": None,
        }

    def inference_model(self, create_image=True,
                        create_noise_map=False,
                        noise_map_sigma=10
                        ):
        noisy_img, noise_map, output = create_data(
            npy_path=self.data_path,
            sigma=self.sigma,
            return_path=False,
            config=self.configs['4'],
            create_image=create_image,
            create_noise_map=create_noise_map,
            noise_map_sigma=noise_map_sigma
        )
        self.noise_image = (noisy_img * 255).clamp(0, 255).cpu().detach().numpy().astype(np.uint8).squeeze(0).squeeze(0)
        self.origin_image = (output * 255).clamp(0, 255).cpu().detach().numpy().astype(np.uint8).squeeze(0).squeeze(0)

        noisy_img, noise_map, output = noisy_img.cuda(), noise_map.cuda(), output.cuda()

        for key, value in self.configs.items():
            if key == "5":
                predicted_noise_level, predicted_output = self.models[key](noisy_img)
            elif key == "4":
                predicted_output = self.models[key](torch.cat((noisy_img, noise_map), dim=1))
            else:
                predicted_output = self.models[key](noisy_img)

            clean_imgs = output
            if value['output_mode'] != 'image':
                denoise_imgs = noisy_img - predicted_output
            else:
                denoise_imgs = predicted_output

            if value['normalize']:
                clean_imgs = clean_imgs * 255
                denoise_imgs = denoise_imgs * 255
                try:
                    predicted_noise_level = predicted_noise_level * 255
                except:
                    pass
                # noise_map = noise_map * 255

            clean_imgs = clean_imgs.clamp(0, 255)
            denoise_imgs = denoise_imgs.clamp(0, 255)

            clean_imgs = clean_imgs.cpu().detach().numpy().astype(np.uint8).squeeze(0).squeeze(0)
            denoise_imgs = denoise_imgs.cpu().detach().numpy().astype(np.uint8).squeeze(0).squeeze(0)

            self.denoise_images[key] = denoise_imgs
            self.psnr[key] = round(psnr(clean_imgs, denoise_imgs), 2)
            self.ssim[key] = round(ssim(clean_imgs, denoise_imgs), 4)

    def run(self, number_data_path, sigma, run_flag=False,
            create_image=True,
            create_noise_map=False,
            noise_map_sigma=10):
        path_image = f"data/test/origin/{number_data_path}"
        if run_flag:
            self.run_flag = run_flag

        if path_image != self.data_path:
            self.data_path = path_image
            self.run_flag = True

        if sigma != self.sigma:
            self.sigma = sigma
            self.run_flag = True

        if self.run_flag:
            self.inference_model(create_image=create_image,
                                 create_noise_map=create_noise_map,
                                 noise_map_sigma=noise_map_sigma
                                 )
            self.run_flag = False

    def get_result(self):
        return {
            "image": {
                "Original": self.origin_image,
                "Noise": self.noise_image,
                "Strategy1": self.denoise_images["1"],
                "Strategy2": self.denoise_images["2"],
                "Strategy3": self.denoise_images["3"],
                "Strategy4": self.denoise_images["4"],
                "Strategy5": self.denoise_images["5"],
            },
            "psnr": {
                "Strategy1": self.psnr["1"],
                "Strategy2": self.psnr["2"],
                "Strategy3": self.psnr["3"],
                "Strategy4": self.psnr["4"],
                "Strategy5": self.psnr["5"],
            },
            "ssim": {
                "Strategy1": self.ssim["1"],
                "Strategy2": self.ssim["2"],
                "Strategy3": self.ssim["3"],
                "Strategy4": self.ssim["4"],
                "Strategy5": self.ssim["5"],
            },
        }
