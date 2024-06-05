config1 = {
    "layers": [10],
    "channels": 1,
    "out_channels": 1,
    "features": 192,
    "n_workers": 10,
    "scheduler_mode": min,
    "scheduler_patience": 1,
    "model_name": "dncnn",
    "loss_function": "L1",
    "alpha_loss": [0.001],
    "sigma_range": [10,25,35,45,55],
    "weight": r"result/layers_dncnn10_Falseclip_Truenormalize_sigrange10_10_image/model_025.pth",
    "data_dir": "data",
    "clip": False,
    "normalize": True,
    "output_mode": 'image',
    "with_map": False
}

config2 = {
    "layers": [10],
    "channels": 1,
    "out_channels": 1,
    "features": 192,
    "n_workers": 10,
    "scheduler_mode": min,
    "scheduler_patience": 1,
    "model_name": "dnresnet",
    "loss_function": "L1",
    "alpha_loss": [0.001],
    "sigma_range": [10,25,35,45,55],
    "weight": r"result/layers_dnresnet10_Falseclip_Truenormalize_sigrange10_10_noise/model_010.pth",
    "data_dir": "data",
    "clip": False,
    "normalize": True,
    "output_mode": 'noise',
    "with_map": False
}

config3 = {
    "layers": [10],
    "channels": 1,
    "out_channels": 1,
    "features": 192,
    "n_workers": 10,
    "scheduler_mode": min,
    "scheduler_patience": 1,
    "model_name": "dnresnet",
    "loss_function": "L1",
    "alpha_loss": [0.001],
    "sigma_range": [10,25,35,45,55],
    "weight": r"result/layers_dnresnet10_Falseclip_Truenormalize_sigrange10_10_image/model_017.pth",
    "data_dir": "data",
    "clip": False,
    "normalize": True,
    "output_mode": 'image',
    "with_map": False
}

config4 = {
    "layers": [10],
    "channels": 2,
    "out_channels": 1,
    "features": 192,
    "n_workers": 10,
    "scheduler_mode": min,
    "scheduler_patience": 1,
    "model_name": "dnresnet",
    "loss_function": "L1",
    "alpha_loss": [0.001],
    "sigma_range": [10,25,35,45,55],
    "weight": r"result/layers_dnresnet10_Falseclip_Truenormalize_sigrange10_10_image_withMap/model_040.pth",
    "data_dir": "data",
    "clip": False,
    "normalize": True,
    "output_mode": 'image',
    "with_map": True
}