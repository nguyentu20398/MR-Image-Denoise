sigmas = [10, 25, 35, 45, 55]

strategy_configs = {
    "1": {
        "model": ["dncnn"],
        "model_configs": {
            "layers": 10,
            "input_channels": 1,
            "out_channels": 1,
            "feature": 192,
            "weigth_path": "result/layers_dncnn10_Falseclip_Truenormalize_sigrange10_10_image/model_025.pth",
        },
        "range_sigma": sigmas,
        "step": 5,
        "data_path": "data",
        "n_workers": 8,
        "batch_size": 1,
        "normalize": True,
        "clip": False,
        "device": 'cuda:0',
        "output_mode": 'image',
        "with_map": False
    },
    "2": {
        "model": ["dnresnet"],
        "model_configs": {
            "layers": 10,
            "input_channels": 1,
            "out_channels": 1,
            "feature": 192,
            "weigth_path": "result/layers_dnresnet10_Falseclip_Truenormalize_sigrange10_10_noise/model_010.pth",
        },
        "range_sigma": sigmas,
        "step": 5,
        "data_path": "data",
        "n_workers": 8,
        "batch_size": 1,
        "normalize": True,
        "clip": False,
        "device": 'cuda:0',
        "output_mode": 'noise',
        "with_map": False
    },
    "3": {
        "model": ["dnresnet"],
        "model_configs": {
            "layers": 10,
            "input_channels": 1,
            "out_channels": 1,
            "feature": 192,
            "weigth_path": "result/layers_dnresnet10_Falseclip_Truenormalize_sigrange10_10_image/model_017.pth",
        },
        "range_sigma": sigmas,
        "step": 5,
        "data_path": "data",
        "n_workers": 8,
        "batch_size": 1,
        "normalize": True,
        "clip": False,
        "device": 'cuda:0',
        "output_mode": 'image',
        "with_map": False
    },

    "4": {
        "model": ["dnresnet"],
        "model_configs": {
            "layers": 10,
            "input_channels": 2,
            "out_channels": 1,
            "feature": 192,
            "weigth_path": "result/layers_dnresnet10_Falseclip_Truenormalize_sigrange10_10_image_withMap/model_040.pth",
        },
        "range_sigma": sigmas,
        "step": 5,
        "data_path": "data",
        "n_workers": 8,
        "batch_size": 1,
        "normalize": True,
        "clip": False,
        "device": 'cuda:0',
        "output_mode": 'image',
        "with_map": True
    },
    "5": {
        "model": ["dnresnet", "fcn"],
        "model_configs": {
            "denoise":
                {"layers": 10,
                 "input_channels": 2,
                 "out_channels": 1,
                 "feature": 192, },
            "noisemap":
                {"model_name": 'fcn',
                 "layers": 5,
                 "input_channels": 1,
                 "feature": 32, },
            "weigth_path": "result/hybrid_dnresnet_fcn_Falseclip_Truenormalize_sigrange10_25_image_withMap/model_003.pth"
        },
        "range_sigma": sigmas,
        "step": 5,
        "data_path": "data",
        "n_workers": 8,
        "batch_size": 1,
        "normalize": True,
        "clip": False,
        "device": 'cuda:0',
        "output_mode": 'image',
        "with_map": False
    },
}
