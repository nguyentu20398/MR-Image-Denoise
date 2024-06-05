import sys

from loguru import logger

from src.model.dncnn import DnCNN
from src.model.dnresnet import DnResnet
from src.model.subnet import SubNet
from src.model.fcn import FCN



def get_model(
    name: str = "dncnn", layers: int = 17, input_channels: int = 1, feature: int = 192, out_channels: int = 1, **kwargs
):
    if name.lower() == "dncnn":
        model = DnCNN(channels=input_channels, num_of_layers=layers, features=feature, out_channels = out_channels)
    elif name.lower() == "dnresnet":
        model = DnResnet(
            depth=layers,
            input_channels=input_channels,
            n_channels=feature,
            kernel_size=3,
            out_channels = out_channels
        )
    elif name.lower() == "subnet":
        model = SubNet(channels=input_channels, num_of_layers=layers, features=feature)
    elif name.lower() == "fcn":
        model = FCN(channels=input_channels, num_of_layers=layers, features=feature)
    else:
        logger.error(f"Model {name} is not supported now. ")
        sys.exit()
    return model
