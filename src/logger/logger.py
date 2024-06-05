import sys
import time

import torch
from loguru import logger

CUDA_LAUNCH_BLOCKING = "1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True


def configure_logging():
    """

    Initialize logging defaults for Project.

    This function does:

    - Assign INFO and DEBUG level to logger file handler and console handler

    """
    logger.remove(0)
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm}</green> | <blue>[{level}]</blue> <cyan>{name}</cyan>:<yellow>{function}</yellow>:<cyan>{line}</cyan> : <green>{message}</green> ",
        colorize=True,
    )

    logger.add(
        f'src/logger/log/{time.strftime("%Y-%m-%d")}.log',
        mode="a",
        format="<green>{time:YYYY-MM-DD HH:mm}</green> | <blue>[{level}]</blue> <cyan>{name}</cyan>:<yellow>{function}</yellow>:<cyan>{line}</cyan> : <green>{message}</green> ",
        rotation="2 days",
    )

    return logger
