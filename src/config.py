import yaml


def load_config(config_path: str = "./configs/train.yml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config
