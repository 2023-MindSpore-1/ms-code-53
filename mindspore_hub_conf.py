"""hub config"""
from src.unet3d_model import UNet3d
from src.config import config as cfg

def unet3d_net(*args, **kwargs):
    return UNet3d(*args, **kwargs)

def create_network(name, *args, **kwargs):
    """create_network about unet3d"""
    if name == "unet3d":
        return unet3d_net(config=cfg, *args, **kwargs)
    raise NotImplementedError(f"{name} is not implemented in the repo")
