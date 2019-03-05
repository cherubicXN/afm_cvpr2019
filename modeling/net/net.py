from .unet import UNet
from .deeplabv3plus import DeepLabv3_plus
from modeling import registry
import torch

@registry.NETS.register("unet")
def build_unet(cfg):
    model = UNet(3, 2)
    return model

@registry.NETS.register("atrous")
def build_atrous(cfg):
    model = DeepLabv3_plus(nInputChannels=3, nOutChannels=2, os=16, pretrained=False, _print=True)
    return model


def build_network(cfg):
    assert cfg.MODEL.ARCH in registry.NETS, \
        "cfg.MODEL.ARCH: {} are not registered in registry".format(cfg.MODEL.ARCH)

    return registry.NETS[cfg.MODEL.ARCH](cfg)
