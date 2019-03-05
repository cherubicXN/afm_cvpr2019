import torch

from . import registry


class FixedInput(object):
    def __init__(self, cfg):
        self.in_res = cfg.INPUT.IN_RES
        self.out_res = cfg.INPUT.OUT_RES

    def __call__(self, image):        
        image_out = torch.nn.functional.interpolate(image, [self.out_res]*2)        
        # shape_info[0][]
        return image_out

        
@registry.INPUT_PREPROCESS.register('fixed')
def build_fixed_input(cfg):
    return FixedInput(cfg)


def build_test_input(cfg):
    assert cfg.INPUT.STRATEGY in registry.INPUT_PREPROCESS, \
        "cfg.INPUT.STRATEGY: {} are not registered in registry".format(cfg.INPUT.STRATEGY)  

    return registry.INPUT_PREPROCESS[cfg.INPUT.STRATEGY](cfg)

