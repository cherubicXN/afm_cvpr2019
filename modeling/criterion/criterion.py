import torch
from modeling import registry

@registry.CRITERIONS.register("l1")
def build_l1_loss(cfg):
    return torch.nn.L1Loss()


def build_criterions(cfg):
    assert cfg.CRITERION.LOSS in registry.CRITERIONS, \
        "cfg.CRITERION.LOSS: {} are not registered in registry".format(cfg.CRITERION.LOSS)    
    return registry.CRITERIONS[cfg.CRITERION.LOSS](cfg)