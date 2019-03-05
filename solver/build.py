import torch

# def make_optimizer(cfg, model):
#     params = []
#     for key, value in model.named_parameters():
#         if not value.requires_grad:
#             continue
        
#         lr = cfg.SOLVER.BASE_LR
#         weight_decay = cfg.SOLVER.WEIGHT_DECAY
#         if "bias" in key:
#             lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
#             weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        
#         params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

#     optimizer = torch.optim.SGD(params, cfg.SOLVER.BASE_LR, 
#     momentum=cfg.SOLVER.MOMENTUM)

#     return optimizer
def make_optimizer(cfg, model):
    optimizer = torch.optim.SGD(model.parameters(),lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    return optimizer

def make_lr_scheduler(cfg, optimizer):
    return torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=cfg.SOLVER.STEPS, gamma=cfg.SOLVER.GAMMA)