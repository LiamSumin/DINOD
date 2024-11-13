import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

def get_optimizer(optimizer_config, model_params):
    optimizer_type = optimizer_config.type
    del optimizer_config['type']  # Remove 'type' to avoid passing it as a parameter

    if optimizer_type == 'AdamW':
        return optim.AdamW(model_params, **optimizer_config)
    elif optimizer_type == 'SGD':
        return optim.SGD(model_params, **optimizer_config)
    elif optimizer_type == 'Adam':
        return optim.Adam(model_params, **optimizer_config)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

def get_lr_scheduler(lr_scheduler_config, optimizer):
    scheduler_type = lr_scheduler_config.type
    del lr_scheduler_config['type']
    if scheduler_type == 'MultiStepLR':

        return lr_scheduler.MultiStepLR(optimizer, **lr_scheduler_config)
    elif scheduler_type == 'CosineAnnealingLR':
        return lr_scheduler.CosineAnnealingLR(optimizer, **lr_scheduler_config)
    elif scheduler_type == 'OneCycleLR':
        return lr_scheduler.OneCycleLR(optimizer, **lr_scheduler_config)
    elif scheduler_type == 'LambdaLR':
        return lr_scheduler.LambdaLR(optimizer, **lr_scheduler_config)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")