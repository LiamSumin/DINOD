import torch
import torch.nn as nn
import torch.amp as amp

def get_scaler(scaler_config):
    scaler_type = scaler_config.type
    del scaler_config['type']
    if scaler_type == 'GradScaler':
        scaler = amp.grad_scaler.GradScaler( **scaler_config)

    else :
        raise NotImplementedError(f"The Scaler {scaler_type} is not implemented.")

    return scaler