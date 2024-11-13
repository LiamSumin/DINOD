import copy
from datetime import timedelta
import torch
from lib.misc.torch.distributed.barrier import torch_distributed_zero_first
from lib.core.runtime.global_constant import get_global_constant

def _build_backbone(backbone_config: dict, load_pretrained=True):
    backbone_config = copy.deepcopy(backbone_config)

    if load_pretrained and 'pretrained' in backbone_config:
            load_pretrained = backbone_config['pretrained']
            del backbone_config['pretrained']

    else:
        backbone_config = {}

    if backbone_config['type'] == 'DINOv2':
        from .dinov2.builder import build_dino_v2_backbone

        backbone = build_dino_v2_backbone(name= backbone_config['name'], load_pretrained=load_pretrained)#, **backbone_config)

    else:
        raise Exception(f'unsupported {backbone_config["type"]}')

    return backbone

def build_backbone(backbone_config: dict, load_pretrained=True):
    return _build_backbone(backbone_config, load_pretrained)

