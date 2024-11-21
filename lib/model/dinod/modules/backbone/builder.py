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


def build_preprocessing(preprocess_config):
    preprocess_type = preprocess_config.pop('type')

    if preprocess_type == "Swinv2":
        from .swinv2 import SwinV2FirstStage, SwinToDINOEmbedding
        preprocess_nn = SwinV2FirstStage(img_size=preprocess_config.img_size, window_size= preprocess_config.window_size)
        H_prime = W_prime = preprocess_config.img_size // preprocess_nn.patch_size
        preprocessing_to_backbone_embedding = SwinToDINOEmbedding(input_dim=preprocess_config.input_dim,
                                                           hidden_dim= preprocess_config.hidden_dim,
                                                           H=H_prime,
                                                           W=W_prime)

    else :
        raise NotImplementedError(f"The preprocessing method {preprocess_type} is not implemented.")

    return preprocess_nn, preprocessing_to_backbone_embedding
