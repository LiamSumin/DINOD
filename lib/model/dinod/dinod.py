from typing import Tuple, List, Optional, Mapping, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from timm.models.layers import trunc_normal_
from lib.model.backbone.dinov2.vision_transformer import DinoVisionTransformer
from lib.model.backbone.dinov2.positional_encoding import interpolate_pos_encoding
from .backbone.layers.patch_embed import PatchEmbedNoSizeCheck
from .backbone.lora.apply import find_all_frozen_nn_linear_names, apply_lora

import numpy as np

from lib.core import register

__all__ = ['DINOD', ]


@register
class DINOD(nn.Module):
    #__inject__ = ['backbone', 'encoder', 'decoder']
    __inject__ = ['backbone', 'decoder']
    def __init__(self,
                 backbone,
                 #encoder,
                 decoder,
                 multi_scale=None
                 ):
        super().__init__()
        self.backbone = backbone
        #self.encoder = encoder
        self.decoder = decoder
        self.multi_scale = multi_scale

    def forward(self, x, targets=None):
        # if self.multi_scale and self.training:
        #     sz = np.random.choice(self.multi_scale)
        #     x = F.interpolate(x, size=[sz, sz])
        x = self.backbone(x)
        #x = self.encoder(x)
        x = self.decoder(x, targets)
        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
