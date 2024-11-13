from typing import Tuple, Mapping, Any
import torch
import torch.nn as nn
from collections import OrderedDict
from timm.models.layers import trunc_normal_
from lib.model.dinod.modules.backbone.dinov2 import DinoVisionTransformer, interpolate_pos_encoding
from .modules.patch_embed import PatchEmbedNoSizeCheck
from .modules.lora.apply import find_all_frozen_nn_linear_names, apply_lora
from lib.model.dinod.modules.decoder import RTDETRTransformer

class DINOD(nn.Module):
    def __init__(self, backbone:DinoVisionTransformer,
                 decoder: RTDETRTransformer,
                 feat_size: Tuple[int, int],
                 patch_size: Tuple[int, int],
                 lora_r: int,
                 lora_alpha: int,
                 lora_dropout: float,
                 use_rslora: bool = False ):
        super().__init__()
        self.feat_size = feat_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbedNoSizeCheck.build(backbone.patch_embed)
        self.blocks = backbone.blocks
        self.norm = backbone.norm
        self.embed_dim = backbone.embed_dim
        self.patch_cnt = [int(self.feat_size[0] / self.patch_size[0]), int(self.feat_size[1] / self.patch_size[1])]

        self.pos_embed = nn.Parameter(torch.empty(1, self.patch_cnt[0] * self.patch_cnt[1], self.embed_dim))
        self.pos_embed.data.copy_(interpolate_pos_encoding(backbone.pos_embed.data[:, 1:, :],
                                                           self.patch_cnt,
                                                           backbone.patch_embed.patches_resolution,
                                                           num_prefix_tokens=0, interpolate_offset=0))

        self.lora_alpha = lora_alpha
        self.use_rslora = use_rslora

        for param in self.parameters():
            param.requires_grad = False

        self.token_type_embed = nn.Parameter(torch.empty(3, self.embed_dim))
        trunc_normal_(self.token_type_embed, std=.02)

        for i_layer, block in enumerate(self.blocks):
            linear_names = find_all_frozen_nn_linear_names(block)
            apply_lora(block, linear_names, lora_r, lora_alpha, lora_dropout, use_rslora)

        self.decoder = decoder

    def _x_feat(self, x:torch.Tensor):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = x + self.token_type_embed[2].view(1, 1, self.embed_dim)
        return x

    def forward(self, x: torch.Tensor, targets=None):
        x = self._x_feat(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        x = self.norm(x)
        x = self.decoder(x, targets)
        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self

    def state_dict(self, **kwargs):
        state_dict = super().state_dict(**kwargs)
        prefix = kwargs.get('prefix', '')
        for key in list(state_dict.keys()):
            if not self.get_parameter(key[len(prefix):]).requires_grad:
                state_dict.pop(key)
        if self.lora_alpha != 1.:
            state_dict[prefix + 'lora_alpha'] = torch.as_tensor(self.lora_alpha)
            state_dict[prefix + 'use_rslora'] = torch.as_tensor(self.use_rslora)
        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], **kwargs):
        if 'lora_alpha' in state_dict:
            state_dict = OrderedDict(**state_dict)
            self.lora_alpha = state_dict['lora_alpha'].item()
            self.use_rslora = state_dict['use_rslora'].item()
            del state_dict['lora_alpha']
            del state_dict['use_rslora']
        return super().load_state_dict(state_dict, **kwargs)