from typing import Tuple,List,Optional,Mapping,Any
import torch
import torch.nn as nn
from collections import OrderedDict
from timm.models.layers import trunc_normal_
from lib.model.backbone.dinov2.vision_transformer import  DinoVisionTransformer
from lib.model.backbone.dinov2.positional_encoding import  interpolate_pos_encoding
from .layers.patch_embed import PatchEmbedNoSizeCheck
from .lora.apply import  find_all_frozen_nn_linear_names, apply_lora
from lib.core import register

__all__=['LoRA_DINOv2']
@register
class LoRA_DINOv2(nn.Module):
    def __init__(self,
                 vit: DinoVisionTransformer,
                 feat_size: Tuple[int, int],
                 lora_r: int,
                 lora_alpha: float,
                 lora_dropout: float,
                 use_rslora:bool = False
                 ):
        super().__init__()
        self.feat_size = feat_size
        self.patch_embed = PatchEmbedNoSizeCheck.build(vit.patch_embed)
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.embed_dim = vit.embed_dim

        self.pos_embed = nn.Parameter(torch.empty(1, feat_size[0]*feat_size[1], self.embed_dim))
        self.pos_embed.data.copy_(interpolate_pos_encoding(vit.pos_embed.data[:, 1:, :],
                                                           self.feat_size,
                                                           vit.patch_embed.patches_resolution,
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

    def _x_feat(self, x: torch.Tensor):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = x + self.token_type_embed[2].view(1, 1, self.embed_dim)
        return x
    def forward(self, x: torch.Tensor):
        x = self._x_feat(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        x = self.norm(x)
        return x

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