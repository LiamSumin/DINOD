from typing import Callable, Optional, Tuple, Union

from torch import Tensor
import torch.nn as nn
from lib.model.backbone.dinov2.layers.patch_embed import PatchEmbed
from timm.layers import to_2tuple

class PatchEmbedNoSizeCheck(nn.Module):
    def __init__(self,
                 in_chans: int = 3,
                 embed_dim: int = 768,
                 norm_layer: Optional[Callable] = None,
                 patch_size: Tuple[int, int] = (16,16)
                 ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size= patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

    @classmethod
    def build(cls, module: PatchEmbed):
        new_module = cls.__new__(cls)
        nn.Module.__init__(new_module)
        new_module.patch_size = module.proj.kernel_size
        new_module.proj = module.proj
        new_module.norm = module.norm
        return new_module
