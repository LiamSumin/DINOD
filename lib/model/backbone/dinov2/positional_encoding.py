import math
import torch
import torch.nn as nn


def interpolate_pos_encoding(pos_embed, new_size, old_size = None, num_prefix_tokens=1, interpolate_offset=0.1):
    num_pos_tokens = pos_embed.shape[1]
    if old_size is None:
        hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))
        old_size = hw, hw
    if new_size[0] == old_size[0] and new_size[1] == old_size[1]:
        return pos_embed

    dim = pos_embed.shape[-1]

    old_dtype = pos_embed.dtype
    if num_prefix_tokens:
        pos_embed_prefix, pos_embed = pos_embed[:, :num_prefix_tokens], pos_embed[:, num_prefix_tokens:]
    else:
        pos_embed_prefix, pos_embed = None, pos_embed

    kwargs = {}
    if interpolate_offset:
        # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
        # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
        sx = float(new_size[0] + interpolate_offset) / old_size[0]
        sy = float(new_size[1] + interpolate_offset) / old_size[1]
        kwargs["scale_factor"] = (sx, sy)
    else:
        # Simply specify an output size instead of a scale factor
        kwargs["size"] = new_size
    pos_embed = nn.functional.interpolate(
        pos_embed.to(torch.float32).reshape(1, old_size[1], old_size[0], dim).permute(0, 3, 1, 2),
        mode="bicubic",
        **kwargs
    )

    pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim).to(old_dtype)
    if pos_embed_prefix is not None:
        pos_embed = torch.cat([pos_embed_prefix, pos_embed], dim=1)
    return pos_embed
