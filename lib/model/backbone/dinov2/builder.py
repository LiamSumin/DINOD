import torch
import ccopy

_default_config = {
'block_chunks': 0,
    'init_values': 1.0e-05,
    'drop_path_uniform': True,
    'img_size': 518
}

def build_dino_v2_backbone(size: str, load_pretrained: bool, **kwargs):
    config = copy.deepcopy(_default_config)
    config.update(kwargs)

    if size == 's':
        from . import vit_small
        model = vit_small(patch_size=14, **config)
    elif size == 'b':
        from . import vit_base
        model = vit_base(patch_size=14, **config)
    elif size == 'l':
        from . import vit_large
        model = vit_large(patch_size=14, **config)
    elif size == 'g':
        from . import vit_giant2
        size = vit_giant2(patch_size=14, **config)
    else:
        raise NotImplementedError(f'Unknown DINO v2 model name: {name}')
    if load_pretrained:
        model.load_state_dict(torch.load(f"pretrained/dinov2_vit{size}14_pretrain.pth", weights_only=True))

    return model