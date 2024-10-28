import torch
import copy
import time
from lib.model.backbone.dinov2.vision_transformer import vit_small, vit_base, vit_large

_default_config = {
    'block_chunks': 0,
    'init_values': 1.0e-05,
    'drop_path_uniform': True,
    'img_size': 518
}
config = copy.deepcopy(_default_config)

def eval_latency():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_sizes = ["s", "b", "l"]
    for model_size in model_sizes:
        latencies = []
        if model_size=="s":
            model = vit_small(patch_size=14, **config).to(device)
        elif model_size=="b":
            model = vit_base(patch_size=14, **config).to(device)
        else:
            model = vit_large(patch_size=14, **config).to(device)


        model.load_state_dict(torch.load(f"pretrained/dinov2_vit{model_size}14_pretrain.pth", weights_only=True))
        tmp = torch.rand((1, 3, 518, 518)).to(device)

        for _ in range(10):  # 10번의 Warmup Forward Pass
            with torch.no_grad():
                tmp_warmup = model.patch_embed(tmp)
                for block in model.blocks:
                    tmp_warmup = block(tmp_warmup)
                tmp_warmup = model.norm(tmp_warmup)
                tmp_warmup = model.head(tmp_warmup)

        print()
        torch.cuda.synchronize()
        s_t_blocks = time.time()
        tmp = model.patch_embed(tmp)
        torch.cuda.synchronize()
        latencies.append(1000 * (time.time() - s_t_blocks))
        for i in range(len(model.blocks)):
            torch.cuda.synchronize()

            s_t_blocks = time.time()
            tmp = model.blocks[i](tmp)
            torch.cuda.synchronize()

            latencies.append(1000 * (time.time() - s_t_blocks))
        torch.cuda.synchronize()

        s_t_blocks = time.time()
        tmp = model.norm(tmp)
        torch.cuda.synchronize()

        latencies.append(1000 * (time.time() - s_t_blocks))
        torch.cuda.synchronize()

        s_t_blocks = time.time()
        tmp = model.head(tmp)
        torch.cuda.synchronize()

        latencies.append(1000 * (time.time() - s_t_blocks))
        print(latencies)
        print(sum(latencies))
        print(tmp.shape)


if __name__ == "__main__":
    eval_latency()
