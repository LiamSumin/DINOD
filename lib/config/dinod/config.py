from easydict import EasyDict as edict
import yaml

cfg = edict()
cfg.task = "detection"

cfg.COMMON = edict()
cfg.COMMON.feat_size = [518, 518]
cfg.COMMON.patch_size = [14, 14]
cfg.COMMON.lora_r = 64
cfg.COMMON.lora_alpha = 64
cfg.COMMON.lora_dropout = 0.
cfg.COMMON.use_rslora: False

cfg.RUNTIME = edict()
cfg.RUNTIME.sync_bn = True
cfg.RUNTIME.find_unused_parameters = False
cfg.RUNTIME.output_dir = "evaluation/DINOD"
cfg.RUNTIME.use_ema = False
cfg.RUNTIME.use_amp = False

cfg.RUNTIME.scaler = edict()
cfg.RUNTIME.scaler.type = "GradScaler"
cfg.RUNTIME.scaler.enabled = True

cfg.RUNTIME.ema = edict()
cfg.RUNTIME.ema.type = "ModelEMA"
cfg.RUNTIME.ema.decay = 0.9999
cfg.RUNTIME.ema.warmups = 2000

# MODEL
cfg.MODEL = edict()
cfg.MODEL.NAME = "DINOD"
cfg.MODEL.TYPE = 'small'
cfg.MODEL.PRETRAINED_PATH = "pretrained"


# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.type = "DINOv2"
cfg.MODEL.BACKBONE.name = "ViT-S/14"
cfg.MODEL.BACKBONE.acc = "default"
cfg.MODEL.BACKBONE.pretrained = cfg.MODEL.PRETRAINED_PATH

# MODEL.DECODER
cfg.MODEL.DECODER = edict()
cfg.MODEL.DECODER.TYPE = "RTDETR_DECODER"

cfg.MODEL.DECODER.CONFIG = edict()
cfg.MODEL.DECODER.CONFIG.feat_channels = [384]
cfg.MODEL.DECODER.CONFIG.feat_strides = [14]
cfg.MODEL.DECODER.CONFIG.hidden_dim = 256
cfg.MODEL.DECODER.CONFIG.num_levels = 3
cfg.MODEL.DECODER.CONFIG.num_queries = 300
cfg.MODEL.DECODER.CONFIG.num_decoder_layers = 6
cfg.MODEL.DECODER.CONFIG.num_denoising = 100
cfg.MODEL.DECODER.CONFIG.eval_idx = -1
cfg.MODEL.DECODER.CONFIG.eval_spatial_size = [518, 518]

# MODEL.POSTPROCESS
cfg.MODEL.POSTPROCESS = edict()
cfg.MODEL.POSTPROCESS.num_top_queries = 300

# OPTIMIZER
cfg.OPTIMIZER = edict()

# OPTIMIZER.OPTIMIZER
cfg.OPTIMIZER.OPTIMIZER = edict()
cfg.OPTIMIZER.OPTIMIZER.type = 'AdamW'
# cfg.OPTIMIZER.OPTIMIZER.params = []
#
# backbone_param = edict()
# backbone_param.params = 'backbone'
# backbone_param.lr = 0.00001
# cfg.OPTIMIZER.OPTIMIZER.params.append(backbone_param)
#
# decoder_param = edict()
# decoder_param.params = '^(?=.*decoder(?=.*bias|.*norm.*weight)).*$'
# decoder_param.weight_decay = 0.
# cfg.OPTIMIZER.OPTIMIZER.params.append(decoder_param)

cfg.OPTIMIZER.OPTIMIZER.lr = 0.0001
cfg.OPTIMIZER.OPTIMIZER.betas = [0.9, 0.999]
cfg.OPTIMIZER.OPTIMIZER.weight_decay = 0.0001

cfg.OPTIMIZER.LR_SCHEDULER = edict()
cfg.OPTIMIZER.LR_SCHEDULER.type = 'MultiStepLR'
cfg.OPTIMIZER.LR_SCHEDULER.milestones = [1000]
cfg.OPTIMIZER.LR_SCHEDULER.gamma = 0.1

# OPTIMIZER.EMA
cfg.OPTIMIZER.EMA = edict()
cfg.OPTIMIZER.EMA.type ='ModelEMA'

cfg.OPTIMIZER.EMA.CONFIG =edict()
cfg.OPTIMIZER.EMA.CONFIG.decay = 0.9999
cfg.OPTIMIZER.EMA.CONFIG.warmups = 2000

# DATALOADER
cfg.DATALOADER = edict()
cfg.DATALOADER.task = "detection"
cfg.DATALOADER.num_classes = 80
cfg.DATALOADER.remap_mscoco_category = True

# DATALOADER.TRAIN
cfg.DATALOADER.TRAIN = edict()
cfg.DATALOADER.TRAIN.type = "DataLoader"
cfg.DATALOADER.TRAIN.dataset = edict()
cfg.DATALOADER.TRAIN.dataset.type = "CocoDetection"
cfg.DATALOADER.TRAIN.dataset.img_folder= "/home2/ObjectDetection/COCO/train2017/"
cfg.DATALOADER.TRAIN.dataset.ann_file= "/home2/ObjectDetection/COCO/annotations/instances_train2017.json"
cfg.DATALOADER.TRAIN.dataset.return_masks= False
cfg.DATALOADER.TRAIN.dataset.transforms = edict()
cfg.DATALOADER.TRAIN.dataset.transforms.type = "Compose"
cfg.DATALOADER.TRAIN.dataset.transforms.ops= [
                    {"type": "RandomPhotometricDistort", "p": 0.5},
                    {"type": "RandomZoomOut", "fill": 0},
                    {"type": "RandomIoUCrop", "p": 0.8},
                    {"type": "SanitizeBoundingBox", "min_size": 1},
                    {"type": "RandomHorizontalFlip"},
                    {"type": "Resize", "size": [518, 518]},
                    # {"type": "Resize", "size": 639, "max_size": 640},
                    # {"type": "PadToSize", "spatial_size": 640},
                    {"type": "ToImageTensor"},
                    {"type": "ConvertDtype"},
                    {"type": "SanitizeBoundingBox", "min_size": 1},
                    {"type": "ConvertBox", "out_fmt": "cxcywh", "normalize": True}
]
cfg.DATALOADER.TRAIN.shuffle = True
cfg.DATALOADER.TRAIN.batch_size=8
cfg.DATALOADER.TRAIN.num_workers=4
cfg.DATALOADER.TRAIN.drop_last=True
cfg.DATALOADER.TRAIN.collate_fn="default_collate_fn"

# DATALOADER.VAL
cfg.DATALOADER.VAL = edict()
cfg.DATALOADER.VAL.type = "DataLoader"
cfg.DATALOADER.VAL.dataset = edict()
cfg.DATALOADER.VAL.dataset.type = "CocoDetection"
cfg.DATALOADER.VAL.dataset.img_folder= "/home2/ObjectDetection/COCO/val2017/"
cfg.DATALOADER.VAL.dataset.ann_file= "/home2/ObjectDetection/COCO/annotations/instances_val2017.json"
cfg.DATALOADER.VAL.dataset.transforms = edict()
cfg.DATALOADER.VAL.dataset.transforms.type = "Compose"
cfg.DATALOADER.VAL.dataset.transforms.ops= [
    {"type": "Resize", "size": [518, 518]},
    {"type": "ToImageTensor"},
    {"type": "ConvertDtype"}
]
cfg.DATALOADER.VAL.shuffle = False
cfg.DATALOADER.VAL.batch_size=8
cfg.DATALOADER.VAL.num_workers=4
cfg.DATALOADER.VAL.drop_last=False
cfg.DATALOADER.VAL.collate_fn="default_collate_fn"


#
cfg.TRAIN = edict()

# TEST
cfg.TEST = edict()


# DATA
cfg.DATA = edict()




def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename, base_cfg=None):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        if base_cfg is not None:
            _update_config(base_cfg, exp_config)
        else:
            _update_config(cfg, exp_config)
