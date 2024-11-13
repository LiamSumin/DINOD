import torch
import torch.nn as nn
import copy
from datetime import datetime
from pathlib import Path
from typing import Dict

from lib.model.dinod.builder import get_DINOD_build_context
from lib.optim.criterion.builder import get_criterion
from lib.model.dinod.modules.postprocessing.builder import get_postprocessor
from lib.optim.ema.builder import get_ema
from lib.utils.misc import dist
from lib.optim.optimizer.builder import  get_optimizer, get_lr_scheduler
from lib.optim.scaler.builder import  get_scaler
from lib.data.builder import get_train_dataloader, get_val_dataloader

class BaseSolver(object):
    def __init__(self, cfg, status):
        self.cfg = copy.deepcopy(cfg)
        self.status = status

    def setup(self, ):
        '''

        :return:
        '''
        cfg = self.cfg
        runtime_config = cfg['RUNTIME']
        model_config = cfg['MODEL']
        train_config = cfg['TRAIN']
        test_config = cfg['TEST']

        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

        self.model = dist.warp_model(get_DINOD_build_context(cfg).to(self.device), runtime_config['find_unused_parameters'], runtime_config['sync_bn'])
        self.criterion = get_criterion().to(self.device)
        self.postprocessor = get_postprocessor(cfg['MODEL']['POSTPROCESS']).to(self.device)
        if cfg.RUNTIME.use_amp == False:
            self.scalar = get_scaler(cfg.RUNTIME.scaler)
        self.ema = get_ema(model=self.model, ema_config=cfg.OPTIMIZER.EMA).to(self.device) if runtime_config.ema is not None else None

        self.output_dir = Path(runtime_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self, ):
        self.setup()
        self.optimizer = get_optimizer(self.cfg.OPTIMIZER.OPTIMIZER, self.model.parameters())
        self.lr_scheduler = get_lr_scheduler(self.cfg.OPTIMIZER.LR_SCHEDULER, self.optimizer)

        self.train_dataloader = get_train_dataloader(self.cfg.DATALOADER.TRAIN)
        self.val_dataloader = get_val_dataloader(self.cfg.DATALOADER.VAL)







