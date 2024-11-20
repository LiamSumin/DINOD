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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.last_epoch = cfg.MODEL.last_epoch

        self.model = dist.warp_model(get_DINOD_build_context(cfg).to(self.device), runtime_config['find_unused_parameters'], runtime_config['sync_bn'])
        self.criterion = get_criterion(cfg.MODEL.CRITERION).to(self.device)
        self.postprocessor = get_postprocessor(cfg.MODEL.POSTPROCESS).to(self.device)


        if cfg.RUNTIME.use_amp == False:
            self.scaler = get_scaler(cfg.RUNTIME.scaler)
        self.ema = get_ema(model=self.model, ema_config=cfg.OPTIMIZER.EMA).to(self.device) if runtime_config.ema is not None else None

        self.output_dir = Path(runtime_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self, ):
        self.setup()
        self.optimizer = get_optimizer(self.cfg.OPTIMIZER.OPTIMIZER, self.model.parameters())
        self.lr_scheduler = get_lr_scheduler(self.cfg.OPTIMIZER.LR_SCHEDULER, self.optimizer)

        if self.cfg.resume:
            print(f"Resume checkpoint from {self.cfg.resume}")
            self.resume(self.cfg.resume)

        self.train_dataloader = get_train_dataloader(self.cfg.DATALOADER.TRAIN)
        self.val_dataloader = get_val_dataloader(self.cfg.DATALOADER.VAL)



    def eval(self, ):
        self.setup()
        if self.cfg.resume:
            print(f"Resume checkpoint from {self.cfg.resume}")
            self.resume(self.cfg.resume)

        self.val_dataloader = get_val_dataloader(self.cfg.DATALOADER.VAL)

    def state_dict(self, last_epoch):
        '''state dict
        '''
        state = {}
        state['model'] = dist.de_parallel(self.model).state_dict()
        state['date'] = datetime.now().isoformat()

        # TODO
        state['last_epoch'] = last_epoch

        if self.optimizer is not None:
            state['optimizer'] = self.optimizer.state_dict()

        if self.lr_scheduler is not None:
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            # state['last_epoch'] = self.lr_scheduler.last_epoch

        if self.ema is not None:
            state['ema'] = self.ema.state_dict()

        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()

        return state

    def load_state_dict(self, state):
        '''load state dict
        '''
        # TODO
        if getattr(self, 'last_epoch', None) and 'last_epoch' in state:
            self.last_epoch = state['last_epoch']
            print('Loading last_epoch')

        if getattr(self, 'model', None) and 'model' in state:
            if dist.is_parallel(self.model):
                self.model.module.load_state_dict(state['model'])
            else:
                self.model.load_state_dict(state['model'])
            print('Loading model.state_dict')

        if getattr(self, 'ema', None) and 'ema' in state:
            self.ema.load_state_dict(state['ema'])
            print('Loading ema.state_dict')

        if getattr(self, 'optimizer', None) and 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])
            print('Loading optimizer.state_dict')

        if getattr(self, 'lr_scheduler', None) and 'lr_scheduler' in state:
            self.lr_scheduler.load_state_dict(state['lr_scheduler'])
            print('Loading lr_scheduler.state_dict')

        if getattr(self, 'scaler', None) and 'scaler' in state:
            self.scaler.load_state_dict(state['scaler'])
            print('Loading scaler.state_dict')

    def save(self, path):
        '''save state
        '''
        state = self.state_dict()
        dist.save_on_master(state, path)

    def resume(self, path):
        '''load resume
        '''
        # for cuda:0 memory
        state = torch.load(path, map_location='cpu')
        self.load_state_dict(state)

    def load_tuning_state(self, path, ):
        """only load model for tuning and skip missed/dismatched keys
        """
        if 'http' in path:
            state = torch.hub.load_state_dict_from_url(path, map_location='cpu')
        else:
            state = torch.load(path, map_location='cpu')

        module = dist.de_parallel(self.model)

        # TODO hard code
        if 'ema' in state:
            stat, infos = self._matched_state(module.state_dict(), state['ema']['module'])
        else:
            stat, infos = self._matched_state(module.state_dict(), state['model'])

        module.load_state_dict(stat, strict=False)
        print(f'Load model.state_dict, {infos}')

    @staticmethod
    def _matched_state(state: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]):
        missed_list = []
        unmatched_list = []
        matched_state = {}
        for k, v in state.items():
            if k in params:
                if v.shape == params[k].shape:
                    matched_state[k] = params[k]
                else:
                    unmatched_list.append(k)
            else:
                missed_list.append(k)

        return matched_state, {'missed': missed_list, 'unmatched': unmatched_list}

    def fit(self, ):
        raise NotImplementedError('')

    def val(self, ):
        raise NotImplementedError('')

