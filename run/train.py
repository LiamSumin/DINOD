import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

import lib.utils.misc.dist as dist
from lib.core import YAMLConfig
from lib.solver import TASKS

import wandb
from datetime import datetime
def wandb_setup(project_name, model_size):
    now = datetime.now()
    date_time = now.strftime("%m_%d_%y-%H-%M")
    run = wandb.init(
        project=project_name,
        name=model_size+"_"+date_time,
        group="DDP",
    )

def main(args, ) -> None:
    """
    main
    """
    dist.init_distributed()
    if args.seed is not None:
        dist.set_seed(args.seed)

    assert not all([args.tuning, args.resume]), \
    "Only support from_scratch or resume or training at one time"

    cfg = YAMLConfig(
        args.config,
        resume = args.resume,
        use_amp=args.amp,
        tuning=args.tuning
    )

    if dist.get_rank() ==0:
        wandb_setup(cfg.model_name, cfg.model_size)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default="../configs/dinod/dinod.yml" )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--test-only', action='store_true', default=False, )
    parser.add_argument('--amp', action='store_true', default=False, )
    parser.add_argument('--seed', type=int, help='seed', )
    args = parser.parse_args()

    main(args)
