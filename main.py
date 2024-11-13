import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import importlib
from lib.solver import TASKS
import lib.utils.misc.dist as dist

if __name__ =="__main__":
    dist.init_distributed()

    config_module = importlib.import_module("lib.config.dinod.config")
    cfg = config_module.cfg
    config_module.update_config_from_file("configs/dinov2_small.yaml")


    solver = TASKS[cfg.task](cfg, 'training')

    solver.fit()