import os
import sys
import signal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import importlib
from lib.solver import TASKS
from lib.utils.misc import dist
import wandb
from datetime import datetime
import argparse


def handle_signal(signal_number, frame):
    print("CTRL+C or termination signal received. Cleaning up...")
    wandb.finish()
    sys.exit(0)


def parse_arguments():
    parser = argparse.ArgumentParser(description="DINOD argument")

    parser.add_argument("--config", type=str, required=True, help="Config for training or inference.")
    parser.add_argument("--resume", type=str)
    # 종료 전에 실행할 작업
    parser.add_argument("--logging", action="store_true", help="Enable logging")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    dist.init_distributed()
    args = parse_arguments()

    config_module = importlib.import_module("lib.config.dinod.config")
    config_module.update_config_from_file(args.config)
    cfg = config_module.cfg
    cfg.resume = args.resume

    cfg.LOGGING.run = args.logging
    if dist.is_main_process() and cfg.LOGGING.run:
        wandb.init(
            project=f"{cfg.MODEL.NAME}",
            name=f"{cfg.MODEL.TYPE}_{os.path.basename(args.config)}_{datetime.now().strftime('%y%m%d')}",
            config=cfg,
        )
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
    solver = TASKS[cfg.task](cfg, 'test')

    solver.val()
