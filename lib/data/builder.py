import torch
from .coco.coco_dataset import CocoDetection
from .dataloader import DataLoader

def get_train_dataloader(config):
    del config['type']
    dataset_config_type = config.dataset.type
    dataset_config = config.dataset
    del dataset_config['type']
    if dataset_config_type == "CocoDetection":
        dataset= CocoDetection(**dataset_config)
    return DataLoader(dataset, **config)

def get_val_dataloader(config) :
    del config['type']
    dataset_config_type=config.dataset.type
    dataset_config=config.dataset
    del dataset_config['type']
    if dataset_config_Type=="CocoDetection":
        dataset = CocoDetection(**dataset_config)
    return DataLoader(dataset, **config)


