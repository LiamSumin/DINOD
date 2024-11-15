import torch
from .coco.coco_dataset import CocoDetection
from .dataloader import DataLoader, default_collate_fn
from .transforms import Compose
from lib.utils.misc import is_dist_available_and_initialized
from torch.utils.data import DistributedSampler


def create_dataloader(config, mode="train"):
    """
    Creates a DataLoader for training or validation.

    Args:
        config (dict): Configuration dictionary containing dataset, transforms, and dataloader settings.
        mode (str): Mode for the dataloader. Either "train" or "val".

    Returns:
        DataLoader: The constructed DataLoader.
    """
    # Remove unnecessary keys
    del config['type']
    dataset_config = config.dataset
    dataset_config_type = dataset_config.pop('type')
    transforms_config = dataset_config.pop('transforms')
    transforms_config.pop('type')

    # Create transforms and dataset
    transforms = Compose(**transforms_config)
    if dataset_config_type == "CocoDetection":
        dataset = CocoDetection(transforms=transforms, **dataset_config)

    # Handle shuffling and distributed sampling
    sampler = None
    if is_dist_available_and_initialized():
        sampler = DistributedSampler(dataset=dataset, shuffle=config.shuffle)
    config.pop('shuffle', None)  # Remove shuffle to avoid conflict with sampler

    # Create DataLoader
    del config['dataset']
    del config['collate_fn']
    loader = DataLoader(dataset=dataset, sampler=sampler, collate_fn= default_collate_fn, **config)
    return loader


def get_train_dataloader(config):
    """
    Wrapper for creating the training DataLoader.

    Args:
        config (dict): Configuration for the training dataloader.

    Returns:
        DataLoader: The constructed training DataLoader.
    """
    return create_dataloader(config, mode="train")


def get_val_dataloader(config):
    """
    Wrapper for creating the validation DataLoader.

    Args:
        config (dict): Configuration for the validation dataloader.

    Returns:
        DataLoader: The constructed validation DataLoader.
    """
    return create_dataloader(config, mode="val")
