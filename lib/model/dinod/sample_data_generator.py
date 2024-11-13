import torch
from typing import Tuple

from lib.model import SampleInputDataGeneratorInterface


class DINOD_DummyDataGenerator(SampleInputDataGeneratorInterface):
    def __init__(self, image_size: Tuple[int, int], feat_size: Tuple[int, int]):
        self._image_size = image_size
        self._feat_size = feat_size

    def get(self, batch_size: int, device: torch.device):
        return {'x' : torch.full((batch_size, 3, self._image_size[1], self._image_size[0]), 0.5, device=device)}

def build_sample_input_data_generator(config: dict):
    common_config = config['COMMON']
    return DINOD_DummyDataGenerator(common_config['IMAGE_SIZE'], common_config['FEAT_SIZE'])