import torch
import torch.nn
from .rtdetr_postprocessor import  RTDETRPostProcessor
import copy

def get_postprocessor(postprocessor_config):
    postprocessor_config = copy.deepcopy(postprocessor_config)
    return RTDETRPostProcessor(**postprocessor_config)