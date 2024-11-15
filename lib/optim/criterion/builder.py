import torch.nn as nn
from .matcher import HungarianMatcher
from .rtdetr_criterion import SetCriterion
def get_criterion(criterion_config):
    MatcherConfig = criterion_config.pop('MATCHER')
    MatcherType = MatcherConfig.pop('type')

    if MatcherType == "HungarianMatcher":
        matcher = HungarianMatcher(**MatcherConfig)
    else :
        raise NotImplementedError(f"The Matcher type {MatcherType} is not implemented.")

    criterion = SetCriterion(matcher=matcher,
                             **criterion_config)

    return criterion