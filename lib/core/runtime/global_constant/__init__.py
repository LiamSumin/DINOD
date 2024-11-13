import shutil
import yaml
from yaml import CSafeLoader as Loader
import os
from typing import Sequence, Optional
import importlib

_global_constants: Optional[dict] = None
__root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))


class _CustomLoader(Loader):
    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]

        super(_CustomLoader, self).__init__(stream)

    def include(self, node):
        if isinstance(node, yaml.ScalarNode):
            filename = os.path.join(self._root, self.construct_scalar(node))
        elif isinstance(node, yaml.SequenceNode):
            filename = os.path.join(self._root, *self.construct_sequence(node))
        else:
            raise TypeError('!include tag must be a scalar or sequence')

        with open(filename, 'r') as f:
            return yaml.load(f, _CustomLoader)


_CustomLoader.add_constructor('!include', _CustomLoader.include)


def _initialize_global_constants():
    import pdb; pdb.set_trace()
    global _global_constants
    config_module = importlib.import_module("lib.config.dinod.config")
    cfg = config_module.cfg
    config_module.update_config_from_file("configs/dinov2_small.yaml")
    print("New configuration is shown below.")
    for key in cfg.keys():
        print("%s configuration:" % key, cfg[key])


def _get_value(constants: dict, paths: Sequence[str]):
    for path in paths:
        if path not in constants:
            raise KeyError(f'Key {".".join(paths)} not found in env')

        constants = constants[path]
    return constants


def get_global_constant(*paths):
    print(paths)
    if _global_constants is None:
        _initialize_global_constants()

    return _get_value(_global_constants, paths)