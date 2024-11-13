import yaml
from yaml import CSafeDumper


def pretty_format(object_):
    return yaml.dump(object_, Dumper=yaml.SafeDumper, indent=2, default_flow_style=False, allow_unicode=True, sort_keys=False).strip()