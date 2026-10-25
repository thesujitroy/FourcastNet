"""Config loading for the mixer comparison, without ruamel.

`Params` is a drop-in for utils/YParams.py::YParams (attribute and item access,
`in`, `update_params`, `log`, the `params` dict) that reads the named config with
PyYAML. ruamel.yaml is not installed in the NAS environment. PyYAML follows YAML
1.1 and reads "5E-4" or "1e-3" as strings; `_clean` turns those back into floats
(and 'None' into None, as YParams does).
"""
import ctypes
import logging
import os
import re

import yaml

_SCIENTIFIC = re.compile(r'^[-+]?(\d+\.?\d*|\.\d+)[eE][-+]?\d+$')


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if value == 'None':
        return None
    if isinstance(value, str) and _SCIENTIFIC.match(value):
        return float(value)
    return value


class Params:
    def __init__(self, yaml_filename, config_name, overrides=None):
        self._yaml_filename = str(yaml_filename)
        self._config_name = config_name
        self.params = {}
        with open(yaml_filename) as stream:
            configs = yaml.safe_load(stream)
        if config_name not in configs:
            raise KeyError(f"no config {config_name!r} in {yaml_filename}; have {sorted(configs)}")
        for key, val in {**configs[config_name], **(overrides or {})}.items():
            self[key] = _clean(val)

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, val):
        self.params[key] = val
        self.__setattr__(key, val)

    def __contains__(self, key):
        return key in self.params

    def get(self, key, default=None):
        return self.params.get(key, default)

    def update_params(self, config):
        for key, val in config.items():
            self[key] = val

    def log(self):
        logging.info("------------------ Configuration ------------------")
        logging.info("Configuration file: " + self._yaml_filename)
        logging.info("Configuration name: " + str(self._config_name))
        for key, val in self.params.items():
            logging.info(str(key) + ' ' + str(val))
        logging.info("---------------------------------------------------")


def parse_overrides(items):
    """['key=value', ...] -> {key: yaml-parsed value}; 'None' maps to None."""
    out = {}
    for item in items or []:
        key, sep, value = item.partition('=')
        if not sep or not key:
            raise ValueError(f"override {item!r} is not key=value")
        out[key] = _clean(yaml.safe_load(value))
    return out


def disable_transparent_hugepages():
    """Opt this process, and the DataLoader workers it forks, out of transparent
    hugepages. numpy madvise()s large arrays MADV_HUGEPAGE; on NAS mil_a100 nodes
    (THP defrag=madvise) page faults then spin in direct compaction once the page
    cache fills memory, freezing data workers for minutes to hours (see HelioFM
    utils/distributed.py). Linux only; a no-op elsewhere."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError:
        return
    PR_SET_THP_DISABLE = 41
    if libc.prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0) != 0:
        logging.warning("prctl(PR_SET_THP_DISABLE) failed: %s", os.strerror(ctypes.get_errno()))
