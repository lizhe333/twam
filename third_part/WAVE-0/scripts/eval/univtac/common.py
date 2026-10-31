"""Small standard-library-only entry points around the unchanged experiment code."""
import importlib.util
import json
import os
from pathlib import Path
import re

PACKAGE = Path(__file__).resolve().parent
TASKS = ('grasp_classify', 'insert_HDMI', 'insert_hole', 'insert_tube',
         'lift_bottle', 'lift_can', 'pull_out_key', 'put_bottle_in_shelf')
SEEDS = list(range(1000000, 1000050))


def workspace():
    default = next(p for p in PACKAGE.parents if (p / 'UniVTAC').is_dir())
    return Path(os.environ.get('TWAM_ROOT', default)).resolve()


def experiment(view):
    return json.loads((PACKAGE / 'experiments.json').read_text())[view]


def name(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,80}', value):
        raise ValueError('Use a descriptive run name containing letters, numbers, hyphens or underscores')
    return value


def load(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
