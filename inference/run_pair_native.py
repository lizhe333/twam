"""Run frozen native evaluator/bridges; Isaac Sim initializes in the native entry."""
import importlib.abc
import importlib.util
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parent


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        files = {
            'policy.A800Compare': HERE / 'pair_policy.py',
            'policy.TactileWave0Wam.deploy_policy': RUNTIME / 'native_tactile.py',
            'policy.Wave0Wam.deploy_policy': RUNTIME / 'native_base.py',
            'policy.Pi05Wam.deploy_policy': RUNTIME / 'native_pi05.py',
        }
        if fullname in files:
            return importlib.util.spec_from_file_location(fullname, files[fullname])


def main():
    os.environ['CBWAM_ROOT'] = os.environ['PAIR_RUNTIME'] = str(RUNTIME)
    sys.path[:0] = [str(RUNTIME / 'src'), str(HERE),
                   str(Path(os.environ['TWAM_ROOT']) / 'UniVTAC/scripts')]
    sys.meta_path.insert(0, Finder())
    from run_native import run_pinned_script
    run_pinned_script(RUNTIME / 'native_eval.py',
                      Path(os.environ['TWAM_ROOT']) / 'UniVTAC/scripts/eval_policy.py')


if __name__ == '__main__':
    main()
