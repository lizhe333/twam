"""Load the pinned bridge and aligned policy after Isaac Sim initializes."""
import importlib.abc
import importlib.util
import os
from pathlib import Path
import sys
import types


HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parent


class PinnedPolicyFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        files = {
            "policy.TactileWave0WamAligned": HERE / "native_policy.py",
            "policy.TactileWave0Wam.deploy_policy": RUNTIME / "native_policy_original.py",
        }
        if fullname in files:
            return importlib.util.spec_from_file_location(fullname, files[fullname])


def run_pinned_script(snapshot, original):
    # The native evaluator locates task_config/, data/ and instructions/ relative
    # to __file__. Execute frozen bytes while preserving its original asset root.
    module = types.ModuleType("__main__")
    module.__file__ = str(original)
    previous, argument = sys.modules.get("__main__"), sys.argv[0]
    sys.modules["__main__"], sys.argv[0] = module, str(original)
    try:
        exec(compile(snapshot.read_bytes(), str(snapshot), "exec"), module.__dict__)
    finally:
        sys.modules["__main__"], sys.argv[0] = previous, argument


def main():
    os.environ["CBWAM_ROOT"] = str(RUNTIME)
    sys.path[:0] = [str(RUNTIME / "src"), str(HERE),
                   str(Path(os.environ["TWAM_ROOT"]) / "UniVTAC/scripts")]
    sys.meta_path.insert(0, PinnedPolicyFinder())
    run_pinned_script(RUNTIME / "native_eval.py",
                      Path(os.environ["TWAM_ROOT"]) / "UniVTAC/scripts/eval_policy.py")


if __name__ == "__main__":
    main()
