"""Native evaluator policy using an isolated TWAM source snapshot."""
import os
from pathlib import Path

# Import from the pinned source BEFORE the old native bridge inserts its usual
# training-checkout path into sys.path. Python retains this package's __path__.
from contact_belief_wam import sim_eval
from aligned_feedback import PROFILE, TrainingAlignedAdapter
from policy.TactileWave0Wam import deploy_policy as native


RUNTIME = Path(os.environ["CBWAM_ROOT"]).resolve()
if not Path(sim_eval.__file__).resolve().is_relative_to(RUNTIME / "src"):
    raise RuntimeError("Aligned evaluation must import its pinned inference source")
# Only this process's copied bridge module changes; the training tree is intact.
native.PROJECT = RUNTIME


class Policy(native.Policy):
    def eval(self, task, observation):
        if self.adapter is None:
            self.adapter = TrainingAlignedAdapter(
                self.client, f"{self.task_name}:{task.cfg.seed}",
                native.prompt(self.task_name), native.robot_action_bounds(task))
        return super().eval(task, observation)

    def record_episode(self, task, outcome):
        super().record_episode(task, outcome)
        task.metadata["policy_execution"].update(
            input_alignment=PROFILE,
            tactile_input="cv2.imencode(.jpg, rgb_marker) -> imdecode(IMREAD_COLOR)",
            belief_history="measured q after execution, anchored at previous replan",
        )
