"""Regression tests for the two real training/inference input mismatches."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "inference"))
sys.path.insert(0, str(PROJECT / "scripts"))
from aligned_feedback import PROFILE, TrainingAlignedAdapter, training_jpeg
import reeval_hdmi_aligned as queue
from run_native import run_pinned_script


def writer():
    spec = importlib.util.spec_from_file_location("univtac_hdf5_writer", PROJECT.parent / "UniVTAC/envs/utils/data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HDF5Handler


def test_tactile_matches_actual_dataset_writer_and_decoder():
    source = np.random.default_rng(5).integers(0, 256, (2, 24, 32, 3), dtype=np.uint8)
    before = source.copy()
    hdf5 = writer()
    encoded, _ = hdf5.img_to_stream(source)
    expected = hdf5.stream_to_img(np.array(encoded, dtype=object))
    actual = np.stack([training_jpeg(x) for x in source])
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, before)
    assert not np.array_equal(source, actual)  # Raw RGB would fail this regression.


class Client:
    metadata = dict(k=10, camera_mode="head_only")
    def __init__(self):
        self.calls = []
    def call(self, **message):
        self.calls.append(message)
        if message["op"] == "infer":
            return dict(actions=np.full((50, 8), .25, np.float32), diagnostics={},
                        arm_limit_overshoots=np.zeros(50, np.float32))
        return {}


class Task:
    def __init__(self, terminal=False):
        self.cfg = SimpleNamespace(seed=1000000, step_lim=600)
        self.step_count = self.take_action_cnt = 0
        self.plan_success = True
        self.terminal = terminal
        self._robot_manager = SimpleNamespace(device="cpu")
    def _get_observations(self):
        t = self.step_count // 2
        image = np.random.default_rng(t).integers(0, 256, (24, 32, 3), dtype=np.uint8)
        return dict(tactile={f"{side}_tactile": {"rgb_marker": image.copy()}
                             for side in ("left", "right")},
                    observation={"head": {"rgb": image}},
                    embodiment={"joint": np.arange(8, dtype=np.float32) / 100 + self.step_count / 1000})
    def take_action(self, value, **kwargs):
        self.step_count += 1
        self.take_action_cnt += 1
        return True, self.terminal


def adapter(client):
    return TrainingAlignedAdapter(client, "test", "insert HDMI", np.array([[-1, 1]] * 8, np.float32))


def test_actual_q_history_has_training_endpoints_and_tactile_offsets():
    client, task = Client(), Task()
    policy = adapter(client)
    actual_frames, actual_q = [], []
    observation = task._get_observations()
    for i in range(21):
        actual_frames.append(training_jpeg(observation["tactile"]["left_tactile"]["rgb_marker"]))
        actual_q.append(observation["embodiment"]["joint"].copy())
        command = policy.next_action(task, observation)
        _, observation = policy.execute(task, command)
    messages = [m for m in client.calls if m["op"] == "infer"]
    assert [m["step"] for m in messages] == [0, 10, 20]
    assert messages[0]["executed_actions"].shape == (0, 8)
    for message, endpoint in zip(messages[1:], [10, 20]):
        np.testing.assert_array_equal(message["executed_actions"], actual_q[endpoint-9:endpoint+1])
        np.testing.assert_array_equal(message["state"], actual_q[endpoint])
        assert not np.all(message["executed_actions"] == .25)
        for sensor in range(2):
            np.testing.assert_array_equal(message["tactile_window"][sensor],
                [actual_frames[i] for i in (endpoint-6, endpoint-4, endpoint-2, endpoint)])
    np.testing.assert_array_equal(messages[0]["tactile_window"][0], [actual_frames[0]] * 4)
    assert task.step_count == 42
    np.testing.assert_array_equal(policy.actions[0]["action"], np.full(8, .25))
    np.testing.assert_array_equal(policy.actions[0]["measured_joint"], actual_q[1])
    policy.close()
    fresh = adapter(client)
    assert fresh.step == 0 and fresh.executed == [] and len(fresh.touch) == 0


def test_terminal_first_physical_step_uses_that_observation():
    policy, task = adapter(Client()), Task(terminal=True)
    command = policy.next_action(task, task._get_observations())
    success, observation = policy.execute(task, command)
    assert success and task.step_count == 1
    np.testing.assert_array_equal(policy.executed[-1], observation["embodiment"]["joint"])


def test_reject_nonfinite_measured_history():
    policy, task = adapter(Client()), Task()
    original = task._get_observations
    def invalid():
        value = original()
        value["embodiment"]["joint"][3] = np.nan
        return value
    task._get_observations = invalid
    with pytest.raises(ValueError, match="finite measured"):
        policy.execute(task, np.zeros(8, np.float32))


@pytest.fixture
def batch():
    return queue.watch.load_runner(PROJECT.parent).batch


def test_runtime_copy_does_not_follow_later_training_edits(tmp_path, batch):
    project = tmp_path / "twam"
    originals = ["twam/src/contact_belief_wam/model.py", "twam/inference/run_native.py",
                 "twam/vendor/wave0/src/model.py", "twam/pyproject.toml", "twam/scripts/cloud_train.sh",
                 "UniVTAC/policy/TactileWave0Wam/deploy_policy.py", "UniVTAC/scripts/eval_policy.py"]
    for name in originals:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# original\n")
    (project / "vendor/transformers").mkdir()
    suite = tmp_path / "suite"
    runtime, record = queue.pin_runtime(tmp_path, suite, batch)
    source = project / "src/contact_belief_wam/model.py"
    source.write_text("# concurrent training edit\n")
    assert (runtime / "src/contact_belief_wam/model.py").read_text() == "# original\n"
    assert queue.verify_runtime(runtime) == record
    (runtime / "inference/run_native.py").write_text("# accidental edit\n")
    with pytest.raises(ValueError, match="Pinned inference source changed"):
        queue.verify_runtime(runtime)


def test_pinned_entry_keeps_native_asset_root_without_executing_live_source(tmp_path):
    original = tmp_path / "UniVTAC/scripts/eval_policy.py"
    original.parent.mkdir(parents=True)
    original.write_text("raise AssertionError('must execute the pinned source')\n")
    config = original.parent.parent / "task_config/clean.yml"
    config.parent.mkdir()
    config.write_text("same native task config")
    snapshot = tmp_path / "runtime/native_eval.py"
    snapshot.parent.mkdir()
    output = tmp_path / "entry.json"
    snapshot.write_text("from pathlib import Path\nimport json, sys\n"
        "assert sys.modules['__main__'].__file__ == __file__\n"
        f"Path({str(output)!r}).write_text(json.dumps(dict(config="
        "(Path(__file__).parent.parent / 'task_config/clean.yml').read_text(), "
        "file=__file__, argument=sys.argv[0])))\n")
    previous, argument = sys.modules["__main__"], sys.argv[0]
    run_pinned_script(snapshot, original)
    result = json.loads(output.read_text())
    assert result == dict(config="same native task config", file=str(original), argument=str(original))
    assert sys.modules["__main__"] is previous and sys.argv[0] == argument


def test_gpu_guard_restarts_idle_period_when_another_job_appears(monkeypatch):
    clock = [0.]
    monkeypatch.setattr(queue.time, "monotonic", lambda: clock[0])
    busy = [False]
    locks = []
    def claim(*unused):
        if busy[0]:
            return None, None, "existing compute process"
        lock = SimpleNamespace(closed=False)
        lock.close = lambda: setattr(lock, "closed", True)
        locks.append(lock)
        return lock, dict(name="RTX 4090", uuid="GPU-test"), None
    guarded = queue.idle_gpu_guard(claim, 60)
    assert guarded(None, None)[0] is None and locks[-1].closed
    clock[0] = 59
    assert guarded(None, None)[0] is None
    busy[0], clock[0] = True, 60
    assert "compute" in guarded(None, None)[2]
    busy[0], clock[0] = False, 61
    assert guarded(None, None)[0] is None
    clock[0] = 121
    lock, _, reason = guarded(None, None)
    assert lock is locks[-1] and not lock.closed and reason is None


def test_launch_uses_pinned_policy_and_same_protocol(tmp_path, monkeypatch, batch):
    captured = {}
    monkeypatch.setattr(queue, "verify_runtime", lambda path: {"fingerprint": "source-hash"})
    def popen(command, **kwargs):
        captured.update(command=command, **kwargs)
        return SimpleNamespace(pid=123)
    monkeypatch.setattr(queue.subprocess, "Popen", popen)
    job = dict(id="HDMI", task="insert_HDMI", attempts=0, checkpoint="/pinned/10000")
    queue.launch(job, PROJECT.parent, tmp_path, "0", 50, 12345, Path("/model/python"),
                 runner=SimpleNamespace(batch=batch))
    deploy = batch.read(tmp_path / "work/HDMI/attempt01/deploy.yml")
    assert deploy["policy_name"] == "TactileWave0WamAligned"
    assert deploy["pi05_eval_protocol"] == batch.EVAL_PROTOCOL
    assert captured["env"]["CBWAM_ROOT"] == str(tmp_path / "runtime")
    assert captured["env"]["TACTILE_WAVE0_CHECKPOINT"] == "/pinned/10000"
    assert str(tmp_path / "runtime/inference/run_native.py") in captured["command"]
    assert captured["start_new_session"] and "PI05_TRACE_DIR" not in captured["env"]
    assert captured["command"][-4:] == ["--start_seed", "1000000", "--run_name", "HDMI"]


def test_compare_keeps_denominator_and_reports_seed_changes(tmp_path, batch):
    batch.write(tmp_path / "comparison_reference.json", dict(
        result=dict(successes=1), result_file="/old/result.json", ema_sha256="same-ema",
        outcomes={"1000000": "success", "1000001": "failed", "1000002": "failed"}))
    result_dir = tmp_path / "results/HDMI"
    batch.write(result_dir / "metadata.json", {
        "1000000": dict(result="error"),
        "1000001": dict(result="success", policy_execution=dict(input_alignment=PROFILE)),
        "1000002": dict(result="success", policy_execution=dict(input_alignment=PROFILE)),
    })
    job = dict(state="completed", runtime_fingerprint="same-source", result=dict(
        native_result_dir=str(result_dir), successes=2, completed_episodes=3))
    queue.compare(tmp_path, dict(jobs=[job]), batch)
    report = batch.read(tmp_path / "comparison.json")
    assert report["aligned_success_rate"] == 2 / 3
    assert report["gained_success_seeds"] == [1000001, 1000002]
    assert report["lost_success_seeds"] == [1000000]
    metadata = batch.read(result_dir / "metadata.json")
    metadata["1000001"]["policy_execution"]["input_alignment"] = "raw-unfixed"
    batch.write(result_dir / "metadata.json", metadata)
    with pytest.raises(ValueError, match="did not use aligned inference"):
        queue.compare(tmp_path, dict(jobs=[job]), batch)
