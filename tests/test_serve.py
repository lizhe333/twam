import importlib
import json
import multiprocessing as mp
import os
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

serve_module = importlib.import_module("contact_belief_wam.serve")
cli_module = importlib.import_module("contact_belief_wam.cli")


def _fake_encoder_worker(connection, source, checkpoint, device, threads):
    try:
        connection.send({"metadata": {"fake": True}})
        while connection.recv() is not None:
            pass
        Path(source).write_text("received stop")
    finally:
        connection.close()


@pytest.mark.skipif(os.name != "posix", reason="serve SIGTERM cleanup is POSIX-specific")
def test_sigterm_closes_and_reaps_private_encoder_child(monkeypatch, tmp_path):
    monkeypatch.setattr(serve_module, "_encoder_worker", _fake_encoder_worker)
    instances = []

    class FakePolicyService:
        def __init__(self, checkpoint, **kwargs):
            self.encoder = serve_module.EncoderProcess(tmp_path / "stop-received", threads=1)
            self.child_pid = self.encoder.process.pid
            self.metadata = {"fake": True}
            instances.append(self)

        def close(self):
            self.encoder.close()

    class FakeSocketServer:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def serve_forever(self):
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(serve_module, "PolicyService", FakePolicyService)
    websocket_server = importlib.import_module("websockets.sync.server")
    monkeypatch.setattr(websocket_server, "serve", lambda *args, **kwargs: FakeSocketServer())

    with pytest.raises(SystemExit) as exit_info:
        serve_module.serve("unused-checkpoint")

    assert exit_info.value.code == 128 + signal.SIGTERM
    assert (tmp_path / "stop-received").read_text() == "received stop"
    assert len(instances) == 1
    child_pid = instances[0].child_pid
    assert all(child.pid != child_pid for child in mp.active_children())


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="process groups are POSIX-specific")
def test_eval_timeout_kills_only_its_isolated_server_group(monkeypatch, tmp_path):
    events = []
    popen_kwargs = {}

    class FakeClient:
        def __init__(self, port, timeout):
            pass

        def close(self):
            pass

    class FakeServerProcess:
        pid = 424242

        def poll(self):
            return None

        def terminate(self):
            events.append("terminate")

        def wait(self, timeout=None):
            events.append(("wait", timeout))
            if timeout is not None:
                raise cli_module.subprocess.TimeoutExpired("policy-server", timeout)
            return -signal.SIGKILL

    def fake_popen(command, **kwargs):
        popen_kwargs.update(kwargs)
        return FakeServerProcess()

    monkeypatch.setattr(cli_module.subprocess, "Popen", fake_popen)
    def fake_run(*args, **kwargs):
        (tmp_path / "evaluation.json").write_text(json.dumps({"episodes": [
            {"seed": 1, "complete_rollout": True},
        ]}))

    monkeypatch.setattr(cli_module.subprocess, "run", fake_run)
    monkeypatch.setattr(importlib.import_module("contact_belief_wam.protocol"), "Client", FakeClient)
    monkeypatch.setattr(cli_module.os, "killpg", lambda pgid, sig: events.append(("killpg", pgid, sig)))
    args = SimpleNamespace(output=tmp_path, checkpoint=tmp_path / "checkpoint", task="insert_hole",
                            seeds="1", device="cpu", port=18780, k=None, max_policy_steps=None)

    cli_module.evaluate_with_server(args)

    assert popen_kwargs["start_new_session"] is True
    assert events == ["terminate", ("wait", 30), ("killpg", 424242, signal.SIGKILL), ("wait", None)]


def test_eval_rejects_zero_exit_without_fresh_complete_episode(monkeypatch, tmp_path):
    events = []

    class FakeClient:
        def __init__(self, port, timeout):
            pass

        def close(self):
            pass

    class FakeServerProcess:
        pid = 424243

        def poll(self):
            return None

        def terminate(self):
            events.append("terminate")

        def wait(self, timeout=None):
            events.append(("wait", timeout))
            return 0

    def fake_popen(command, **kwargs):
        return FakeServerProcess()

    output = tmp_path / "eval-output"
    output.mkdir()
    (output / "evaluation.json").write_text(json.dumps({"episodes": [
        {"seed": 1, "complete_rollout": True},
    ]}))
    monkeypatch.setattr(cli_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(cli_module.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(importlib.import_module("contact_belief_wam.protocol"), "Client", FakeClient)
    monkeypatch.setattr(cli_module.os, "killpg", lambda *args: None)
    args = SimpleNamespace(output=output, checkpoint=tmp_path / "checkpoint", task="insert_hole",
                           seeds="1", device="cpu", port=18780, k=None, max_policy_steps=None)

    with pytest.raises(RuntimeError, match="without a readable evaluation.json"):
        cli_module.evaluate_with_server(args)

    assert not (output / "evaluation.json").exists()
    assert events == ["terminate", ("wait", 30)]
