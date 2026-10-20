import importlib

import numpy as np
import pytest

from contact_belief_wam.protocol import Client, pack, unpack


def test_pack_unpack_round_trips_nested_noncontiguous_numpy_arrays():
    image = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    state = np.arange(24, dtype=">f4").reshape(2, 3, 4)[:, ::2, :]
    assert not state.flags.c_contiguous
    message = {
        "image": image,
        "observation": {"state": state, "step": np.int64(12)},
        "valid": True,
    }

    restored = unpack(pack(message))

    np.testing.assert_array_equal(restored["image"], image)
    np.testing.assert_array_equal(restored["observation"]["state"], state)
    assert restored["image"].dtype == image.dtype
    assert restored["observation"]["state"].dtype == state.dtype
    assert restored["observation"]["state"].flags.c_contiguous
    assert restored["observation"]["step"] == 12
    assert type(restored["observation"]["step"]) is int
    assert restored["valid"] is True


@pytest.mark.parametrize("modern_signature", [False, True], ids=["legacy-websockets", "modern-websockets"])
def test_client_only_passes_supported_websocket_options(monkeypatch, modern_signature):
    captured = {}

    if modern_signature:
        def connect(uri, *, max_size=None, compression=None, open_timeout=30, ping_interval=20, proxy=None):
            captured.update(max_size=max_size, compression=compression, open_timeout=open_timeout,
                            ping_interval=ping_interval, proxy=proxy)
            return _FakeWebSocket()
    else:
        def connect(uri, *, max_size=None, compression=None, open_timeout=30):
            captured.update(max_size=max_size, compression=compression, open_timeout=open_timeout)
            return _FakeWebSocket()

    client_module = importlib.import_module("websockets.sync.client")
    monkeypatch.setattr(client_module, "connect", connect)
    client = Client()
    client.close()

    assert captured["max_size"] is None
    assert captured["compression"] is None
    assert captured["open_timeout"] == 30
    for name in ("ping_interval", "proxy"):
        if modern_signature:
            assert captured[name] is None
        else:
            assert name not in captured


class _FakeWebSocket:
    def recv(self, timeout=None):
        return pack({"fake": True})

    def close(self):
        pass
