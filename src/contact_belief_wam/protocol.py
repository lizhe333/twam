"""Small NumPy wire format shared by the two independent Python environments."""
import msgpack
import numpy as np


def _encode(value):
    if isinstance(value, np.ndarray):
        value = np.ascontiguousarray(value)
        return {"__array__": True, "dtype": value.dtype.str, "shape": value.shape, "data": value.tobytes()}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _decode(value):
    if value.get("__array__"):
        dtype = np.dtype(value["dtype"])
        if dtype.hasobject:
            raise ValueError("Object arrays are not supported")
        return np.frombuffer(value["data"], dtype=dtype).reshape(value["shape"]).copy()
    return value


def pack(value):
    return msgpack.packb(value, default=_encode, use_bin_type=True)


def unpack(value):
    return msgpack.unpackb(value, object_hook=_decode, raw=False)


class Client:
    def __init__(self, host="127.0.0.1", port=18780, timeout=1800):
        import inspect
        from websockets.sync.client import connect
        # Isaac can prepend its older bundled websockets after AppLauncher,
        # even when the simulator environment has v15 installed. Older sync
        # clients neither expose these options nor enable automatic proxies.
        supported = inspect.signature(connect).parameters
        options = {key: None for key in ("ping_interval", "proxy") if key in supported}
        self.ws = connect(f"ws://{host}:{port}", max_size=None, compression=None,
                          open_timeout=30, **options)
        self.timeout = timeout
        self.metadata = unpack(self.ws.recv(timeout=timeout))

    def call(self, **message):
        self.ws.send(pack(message))
        result = unpack(self.ws.recv(timeout=self.timeout))
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def close(self):
        self.ws.close()
