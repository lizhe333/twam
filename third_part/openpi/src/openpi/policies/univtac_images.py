"""Image decoding and preprocessing shared by dataset and live inputs."""
from __future__ import annotations

from typing import Any

import numpy as np


def _as_numpy(image: Any) -> np.ndarray:
    """Move a numpy or torch image to CPU without changing its channel order."""
    if hasattr(image, "detach"):
        image = image.detach()
    if hasattr(image, "cpu"):
        image = image.cpu()
    if hasattr(image, "numpy"):
        image = image.numpy()
    return np.asarray(image)


def to_rgb_uint8(image: Any) -> np.ndarray:
    """Return one RGB image as HWC uint8.

    Accepted inputs are HWC uint8 images and CHW floating point LeRobot images
    in [0, 1]. No BGR channel swap is applied:
    UniVTAC encoded its RGB numerical arrays with OpenCV and its stored JPEGs
    decode to those same channel values through ``IMREAD_COLOR``.
    """
    array = _as_numpy(image)
    if array.ndim != 3:
        raise ValueError(f"Expected one HWC or CHW image, got shape {array.shape}")

    if array.dtype == np.uint8 and array.shape[-1] == 3:
        result = array
    elif array.shape[0] == 3 and np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError("Image contains NaN or infinity")
        low = float(array.min())
        high = float(array.max())
        if low < 0.0 or high > 1.0:
            raise ValueError(f"Float image must be in [0, 1], got [{low}, {high}]")
        # Match OpenPI's client-side convert_to_uint8 convention.
        result = (array * 255.0).astype(np.uint8).transpose(1, 2, 0)
    else:
        raise ValueError(
            f"Expected HWC uint8 or CHW floating image with 3 channels, got {array.shape} {array.dtype}"
        )

    if result.shape[0] < 1 or result.shape[1] < 1:
        raise ValueError(f"Image dimensions must be nonempty, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("Image contains NaN or infinity")
    return np.ascontiguousarray(result)


def decode_jpeg_rgb(encoded: bytes | bytearray | memoryview | np.bytes_) -> np.ndarray:
    """Decode a UniVTAC JPEG byte string as HWC uint8, preserving stored order."""
    if not isinstance(encoded, (bytes, bytearray, memoryview, np.bytes_)):
        raise TypeError(f"Expected encoded JPEG bytes, got {type(encoded).__name__}")
    payload = bytes(encoded)
    if not payload:
        raise ValueError("Cannot decode an empty JPEG payload")

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency supplied by OpenPI env
        raise RuntimeError("OpenCV is required to decode UniVTAC JPEG frames") from exc

    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("OpenCV could not decode a UniVTAC JPEG frame")
    # Do not call cvtColor here. UniVTAC fed RGB numerical arrays to imencode;
    # IMREAD_COLOR returns those values in the original numerical order.
    return to_rgb_uint8(decoded)


def resize_rgb(image: Any, height: int = 224, width: int = 224) -> np.ndarray:
    """Apply OpenPI client's official aspect-preserving resize with black padding.

    The result is HWC uint8. Conversion before resizing makes this function
    usable for both the source HWC images and LeRobot CHW float images.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"Target image dimensions must be positive, got {(height, width)}")
    rgb = to_rgb_uint8(image)
    try:
        from openpi_client.image_tools import resize_with_pad
    except ImportError as exc:  # pragma: no cover - dependency supplied by OpenPI env
        raise RuntimeError("openpi-client is required for the official resize_with_pad transform") from exc
    resized = np.asarray(resize_with_pad(rgb, height, width))
    if resized.shape != (height, width, 3) or resized.dtype != np.uint8:
        raise ValueError(f"OpenPI resize returned {resized.shape} {resized.dtype}")
    return np.ascontiguousarray(resized)
