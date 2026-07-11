"""Reference / input image → temporal spike tensor for 3DCG generators.

Rate-codes luminance (and optional edge energy) into binary spikes over
``time_bins``. Pure NumPy; no network, no OpenCV required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union
import numpy as np

ArrayLike = Union[np.ndarray, str, Path]


def load_image_array(path_or_array: ArrayLike) -> np.ndarray:
    """Load an image as float HxWxC in [0, 1].

    Accepts a numpy array, or a path to PNG/JPEG via Pillow when available,
    otherwise a minimal raw path fallback for synthetic tests.
    """
    if isinstance(path_or_array, np.ndarray):
        arr = np.asarray(path_or_array, dtype=np.float64)
    else:
        path = Path(path_or_array)
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            from PIL import Image

            img = Image.open(path).convert("RGB")
            arr = np.asarray(img, dtype=np.float64) / 255.0
        except ImportError:
            # Fallback: interpret as tiny PNG via pure numpy is hard; require PIL
            # or pass arrays in tests. For unit tests we always pass arrays.
            raise ImportError(
                "Install Pillow to load image files, or pass a numpy array."
            ) from None
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"expected HxW or HxWxC image, got shape {arr.shape}")
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr.astype(np.float64), 0.0, 1.0)


def _luminance(image: np.ndarray) -> np.ndarray:
    if image.shape[2] == 1:
        return image[:, :, 0]
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _edge_energy(luma: np.ndarray) -> np.ndarray:
    """Simple 3x3 Sobel-like edge magnitude, zero-padded."""
    # kernels
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    pad = np.pad(luma, 1, mode="edge")
    h, w = luma.shape
    gx = np.zeros_like(luma)
    gy = np.zeros_like(luma)
    for i in range(3):
        for j in range(3):
            gx += kx[i, j] * pad[i : i + h, j : j + w]
            gy += ky[i, j] * pad[i : i + h, j : j + w]
    mag = np.sqrt(gx * gx + gy * gy)
    m = float(mag.max()) if mag.size else 1.0
    if m > 0:
        mag = mag / m
    return mag


def spike_feature_size(height: int, width: int, *, include_edges: bool = True) -> int:
    base = int(height) * int(width)
    return base * 2 if include_edges else base


def image_to_spikes(
    image: ArrayLike,
    *,
    time_bins: int = 8,
    threshold: float = 0.5,
    seed: int = 0,
    include_edges: bool = True,
    max_side: int = 32,
) -> np.ndarray:
    """Convert image to binary spikes ``[time_bins, features]``.

    Features = flattened luminance rates, optionally concatenated with edge
    energy rates. Each bin draws Bernoulli spikes with p = rate (deterministic
    via seed + thresholded cumulative noise for reproducibility without
    floating drift issues).
    """
    if time_bins < 1:
        raise ValueError("time_bins must be >= 1")
    img = load_image_array(image)
    h, w = img.shape[:2]
    # Downscale if needed (nearest) to bound feature size.
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        ys = (np.linspace(0, h - 1, nh)).astype(int)
        xs = (np.linspace(0, w - 1, nw)).astype(int)
        img = img[ys][:, xs]
        h, w = img.shape[:2]

    luma = _luminance(img)
    rates = [luma.reshape(-1)]
    if include_edges:
        rates.append(_edge_energy(luma).reshape(-1))
    rate_vec = np.concatenate(rates, axis=0)
    rate_vec = np.clip(rate_vec, 0.0, 1.0)

    # Deterministic spikes: for each feature, spike in bin t if
    # frac(seed-hash + rate * (t+1)) > threshold adjusted... use cumulative
    # rate coding: spike if random < rate for each bin independently.
    rng = np.random.default_rng(seed)
    noise = rng.random((time_bins, rate_vec.shape[0]))
    # Scale rates by threshold so low-contrast images still produce some spikes
    # when threshold is low; default threshold=0.5 means p=rate.
    p = np.clip(rate_vec * (1.0 / max(threshold, 1e-6)) * 0.5, 0.0, 1.0)
    spikes = (noise < p[None, :]).astype(np.float32)
    return spikes
