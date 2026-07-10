"""Optional render-based perceptual similarity (SSIM), gated on pyrender."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .asset import Asset


def render_available() -> bool:
    try:
        import pyrender  # noqa: F401
    except Exception:
        return False
    return True


def ssim(image_a: np.ndarray, image_b: np.ndarray) -> float:
    a = np.asarray(image_a, dtype=np.float64)
    b = np.asarray(image_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("images must have the same shape")
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    denominator = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    return float(numerator / denominator)


def _render_orbit(asset: Asset, views: int, resolution: int) -> list[np.ndarray]:  # pragma: no cover
    import pyrender
    import trimesh

    mesh = trimesh.Trimesh(vertices=asset.vertices, faces=asset.faces, process=False)
    render_mesh = pyrender.Mesh.from_trimesh(mesh)
    images: list[np.ndarray] = []
    for i in range(views):
        scene = pyrender.Scene()
        scene.add(render_mesh)
        angle = 2.0 * np.pi * i / views
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        pose = np.eye(4)
        pose[0, 3] = 2.5 * np.cos(angle)
        pose[2, 3] = 2.5 * np.sin(angle)
        scene.add(camera, pose=pose)
        scene.add(pyrender.DirectionalLight(), pose=pose)
        renderer = pyrender.OffscreenRenderer(resolution, resolution)
        color, _ = renderer.render(scene)
        renderer.delete()
        images.append(np.asarray(color, dtype=np.float64).mean(axis=2) / 255.0)
    return images


def render_similarity(
    candidate: Asset,
    reference: Asset,
    *,
    views: int = 4,
    resolution: int = 128,
) -> Optional[float]:
    if not render_available():
        return None
    cand_images = _render_orbit(candidate, views, resolution)  # pragma: no cover
    ref_images = _render_orbit(reference, views, resolution)  # pragma: no cover
    scores = [ssim(c, r) for c, r in zip(cand_images, ref_images)]  # pragma: no cover
    return float(np.mean(scores)) if scores else None  # pragma: no cover
