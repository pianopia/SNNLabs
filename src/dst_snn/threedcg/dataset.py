"""Synthetic offline dataset for Track1/2 supervised + quality training.

No network. Each sample: synthetic RGB silhouette of a primitive or composite
mesh, spike tensor via ``image_to_spikes``, labels for nearest shape class +
AABB extents, occupancy grid, and a named ``family`` for diversity metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import trimesh

from benchmarks.threedcg.asset import Asset, asset_from_trimesh
from src.dst_snn.threedcg.image_spikes import image_to_spikes

ShapeName = Literal["box", "sphere", "cylinder"]
SHAPE_TO_ID: dict[str, int] = {"box": 0, "sphere": 1, "cylinder": 2}
ID_TO_SHAPE: tuple[ShapeName, ...] = ("box", "sphere", "cylinder")

# Commercial-ish families: not only three primitives.
# ``nearest_shape`` is the Track-1 class head label (box/sphere/cylinder).
FAMILY_SPECS: dict[str, ShapeName] = {
    "box": "box",
    "sphere": "sphere",
    "cylinder": "cylinder",
    "capsule": "cylinder",
    "l_beam": "box",
    "t_joint": "box",
    "arch": "box",
    "body": "cylinder",
    "platform": "box",
    "pillar": "cylinder",
    "wedge": "box",
    "ring": "sphere",
}

FAMILIES: tuple[str, ...] = tuple(FAMILY_SPECS.keys())


@dataclass(frozen=True)
class Sample:
    image: np.ndarray  # HxWx3 float
    spikes: np.ndarray  # [T, F]
    shape_id: int
    extents: tuple[float, float, float]
    occupancy: np.ndarray  # [R,R,R] float 0/1
    asset: Asset
    family: str = "box"


def _concat(meshes: Sequence[trimesh.Trimesh]) -> trimesh.Trimesh:
    clean = [m for m in meshes if m is not None and len(m.vertices) > 0]
    if not clean:
        return trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    if len(clean) == 1:
        return clean[0]
    return trimesh.util.concatenate(clean)


def _make_mesh_family(
    family: str,
    extents: Sequence[float],
    rng: np.random.Generator,
) -> trimesh.Trimesh:
    """Build a mesh for a commercial-ish family (composites allowed)."""
    ex, ey, ez = (float(max(0.25, e)) for e in extents[:3])
    if family == "box":
        mesh = trimesh.creation.box(extents=(ex, ey, ez))
    elif family == "sphere":
        r = 0.5 * max(ex, ey, ez)
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=r)
    elif family == "cylinder":
        mesh = trimesh.creation.cylinder(radius=0.5 * max(ex, ez), height=ey)
    elif family == "capsule":
        r = 0.35 * max(ex, ez)
        body = trimesh.creation.cylinder(radius=r, height=max(ey * 0.7, 0.3))
        top = trimesh.creation.icosphere(subdivisions=1, radius=r)
        top.apply_translation([0, ey * 0.4, 0])
        bot = trimesh.creation.icosphere(subdivisions=1, radius=r)
        bot.apply_translation([0, -ey * 0.4, 0])
        mesh = _concat([body, top, bot])
    elif family == "l_beam":
        a = trimesh.creation.box(extents=(ex, ey * 0.35, ez * 0.45))
        b = trimesh.creation.box(extents=(ex * 0.35, ey, ez * 0.45))
        b.apply_translation([ex * 0.25, ey * 0.2, 0])
        mesh = _concat([a, b])
    elif family == "t_joint":
        stem = trimesh.creation.box(extents=(ex * 0.35, ey, ez * 0.4))
        bar = trimesh.creation.box(extents=(ex, ey * 0.3, ez * 0.4))
        bar.apply_translation([0, ey * 0.4, 0])
        mesh = _concat([stem, bar])
    elif family == "arch":
        # inverted U: two pillars + top beam
        r = 0.18 * max(ex, ez)
        left = trimesh.creation.cylinder(radius=r, height=ey * 0.85)
        left.apply_translation([-ex * 0.3, 0, 0])
        right = trimesh.creation.cylinder(radius=r, height=ey * 0.85)
        right.apply_translation([ex * 0.3, 0, 0])
        top = trimesh.creation.box(extents=(ex * 0.85, ey * 0.2, ez * 0.35))
        top.apply_translation([0, ey * 0.45, 0])
        mesh = _concat([left, right, top])
    elif family == "body":
        # torso + head + simple limb stubs (character proxy)
        torso = trimesh.creation.cylinder(radius=0.28 * max(ex, ez), height=ey * 0.55)
        head = trimesh.creation.icosphere(subdivisions=1, radius=0.22 * max(ex, ez))
        head.apply_translation([0, ey * 0.42, 0])
        arm_l = trimesh.creation.box(extents=(ex * 0.55, ey * 0.12, ez * 0.18))
        arm_l.apply_translation([0, ey * 0.1, 0])
        leg = trimesh.creation.cylinder(radius=0.12 * max(ex, ez), height=ey * 0.35)
        leg.apply_translation([0, -ey * 0.4, 0])
        mesh = _concat([torso, head, arm_l, leg])
    elif family == "platform":
        base = trimesh.creation.box(extents=(ex * 1.2, ey * 0.18, ez * 1.2))
        post = trimesh.creation.cylinder(radius=0.12 * max(ex, ez), height=ey * 0.7)
        post.apply_translation([0, ey * 0.35, 0])
        top = trimesh.creation.box(extents=(ex * 0.5, ey * 0.12, ez * 0.5))
        top.apply_translation([0, ey * 0.65, 0])
        mesh = _concat([base, post, top])
    elif family == "pillar":
        shaft = trimesh.creation.cylinder(radius=0.22 * max(ex, ez), height=ey)
        cap = trimesh.creation.box(extents=(ex * 0.7, ey * 0.12, ez * 0.7))
        cap.apply_translation([0, ey * 0.5, 0])
        base = trimesh.creation.box(extents=(ex * 0.75, ey * 0.1, ez * 0.75))
        base.apply_translation([0, -ey * 0.5, 0])
        mesh = _concat([shaft, cap, base])
    elif family == "wedge":
        # approximate wedge via scaled box + smaller box
        a = trimesh.creation.box(extents=(ex, ey * 0.5, ez))
        b = trimesh.creation.box(extents=(ex * 0.55, ey * 0.55, ez * 0.55))
        b.apply_translation([ex * 0.15, ey * 0.25, 0])
        mesh = _concat([a, b])
    elif family == "ring":
        # torus-ish: outer sphere shell approx via two cylinders
        outer = trimesh.creation.cylinder(radius=0.45 * max(ex, ez), height=ey * 0.35)
        inner = trimesh.creation.cylinder(radius=0.22 * max(ex, ez), height=ey * 0.5)
        # no true CSG difference offline — stack as low-poly ring proxy
        mesh = outer
        try:
            # prefer hollow look: keep outer only if difference unavailable
            if hasattr(outer, "difference"):
                mesh = outer.difference(inner)
                if mesh is None or len(getattr(mesh, "vertices", [])) == 0:
                    mesh = outer
        except Exception:
            mesh = outer
    else:
        mesh = trimesh.creation.box(extents=(ex, ey, ez))

    # small random yaw for silhouette diversity
    angle = float(rng.uniform(-0.45, 0.45))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 1, 0]))
    return mesh


def _make_mesh(shape: ShapeName, extents: Sequence[float], rng: np.random.Generator) -> trimesh.Trimesh:
    return _make_mesh_family(shape, extents, rng)


def render_silhouette(
    mesh: trimesh.Trimesh,
    *,
    size: int = 32,
    bg: float = 0.08,
    fg: float = 0.92,
) -> np.ndarray:
    """Orthographic XY silhouette (no GPU): project vertices to a binary-ish image."""
    img = np.full((size, size, 3), bg, dtype=np.float64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if verts.size == 0:
        return img
    # center and scale to fit
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    center = 0.5 * (lo + hi)
    span = np.maximum(hi - lo, 1e-6)
    scale = 0.8 * size / float(span.max())
    xy = (verts[:, [0, 1]] - center[[0, 1]]) * scale + (size / 2.0)
    # paint faces as filled triangles (coarse)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    for tri in faces:
        pts = xy[tri]
        # bounding box raster
        minx = int(max(0, np.floor(pts[:, 0].min())))
        maxx = int(min(size - 1, np.ceil(pts[:, 0].max())))
        miny = int(max(0, np.floor(pts[:, 1].min())))
        maxy = int(min(size - 1, np.ceil(pts[:, 1].max())))
        if maxx < minx or maxy < miny:
            continue
        # barycentric fill
        v0, v1, v2 = pts
        for y in range(miny, maxy + 1):
            for x in range(minx, maxx + 1):
                p = np.array([x + 0.5, y + 0.5])
                den = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
                if abs(den) < 1e-9:
                    continue
                a = ((v1[1] - v2[1]) * (p[0] - v2[0]) + (v2[0] - v1[0]) * (p[1] - v2[1])) / den
                b = ((v2[1] - v0[1]) * (p[0] - v2[0]) + (v0[0] - v2[0]) * (p[1] - v2[1])) / den
                c = 1.0 - a - b
                if a >= 0 and b >= 0 and c >= 0:
                    img[size - 1 - y, x, :] = fg
    return img


def mesh_to_occupancy(
    mesh: trimesh.Trimesh,
    *,
    resolution: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voxelize AABB of mesh by vertex occupancy (fast, offline).

    Returns ``(grid, origin, extents)``.
    """
    res = max(2, int(resolution))
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if verts.size == 0:
        return np.zeros((res, res, res)), np.zeros(3), np.ones(3)
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    extents = np.maximum(hi - lo, 1e-6)
    origin = lo.copy()
    grid = np.zeros((res, res, res), dtype=np.float64)
    norm = (verts - origin) / extents
    idx = np.clip((norm * res).astype(int), 0, res - 1)
    for i, j, k in idx:
        grid[i, j, k] = 1.0
    if grid.sum() == 0:
        grid[res // 2, res // 2, res // 2] = 1.0
    return grid, origin, extents


def make_sample(
    *,
    shape: ShapeName | None = None,
    family: str | None = None,
    extents: Sequence[float],
    seed: int,
    time_bins: int = 8,
    image_size: int = 32,
    resolution: int = 8,
    max_side: int = 32,
) -> Sample:
    rng = np.random.default_rng(seed)
    fam = family or shape or "box"
    if fam not in FAMILY_SPECS:
        fam = "box"
    nearest = FAMILY_SPECS[fam]
    mesh = _make_mesh_family(fam, extents, rng)
    asset = asset_from_trimesh(mesh)
    image = render_silhouette(mesh, size=image_size)
    spikes = image_to_spikes(image, time_bins=time_bins, seed=seed, max_side=max_side)
    occ, _, _ = mesh_to_occupancy(mesh, resolution=resolution)
    ex = tuple(float(max(0.2, e)) for e in extents[:3])
    while len(ex) < 3:
        ex = (*ex, ex[-1])
    return Sample(
        image=image,
        spikes=spikes,
        shape_id=SHAPE_TO_ID[nearest],
        extents=(ex[0], ex[1], ex[2]),
        occupancy=occ,
        asset=asset,
        family=fam,
    )


def make_batch(
    n: int,
    *,
    seed: int = 0,
    time_bins: int = 8,
    resolution: int = 8,
    image_size: int = 32,
    diverse: bool = True,
    families: Sequence[str] | None = None,
) -> list[Sample]:
    """Build a batch. When ``diverse=True``, stratify across commercial families."""
    rng = np.random.default_rng(seed)
    fam_list = list(families) if families else (list(FAMILIES) if diverse else ["box", "sphere", "cylinder"])
    out: list[Sample] = []
    for i in range(n):
        # round-robin + noise so every family appears when n >= len(families)
        family = fam_list[i % len(fam_list)]
        if rng.random() < 0.15:
            family = fam_list[int(rng.integers(0, len(fam_list)))]
        # family-conditioned extent priors (more realistic proportions)
        if family in {"pillar", "capsule", "body"}:
            extents = (
                float(rng.uniform(0.35, 1.1)),
                float(rng.uniform(0.9, 2.0)),
                float(rng.uniform(0.35, 1.1)),
            )
        elif family in {"platform", "ring"}:
            extents = (
                float(rng.uniform(0.8, 1.8)),
                float(rng.uniform(0.35, 1.0)),
                float(rng.uniform(0.8, 1.8)),
            )
        elif family in {"l_beam", "t_joint", "arch", "wedge"}:
            extents = (
                float(rng.uniform(0.6, 1.7)),
                float(rng.uniform(0.5, 1.6)),
                float(rng.uniform(0.4, 1.2)),
            )
        else:
            extents = (
                float(rng.uniform(0.4, 1.6)),
                float(rng.uniform(0.4, 1.8)),
                float(rng.uniform(0.4, 1.6)),
            )
        out.append(
            make_sample(
                family=family,
                extents=extents,
                seed=seed * 10007 + i,
                time_bins=time_bins,
                image_size=image_size,
                resolution=resolution,
            )
        )
    return out
