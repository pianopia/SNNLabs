"""External + local mesh corpus for commercial 3DCG quality training.

Layout (see ``benchmarks/threedcg/corpus.md``)::

    data/threedcg/<asset_id>/
      reference.glb
      input.png          # optional; generated from mesh if missing
      meta.json

Supports:
- existing catalog under ``data/threedcg``
- bulk import of ``.glb/.gltf/.obj/.ply/.stl`` trees
- ShapeNet-like trees (``*/models/model_normalized.obj``)

No network downloads — callers pass local paths (licensed SketchFab, ShapeNet
mirrors, internal packs, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence
import hashlib

import numpy as np
import trimesh

from benchmarks.threedcg.asset import Asset, asset_from_trimesh, load_asset
from src.dst_snn.threedcg.dataset import (
    FAMILY_SPECS,
    ID_TO_SHAPE,
    SHAPE_TO_ID,
    Sample,
    mesh_to_occupancy,
    render_silhouette,
)
from src.dst_snn.threedcg.image_spikes import image_to_spikes

MESH_EXTENSIONS = {".glb", ".gltf", ".obj", ".ply", ".stl", ".off"}
DEFAULT_CORPUS_ROOT = Path("data/threedcg")

# Map free-text categories → training family + nearest primitive class.
_CATEGORY_TO_FAMILY: dict[str, str] = {
    "rigid_prop": "box",
    "hard_surface": "box",
    "organic": "sphere",
    "organic_character": "body",
    "character": "body",
    "foliage": "platform",
    "architecture": "arch",
    "vehicle": "l_beam",
    "furniture": "platform",
    "prop": "box",
    "weapon": "wedge",
    "animal": "body",
}


@dataclass(frozen=True)
class CorpusEntry:
    asset_id: str
    reference_path: Path
    image_path: Optional[Path]
    meta: dict[str, Any] = field(default_factory=dict)
    category: str = "unknown"
    family: str = "box"
    license: str = "unknown"
    source: str = "local-layout"

    @property
    def shape_id(self) -> int:
        nearest = FAMILY_SPECS.get(self.family, "box")
        return SHAPE_TO_ID[nearest]


def _slug(text: str, *, max_len: int = 48) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", text.strip()).strip("-").lower()
    if not s:
        s = "asset"
    return s[:max_len]


def _read_meta(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_png_rgb(path: Path, image: np.ndarray) -> None:
    """Write HxWx3 float/uint image as PNG (PIL if available)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    try:
        from PIL import Image

        Image.fromarray(arr, mode="RGB").save(path)
        return
    except Exception:
        pass
    # Fallback: solid color PNG
    png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
        b"\x00\x00\x00\x03\x00\x01\x00\x05\xfe\xd4\xef\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    path.write_bytes(png)


def load_mesh_any(path: Path) -> trimesh.Trimesh:
    """Load a mesh file (glb/obj/…) into a single Trimesh."""
    loaded = trimesh.load(str(path), force="scene")
    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    elif isinstance(loaded, trimesh.Scene):
        geoms = list(loaded.geometry.values())
        if not geoms:
            raise ValueError(f"no geometry in {path}")
        mesh = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
    else:
        raise TypeError(f"unsupported load type for {path}: {type(loaded)}")
    if len(mesh.vertices) == 0:
        raise ValueError(f"empty mesh: {path}")
    # Normalize to unit-ish AABB for stable training.
    mesh = mesh.copy()
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    bounds = mesh.bounds
    span = np.maximum(bounds[1] - bounds[0], 1e-6)
    scale = 1.6 / float(span.max())
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(scale)
    return mesh


def infer_family(category: str | None, asset_id: str = "", path: str = "") -> str:
    key = (category or "").lower().replace("-", "_").replace(" ", "_")
    if key in _CATEGORY_TO_FAMILY:
        return _CATEGORY_TO_FAMILY[key]
    blob = f"{category or ''} {asset_id} {path}".lower()
    if any(w in blob for w in ("char", "human", "body", "person", "avatar")):
        return "body"
    if any(w in blob for w in ("arch", "bridge", "building", "door")):
        return "arch"
    if any(w in blob for w in ("tree", "plant", "foliage", "leaf")):
        return "platform"
    if any(w in blob for w in ("pillar", "column", "pole")):
        return "pillar"
    if any(w in blob for w in ("sphere", "ball", "organic")):
        return "sphere"
    if any(w in blob for w in ("cyl", "pipe", "tube", "capsule")):
        return "capsule"
    if any(w in blob for w in ("l_beam", "beam", "joint", "mech")):
        return "l_beam"
    if "ring" in blob or "torus" in blob:
        return "ring"
    return "box"


def discover_mesh_files(
    src: Path,
    *,
    recursive: bool = True,
    shapenet: bool = False,
    max_files: int | None = None,
) -> list[Path]:
    """Find mesh files under ``src``."""
    src = Path(src)
    if not src.exists():
        return []
    found: list[Path] = []
    if src.is_file() and src.suffix.lower() in MESH_EXTENSIONS:
        return [src]

    if shapenet:
        # One mesh per model folder: prefer model_normalized.*, else first mesh in models/
        model_dirs = sorted({p.parent for p in src.glob("**/models") if p.is_dir()})
        for models_dir in model_dirs:
            preferred = None
            for name in ("model_normalized.obj", "model_normalized.glb", "model_normalized.ply"):
                cand = models_dir / name
                if cand.is_file():
                    preferred = cand
                    break
            if preferred is None:
                meshes = sorted(
                    p for p in models_dir.iterdir() if p.suffix.lower() in MESH_EXTENSIONS
                )
                preferred = meshes[0] if meshes else None
            if preferred is None:
                continue
            found.append(preferred)
            if max_files and len(found) >= max_files:
                return found
        return found

    iterator: Iterable[Path] = src.rglob("*") if recursive else src.glob("*")
    for p in sorted(iterator):
        if not p.is_file() or p.suffix.lower() not in MESH_EXTENSIONS:
            continue
        # Prefer reference.glb inside corpus-style folders; skip random sidecars later via max.
        found.append(p)
        if max_files and len(found) >= max_files:
            break
    return found


def _asset_id_for_path(path: Path, *, shapenet: bool = False, prefix: str = "") -> str:
    if shapenet and path.parent.name == "models":
        model_id = path.parent.parent.name
        synset = path.parent.parent.parent.name
        base = f"{synset}_{model_id}"
    else:
        base = path.stem
        if base in {"model_normalized", "model", "mesh", "reference"}:
            base = path.parent.name
    slug = _slug(f"{prefix}{base}" if prefix else base)
    # disambiguate collisions with short hash
    h = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:6]
    return f"{slug}-{h}"


def import_mesh_file(
    src: Path,
    dest_root: Path | str = DEFAULT_CORPUS_ROOT,
    *,
    asset_id: str | None = None,
    category: str | None = None,
    license: str = "external-user-provided",
    source: str = "imported",
    shapenet: bool = False,
    overwrite: bool = False,
    image_size: int = 64,
) -> CorpusEntry:
    """Copy/normalize one external mesh into the corpus layout."""
    src = Path(src)
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    aid = asset_id or _asset_id_for_path(src, shapenet=shapenet)
    out_dir = dest_root / aid
    if out_dir.exists() and not overwrite and (out_dir / "reference.glb").is_file():
        meta = _read_meta(out_dir / "meta.json")
        return CorpusEntry(
            asset_id=aid,
            reference_path=out_dir / "reference.glb",
            image_path=(out_dir / "input.png") if (out_dir / "input.png").is_file() else None,
            meta=meta,
            category=str(meta.get("category", category or "unknown")),
            family=str(meta.get("family", infer_family(category, aid, str(src)))),
            license=str(meta.get("license", license)),
            source=str(meta.get("source", source)),
        )

    mesh = load_mesh_any(src)
    out_dir.mkdir(parents=True, exist_ok=True)
    glb = out_dir / "reference.glb"
    mesh.export(glb)

    # Silhouette input from geometry (better than solid color for SNN).
    sil = render_silhouette(mesh, size=image_size)
    img_path = out_dir / "input.png"
    _write_png_rgb(img_path, sil)

    cat = category or "external"
    fam = infer_family(cat, aid, str(src))
    meta = {
        "license": license,
        "category": cat,
        "family": fam,
        "rigged": False,
        "source": source,
        "import_from": str(src),
        "n_vertices": int(len(mesh.vertices)),
        "n_faces": int(len(mesh.faces)),
        "poly_band": (
            "low" if len(mesh.faces) < 5000 else "mid" if len(mesh.faces) < 50000 else "high"
        ),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return CorpusEntry(
        asset_id=aid,
        reference_path=glb,
        image_path=img_path,
        meta=meta,
        category=cat,
        family=fam,
        license=license,
        source=source,
    )


def import_directory(
    src: Path | str,
    dest_root: Path | str = DEFAULT_CORPUS_ROOT,
    *,
    recursive: bool = True,
    shapenet: bool = False,
    max_assets: int | None = None,
    license: str = "external-user-provided",
    category: str | None = None,
    overwrite: bool = False,
    prefix: str = "",
) -> list[CorpusEntry]:
    """Bulk-import meshes from a local directory into corpus layout."""
    files = discover_mesh_files(
        Path(src), recursive=recursive, shapenet=shapenet, max_files=max_assets
    )
    entries: list[CorpusEntry] = []
    for path in files:
        aid = _asset_id_for_path(path, shapenet=shapenet, prefix=prefix)
        try:
            entry = import_mesh_file(
                path,
                dest_root,
                asset_id=aid,
                category=category,
                license=license,
                source="shapenet-like" if shapenet else "imported",
                shapenet=shapenet,
                overwrite=overwrite,
            )
            entries.append(entry)
        except Exception as exc:
            print(f"skip {path}: {exc}")
        if max_assets and len(entries) >= max_assets:
            break
    return entries


def scan_corpus_layout(root: Path | str = DEFAULT_CORPUS_ROOT) -> list[CorpusEntry]:
    """Scan ``data/threedcg/<id>/reference.glb`` layout."""
    root = Path(root)
    if not root.is_dir():
        return []
    entries: list[CorpusEntry] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        glb = child / "reference.glb"
        if not glb.is_file():
            # also accept any single mesh as reference
            meshes = [p for p in child.iterdir() if p.suffix.lower() in MESH_EXTENSIONS]
            if not meshes:
                continue
            glb = meshes[0]
        meta = _read_meta(child / "meta.json")
        aid = child.name
        cat = str(meta.get("category", "unknown"))
        fam = str(meta.get("family") or infer_family(cat, aid, str(glb)))
        img = child / "input.png"
        entries.append(
            CorpusEntry(
                asset_id=aid,
                reference_path=glb,
                image_path=img if img.is_file() else None,
                meta=meta,
                category=cat,
                family=fam,
                license=str(meta.get("license", "unknown")),
                source=str(meta.get("source", "local-layout")),
            )
        )
    return entries


def write_catalog(
    entries: Sequence[CorpusEntry],
    root: Path | str = DEFAULT_CORPUS_ROOT,
    *,
    source: str = "mixed",
) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    assets = []
    for e in entries:
        # Stable relative path under data/threedcg/
        rel = f"{root.name}/{e.asset_id}/reference.glb"
        try:
            # Prefer path relative to repo root (parent of data/)
            if root.name == "threedcg" and root.parent.name == "data":
                rel = f"data/threedcg/{e.asset_id}/{e.reference_path.name}"
        except Exception:
            pass
        assets.append(
            {
                "asset_id": e.asset_id,
                "path": rel,
                "category": e.category,
                "family": e.family,
                "license": e.license,
                "source": e.source,
                "n_vertices": e.meta.get("n_vertices"),
                "n_faces": e.meta.get("n_faces"),
            }
        )
    catalog = {
        "version": 2,
        "source": source,
        "n_assets": len(assets),
        "assets": assets,
    }
    path = root / "catalog.json"
    path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    return path


def rebuild_catalog(root: Path | str = DEFAULT_CORPUS_ROOT) -> Path:
    entries = scan_corpus_layout(root)
    sources = sorted({e.source for e in entries}) or ["empty"]
    return write_catalog(entries, root, source="+".join(sources))


def entry_to_sample(
    entry: CorpusEntry,
    *,
    seed: int = 0,
    time_bins: int = 8,
    image_size: int = 32,
    resolution: int = 8,
    max_side: int = 32,
    regenerate_image: bool = False,
) -> Sample:
    """Load a corpus entry as a training Sample (spikes + reference asset)."""
    # Prefer Asset loader for glb with skin/uv; fall back to trimesh.
    try:
        if entry.reference_path.suffix.lower() in {".glb", ".gltf"}:
            asset = load_asset(str(entry.reference_path))
            mesh = trimesh.Trimesh(vertices=asset.vertices, faces=asset.faces, process=False)
        else:
            mesh = load_mesh_any(entry.reference_path)
            asset = asset_from_trimesh(mesh)
    except Exception:
        mesh = load_mesh_any(entry.reference_path)
        asset = asset_from_trimesh(mesh)

    # Image: use provided PNG if valid, else silhouette
    image: np.ndarray
    if entry.image_path and entry.image_path.is_file() and not regenerate_image:
        try:
            from src.dst_snn.threedcg.image_spikes import load_image_array

            image = load_image_array(str(entry.image_path))
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
        except Exception:
            image = render_silhouette(mesh, size=image_size)
    else:
        image = render_silhouette(mesh, size=image_size)

    spikes = image_to_spikes(image, time_bins=time_bins, seed=seed, max_side=max_side)
    occ, _, extents_arr = mesh_to_occupancy(mesh, resolution=resolution)
    # extents from mesh AABB
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if verts.size:
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        ex = tuple(float(max(0.2, x)) for x in (hi - lo))
    else:
        ex = (1.0, 1.0, 1.0)
    while len(ex) < 3:
        ex = (*ex, ex[-1])

    return Sample(
        image=np.asarray(image, dtype=np.float64),
        spikes=spikes,
        shape_id=entry.shape_id,
        extents=(ex[0], ex[1], ex[2]),
        occupancy=occ,
        asset=asset,
        family=entry.family,
    )


@dataclass
class MeshCorpus:
    """In-memory index over a corpus root for stratified sampling."""

    root: Path
    entries: list[CorpusEntry] = field(default_factory=list)

    @classmethod
    def open(cls, root: Path | str = DEFAULT_CORPUS_ROOT) -> "MeshCorpus":
        root = Path(root)
        entries = scan_corpus_layout(root)
        return cls(root=root, entries=entries)

    def __len__(self) -> int:
        return len(self.entries)

    def by_family(self) -> dict[str, list[CorpusEntry]]:
        out: dict[str, list[CorpusEntry]] = {}
        for e in self.entries:
            out.setdefault(e.family, []).append(e)
        return out

    def by_category(self) -> dict[str, list[CorpusEntry]]:
        out: dict[str, list[CorpusEntry]] = {}
        for e in self.entries:
            out.setdefault(e.category, []).append(e)
        return out

    def sample_entries(
        self,
        n: int,
        *,
        seed: int = 0,
        stratify: bool = True,
    ) -> list[CorpusEntry]:
        if not self.entries or n <= 0:
            return []
        rng = np.random.default_rng(seed)
        if not stratify:
            idx = rng.integers(0, len(self.entries), size=n)
            return [self.entries[int(i)] for i in idx]
        # round-robin families
        groups = list(self.by_family().values())
        if not groups:
            groups = [self.entries]
        out: list[CorpusEntry] = []
        for i in range(n):
            g = groups[i % len(groups)]
            out.append(g[int(rng.integers(0, len(g)))])
        return out

    def make_batch(
        self,
        n: int,
        *,
        seed: int = 0,
        time_bins: int = 8,
        resolution: int = 8,
        image_size: int = 32,
        mix_synthetic: float = 0.0,
        synthetic_diverse: bool = True,
    ) -> list[Sample]:
        """Build a training batch from corpus, optionally mixed with synthetic families."""
        from src.dst_snn.threedcg.dataset import make_batch as synthetic_batch

        mix = float(np.clip(mix_synthetic, 0.0, 1.0))
        n_syn = int(round(n * mix)) if self.entries else n
        n_ext = n - n_syn
        samples: list[Sample] = []
        if n_ext > 0 and self.entries:
            for i, entry in enumerate(self.sample_entries(n_ext, seed=seed, stratify=True)):
                samples.append(
                    entry_to_sample(
                        entry,
                        seed=seed * 10007 + i,
                        time_bins=time_bins,
                        image_size=image_size,
                        resolution=resolution,
                        max_side=image_size,
                    )
                )
        if n_syn > 0:
            samples.extend(
                synthetic_batch(
                    n_syn,
                    seed=seed + 17,
                    time_bins=time_bins,
                    resolution=resolution,
                    image_size=image_size,
                    diverse=synthetic_diverse,
                )
            )
        # shuffle with deterministic RNG
        rng = np.random.default_rng(seed + 3)
        order = rng.permutation(len(samples))
        return [samples[int(i)] for i in order]
