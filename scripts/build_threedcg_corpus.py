#!/usr/bin/env python3
"""Build offline synthetic multi-asset 3DCG corpus (licensed-ready layout).

Produces several categories under ``data/threedcg/<asset_id>/`` matching
``benchmarks/threedcg/corpus.md``. Not SketchFab content — redistribution-safe
synthetic stand-ins until licensed references are dropped into the same layout.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import trimesh

from benchmarks.threedcg.asset import load_asset
from benchmarks.threedcg.generator import run_generator


def _write_png(path: Path, rgb: tuple[int, int, int] = (160, 170, 190)) -> None:
    try:
        from PIL import Image

        Image.new("RGB", (128, 128), color=rgb).save(path)
        return
    except ImportError:
        pass
    png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
        b"\x00\x00\x00\x03\x00\x01\x00\x05\xfe\xd4\xef\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    path.write_bytes(png)


def _export(asset_id: str, mesh: trimesh.Trimesh, meta: dict) -> Path:
    out_dir = ROOT / "data" / "threedcg" / asset_id
    out_dir.mkdir(parents=True, exist_ok=True)
    glb = out_dir / "reference.glb"
    mesh.export(glb)
    _write_png(out_dir / "input.png")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return glb


def build_unit_box() -> Path:
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    return _export(
        "unit-box",
        mesh,
        {
            "license": "synthetic-internal",
            "category": "rigid_prop",
            "family": "box",
            "source": "synthetic-internal",
            "rigged": False,
            "poly_band": "low",
            "note": "Axis-aligned unit box.",
        },
    )


def build_icosphere() -> Path:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.75)
    return _export(
        "organic-sphere",
        mesh,
        {
            "license": "synthetic-internal",
            "category": "organic",
            "family": "sphere",
            "source": "synthetic-internal",
            "rigged": False,
            "poly_band": "low",
            "note": "Icosphere stand-in for organic silhouette.",
        },
    )


def build_cylinder_prop() -> Path:
    mesh = trimesh.creation.cylinder(radius=0.35, height=1.2)
    return _export(
        "hard-surface-cylinder",
        mesh,
        {
            "license": "synthetic-internal",
            "category": "hard_surface",
            "family": "cylinder",
            "source": "synthetic-internal",
            "rigged": False,
            "poly_band": "low",
            "note": "Cylinder hard-surface prop.",
        },
    )


def build_capsule_character() -> Path:
    """Two stacked capsules as a minimal 'character' massing (unrigged)."""
    body = trimesh.creation.capsule(height=1.0, radius=0.25)
    head = trimesh.creation.icosphere(subdivisions=1, radius=0.22)
    head.apply_translation([0, 0.85, 0])
    mesh = trimesh.util.concatenate([body, head])
    return _export(
        "character-massing",
        mesh,
        {
            "license": "synthetic-internal",
            "category": "organic_character",
            "family": "body",
            "source": "synthetic-internal",
            "rigged": False,
            "poly_band": "low",
            "note": "Unrigged character massing; replace with licensed rigged GLB for skin metrics.",
        },
    )


def build_foliage_proxy() -> Path:
    """Random-ish cone cluster as foliage proxy (deterministic seed)."""
    rng = np.random.default_rng(0)
    meshes = []
    for i in range(5):
        cone = trimesh.creation.cone(radius=0.3 + 0.05 * i, height=0.8 + 0.1 * i)
        offset = rng.normal(scale=0.15, size=3)
        offset[1] = abs(offset[1])
        cone.apply_translation(offset)
        meshes.append(cone)
    mesh = trimesh.util.concatenate(meshes)
    return _export(
        "foliage-proxy",
        mesh,
        {
            "license": "synthetic-internal",
            "category": "foliage",
            "family": "platform",
            "source": "synthetic-internal",
            "rigged": False,
            "poly_band": "low",
            "note": "Foliage category stand-in.",
        },
    )


def build_diverse_families() -> list[Path]:
    """Extra synthetic families so the local corpus is non-trivial without external packs."""
    from src.dst_snn.threedcg.dataset import FAMILIES, make_sample

    paths = []
    for i, fam in enumerate(FAMILIES):
        if fam in {"box", "sphere", "cylinder"}:
            continue  # already covered by classic builders
        sample = make_sample(
            family=fam,
            extents=(0.9 + 0.05 * (i % 3), 1.2 + 0.08 * ((i + 1) % 3), 0.85 + 0.05 * ((i + 2) % 3)),
            seed=100 + i,
        )
        mesh = trimesh.Trimesh(
            vertices=sample.asset.vertices,
            faces=sample.asset.faces,
            process=False,
        )
        asset_id = f"syn-{fam}"
        paths.append(
            _export(
                asset_id,
                mesh,
                {
                    "license": "synthetic-internal",
                    "category": fam,
                    "family": fam,
                    "rigged": False,
                    "poly_band": "low",
                    "source": "synthetic-internal",
                    "note": f"Synthetic stand-in for family={fam}; replace with licensed GLB.",
                },
            )
        )
    return paths


def main() -> None:
    paths = [
        build_unit_box(),
        build_icosphere(),
        build_cylinder_prop(),
        build_capsule_character(),
        build_foliage_proxy(),
        *build_diverse_families(),
    ]
    catalog = []
    for glb in paths:
        asset_id = glb.parent.name
        ref = load_asset(str(glb))
        result = run_generator(ref, asset_id=asset_id, kind="primitive_fit")
        report = glb.parent / "generator_primitive_fit.json"
        report.write_text(result.to_json() + "\n", encoding="utf-8")
        meta = json.loads((glb.parent / "meta.json").read_text(encoding="utf-8"))
        # Ensure silhouette input exists for SNN training
        try:
            from src.dst_snn.threedcg.corpus import _write_png_rgb, load_mesh_any
            from src.dst_snn.threedcg.dataset import render_silhouette

            mesh = load_mesh_any(glb)
            sil = render_silhouette(mesh, size=64)
            _write_png_rgb(glb.parent / "input.png", sil)
        except Exception:
            pass
        catalog.append(
            {
                "asset_id": asset_id,
                "path": str(glb.relative_to(ROOT)),
                "category": meta.get("category"),
                "family": meta.get("family"),
                "license": meta.get("license"),
                "source": meta.get("source", "synthetic-internal"),
                "generator_quality": result.metrics.quality,
            }
        )
        print(f"{asset_id}: generator quality={result.metrics.quality:.4f}")

    catalog_path = ROOT / "data" / "threedcg" / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {"version": 2, "source": "synthetic-internal", "n_assets": len(catalog), "assets": catalog},
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {catalog_path} ({len(catalog)} assets)")


if __name__ == "__main__":
    main()
