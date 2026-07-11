#!/usr/bin/env python3
"""Export Track1/Track2 generations as GLB into EDEN/public/generated.

Also refreshes ``manifest.json`` so EDEN can auto-spawn all exports.

Usage:
  python scripts/export_threedcg_for_eden.py --track track1_sequence --name snn-build
  python scripts/export_threedcg_for_eden.py --diverse-pack
  python scripts/export_threedcg_for_eden.py --refresh-manifest-only
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import trimesh

from benchmarks.threedcg.asset import load_asset
from benchmarks.threedcg.scorer import score_to_result
from src.dst_snn.threedcg.dataset import FAMILIES, ID_TO_SHAPE, make_sample
from src.dst_snn.threedcg.ops import ops_to_asset
from src.dst_snn.threedcg.pipeline import generate_from_image, synthetic_box_image
from src.dst_snn.threedcg.sequence import template_program


def asset_to_glb(asset, path: Path) -> None:
    mesh = trimesh.Trimesh(
        vertices=asset.vertices,
        faces=asset.faces,
        process=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


def write_manifest(out_dir: Path) -> Path:
    """Scan sidecars + glbs and write manifest.json for EDEN auto-spawn."""
    assets = []
    for glb in sorted(out_dir.glob("*.glb")):
        name = glb.stem
        side = out_dir / f"{name}.json"
        meta = {}
        if side.is_file():
            try:
                meta = json.loads(side.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        assets.append(
            {
                "name": name,
                "url": f"/generated/{name}.glb",
                "track": meta.get("track"),
                "family": meta.get("family"),
                "quality": meta.get("quality"),
                "n_vertices": meta.get("n_vertices"),
                "n_faces": meta.get("n_faces"),
                "has_uv": meta.get("has_uv"),
                "has_skin": meta.get("has_skin"),
                "bones": meta.get("bones") or [],
            }
        )
    manifest = {
        "version": 2,
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "assets": assets,
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def _pick_ckpt(ckpt_dir: Path, prefer_quality: bool, *names: str) -> Path | None:
    ordered = list(names)
    if prefer_quality:
        # quality variants first if present in names list already
        pass
    for n in ordered:
        ckpt = ckpt_dir / n
        if ckpt.is_file():
            return ckpt
    return None


def _write_one(
    *,
    asset,
    out_dir: Path,
    name: str,
    track: str,
    seed: int,
    family: str | None = None,
    quality: float | None = None,
    extra: dict | None = None,
) -> dict:
    out = out_dir / f"{name}.glb"
    asset_to_glb(asset, out)
    meta = {
        "track": track,
        "seed": seed,
        "family": family,
        "quality": quality,
        "path": str(out.relative_to(ROOT)) if str(out).startswith(str(ROOT)) else str(out),
        "n_vertices": int(len(asset.vertices)),
        "n_faces": int(len(asset.faces)),
        "bones": list(asset.bones or []),
        "has_uv": asset.uv is not None,
        "has_skin": asset.skin_weights is not None,
        "url": f"/generated/{name}.glb",
        **(extra or {}),
    }
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {out} family={family} verts={meta['n_vertices']} faces={meta['n_faces']}"
        + (f" q={quality:.3f}" if quality is not None else "")
    )
    return meta


def export_diverse_pack(
    *,
    out_dir: Path,
    ckpt_dir: Path,
    seed: int,
    backend: str,
    prefer_quality: bool,
    families: list[str] | None = None,
) -> list[dict]:
    """Export one GLB per commercial family from sequence templates + trained decode."""
    fams = families or list(FAMILIES)
    results = []
    seq_ckpt = _pick_ckpt(
        ckpt_dir,
        prefer_quality,
        "track1_seq_quality.pt",
        "track1_seq.pt",
    )
    t2_ckpt = _pick_ckpt(
        ckpt_dir,
        prefer_quality,
        "track2_quality.pt",
        "track2.pt",
    )

    for i, family in enumerate(fams):
        sample = make_sample(
            family=family,
            extents=(0.9 + 0.1 * (i % 3), 1.1 + 0.15 * ((i + 1) % 3), 0.85 + 0.1 * ((i + 2) % 3)),
            seed=seed * 1000 + i,
            time_bins=8,
            image_size=32,
            resolution=8,
        )
        # 1) Teacher-quality template (ground-truth style commercial recipe)
        ops = template_program(
            ID_TO_SHAPE[sample.shape_id],
            sample.extents,
            family=family,
        )
        teacher = ops_to_asset(ops)
        q_teacher = float(score_to_result(teacher, sample.asset, asset_id=f"t-{family}").metrics.quality)
        results.append(
            _write_one(
                asset=teacher,
                out_dir=out_dir,
                name=f"fam-{family}-teacher",
                track="template_teacher",
                seed=seed + i,
                family=family,
                quality=q_teacher,
            )
        )

        # 2) Trained sequence model conditioned on silhouette
        kwargs = {"seed": seed + i, "mesh_backend": backend, "reference": sample.asset}
        if seq_ckpt is not None:
            kwargs["track1_checkpoint"] = str(seq_ckpt)
            try:
                pred = generate_from_image(sample.image, track="track1_sequence", **kwargs)
                q_pred = float(score_to_result(pred, sample.asset, asset_id=f"p-{family}").metrics.quality)
                results.append(
                    _write_one(
                        asset=pred,
                        out_dir=out_dir,
                        name=f"fam-{family}-seq",
                        track="track1_sequence",
                        seed=seed + i,
                        family=family,
                        quality=q_pred,
                        extra={"checkpoint": str(seq_ckpt.name)},
                    )
                )
            except Exception as exc:
                print(f"warn: seq export failed for {family}: {exc}")

        # 3) Occupancy track for a subset (complex families benefit)
        if t2_ckpt is not None and family in {"body", "arch", "l_beam", "platform", "pillar"}:
            kwargs2 = {
                "seed": seed + i,
                "mesh_backend": backend,
                "reference": sample.asset,
                "track2_checkpoint": str(t2_ckpt),
            }
            try:
                pred2 = generate_from_image(sample.image, track="track2_trained", **kwargs2)
                q2 = float(score_to_result(pred2, sample.asset, asset_id=f"o-{family}").metrics.quality)
                results.append(
                    _write_one(
                        asset=pred2,
                        out_dir=out_dir,
                        name=f"fam-{family}-occ",
                        track="track2_trained",
                        seed=seed + i,
                        family=family,
                        quality=q2,
                        extra={"checkpoint": str(t2_ckpt.name)},
                    )
                )
            except Exception as exc:
                print(f"warn: occ export failed for {family}: {exc}")

    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--track", default="track1_sequence")
    p.add_argument("--name", default="snn-generated")
    p.add_argument("--reference", default=str(ROOT / "data" / "threedcg" / "unit-box" / "reference.glb"))
    p.add_argument("--image", default=None)
    p.add_argument("--synthetic", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--ckpt-dir",
        type=Path,
        default=ROOT / "artifacts" / "threedcg" / "checkpoints",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "EDEN" / "public" / "generated",
    )
    p.add_argument("--backend", default="trimesh")
    p.add_argument(
        "--refresh-manifest-only",
        action="store_true",
        help="Only rebuild manifest.json from existing GLBs",
    )
    p.add_argument(
        "--prefer-quality",
        action="store_true",
        default=True,
        help="Prefer *_quality.pt checkpoints when present",
    )
    p.add_argument(
        "--diverse-pack",
        action="store_true",
        help="Export multi-family commercial pack (teacher + trained seq/occ)",
    )
    p.add_argument(
        "--clear-generated",
        action="store_true",
        help="Remove old GLBs/json in out-dir before diverse pack (keeps manifest rewrite)",
    )
    p.add_argument(
        "--from-corpus",
        action="store_true",
        help="Export reference GLBs from data/threedcg corpus into EDEN/public/generated",
    )
    p.add_argument(
        "--corpus-root",
        type=Path,
        default=ROOT / "data" / "threedcg",
    )
    p.add_argument("--max-corpus", type=int, default=24, help="Max corpus assets to export")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.refresh_manifest_only:
        path = write_manifest(args.out_dir)
        print(f"wrote {path}")
        return

    if args.from_corpus:
        from src.dst_snn.threedcg.corpus import MeshCorpus
        from benchmarks.threedcg.scorer import score_to_result

        if args.clear_generated:
            for pth in list(args.out_dir.glob("*.glb")) + list(args.out_dir.glob("*.json")):
                if pth.name == "manifest.json":
                    continue
                pth.unlink(missing_ok=True)
        corpus = MeshCorpus.open(args.corpus_root)
        rows = []
        for i, entry in enumerate(corpus.entries[: max(0, args.max_corpus)]):
            try:
                asset = load_asset(str(entry.reference_path))
                # self-score as baseline reference quality
                q = float(score_to_result(asset, asset, asset_id=entry.asset_id).metrics.quality)
                rows.append(
                    _write_one(
                        asset=asset,
                        out_dir=args.out_dir,
                        name=f"corpus-{entry.asset_id}",
                        track="corpus_reference",
                        seed=args.seed + i,
                        family=entry.family,
                        quality=q,
                        extra={
                            "category": entry.category,
                            "license": entry.license,
                            "source": entry.source,
                            "asset_id": entry.asset_id,
                        },
                    )
                )
            except Exception as exc:
                print(f"skip {entry.asset_id}: {exc}")
        manifest = write_manifest(args.out_dir)
        print(f"from-corpus: {len(rows)} assets → {manifest}")
        return

    if args.diverse_pack:
        if args.clear_generated:
            for pth in list(args.out_dir.glob("*.glb")) + list(args.out_dir.glob("*.json")):
                if pth.name == "manifest.json":
                    continue
                pth.unlink(missing_ok=True)
        rows = export_diverse_pack(
            out_dir=args.out_dir,
            ckpt_dir=args.ckpt_dir,
            seed=args.seed,
            backend=args.backend,
            prefer_quality=args.prefer_quality,
        )
        manifest = write_manifest(args.out_dir)
        qs = [r["quality"] for r in rows if r.get("quality") is not None]
        print(
            f"diverse pack: {len(rows)} assets | mean_quality={sum(qs)/len(qs):.3f}"
            if qs
            else f"diverse pack: {len(rows)} assets"
        )
        print(f"manifest: {manifest}")
        return

    if args.image:
        image = args.image
    else:
        image = synthetic_box_image(size=32)

    ref = None
    ref_path = Path(args.reference)
    if ref_path.is_file():
        ref = load_asset(str(ref_path))

    kwargs = {"seed": args.seed, "mesh_backend": args.backend, "reference": ref}

    if args.track in {"track1_trained", "track1_sequence"}:
        if args.track == "track1_sequence":
            ckpt = _pick_ckpt(args.ckpt_dir, args.prefer_quality, "track1_seq_quality.pt", "track1_seq.pt")
        else:
            ckpt = _pick_ckpt(args.ckpt_dir, args.prefer_quality, "track1_quality.pt", "track1.pt")
        if ckpt is not None:
            kwargs["track1_checkpoint"] = str(ckpt)
    if args.track in {"track2_trained", "track2_sdf"}:
        if args.track == "track2_sdf":
            ckpt = _pick_ckpt(args.ckpt_dir, args.prefer_quality, "track2_sdf.pt")
        else:
            ckpt = _pick_ckpt(args.ckpt_dir, args.prefer_quality, "track2_quality.pt", "track2.pt")
        if ckpt is not None:
            kwargs["track2_checkpoint"] = str(ckpt)

    asset = generate_from_image(image, track=args.track, **kwargs)
    meta = _write_one(
        asset=asset,
        out_dir=args.out_dir,
        name=args.name,
        track=args.track,
        seed=args.seed,
    )
    manifest = write_manifest(args.out_dir)
    print(f"EDEN URL path: /generated/{args.name}.glb")
    print(f"manifest: {manifest} ({len(json.loads(manifest.read_text())['assets'])} assets)")


if __name__ == "__main__":
    main()
