#!/usr/bin/env python3
"""Import external meshes into data/threedcg corpus layout.

No network downloads. Point ``--src`` at local licensed assets (SketchFab glTF
exports, ShapeNet mirrors, internal packs, etc.).

Examples:
  # Flat / nested folder of GLB/OBJ:
  python scripts/import_threedcg_external.py --src /path/to/meshes --max 100

  # ShapeNetCore-like tree:
  python scripts/import_threedcg_external.py --src /data/ShapeNetCore.v2 --shapenet --max 200

  # Rebuild catalog only:
  python scripts/import_threedcg_external.py --rebuild-catalog
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dst_snn.threedcg.corpus import (
    MeshCorpus,
    import_directory,
    rebuild_catalog,
    scan_corpus_layout,
    write_catalog,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src",
        type=Path,
        default=None,
        help="Local directory or file of meshes to import",
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=ROOT / "data" / "threedcg",
        help="Corpus root (default: data/threedcg)",
    )
    p.add_argument("--shapenet", action="store_true", help="Parse ShapeNet-like models/ layout")
    p.add_argument("--max", type=int, default=None, help="Max assets to import")
    p.add_argument("--license", default="external-user-provided")
    p.add_argument("--category", default=None, help="Default category tag for imports")
    p.add_argument("--prefix", default="", help="Prefix for generated asset_id")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--rebuild-catalog",
        action="store_true",
        help="Only rescan dest and rewrite catalog.json",
    )
    p.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not recurse when discovering meshes",
    )
    args = p.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)

    if args.rebuild_catalog and args.src is None:
        path = rebuild_catalog(args.dest)
        entries = scan_corpus_layout(args.dest)
        print(f"catalog {path} ({len(entries)} assets)")
        cats = {}
        for e in entries:
            cats[e.category] = cats.get(e.category, 0) + 1
        for k, v in sorted(cats.items()):
            print(f"  {k}: {v}")
        return

    if args.src is None:
        p.error("--src is required unless --rebuild-catalog")

    if not args.src.exists():
        p.error(f"--src does not exist: {args.src}")

    print(f"importing from {args.src} → {args.dest} (shapenet={args.shapenet})")
    entries = import_directory(
        args.src,
        args.dest,
        recursive=not args.no_recursive,
        shapenet=args.shapenet,
        max_assets=args.max,
        license=args.license,
        category=args.category,
        overwrite=args.overwrite,
        prefix=args.prefix,
    )
    all_entries = scan_corpus_layout(args.dest)
    catalog = write_catalog(all_entries, args.dest, source="mixed-external")
    print(f"imported {len(entries)} new/updated entries")
    print(f"corpus total: {len(all_entries)}")
    print(f"catalog: {catalog}")

    # quick sanity: loadable samples
    corpus = MeshCorpus.open(args.dest)
    if len(corpus) > 0:
        batch = corpus.make_batch(min(4, len(corpus)), seed=0, mix_synthetic=0.0)
        print(f"sample batch ok: {len(batch)} | families={[s.family for s in batch]}")


if __name__ == "__main__":
    main()
