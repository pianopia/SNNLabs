#!/usr/bin/env python3
"""Create a minimal offline 3DCG reference corpus entry (unit-box).

This is a synthetic stand-in for a SketchFab reference so scorer smoke tests
can run without network downloads. Replace with licensed real assets under
the same layout when available.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import trimesh

from benchmarks.threedcg.asset import load_asset
from benchmarks.threedcg.baseline import run_baseline


def main() -> None:
    out_dir = ROOT / "data" / "threedcg" / "unit-box"
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    glb_path = out_dir / "reference.glb"
    mesh.export(glb_path)

    # Simple solid input image (generation input stand-in).
    try:
        from PIL import Image
    except ImportError:
        Image = None
    if Image is not None:
        img = Image.new("RGB", (128, 128), color=(180, 180, 200))
        img.save(out_dir / "input.png")
    else:
        # Minimal valid 1x1 PNG
        png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
            b"\x00\x00\x00\x03\x00\x01\x00\x05\xfe\xd4\xef\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        (out_dir / "input.png").write_bytes(png)

    meta = {
        "license": "synthetic-internal",
        "category": "rigid_prop",
        "rigged": False,
        "note": "Offline synthetic unit box for scorer smoke tests.",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    asset = load_asset(str(glb_path))
    result = run_baseline(asset, asset_id="unit-box")
    report_path = out_dir / "baseline_result.json"
    report_path.write_text(result.to_json() + "\n", encoding="utf-8")
    print(f"wrote {glb_path}")
    print(f"baseline quality={result.metrics.quality:.4f}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
