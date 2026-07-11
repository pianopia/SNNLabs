"""External mesh corpus import + sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from src.dst_snn.threedcg.corpus import (
    MeshCorpus,
    import_directory,
    import_mesh_file,
    rebuild_catalog,
    scan_corpus_layout,
)


def _write_box_glb(path: Path, extents=(1.0, 1.2, 0.8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.creation.box(extents=list(extents)).export(path)


def test_import_and_sample(tmp_path: Path):
    src = tmp_path / "external"
    src.mkdir()
    _write_box_glb(src / "crate.glb")
    _write_box_glb(src / "pillar.obj")  # obj via trimesh

    dest = tmp_path / "corpus"
    entries = import_directory(src, dest, max_assets=10, license="test")
    assert len(entries) >= 1
    assert (dest / entries[0].asset_id / "reference.glb").is_file()
    assert (dest / entries[0].asset_id / "meta.json").is_file()
    assert (dest / entries[0].asset_id / "input.png").is_file()

    catalog = rebuild_catalog(dest)
    assert catalog.is_file()

    corpus = MeshCorpus.open(dest)
    assert len(corpus) >= 1
    batch = corpus.make_batch(4, seed=0, mix_synthetic=0.5)
    assert len(batch) == 4
    assert all(s.spikes.ndim == 2 for s in batch)
    assert all(len(s.asset.vertices) > 0 for s in batch)


def test_scan_existing_layout(tmp_path: Path):
    root = tmp_path / "data" / "threedcg"
    asset = root / "demo-prop"
    _write_box_glb(asset / "reference.glb")
    (asset / "meta.json").write_text(
        '{"license":"test","category":"hard_surface","family":"box"}\n',
        encoding="utf-8",
    )
    entries = scan_corpus_layout(root)
    assert len(entries) == 1
    assert entries[0].family == "box"
    sample = MeshCorpus.open(root).make_batch(1, seed=1, mix_synthetic=0.0)[0]
    assert sample.family == "box"
    assert sample.shape_id == 0


def test_import_single_overwrite(tmp_path: Path):
    src = tmp_path / "a.glb"
    _write_box_glb(src, extents=(0.5, 0.5, 0.5))
    dest = tmp_path / "corpus"
    e1 = import_mesh_file(src, dest, asset_id="one", overwrite=True)
    e2 = import_mesh_file(src, dest, asset_id="one", overwrite=False)
    assert e1.asset_id == e2.asset_id == "one"
    assert e1.reference_path == e2.reference_path
