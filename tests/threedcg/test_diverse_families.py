"""Diverse commercial-ish families for 3DCG quality training."""

from __future__ import annotations

from src.dst_snn.threedcg.dataset import FAMILIES, make_batch, make_sample
from src.dst_snn.threedcg.ops import ops_to_asset
from src.dst_snn.threedcg.sequence import template_program
from src.dst_snn.threedcg.quality_loop import score_quality


def test_all_families_produce_meshes():
    for fam in FAMILIES:
        s = make_sample(family=fam, extents=(1.0, 1.2, 0.9), seed=0)
        assert s.family == fam
        assert len(s.asset.vertices) > 0
        assert len(s.asset.faces) > 0
        assert s.image.shape[-1] == 3
        assert s.spikes.ndim == 2


def test_diverse_batch_covers_many_families():
    batch = make_batch(24, seed=1, diverse=True)
    fams = {s.family for s in batch}
    assert len(fams) >= 8


def test_family_templates_score_against_self():
    for fam in ("body", "l_beam", "arch", "pillar", "capsule"):
        s = make_sample(family=fam, extents=(1.0, 1.4, 0.9), seed=2)
        ops = template_program("box", s.extents, family=fam)
        cand = ops_to_asset(ops)
        q = score_quality(cand, s.asset, asset_id=fam)
        # teacher vs reference mesh of same family should be non-trivial quality
        assert q > 0.25
        assert len(cand.vertices) > 0
