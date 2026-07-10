from __future__ import annotations

from src.dst_snn.web_autonomous_learner import compute_novelty


def test_all_novel_when_nothing_known():
    active = [{"token": "text:a"}, {"token": "text:b"}]
    assert compute_novelty(active, known_before=set()) == 1.0


def test_none_novel_when_all_known():
    active = [{"token": "text:a"}, {"token": "text:b"}]
    assert compute_novelty(active, known_before={"text:a", "text:b"}) == 0.0


def test_half_novel():
    active = [{"token": "text:a"}, {"token": "text:b"}]
    assert compute_novelty(active, known_before={"text:a"}) == 0.5


def test_empty_active_is_zero():
    assert compute_novelty([], known_before={"text:a"}) == 0.0
