from __future__ import annotations

import numpy as np

from src.dst_snn.threedcg.ops import OP_ADD_BOX, OP_ADD_SPHERE
from src.dst_snn.threedcg.track1_policy import (
    Track1OpHead,
    decode_ops_from_spikes,
    scripted_box_policy,
)


def test_scripted_box_policy_uses_hint():
    spikes = np.zeros((4, 16), dtype=np.float32)
    ops = scripted_box_policy(spikes, extents_hint=[2.0, 1.0, 0.5])
    assert ops[0].name == OP_ADD_BOX
    assert ops[0].params["extents"] == [2.0, 1.0, 0.5]


def test_decode_scripted_shapes():
    spikes = np.ones((4, 8), dtype=np.float32) * 0.5
    ops = decode_ops_from_spikes(spikes, mode="scripted", shape="sphere")
    assert ops[0].name == OP_ADD_SPHERE


def test_torch_head_smoke():
    spikes = np.random.default_rng(0).random((4, 16)).astype(np.float32)
    head = Track1OpHead(16, seed=0)
    ops = head.decode(spikes)
    assert ops[0].name in {OP_ADD_BOX, OP_ADD_SPHERE, "ADD_CYLINDER"}
