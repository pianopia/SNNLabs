from __future__ import annotations

from src.dst_snn.eval.baselines.frame_cnn import FrameCnnClassifier
from src.dst_snn.eval.baselines.spatial_ops import (
    conv2d_mac_ops,
    estimate_sew_macs,
    estimate_three_stage_conv_macs,
)


def test_conv2d_mac_ops():
    assert conv2d_mac_ops(2, 4, 3, 8, 8) == 2 * 4 * 9 * 8 * 8


def test_three_stage_scales_with_time():
    one = estimate_three_stage_conv_macs(
        in_channels=2, channels=(8, 16, 16), height=16, width=16, time_bins=1, num_classes=11
    )
    four = estimate_three_stage_conv_macs(
        in_channels=2, channels=(8, 16, 16), height=16, width=16, time_bins=4, num_classes=11
    )
    assert four == 4 * one


def test_frame_cnn_matches_helper():
    model = FrameCnnClassifier(2, 11, channels=(8, 16, 16))
    assert model.mac_ops_per_inference(4, 16, 16) == estimate_three_stage_conv_macs(
        in_channels=2, channels=(8, 16, 16), height=16, width=16, time_bins=4, num_classes=11
    )


def test_sew_macs_positive_and_scales():
    a = estimate_sew_macs(
        in_channels=2, width=8, blocks_per_stage=1, height=16, width_px=16, time_bins=1, num_classes=11
    )
    b = estimate_sew_macs(
        in_channels=2, width=8, blocks_per_stage=1, height=16, width_px=16, time_bins=2, num_classes=11
    )
    assert a > 0
    assert b == 2 * a
