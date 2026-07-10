"""Shared spatial MAC estimators for fair SNN vs CNN energy proxies.

Counts dense multiply-accumulates for 3×3 conv stages with strides (1, 2, 2)
plus an optional final linear classifier. Used by Frame-CNN baselines and as
the dense MAC side of Conv-PLIF energy comparisons.
"""

from __future__ import annotations


def conv2d_mac_ops(in_c: int, out_c: int, kernel: int, h_out: int, w_out: int) -> float:
    if min(in_c, out_c, kernel, h_out, w_out) < 0:
        raise ValueError("sizes must be non-negative")
    return float(in_c * out_c * kernel * kernel * h_out * w_out)


def estimate_three_stage_conv_macs(
    *,
    in_channels: int,
    channels: tuple[int, int, int],
    height: int,
    width: int,
    time_bins: int,
    num_classes: int = 0,
) -> float:
    """MACs for three 3×3 convs (strides 1,2,2) evaluated each time bin + FC."""
    if time_bins < 0 or height < 0 or width < 0:
        raise ValueError("time_bins/height/width must be non-negative")
    c1, c2, c3 = channels
    h1, w1 = max(1, height), max(1, width)
    h2, w2 = max(1, height // 2), max(1, width // 2)
    h3, w3 = max(1, height // 4), max(1, width // 4)
    per_step = (
        conv2d_mac_ops(in_channels, c1, 3, h1, w1)
        + conv2d_mac_ops(c1, c2, 3, h2, w2)
        + conv2d_mac_ops(c2, c3, 3, h3, w3)
        + float(max(0, c3) * max(0, num_classes))
    )
    return per_step * float(time_bins)


def estimate_sew_macs(
    *,
    in_channels: int,
    width: int,
    blocks_per_stage: int,
    height: int,
    width_px: int,
    time_bins: int,
    num_classes: int = 0,
) -> float:
    """Order-of-magnitude MACs matching ``SewConvPLIFClassifier`` topology.

    stem 3×3 s1 → stage1 residuals at ``width`` → stage2 (first s2) → expand
    to 2×width s2 → stage3 residuals → FC.
    """
    if blocks_per_stage < 1:
        raise ValueError("blocks_per_stage must be >= 1")
    h, w = max(1, height), max(1, width_px)
    h2, w2 = max(1, h // 2), max(1, w // 2)
    h4, w4 = max(1, h // 4), max(1, w // 4)
    # stem
    per = conv2d_mac_ops(in_channels, width, 3, h, w)
    # stage1: each residual ≈ two 3×3 at full res
    per += 2 * blocks_per_stage * conv2d_mac_ops(width, width, 3, h, w)
    # stage2 first block stride 2 + remaining blocks
    per += 2 * conv2d_mac_ops(width, width, 3, h2, w2)  # first residual pair (s2 then s1)
    if blocks_per_stage > 1:
        per += 2 * (blocks_per_stage - 1) * conv2d_mac_ops(width, width, 3, h2, w2)
    # expand  width -> 2*width stride 2
    per += conv2d_mac_ops(width, width * 2, 3, h4, w4)
    # stage3 residuals at 2*width
    per += 2 * blocks_per_stage * conv2d_mac_ops(width * 2, width * 2, 3, h4, w4)
    per += float(width * 2 * max(0, num_classes))
    return per * float(time_bins)
