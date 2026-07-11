import numpy as np

from src.dst_snn.foundation.events import (
    ModalitySpec,
    SignedEventEncoder,
    align_and_fuse_modalities,
    compressed_time_steps,
    temporal_compress,
)


def test_signed_encoder_preserves_polarity_and_silences_static_input():
    encoder = SignedEventEncoder(2, threshold=0.25, max_event_level=3)
    events = encoder.encode(np.array([[0.75, -0.5], [0.75, -0.5]], dtype=np.float32))
    assert events[0].tolist() == [3, -2]
    assert events[1].tolist() == [0, 0]


def test_temporal_compression_preserves_signed_sum_without_clipping():
    source = np.array([[1, -1], [1, 0], [0, -1], [-1, 1]], dtype=np.int8)
    compressed = temporal_compress(source, 2, max_event_level=7)
    assert compressed.shape == (2, 2)
    np.testing.assert_array_equal(compressed.sum(axis=0), source.sum(axis=0))
    assert compressed_time_steps(16) == 4


def test_modalities_use_different_temporal_scales_and_keep_slices():
    streams = {
        "text": np.ones((8, 2), dtype=np.int8),
        "image": np.ones((8, 3), dtype=np.int8),
    }
    specs = {
        "text": ModalitySpec("text", time_steps=4),
        "image": ModalitySpec("image", time_steps=2),
    }
    fused, slices = align_and_fuse_modalities(streams, specs=specs)
    assert fused.shape == (4, 5)
    assert slices == {"text": slice(0, 2), "image": slice(2, 5)}
    assert np.all(fused[2:, slices["image"]] == 0)
