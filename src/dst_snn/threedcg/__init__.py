"""SNN image→3D construction (Track 1 mesh ops + Track 2 occupancy)."""

from .bpy_adapter import bpy_available
from .image_spikes import image_to_spikes, load_image_array, spike_feature_size
from .mesh_backend import backend_info, execute_ops_backend, resolve_backend
from .ops import MeshOp, VOCABULARY, execute_ops, execute_ops_with_state, ops_to_asset
from .pipeline import generate_from_image, run_pipeline_score
from .sdf import Track2SdfHead, mesh_to_sdf, sdf_to_mesh
from .sequence import Track1SequenceHead, template_program
from .track1_policy import Track1OpHead, decode_ops_from_spikes, scripted_box_policy
from .track2_occupancy import (
    Track2OccupancyHead,
    occupancy_to_mesh,
    spikes_to_occupancy,
    track2_from_spikes,
)
from .train import (
    load_track1_head,
    load_track1_sequence_head,
    load_track2_head,
    load_track2_sdf_head,
    train_track1,
    train_track1_sequence,
    train_track2,
    train_track2_sdf,
)

__all__ = [
    "MeshOp",
    "Track1OpHead",
    "Track1SequenceHead",
    "Track2OccupancyHead",
    "Track2SdfHead",
    "VOCABULARY",
    "backend_info",
    "bpy_available",
    "decode_ops_from_spikes",
    "execute_ops",
    "execute_ops_backend",
    "execute_ops_with_state",
    "generate_from_image",
    "image_to_spikes",
    "load_image_array",
    "load_track1_head",
    "load_track1_sequence_head",
    "load_track2_head",
    "load_track2_sdf_head",
    "mesh_to_sdf",
    "occupancy_to_mesh",
    "ops_to_asset",
    "resolve_backend",
    "run_pipeline_score",
    "scripted_box_policy",
    "sdf_to_mesh",
    "spike_feature_size",
    "spikes_to_occupancy",
    "template_program",
    "track2_from_spikes",
    "train_track1",
    "train_track1_sequence",
    "train_track2",
    "train_track2_sdf",
]
