"""Multi-op sequence templates and Track1 sequence head."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from src.dst_snn.threedcg.ops import (
    OP_ADD_ARMATURE,
    OP_ADD_BOX,
    OP_ADD_CYLINDER,
    OP_ADD_SPHERE,
    OP_ASSIGN_MATERIAL,
    OP_AUTO_WEIGHTS,
    OP_BEVEL,
    OP_EXTRUDE,
    OP_FINISH,
    OP_SMART_UV,
    OP_SUBDIVIDE,
    SEQUENCE_VOCAB,
    MeshOp,
)

SEQ_TO_ID = {name: i for i, name in enumerate(SEQUENCE_VOCAB)}
ID_TO_SEQ = list(SEQUENCE_VOCAB)
SEQ_LEN = 8


def template_program(
    shape: str,
    extents: Sequence[float],
    *,
    family: str | None = None,
) -> List[MeshOp]:
    """Teacher program for supervised sequence learning.

    ``family`` selects commercial-ish multi-part recipes when provided;
    otherwise falls back to primitive ``shape`` (box/sphere/cylinder).
    """
    ex = [float(max(0.3, e)) for e in extents[:3]]
    while len(ex) < 3:
        ex.append(ex[-1])
    key = (family or shape or "box").lower()

    if key == "sphere":
        ops = [
            MeshOp(OP_ADD_SPHERE, {"radius": 0.5 * max(ex)}),
            MeshOp(OP_SUBDIVIDE, {"cuts": 1}),
            MeshOp(OP_BEVEL, {"offset": 0.02}),
            MeshOp(OP_SMART_UV, {}),
            MeshOp(OP_ASSIGN_MATERIAL, {"name": "Body", "albedo": True}),
            MeshOp(OP_FINISH, {}),
        ]
    elif key == "cylinder":
        ops = [
            MeshOp(OP_ADD_CYLINDER, {"radius": 0.5 * max(ex[0], ex[2]), "height": ex[1]}),
            MeshOp(OP_EXTRUDE, {"distance": 0.15, "axis": 1}),
            MeshOp(OP_BEVEL, {"offset": 0.03}),
            MeshOp(OP_SMART_UV, {}),
            MeshOp(OP_ASSIGN_MATERIAL, {"name": "Body"}),
            MeshOp(OP_ADD_ARMATURE, {"bones": 3}),
            MeshOp(OP_AUTO_WEIGHTS, {}),
            MeshOp(OP_FINISH, {}),
        ]
    elif key == "capsule":
        r = 0.35 * max(ex[0], ex[2])
        ops = [
            MeshOp(OP_ADD_CYLINDER, {"radius": r, "height": max(ex[1] * 0.7, 0.3)}),
            MeshOp(OP_ADD_SPHERE, {"radius": r, "center": [0.0, ex[1] * 0.4, 0.0]}),
            MeshOp(OP_SUBDIVIDE, {"cuts": 1}),
            MeshOp(OP_SMART_UV, {}),
            MeshOp(OP_ASSIGN_MATERIAL, {"name": "Capsule", "albedo": True}),
            MeshOp(OP_ADD_ARMATURE, {"bones": 3}),
            MeshOp(OP_AUTO_WEIGHTS, {}),
            MeshOp(OP_FINISH, {}),
        ]
    elif key == "l_beam":
        ops = [
            MeshOp(OP_ADD_BOX, {"extents": [ex[0], ex[1] * 0.35, ex[2] * 0.45]}),
            MeshOp(
                OP_ADD_BOX,
                {
                    "extents": [ex[0] * 0.35, ex[1], ex[2] * 0.45],
                    "center": [ex[0] * 0.25, ex[1] * 0.2, 0.0],
                },
            ),
            MeshOp(OP_BEVEL, {"offset": 0.03}),
            MeshOp(OP_SMART_UV, {}),
            MeshOp(OP_ASSIGN_MATERIAL, {"name": "Metal", "roughness": True, "metallic": True}),
            MeshOp(OP_FINISH, {}),
        ]
    elif key == "t_joint":
        ops = [
            MeshOp(OP_ADD_BOX, {"extents": [ex[0] * 0.35, ex[1], ex[2] * 0.4]}),
            MeshOp(
                OP_ADD_BOX,
                {
                    "extents": [ex[0], ex[1] * 0.3, ex[2] * 0.4],
                    "center": [0.0, ex[1] * 0.4, 0.0],
                },
            ),
            MeshOp(OP_BEVEL, {"offset": 0.025}),
            MeshOp(OP_SMART_UV, {}),
            MeshOp(OP_ASSIGN_MATERIAL, {"name": "Joint"}),
            MeshOp(OP_FINISH, {}),
        ]
    elif key == "arch":
        r = 0.18 * max(ex[0], ex[2])
        ops = [
            MeshOp(
                OP_ADD_CYLINDER,
                {"radius": r, "height": ex[1] * 0.85, "center": [-ex[0] * 0.3, 0.0, 0.0]},
            ),
            MeshOp(
                OP_ADD_CYLINDER,
                {"radius": r, "height": ex[1] * 0.85, "center": [ex[0] * 0.3, 0.0, 0.0]},
            ),
            MeshOp(
                OP_ADD_BOX,
                {
                    "extents": [ex[0] * 0.85, ex[1] * 0.2, ex[2] * 0.35],
                    "center": [0.0, ex[1] * 0.45, 0.0],
                },
            ),
            MeshOp(OP_SMART_UV, {}),
            MeshOp(OP_ASSIGN_MATERIAL, {"name": "Stone", "roughness": True}),
            MeshOp(OP_FINISH, {}),
        ]
    elif key == "body":
        ops = [
            MeshOp(OP_ADD_CYLINDER, {"radius": 0.28 * max(ex[0], ex[2]), "height": ex[1] * 0.55}),
            MeshOp(
                OP_ADD_SPHERE,
                {"radius": 0.22 * max(ex[0], ex[2]), "center": [0.0, ex[1] * 0.42, 0.0]},
            ),
            MeshOp(OP_SUBDIVIDE, {"cuts": 1}),
            MeshOp(OP_SMART_UV, {}),
            MeshOp(OP_ASSIGN_MATERIAL, {"name": "Skin", "albedo": True}),
            MeshOp(OP_ADD_ARMATURE, {"bones": 4}),
            MeshOp(OP_AUTO_WEIGHTS, {}),
            MeshOp(OP_FINISH, {}),
        ]
    elif key == "platform":
        ops = [
            MeshOp(OP_ADD_BOX, {"extents": [ex[0] * 1.2, ex[1] * 0.18, ex[2] * 1.2]}),
            MeshOp(
                OP_ADD_CYLINDER,
                {
                    "radius": 0.12 * max(ex[0], ex[2]),
                    "height": ex[1] * 0.7,
                    "center": [0.0, ex[1] * 0.35, 0.0],
                },
            ),
            MeshOp(
                OP_ADD_BOX,
                {
                    "extents": [ex[0] * 0.5, ex[1] * 0.12, ex[2] * 0.5],
                    "center": [0.0, ex[1] * 0.65, 0.0],
                },
            ),
            MeshOp(OP_SMART_UV, {}),
            MeshOp(OP_ASSIGN_MATERIAL, {"name": "Prop"}),
            MeshOp(OP_FINISH, {}),
        ]
    elif key == "pillar":
        ops = [
            MeshOp(OP_ADD_CYLINDER, {"radius": 0.22 * max(ex[0], ex[2]), "height": ex[1]}),
            MeshOp(
                OP_ADD_BOX,
                {
                    "extents": [ex[0] * 0.7, ex[1] * 0.12, ex[2] * 0.7],
                    "center": [0.0, ex[1] * 0.5, 0.0],
                },
            ),
            MeshOp(OP_BEVEL, {"offset": 0.03}),
            MeshOp(OP_SMART_UV, {}),
            MeshOp(OP_ASSIGN_MATERIAL, {"name": "Column", "roughness": True}),
            MeshOp(OP_ADD_ARMATURE, {"bones": 3}),
            MeshOp(OP_AUTO_WEIGHTS, {}),
            MeshOp(OP_FINISH, {}),
        ]
    elif key == "wedge":
        ops = [
            MeshOp(OP_ADD_BOX, {"extents": [ex[0], ex[1] * 0.5, ex[2]]}),
            MeshOp(
                OP_ADD_BOX,
                {
                    "extents": [ex[0] * 0.55, ex[1] * 0.55, ex[2] * 0.55],
                    "center": [ex[0] * 0.15, ex[1] * 0.25, 0.0],
                },
            ),
            MeshOp(OP_EXTRUDE, {"distance": 0.12, "axis": 2}),
            MeshOp(OP_BEVEL, {"offset": 0.03}),
            MeshOp(OP_SMART_UV, {}),
            MeshOp(OP_ASSIGN_MATERIAL, {"name": "Wedge"}),
            MeshOp(OP_FINISH, {}),
        ]
    elif key == "ring":
        ops = [
            MeshOp(OP_ADD_CYLINDER, {"radius": 0.45 * max(ex[0], ex[2]), "height": ex[1] * 0.35}),
            MeshOp(OP_SUBDIVIDE, {"cuts": 1}),
            MeshOp(OP_BEVEL, {"offset": 0.04}),
            MeshOp(OP_SMART_UV, {}),
            MeshOp(OP_ASSIGN_MATERIAL, {"name": "Ring", "metallic": True}),
            MeshOp(OP_FINISH, {}),
        ]
    else:
        # default box / refined crate
        ops = [
            MeshOp(OP_ADD_BOX, {"extents": ex}),
            MeshOp(OP_EXTRUDE, {"distance": 0.2, "axis": 2}),
            MeshOp(OP_SUBDIVIDE, {"cuts": 1}),
            MeshOp(OP_BEVEL, {"offset": 0.04}),
            MeshOp(OP_SMART_UV, {}),
            MeshOp(OP_ASSIGN_MATERIAL, {"name": "Body", "roughness": True}),
            MeshOp(OP_ADD_ARMATURE, {"bones": 3}),
            MeshOp(OP_AUTO_WEIGHTS, {}),
            MeshOp(OP_FINISH, {}),
        ]
    return ops


def program_to_ids(ops: Sequence[MeshOp], *, seq_len: int = SEQ_LEN) -> np.ndarray:
    ids = []
    for op in ops:
        if op.name in SEQ_TO_ID:
            ids.append(SEQ_TO_ID[op.name])
        if op.name == OP_FINISH:
            break
    # pad with FINISH
    fin = SEQ_TO_ID[OP_FINISH]
    while len(ids) < seq_len:
        ids.append(fin)
    return np.asarray(ids[:seq_len], dtype=np.int64)


def ids_to_program(ids: Sequence[int], *, extents: Sequence[float] | None = None) -> List[MeshOp]:
    """Decode discrete op ids into MeshOps.

    Multiple primitive ADDs get progressive centers so multi-part assemblies
    (body/L/arch) are expressible without TRANSLATE in the sequence vocab.
    """
    ex = [float(e) for e in (extents or (1.0, 1.0, 1.0))]
    while len(ex) < 3:
        ex.append(1.0)
    ops: List[MeshOp] = []
    prim_i = 0
    # progressive offsets for 2nd/3rd primitive (character / structure recipes)
    prim_centers = [
        None,
        [0.0, 0.45 * ex[1], 0.0],
        [0.35 * ex[0], 0.1 * ex[1], 0.0],
        [-0.3 * ex[0], 0.0, 0.0],
    ]

    def _center_for(i: int):
        if i < len(prim_centers):
            return prim_centers[i]
        return [0.25 * ex[0] * ((i % 2) * 2 - 1), 0.15 * ex[1] * i, 0.0]

    for i in ids:
        name = ID_TO_SEQ[int(i) % len(ID_TO_SEQ)]
        if name == OP_ADD_BOX:
            params: dict = {"extents": list(ex)}
            c = _center_for(prim_i)
            if c is not None:
                params["center"] = c
                params["extents"] = [ex[0] * (0.7 if prim_i else 1.0), ex[1] * (0.5 if prim_i else 1.0), ex[2] * 0.8]
            ops.append(MeshOp(name, params))
            prim_i += 1
        elif name == OP_ADD_SPHERE:
            params = {"radius": 0.5 * max(ex) * (0.7 if prim_i else 1.0)}
            c = _center_for(prim_i)
            if c is not None:
                params["center"] = c
            ops.append(MeshOp(name, params))
            prim_i += 1
        elif name == OP_ADD_CYLINDER:
            params = {
                "radius": 0.5 * max(ex[0], ex[2]) * (0.75 if prim_i else 1.0),
                "height": ex[1] * (0.7 if prim_i else 1.0),
            }
            c = _center_for(prim_i)
            if c is not None:
                params["center"] = c
            ops.append(MeshOp(name, params))
            prim_i += 1
        elif name == OP_EXTRUDE:
            ops.append(MeshOp(name, {"distance": 0.2, "axis": 2 if prim_i == 0 else 1}))
        elif name == OP_SUBDIVIDE:
            ops.append(MeshOp(name, {"cuts": 1}))
        elif name == OP_BEVEL:
            ops.append(MeshOp(name, {"offset": 0.04}))
        elif name == OP_SMART_UV:
            ops.append(MeshOp(name, {}))
        elif name == OP_ASSIGN_MATERIAL:
            ops.append(MeshOp(name, {"name": "Body", "albedo": True, "roughness": True}))
        elif name == OP_ADD_ARMATURE:
            ops.append(MeshOp(name, {"bones": 3 + min(2, prim_i)}))
        elif name == OP_AUTO_WEIGHTS:
            ops.append(MeshOp(name, {}))
        elif name == OP_FINISH:
            ops.append(MeshOp(name, {}))
            break
        else:
            ops.append(MeshOp(OP_FINISH, {}))
            break
    if not ops or ops[-1].name != OP_FINISH:
        ops.append(MeshOp(OP_FINISH, {}))
    return ops


class Track1SequenceHead:
    """Predict a short op program + extents from mean-pooled spikes."""

    def __init__(self, in_features: int, *, seq_len: int = SEQ_LEN, seed: int = 0) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install PyTorch for Track1SequenceHead.") from exc

        self._torch = torch
        self.in_features = int(in_features)
        self.seq_len = int(seq_len)
        self.n_ops = len(SEQUENCE_VOCAB)
        g = torch.Generator().manual_seed(seed)
        self.backbone = nn.Sequential(
            nn.Linear(self.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.op_head = nn.Linear(128, self.seq_len * self.n_ops)
        self.extent_head = nn.Linear(128, 3)
        with torch.no_grad():
            for p in list(self.backbone.parameters()) + list(self.op_head.parameters()) + list(
                self.extent_head.parameters()
            ):
                p.normal_(0.0, 0.02, generator=g)

    def parameters(self):
        return (
            list(self.backbone.parameters())
            + list(self.op_head.parameters())
            + list(self.extent_head.parameters())
        )

    def _features(self, spikes: np.ndarray):
        torch = self._torch
        x = torch.as_tensor(np.asarray(spikes, dtype=np.float32).mean(axis=0), dtype=torch.float32)
        if x.numel() != self.in_features:
            buf = torch.zeros(self.in_features, dtype=torch.float32)
            n = min(self.in_features, int(x.numel()))
            buf[:n] = x.reshape(-1)[:n]
            return buf
        return x

    def forward(self, features):
        h = self.backbone(features)
        op_logits = self.op_head(h).view(self.seq_len, self.n_ops)
        extents = self._torch.nn.functional.softplus(self.extent_head(h)) + 0.2
        return op_logits, extents

    def decode(self, spikes: np.ndarray) -> List[MeshOp]:
        torch = self._torch
        x = self._features(spikes)
        with torch.no_grad():
            op_logits, extents = self.forward(x)
            ids = op_logits.argmax(dim=-1).cpu().tolist()
            ex = extents.cpu().tolist()
        return ids_to_program(ids, extents=ex)

    def state_dict(self):
        return {
            "backbone": self.backbone.state_dict(),
            "op_head": self.op_head.state_dict(),
            "extent_head": self.extent_head.state_dict(),
            "in_features": self.in_features,
            "seq_len": self.seq_len,
        }

    def load_state_dict(self, data: dict) -> None:
        self.backbone.load_state_dict(data["backbone"])
        self.op_head.load_state_dict(data["op_head"])
        self.extent_head.load_state_dict(data["extent_head"])
