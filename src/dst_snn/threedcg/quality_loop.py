"""Quality closed-loop training: scorer quality + soft Chamfer proxies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence
import json

import numpy as np
import trimesh

from benchmarks.threedcg.asset import Asset, asset_from_trimesh
from benchmarks.threedcg.scorer import score_to_result
from src.dst_snn.threedcg.dataset import ID_TO_SHAPE, make_batch
from src.dst_snn.threedcg.ops import ops_to_asset
from src.dst_snn.threedcg.sequence import Track1SequenceHead, ids_to_program
from src.dst_snn.threedcg.track1_policy import Track1OpHead
from src.dst_snn.threedcg.track2_occupancy import Track2OccupancyHead, occupancy_to_mesh
from src.dst_snn.threedcg.train import TrainResult, _mean_pool


def score_quality(candidate: Asset, reference: Asset, *, asset_id: str = "ql") -> float:
    """Non-differentiable scorer composite quality in [0, 1]."""
    result = score_to_result(candidate, reference, asset_id=asset_id)
    return float(result.metrics.quality)


def _sample_asset_points(asset: Asset, n: int = 256, seed: int = 0) -> np.ndarray:
    mesh = trimesh.Trimesh(vertices=asset.vertices, faces=asset.faces, process=False)
    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        return np.zeros((n, 3), dtype=np.float64)
    try:
        pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
        return np.asarray(pts, dtype=np.float64)
    except Exception:
        v = np.asarray(mesh.vertices, dtype=np.float64)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(v), size=n)
        return v[idx]


def soft_chamfer_torch(pred: "object", ref: "object") -> "object":
    """Differentiable bidirectional mean-min distance (soft Chamfer)."""
    import torch

    # pred, ref: [N,3], [M,3]
    d = torch.cdist(pred, ref)  # [N,M]
    ab = d.min(dim=1).values.mean()
    ba = d.min(dim=0).values.mean()
    return 0.5 * (ab + ba)


def sample_primitive_points_torch(
    shape_id: int,
    extents: "object",
    *,
    n: int = 256,
    seed: int = 0,
) -> "object":
    """Sample surface-ish points on box/sphere/cylinder from extents (diff through extents)."""
    import torch

    torch.manual_seed(seed)
    ex = extents.clamp(min=0.05)
    # unit samples then scale
    if shape_id == 1:  # sphere
        # gaussian normalize
        g = torch.randn(n, 3)
        g = g / (g.norm(dim=1, keepdim=True) + 1e-6)
        r = 0.5 * ex.max()
        return g * r
    if shape_id == 2:  # cylinder along Y
        theta = torch.rand(n) * 2 * 3.14159265
        y = (torch.rand(n) - 0.5) * ex[1]
        rad = 0.5 * torch.max(ex[0], ex[2])
        x = rad * torch.cos(theta)
        z = rad * torch.sin(theta)
        return torch.stack([x, y, z], dim=1)
    # box: faces — sample each face roughly
    u = torch.rand(n, 3) - 0.5
    # project to surface by fixing one axis to ±0.5
    axis = torch.randint(0, 3, (n,))
    sign = torch.where(torch.rand(n) > 0.5, 1.0, -1.0)
    pts = u.clone()
    for a in range(3):
        mask = axis == a
        pts[mask, a] = 0.5 * sign[mask]
    return pts * ex.unsqueeze(0)


def _load_training_batch(
    n_samples: int,
    *,
    seed: int,
    time_bins: int,
    image_size: int,
    resolution: int,
    corpus_root: Path | str | None,
    mix_synthetic: float,
):
    """Prefer external corpus when available; otherwise synthetic families."""
    if corpus_root is not None:
        from src.dst_snn.threedcg.corpus import MeshCorpus

        corpus = MeshCorpus.open(corpus_root)
        if len(corpus) > 0:
            return corpus.make_batch(
                n_samples,
                seed=seed,
                time_bins=time_bins,
                image_size=image_size,
                resolution=resolution,
                mix_synthetic=mix_synthetic,
                synthetic_diverse=True,
            )
    return make_batch(
        n_samples,
        seed=seed,
        time_bins=time_bins,
        image_size=image_size,
        resolution=resolution,
        diverse=True,
    )


def train_track1_quality(
    *,
    epochs: int = 30,
    n_samples: int = 48,
    seed: int = 0,
    lr: float = 5e-3,
    lambda_proxy: float = 0.5,
    lambda_rl: float = 0.3,
    supervised_weight: float = 0.5,
    out_path: Path | str = "artifacts/threedcg/checkpoints/track1_quality.pt",
    init_checkpoint: Optional[Path | str] = None,
    corpus_root: Path | str | None = None,
    mix_synthetic: float = 0.25,
) -> TrainResult:
    """Hybrid supervised + soft Chamfer + REINFORCE on shape class."""
    try:
        import torch
        from torch import nn
        from torch.distributions import Categorical
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyTorch required") from exc

    samples = _load_training_batch(
        n_samples,
        seed=seed,
        time_bins=6,
        image_size=24,
        resolution=6,
        corpus_root=corpus_root,
        mix_synthetic=mix_synthetic,
    )
    in_features = int(samples[0].spikes.shape[1])
    head = Track1OpHead(in_features, seed=seed)
    if init_checkpoint and Path(init_checkpoint).is_file():
        data = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        if int(data.get("in_features", -1)) == in_features:
            head.net.load_state_dict(data["state_dict"])
        else:
            print(
                f"skip init {init_checkpoint}: in_features "
                f"{data.get('in_features')} != {in_features}"
            )
    optimizer = torch.optim.Adam(head.net.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    history: list[float] = []
    quality_hist: list[float] = []
    family_quality: dict[str, list[float]] = {}
    baseline = 0.5

    # Precompute reference point clouds (fixed)
    ref_pts_np = [_sample_asset_points(s.asset, n=160, seed=i) for i, s in enumerate(samples)]

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        qs = []
        for i, s in enumerate(samples):
            x = _mean_pool(s.spikes, in_features)
            out = head.net(x)
            class_logits = out[:3]
            extents = torch.nn.functional.softplus(out[3:6]) + 0.2

            # Supervised
            loss_sup = ce(class_logits.unsqueeze(0), torch.tensor([s.shape_id]))
            loss_sup = loss_sup + 0.4 * nn.functional.mse_loss(
                extents, torch.tensor(s.extents, dtype=torch.float32)
            )

            # Differentiable proxy Chamfer (uses argmax class as non-diff id, extents diff)
            with torch.no_grad():
                cls = int(class_logits.argmax().item())
            pred_pts = sample_primitive_points_torch(cls, extents, n=128, seed=i)
            ref_t = torch.as_tensor(ref_pts_np[i], dtype=torch.float32)
            # normalize both for scale-invariance
            pred_c = pred_pts - pred_pts.mean(dim=0, keepdim=True)
            ref_c = ref_t - ref_t.mean(dim=0, keepdim=True)
            pred_c = pred_c / (pred_c.abs().max() + 1e-6)
            ref_c = ref_c / (ref_c.abs().max() + 1e-6)
            loss_proxy = soft_chamfer_torch(pred_c, ref_c)

            # REINFORCE on class with scorer quality
            dist = Categorical(logits=class_logits)
            sampled = dist.sample()
            logp = dist.log_prob(sampled)
            # build mesh for quality (numpy path)
            shape = ID_TO_SHAPE[int(sampled.item())]
            from src.dst_snn.threedcg.track1_policy import scripted_shape_policy
            from src.dst_snn.threedcg.ops import ops_to_asset as o2a

            with torch.no_grad():
                ex_list = extents.detach().cpu().tolist()
                ops = scripted_shape_policy(
                    s.spikes, shape=shape, extents_hint=ex_list  # type: ignore[arg-type]
                )
                cand = o2a(ops)
                q = score_quality(cand, s.asset)
            qs.append(q)
            fam = getattr(s, "family", ID_TO_SHAPE[s.shape_id])
            family_quality.setdefault(fam, []).append(q)
            advantage = q - baseline
            loss_rl = -logp * float(advantage)

            losses.append(
                supervised_weight * loss_sup
                + lambda_proxy * loss_proxy
                + lambda_rl * loss_rl
            )
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        mean_q = float(np.mean(qs)) if qs else 0.0
        baseline = 0.9 * baseline + 0.1 * mean_q
        history.append(float(loss.detach().item()))
        quality_hist.append(mean_q)

    fam_summary = {
        k: float(np.mean(v)) for k, v in sorted(family_quality.items()) if v
    }
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "track": "track1_quality",
            "in_features": in_features,
            "state_dict": head.net.state_dict(),
            "history": history,
            "quality_history": quality_hist,
            "family_quality": fam_summary,
        },
        path,
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "track": "track1_quality",
                "final_loss": history[-1] if history else None,
                "final_quality": quality_hist[-1] if quality_hist else None,
                "first_quality": quality_hist[0] if quality_hist else None,
                "family_quality": fam_summary,
                "n_families": len(fam_summary),
                "epochs": epochs,
                "checkpoint": str(path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return TrainResult(
        track="track1_quality",
        epochs=epochs,
        final_loss=history[-1] if history else float("inf"),
        history=history,
        checkpoint=str(path),
        in_features=in_features,
        extra={
            "first_loss": history[0] if history else None,
            "quality_history": quality_hist,
            "final_quality": quality_hist[-1] if quality_hist else None,
            "first_quality": quality_hist[0] if quality_hist else None,
            "family_quality": fam_summary,
        },
    )


def train_track1_sequence_quality(
    *,
    epochs: int = 25,
    n_samples: int = 32,
    seed: int = 0,
    lr: float = 5e-3,
    lambda_rl: float = 1.0,
    supervised_weight: float = 0.35,
    out_path: Path | str = "artifacts/threedcg/checkpoints/track1_seq_quality.pt",
    init_checkpoint: Optional[Path | str] = None,
    corpus_root: Path | str | None = None,
    mix_synthetic: float = 0.25,
) -> TrainResult:
    """REINFORCE on op sequences with scorer quality reward + light supervised CE."""
    try:
        import torch
        from torch import nn
        from torch.distributions import Categorical
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyTorch required") from exc

    from src.dst_snn.threedcg.sequence import SEQ_LEN, program_to_ids, template_program

    samples = _load_training_batch(
        n_samples,
        seed=seed,
        time_bins=6,
        image_size=24,
        resolution=6,
        corpus_root=corpus_root,
        mix_synthetic=mix_synthetic,
    )
    in_features = int(samples[0].spikes.shape[1])
    head = Track1SequenceHead(in_features, seq_len=SEQ_LEN, seed=seed)
    if init_checkpoint and Path(init_checkpoint).is_file():
        data = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        if int(data.get("in_features", -1)) == in_features:
            head.load_state_dict(data["state_dict"])
        else:
            print(
                f"skip init {init_checkpoint}: in_features "
                f"{data.get('in_features')} != {in_features}"
            )
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    history: list[float] = []
    quality_hist: list[float] = []
    family_quality: dict[str, list[float]] = {}
    baseline = 0.5

    teacher_ids = []
    for s in samples:
        shape = ID_TO_SHAPE[s.shape_id]
        fam = getattr(s, "family", shape)
        teacher_ids.append(
            program_to_ids(
                template_program(shape, s.extents, family=fam),
                seq_len=SEQ_LEN,
            )
        )

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        qs = []
        for s, teach in zip(samples, teacher_ids):
            x = _mean_pool(s.spikes, in_features)
            op_logits, extents = head.forward(x)
            # supervised teacher (family-specific multi-part recipes)
            loss_sup = ce(op_logits, torch.as_tensor(teach, dtype=torch.long))
            loss_sup = loss_sup + 0.3 * nn.functional.mse_loss(
                extents, torch.tensor(s.extents, dtype=torch.float32)
            )

            # sample program
            logps = []
            ids = []
            for t in range(head.seq_len):
                dist = Categorical(logits=op_logits[t])
                a = dist.sample()
                logps.append(dist.log_prob(a))
                ids.append(int(a.item()))
            with torch.no_grad():
                ex = extents.detach().cpu().tolist()
                ops = ids_to_program(ids, extents=ex)
                cand = ops_to_asset(ops)
                q = score_quality(cand, s.asset)
            qs.append(q)
            fam = getattr(s, "family", ID_TO_SHAPE[s.shape_id])
            family_quality.setdefault(fam, []).append(q)
            advantage = q - baseline
            loss_rl = -torch.stack(logps).sum() * float(advantage)
            losses.append(supervised_weight * loss_sup + lambda_rl * loss_rl)
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        mean_q = float(np.mean(qs)) if qs else 0.0
        baseline = 0.9 * baseline + 0.1 * mean_q
        history.append(float(loss.detach().item()))
        quality_hist.append(mean_q)

    fam_summary = {
        k: float(np.mean(v)) for k, v in sorted(family_quality.items()) if v
    }
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "track": "track1_seq_quality",
            "in_features": in_features,
            "seq_len": SEQ_LEN,
            "state_dict": head.state_dict(),
            "history": history,
            "quality_history": quality_hist,
            "family_quality": fam_summary,
        },
        path,
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "track": "track1_seq_quality",
                "final_loss": history[-1] if history else None,
                "final_quality": quality_hist[-1] if quality_hist else None,
                "first_quality": quality_hist[0] if quality_hist else None,
                "family_quality": fam_summary,
                "n_families": len(fam_summary),
                "checkpoint": str(path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return TrainResult(
        track="track1_seq_quality",
        epochs=epochs,
        final_loss=history[-1] if history else float("inf"),
        history=history,
        checkpoint=str(path),
        in_features=in_features,
        extra={
            "first_loss": history[0] if history else None,
            "quality_history": quality_hist,
            "final_quality": quality_hist[-1] if quality_hist else None,
            "first_quality": quality_hist[0] if quality_hist else None,
            "family_quality": fam_summary,
        },
    )


def train_track2_quality(
    *,
    epochs: int = 25,
    n_samples: int = 32,
    seed: int = 0,
    lr: float = 5e-3,
    lambda_quality: float = 0.4,
    out_path: Path | str = "artifacts/threedcg/checkpoints/track2_quality.pt",
    init_checkpoint: Optional[Path | str] = None,
    resolution: int = 6,
    corpus_root: Path | str | None = None,
    mix_synthetic: float = 0.25,
) -> TrainResult:
    """Occupancy BCE + quality-weighted occupancy matching (proxy).

    Scorer quality is measured each step and used to scale an extra push toward
    reference occupancy (still differentiable via occupancy BCE).
    """
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyTorch required") from exc

    samples = _load_training_batch(
        n_samples,
        seed=seed,
        time_bins=6,
        image_size=24,
        resolution=resolution,
        corpus_root=corpus_root,
        mix_synthetic=mix_synthetic,
    )
    in_features = int(samples[0].spikes.shape[1])
    head = Track2OccupancyHead(in_features, resolution=resolution, seed=seed)
    if init_checkpoint and Path(init_checkpoint).is_file():
        data = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        if int(data.get("in_features", -1)) == in_features and int(data.get("resolution", resolution)) == resolution:
            head.net.load_state_dict(data["state_dict"])
        else:
            print(
                f"skip init {init_checkpoint}: "
                f"in_features/resolution mismatch vs model"
            )
    optimizer = torch.optim.Adam(head.net.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    history: list[float] = []
    quality_hist: list[float] = []
    family_quality: dict[str, list[float]] = {}

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        qs = []
        for s in samples:
            x = _mean_pool(s.spikes, in_features)
            logits = head.forward_logits(x)
            target = torch.as_tensor(s.occupancy.reshape(-1), dtype=torch.float32)
            loss_bce = bce(logits, target)
            with torch.no_grad():
                probs = torch.sigmoid(logits).cpu().numpy().reshape(resolution, resolution, resolution)
                mesh = occupancy_to_mesh(probs > 0.35)
                cand = asset_from_trimesh(mesh)
                q = score_quality(cand, s.asset)
            qs.append(q)
            fam = getattr(s, "family", "unknown")
            family_quality.setdefault(fam, []).append(q)
            # quality-gated extra occupancy fit (higher quality → less pressure; low q → more)
            gate = float(max(0.1, 1.0 - q))
            losses.append(loss_bce * (1.0 + lambda_quality * gate))
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().item()))
        quality_hist.append(float(np.mean(qs)) if qs else 0.0)

    fam_summary = {
        k: float(np.mean(v)) for k, v in sorted(family_quality.items()) if v
    }
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "track": "track2_quality",
            "in_features": in_features,
            "resolution": resolution,
            "state_dict": head.net.state_dict(),
            "history": history,
            "quality_history": quality_hist,
            "family_quality": fam_summary,
        },
        path,
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "track": "track2_quality",
                "final_loss": history[-1] if history else None,
                "final_quality": quality_hist[-1] if quality_hist else None,
                "first_quality": quality_hist[0] if quality_hist else None,
                "family_quality": fam_summary,
                "n_families": len(fam_summary),
                "checkpoint": str(path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return TrainResult(
        track="track2_quality",
        epochs=epochs,
        final_loss=history[-1] if history else float("inf"),
        history=history,
        checkpoint=str(path),
        in_features=in_features,
        extra={
            "first_loss": history[0] if history else None,
            "quality_history": quality_hist,
            "final_quality": quality_hist[-1] if quality_hist else None,
            "first_quality": quality_hist[0] if quality_hist else None,
            "resolution": resolution,
            "family_quality": fam_summary,
        },
    )
