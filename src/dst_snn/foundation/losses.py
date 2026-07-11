"""Multi-objective losses for accurate, sparse and early-exiting SNNs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class FoundationLossWeights:
    task: float = 1.0
    distillation: float = 1.0
    feature_alignment: float = 1.0
    spike_budget: float = 0.05
    early_exit: float = 0.25
    temperature: float = 2.0
    target_spike_rate: float = 0.2


@dataclass
class FoundationLossOutput:
    total: Tensor
    task: Tensor
    distillation: Tensor
    feature_alignment: Tensor
    spike_budget: Tensor
    early_exit: Tensor

    def detached(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach().item())
            for name in (
                "total",
                "task",
                "distillation",
                "feature_alignment",
                "spike_budget",
                "early_exit",
            )
        }


def foundation_loss(
    student_logits: Tensor,
    targets: Tensor,
    *,
    teacher_logits: Tensor | None = None,
    student_features: Sequence[Tensor] = (),
    teacher_features: Sequence[Tensor] = (),
    event_tensors: Sequence[Tensor] = (),
    early_exit_logits: Sequence[Tensor] = (),
    weights: FoundationLossWeights | None = None,
) -> FoundationLossOutput:
    cfg = weights or FoundationLossWeights()
    task = F.cross_entropy(student_logits, targets)
    zero = student_logits.new_zeros(())
    distillation = zero
    if teacher_logits is not None:
        temperature = max(cfg.temperature, 1e-6)
        distillation = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits.detach() / temperature, dim=-1),
            reduction="batchmean",
        ) * temperature * temperature

    aligned = []
    for student, teacher in zip(student_features, teacher_features):
        if student.shape != teacher.shape:
            raise ValueError("aligned teacher/student features must share shape")
        aligned.append(1.0 - F.cosine_similarity(student, teacher.detach(), dim=-1).mean())
    feature_alignment = torch.stack(aligned).mean() if aligned else zero

    rates = [events.ne(0).to(student_logits.dtype).mean() for events in event_tensors]
    spike_rate = torch.stack(rates).mean() if rates else zero
    spike_budget = F.relu(spike_rate - cfg.target_spike_rate).pow(2)

    exit_losses = [F.cross_entropy(logits, targets) for logits in early_exit_logits]
    early_exit = torch.stack(exit_losses).mean() if exit_losses else zero
    total = (
        cfg.task * task
        + cfg.distillation * distillation
        + cfg.feature_alignment * feature_alignment
        + cfg.spike_budget * spike_budget
        + cfg.early_exit * early_exit
    )
    return FoundationLossOutput(
        total, task, distillation, feature_alignment, spike_budget, early_exit
    )
