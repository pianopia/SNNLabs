"""Progressive ANN-to-SNN block replacement and one-step trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from .losses import FoundationLossOutput, FoundationLossWeights, foundation_loss
from .torch_ssm import SpikingSSMOutput


class ProgressiveBlockReplacement(nn.Module):
    """Use student blocks for a prefix and frozen teacher blocks thereafter."""

    def __init__(
        self,
        teacher_blocks: Sequence[nn.Module],
        student_blocks: Sequence[nn.Module],
        *,
        dim: int,
    ) -> None:
        super().__init__()
        if not teacher_blocks or len(teacher_blocks) != len(student_blocks):
            raise ValueError("teacher/student block lists must be non-empty and equal length")
        self.teacher_blocks = nn.ModuleList(teacher_blocks)
        self.student_blocks = nn.ModuleList(student_blocks)
        self.connectors = nn.ModuleList([nn.Linear(dim, dim) for _ in student_blocks])
        for connector in self.connectors:
            nn.init.eye_(connector.weight)
            nn.init.zeros_(connector.bias)
        for block in self.teacher_blocks:
            for parameter in block.parameters():
                parameter.requires_grad_(False)
            block.eval()
        self.active_student_blocks = 0

    @property
    def depth(self) -> int:
        return len(self.teacher_blocks)

    def set_active_student_blocks(self, count: int) -> None:
        if not 0 <= count <= self.depth:
            raise ValueError("replacement count outside model depth")
        self.active_student_blocks = int(count)

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        hidden = x
        features = []
        events = []
        for index, (teacher, student) in enumerate(zip(self.teacher_blocks, self.student_blocks)):
            if index < self.active_student_blocks:
                result = student(hidden)
                if isinstance(result, SpikingSSMOutput):
                    hidden = result.hidden
                    events.append(result.events)
                else:
                    hidden = result
                hidden = self.connectors[index](hidden)
            else:
                # Teacher parameters are frozen, but autograd must still pass
                # through the teacher suffix into the replaced student prefix.
                hidden = teacher(hidden)
            features.append(hidden)
        return hidden, features, events


@dataclass
class ReplacementStepResult:
    losses: FoundationLossOutput
    spike_rate: float


class BlockwiseReplacementTrainer:
    def __init__(
        self,
        model: ProgressiveBlockReplacement,
        head: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        weights: FoundationLossWeights | None = None,
    ) -> None:
        self.model = model
        self.head = head
        self.optimizer = optimizer
        self.weights = weights

    def step(
        self,
        inputs: Tensor,
        targets: Tensor,
        *,
        teacher_logits: Tensor,
        teacher_features: Sequence[Tensor] = (),
    ) -> ReplacementStepResult:
        self.model.train()
        self.head.train()
        self.optimizer.zero_grad(set_to_none=True)
        hidden, features, events = self.model(inputs)
        logits = self.head(hidden[:, -1])
        losses = foundation_loss(
            logits,
            targets,
            teacher_logits=teacher_logits,
            student_features=features[: len(teacher_features)],
            teacher_features=teacher_features,
            event_tensors=events,
            weights=self.weights,
        )
        losses.total.backward()
        self.optimizer.step()
        rate = (
            float(torch.stack([item.ne(0).float().mean() for item in events]).mean().item())
            if events
            else 0.0
        )
        return ReplacementStepResult(losses, rate)
