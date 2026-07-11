from pathlib import Path

import torch
from torch import nn

from src.dst_snn.foundation.distillation import (
    FeatureCacheDataset,
    TorchTeacherAdapter,
    cache_teacher_features,
)
from src.dst_snn.foundation.replacement import ProgressiveBlockReplacement
from src.dst_snn.foundation.replacement import BlockwiseReplacementTrainer
from src.dst_snn.foundation.torch_ssm import SignedSpikingSSMBlock


class TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4, 4)
        self.head = nn.Linear(4, 3)

    def forward(self, x):
        return self.head(torch.relu(self.hidden(x)))


def test_teacher_adapter_cache_round_trip(tmp_path: Path):
    teacher = TinyTeacher()
    adapter = TorchTeacherAdapter(teacher, feature_layers=["hidden"])
    batches = [
        (torch.randn(2, 4), torch.tensor([0, 1])),
        (torch.randn(1, 4), torch.tensor([2])),
    ]
    path = cache_teacher_features(batches, adapter, tmp_path / "teacher.pt")
    dataset = FeatureCacheDataset(path)
    assert len(dataset) == 3
    assert dataset[0]["teacher_logits"].shape == (3,)
    assert dataset[0]["teacher_features"]["hidden"].shape == (4,)
    assert not any(parameter.requires_grad for parameter in teacher.parameters())
    adapter.close()


def test_progressive_replacement_switches_prefix_to_snn():
    teacher_blocks = [nn.Linear(4, 4), nn.Linear(4, 4)]
    students = [SignedSpikingSSMBlock(4), SignedSpikingSSMBlock(4)]
    model = ProgressiveBlockReplacement(teacher_blocks, students, dim=4)
    x = torch.randn(2, 3, 4)
    model.set_active_student_blocks(1)
    output, features, events = model(x)
    assert output.shape == x.shape
    assert len(features) == 2
    assert len(events) == 1
    assert not any(parameter.requires_grad for block in model.teacher_blocks for parameter in block.parameters())


def test_blockwise_trainer_updates_student_through_frozen_teacher_suffix():
    teacher_blocks = [nn.Linear(4, 4), nn.Linear(4, 4)]
    students = [SignedSpikingSSMBlock(4), SignedSpikingSSMBlock(4)]
    model = ProgressiveBlockReplacement(teacher_blocks, students, dim=4)
    model.set_active_student_blocks(1)
    head = nn.Linear(4, 3)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable + list(head.parameters()), lr=0.01)
    trainer = BlockwiseReplacementTrainer(model, head, optimizer)
    inputs = torch.randn(5, 3, 4)
    targets = torch.tensor([0, 1, 2, 0, 1])
    before = students[0].input_projection.weight.detach().clone()
    result = trainer.step(
        inputs,
        targets,
        teacher_logits=torch.randn(5, 3),
    )
    assert result.losses.total.item() > 0
    assert not torch.equal(before, students[0].input_projection.weight.detach())
