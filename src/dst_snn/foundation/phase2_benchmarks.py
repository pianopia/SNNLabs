"""Small offline benchmarks for Phase 2 mechanics, not capability claims."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .losses import FoundationLossWeights, foundation_loss
from .torch_ssm import SignedSpikingSSMBlock, SpikingSSMBackbone


@dataclass(frozen=True)
class TaskBenchmarkResult:
    task: str
    teacher_score: float
    initial_student_score: float
    final_student_score: float
    first_loss: float
    final_loss: float
    spike_rate: float

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


class _DenseTextTeacher(nn.Module):
    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.rnn = nn.GRU(dim, dim, batch_first=True)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        hidden, _ = self.rnn(self.embedding(tokens))
        return self.head(hidden), hidden


class _SpikingTextStudent(nn.Module):
    def __init__(self, vocab_size: int, dim: int, depth: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.backbone = SpikingSSMBackbone(dim, depth=depth, max_level=3)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, tokens: Tensor):
        hidden, layers = self.backbone(self.embedding(tokens))
        return self.head(hidden), hidden, layers


def _text_data(samples: int, length: int, vocab_size: int, seed: int) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, vocab_size, (samples, 1), generator=generator)
    positions = torch.arange(length + 1).view(1, -1)
    sequence = (starts + positions) % vocab_size
    return sequence[:, :-1].long(), sequence[:, 1:].long()


def run_text_next_token_benchmark(
    *,
    seed: int = 0,
    samples: int = 96,
    length: int = 8,
    vocab_size: int = 16,
    dim: int = 16,
    teacher_epochs: int = 35,
    student_epochs: int = 45,
) -> TaskBenchmarkResult:
    torch.manual_seed(seed)
    inputs, targets = _text_data(samples, length, vocab_size, seed)
    teacher = _DenseTextTeacher(vocab_size, dim)
    teacher_optimizer = torch.optim.Adam(teacher.parameters(), lr=0.03)
    for _ in range(teacher_epochs):
        teacher_optimizer.zero_grad(set_to_none=True)
        logits, _ = teacher(inputs)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        teacher_optimizer.step()
    teacher.eval()
    with torch.no_grad():
        teacher_logits, teacher_features = teacher(inputs)
        teacher_score = float(
            (teacher_logits.argmax(-1) == targets).float().mean().item()
        )

    student = _SpikingTextStudent(vocab_size, dim)
    optimizer = torch.optim.Adam(student.parameters(), lr=0.02)
    with torch.no_grad():
        initial_logits, _, _ = student(inputs)
        initial_score = float((initial_logits.argmax(-1) == targets).float().mean().item())
    history = []
    final_layers = []
    weights = FoundationLossWeights(
        task=1.0,
        distillation=0.75,
        feature_alignment=0.4,
        spike_budget=0.05,
        early_exit=0.15,
        target_spike_rate=0.25,
    )
    for _ in range(student_epochs):
        optimizer.zero_grad(set_to_none=True)
        logits, hidden, layers = student(inputs)
        losses = foundation_loss(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
            teacher_logits=teacher_logits.reshape(-1, vocab_size),
            student_features=[hidden.reshape(-1, dim)],
            teacher_features=[teacher_features.reshape(-1, dim)],
            event_tensors=[layer.events for layer in layers],
            early_exit_logits=[student.head(layers[0].hidden).reshape(-1, vocab_size)],
            weights=weights,
        )
        losses.total.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        history.append(float(losses.total.detach().item()))
        final_layers = layers
    with torch.no_grad():
        final_logits, _, final_layers = student(inputs)
        final_score = float((final_logits.argmax(-1) == targets).float().mean().item())
        spike_rate = float(
            torch.stack([layer.events.ne(0).float().mean() for layer in final_layers])
            .mean()
            .item()
        )
    return TaskBenchmarkResult(
        "text_next_token",
        teacher_score,
        initial_score,
        final_score,
        history[0],
        history[-1],
        spike_rate,
    )


class _SpikingProjection(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, embed_dim)
        self.block = SignedSpikingSSMBlock(embed_dim, max_level=3)
        self.output = nn.Linear(embed_dim, embed_dim)

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        # Two steps expose temporal state while keeping this smoke benchmark tiny.
        sequence = self.input(values).unsqueeze(1).repeat(1, 2, 1)
        result = self.block(sequence)
        embedding = F.normalize(self.output(result.hidden[:, -1]), dim=-1)
        return embedding, result.events


def run_image_text_retrieval_benchmark(
    *,
    seed: int = 0,
    pairs: int = 16,
    image_dim: int = 20,
    text_dim: int = 12,
    embed_dim: int = 16,
    epochs: int = 60,
) -> TaskBenchmarkResult:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1)
    teacher_latent = F.normalize(torch.randn(pairs, embed_dim, generator=generator), dim=-1)
    image_basis = torch.randn(embed_dim, image_dim, generator=generator)
    text_basis = torch.randn(embed_dim, text_dim, generator=generator)
    images = teacher_latent @ image_basis + 0.02 * torch.randn(
        pairs, image_dim, generator=generator
    )
    texts = teacher_latent @ text_basis + 0.02 * torch.randn(
        pairs, text_dim, generator=generator
    )
    targets = torch.arange(pairs)
    teacher_logits = teacher_latent @ teacher_latent.T * 10.0
    teacher_score = float((teacher_logits.argmax(-1) == targets).float().mean().item())

    image_student = _SpikingProjection(image_dim, embed_dim)
    text_student = _SpikingProjection(text_dim, embed_dim)
    parameters = list(image_student.parameters()) + list(text_student.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.02)

    def forward():
        image_embedding, image_events = image_student(images)
        text_embedding, text_events = text_student(texts)
        return image_embedding @ text_embedding.T * 10.0, image_embedding, text_embedding, [image_events, text_events]

    with torch.no_grad():
        initial_logits, _, _, _ = forward()
        initial_score = float((initial_logits.argmax(-1) == targets).float().mean().item())
    history = []
    final_events = []
    weights = FoundationLossWeights(
        task=1.0,
        distillation=0.5,
        feature_alignment=0.5,
        spike_budget=0.05,
        early_exit=0.0,
        target_spike_rate=0.25,
    )
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits, image_embedding, text_embedding, events = forward()
        losses = foundation_loss(
            logits,
            targets,
            teacher_logits=teacher_logits,
            student_features=[image_embedding, text_embedding],
            teacher_features=[teacher_latent, teacher_latent],
            event_tensors=events,
            weights=weights,
        )
        losses.total.backward()
        optimizer.step()
        history.append(float(losses.total.detach().item()))
        final_events = events
    with torch.no_grad():
        logits, _, _, final_events = forward()
        final_score = float((logits.argmax(-1) == targets).float().mean().item())
        spike_rate = float(
            torch.stack([events.ne(0).float().mean() for events in final_events]).mean().item()
        )
    return TaskBenchmarkResult(
        "image_text_retrieval_r1",
        teacher_score,
        initial_score,
        final_score,
        history[0],
        history[-1],
        spike_rate,
    )
