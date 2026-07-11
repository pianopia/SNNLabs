import torch

from src.dst_snn.foundation.losses import FoundationLossWeights, foundation_loss
from src.dst_snn.foundation.torch_ssm import (
    SignedSpikingSSMBlock,
    SpikingSSMBackbone,
    signed_integer_spike,
)


def test_signed_integer_spike_has_levels_polarity_and_gradient():
    x = torch.tensor([-4.2, -0.7, 0.2, 1.6, 4.0], requires_grad=True)
    events = signed_integer_spike(x, max_level=3)
    assert events.tolist() == [-3.0, -1.0, 0.0, 2.0, 3.0]
    events.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_spiking_ssm_is_trainable_and_streaming_state_matches_chunks():
    torch.manual_seed(2)
    block = SignedSpikingSSMBlock(4, state_dim=6, max_level=3)
    x = torch.randn(2, 5, 4, requires_grad=True)
    whole = block(x)
    first = block(x[:, :2])
    second = block(x[:, 2:], first.final_state)
    torch.testing.assert_close(
        torch.cat([first.hidden, second.hidden], dim=1), whole.hidden
    )
    whole.hidden.square().mean().backward()
    assert block.input_projection.weight.grad is not None
    assert tuple(whole.events.shape) == (2, 5, 6)


def test_backbone_and_multiterm_loss_backpropagate():
    torch.manual_seed(4)
    backbone = SpikingSSMBackbone(8, depth=2)
    head = torch.nn.Linear(8, 3)
    x = torch.randn(4, 5, 8)
    hidden, layers = backbone(x)
    logits = head(hidden[:, -1])
    teacher_logits = torch.randn_like(logits)
    losses = foundation_loss(
        logits,
        torch.tensor([0, 1, 2, 0]),
        teacher_logits=teacher_logits,
        student_features=[hidden[:, -1]],
        teacher_features=[torch.randn_like(hidden[:, -1])],
        event_tensors=[layer.events for layer in layers],
        early_exit_logits=[head(hidden[:, 1])],
        weights=FoundationLossWeights(target_spike_rate=0.1),
    )
    losses.total.backward()
    assert losses.total.item() > 0
    assert backbone.blocks[0].input_projection.weight.grad is not None
