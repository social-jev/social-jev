import pytest
import torch
from torch.nn import functional

from social_jev.training import BrierCrossEntropy, BrierLoss, CrossEntropy, FocalLoss, TrainConfig


@pytest.mark.parametrize(
    ("loss_fn", "expected"),
    [
        (CrossEntropy(), -torch.log(torch.tensor(0.8)).item()),
        (FocalLoss(), -(0.2**2) * torch.log(torch.tensor(0.8)).item()),
        (BrierLoss(), 0.08),
        (BrierCrossEntropy(0.3), 0.3 * 0.08 - 0.7 * torch.log(torch.tensor(0.8)).item()),
    ],
)
def test_losses_match_known_probabilities_and_ignore_padding(loss_fn, expected):
    logits = torch.tensor([[0.8, 0.2, 10.0]]).log().requires_grad_()
    labels = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, False]])
    losses = loss_fn(logits, labels, mask)
    assert losses.shape == (1,)
    assert losses.item() == pytest.approx(expected)
    losses.sum().backward()
    assert logits.grad is not None
    assert logits.grad.isfinite().all()
    assert logits.grad[0, 0] < 0
    assert logits.grad[0, 1] > 0
    assert logits.grad[0, 2] == 0


@pytest.mark.parametrize(
    "loss_fn", [CrossEntropy(), FocalLoss(), BrierLoss(), BrierCrossEntropy(0.3)]
)
def test_distribution_labels_are_not_collapsed(loss_fn):
    logits = torch.tensor([[0.8, 0.2]]).log().requires_grad_()
    mask = torch.ones_like(logits, dtype=torch.bool)
    labels = torch.tensor([[0.25, 0.75]])
    actual = loss_fn(logits, labels, mask)
    probabilities = logits.softmax(-1)
    ce = -(labels * logits.log_softmax(-1)).sum(-1)
    brier = (probabilities - labels).square().sum(-1)
    if isinstance(loss_fn, CrossEntropy):
        expected = ce
    elif isinstance(loss_fn, FocalLoss):
        expected = -(labels * (1 - probabilities).square() * logits.log_softmax(-1)).sum(-1)
    elif isinstance(loss_fn, BrierLoss):
        expected = brier
    else:
        expected = 0.3 * brier + 0.7 * ce
    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), logits)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert actual.item() != pytest.approx(loss_fn(logits, torch.tensor([[1.0, 0.0]]), mask).item())


def test_cross_entropy_matches_integer_label_loss_and_gradients():
    logits = torch.tensor([[0.3, -0.8, 0.2], [1.2, 0.2, -0.4]], requires_grad=True)
    indices = torch.tensor([0, 2])
    labels = functional.one_hot(indices, num_classes=3).float()
    actual = CrossEntropy()(logits, labels, torch.ones_like(logits, dtype=torch.bool))
    expected = functional.cross_entropy(logits, indices, reduction="none")
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), logits)[0],
    )


def test_config_selects_independent_default_and_custom_losses():
    first = TrainConfig(steps=2)
    second = TrainConfig(steps=2)
    assert isinstance(first.loss_fn, FocalLoss)
    assert first.loss_fn.gamma == 2
    assert first.loss_fn is not second.loss_fn
    custom = BrierCrossEntropy(brier_weight=0.25)
    assert TrainConfig(steps=2, loss_fn=custom).loss_fn is custom
