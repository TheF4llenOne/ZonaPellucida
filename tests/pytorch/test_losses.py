import numpy as np
import pytest

torch = pytest.importorskip("torch")

from zona_pellucida.backends.pytorch.losses import (  # noqa: E402
    WeightedBCEWithLogitsLoss,
    get_loss,
)


def _data():
    generator = torch.Generator().manual_seed(0)
    targets = (torch.rand(2, 1, 8, 8, generator=generator) > 0.8).float()
    logits = torch.randn(2, 1, 8, 8, generator=generator) * 3
    return logits, targets


def test_unit_weights_equal_bce_with_logits():
    logits, targets = _data()
    weighted = WeightedBCEWithLogitsLoss(1.0, 1.0)(logits, targets)
    reference = torch.nn.BCEWithLogitsLoss()(logits, targets)
    assert weighted.item() == pytest.approx(reference.item(), rel=1e-6)


def test_weighted_loss_formula():
    logits, targets = _data()
    w0, w1 = 0.6, 4.0
    p = torch.sigmoid(logits).double().numpy()
    y = targets.double().numpy()
    expected = -np.mean(w1 * y * np.log(p) + w0 * (1 - y) * np.log(1 - p))
    loss = WeightedBCEWithLogitsLoss(w0, w1)(logits, targets)
    assert loss.item() == pytest.approx(expected, rel=1e-5)


def test_saturated_logits_still_have_gradients():
    # Confidently wrong pixels (logit -30 on zona) keep learning: the loss
    # is computed from the logits, without clipping.
    logits = torch.tensor([[[[-30.0]]]], requires_grad=True)
    loss = WeightedBCEWithLogitsLoss(0.5, 10.0)(logits, torch.ones(1, 1, 1, 1))
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad.item() == pytest.approx(-10.0, rel=1e-6)


def test_get_loss():
    assert isinstance(get_loss("unweighted"), torch.nn.BCEWithLogitsLoss)
    loss = get_loss("weighted", {0: 0.5, 1: 10.0})
    assert (loss.w0, loss.w1) == (0.5, 10.0)
    with pytest.raises(ValueError):
        get_loss("weighted")
    with pytest.raises(ValueError):
        get_loss("focal")
