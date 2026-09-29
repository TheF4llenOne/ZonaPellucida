import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from zona_pellucida.backends.pytorch.model import UNet  # noqa: E402


def test_output_is_one_logit_per_pixel():
    model = UNet().eval()
    with torch.no_grad():
        out = model(torch.randn(2, 1, 32, 32))
    assert out.shape == (2, 1, 32, 32)
    # Logits, not probabilities: the sigmoid is in the loss / predict.
    assert (out < 0).any() or (out > 1).any()


def test_keras_initialisation():
    torch.manual_seed(0)
    model = UNet()
    conv = model.conv5[1][0]  # 3x3 conv, 512 -> 512 (many weights)
    fan_in = conv.in_channels * 9
    assert conv.weight.std().item() == pytest.approx(
        math.sqrt(2.0 / fan_in), rel=0.02)
    # Truncated at 2 std of the (corrected) normal distribution.
    std = math.sqrt(2.0 / fan_in) / 0.87962566103423978
    assert conv.weight.abs().max().item() <= 2 * std + 1e-6
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            assert torch.count_nonzero(module.bias) == 0
    bn = model.conv1[0][2]
    assert (bn.eps, bn.momentum) == (1e-3, 0.01)


def _layers_with_weights(keras_model):
    return [layer for layer in keras_model.layers if layer.get_weights()]


def _torch_layers(model):
    kinds = (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.BatchNorm2d)
    return [m for m in model.modules() if isinstance(m, kinds)]


def _copy_keras_weights(keras_model, torch_model):
    """Copy the weights of the Keras fcn_8 into UNet (same layer order)."""
    keras_layers = _layers_with_weights(keras_model)
    torch_layers = _torch_layers(torch_model)
    assert len(keras_layers) == len(torch_layers)
    with torch.no_grad():
        for k_layer, t_layer in zip(keras_layers, torch_layers):
            weights = [torch.from_numpy(np.array(w))
                       for w in k_layer.get_weights()]
            if isinstance(t_layer, torch.nn.BatchNorm2d):
                gamma, beta, mean, var = weights
                t_layer.weight.copy_(gamma)
                t_layer.bias.copy_(beta)
                t_layer.running_mean.copy_(mean)
                t_layer.running_var.copy_(var)
            else:
                # Keras (kh, kw, a, b) -> PyTorch (b, a, kh, kw)
                kernel, bias = weights
                t_layer.weight.copy_(kernel.permute(3, 2, 0, 1))
                t_layer.bias.copy_(bias)


@pytest.mark.slow
def test_same_network_as_the_keras_thesis_model():
    """With the weights of the Keras fcn_8, UNet gives the same output."""
    keras = pytest.importorskip("keras")
    from zona_pellucida.backends.keras.model import fcn_8

    keras.utils.set_random_seed(0)
    keras_model = fcn_8(64, 64, 1)
    # Non-trivial batch-norm statistics, so they are compared too.
    for layer in keras_model.layers:
        weights = layer.get_weights()
        if len(weights) == 4:
            rng = np.random.default_rng(len(layer.name))
            layer.set_weights([rng.uniform(0.5, 1.5, w.shape).astype(
                np.float32) for w in weights])
    torch_model = UNet().eval()
    _copy_keras_weights(keras_model, torch_model)

    keras_trainable = sum(int(np.prod(w.shape))
                          for w in keras_model.trainable_weights)
    torch_trainable = sum(p.numel() for p in torch_model.parameters())
    assert torch_trainable == keras_trainable

    x = np.random.default_rng(1).random((2, 64, 64, 1)).astype(np.float32)
    keras_out = keras_model.predict(x, verbose=0)
    with torch.no_grad():
        torch_out = torch.sigmoid(torch_model(
            torch.from_numpy(x.transpose(0, 3, 1, 2)))).numpy()
    np.testing.assert_allclose(torch_out.transpose(0, 2, 3, 1), keras_out,
                               atol=1e-4)
