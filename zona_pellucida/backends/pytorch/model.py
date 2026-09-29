"""U-Net-style FCN in PyTorch: the same network as the Keras `fcn_8`.

(`fcn_8` is the historical name of the thesis model; the architecture is a
U-Net, not the FCN-8s of Long et al.)

Same layers, order (conv -> ReLU -> batch norm), skip connections and
dropout. Initialisation and batch-norm settings follow the Keras defaults
so both frameworks start from the same kind of network. The network
returns logits; the sigmoid is applied in the loss and in `predict`.
"""

import math

import torch
from torch import nn

# Keras BatchNormalization: momentum 0.99 (PyTorch convention: 0.01),
# epsilon 1e-3.
BN_MOMENTUM = 0.01
BN_EPSILON = 1e-3


def conv_block(in_channels, out_channels):
    """3x3 'same' convolution + ReLU + batch normalisation."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels, eps=BN_EPSILON, momentum=BN_MOMENTUM),
    )


class UNet(nn.Module):
    """Contracting path: 5 blocks of two conv blocks (32...512 filters)
    with 2x2 max pooling between them. Expanding path: 4 blocks of a 2x2
    transposed conv, skip concatenation and one conv block. Output: one
    logit per pixel (zona pellucida vs background)."""

    def __init__(self, in_channels=1, dropout=0.5):
        super().__init__()
        # Contracting path
        self.conv1 = nn.Sequential(conv_block(in_channels, 32),
                                   conv_block(32, 32))
        self.conv2 = nn.Sequential(conv_block(32, 64), conv_block(64, 64))
        self.conv3 = nn.Sequential(conv_block(64, 128), conv_block(128, 128))
        self.conv4 = nn.Sequential(conv_block(128, 256),
                                   conv_block(256, 256))
        self.conv5 = nn.Sequential(conv_block(256, 512),
                                   conv_block(512, 512))
        self.pool = nn.MaxPool2d(2)
        # Expanding path
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = conv_block(512, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = conv_block(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = conv_block(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = conv_block(64, 32)
        self.dropout = nn.Dropout(dropout)  # Adding dropout for regularization
        self.head = nn.Conv2d(32, 1, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """Keras initialisation: he_normal (truncated normal) for the 3x3
        convolutions, glorot_uniform for the transposed and the output
        convolutions, zero biases."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) and module.kernel_size == (3, 3):
                fan_in = module.in_channels * 9
                # Keras corrects the std for the truncation at 2 std.
                std = math.sqrt(2.0 / fan_in) / 0.87962566103423978
                nn.init.trunc_normal_(module.weight, std=std,
                                      a=-2 * std, b=2 * std)
            elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(module.weight)
            else:
                continue
            nn.init.zeros_(module.bias)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))
        conv6 = self.conv6(torch.cat([self.up6(conv5), conv4], dim=1))
        conv7 = self.conv7(torch.cat([self.up7(conv6), conv3], dim=1))
        conv8 = self.conv8(torch.cat([self.up8(conv7), conv2], dim=1))
        conv9 = self.conv9(torch.cat([self.up9(conv8), conv1], dim=1))
        return self.head(self.dropout(conv9))
