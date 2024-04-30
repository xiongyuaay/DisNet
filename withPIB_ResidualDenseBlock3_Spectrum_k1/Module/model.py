# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# ==============================================================================
# File description: Realize the model definition function.
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualDenseBlock(nn.Module):
    """Achieves densely connected convolutional layers.
    `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.

    Args:
        channels (int): The number of channels in the input image.
        growths (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growths: int) -> None:
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels + growths * 0, growths, (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(channels + growths * 1, growths, (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(channels + growths * 2, growths, (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(channels + growths * 3, growths, (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(channels + growths * 4, channels, (1, 1), (1, 1))

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out1 = self.leaky_relu(self.conv1(x))
        out2 = self.leaky_relu(self.conv2(torch.cat([x, out1], 1)))
        out3 = self.leaky_relu(self.conv3(torch.cat([x, out1, out2], 1)))
        out4 = self.leaky_relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))
        out5 = self.identity(self.conv5(torch.cat([x, out1, out2, out3, out4], 1)))
        out = torch.mul(out5, 0.2)
        out = torch.add(out, identity)

        return out


class ResidualResidualDenseBlock(nn.Module):
    """Multi-layer residual dense convolution block.

    Args:
        channels (int): The number of channels in the input image.
        growths (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growths: int) -> None:
        super(ResidualResidualDenseBlock, self).__init__()
        self.rdb1 = ResidualDenseBlock(channels, growths)
        self.rdb2 = ResidualDenseBlock(channels, growths)
        self.rdb3 = ResidualDenseBlock(channels, growths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        out = torch.mul(out, 0.2)
        out = torch.add(out, identity)

        return out



class DisNet(nn.Module):
    def __init__(self, args) -> None:
        super(DisNet, self).__init__()

        self.scale = args.scale
        self.colors = args.colors
        self.window_sizes = args.window_sizes
        self.Module2_depth  = args.Module2_depth
        self.channels  = args.channels
        self.r_expand = args.r_expand

        # The first layer of convolutional layer.
        self.conv1 = nn.Conv2d(self.colors, self.channels, (1, 1), (1, 1))

        # Feature extraction backbone network.
        trunk = []
        for _ in range(self.Module2_depth):
            trunk.append(ResidualDenseBlock(self.channels, 30))
        self.trunk = nn.Sequential(*trunk)

        # After the feature extraction network, reconnect a layer of convolutional blocks.
        self.conv2 = nn.Conv2d(self.channels, self.channels, (1, 1), (1, 1))

        # Upsampling convolutional layer.
        self.upsampling = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        # Reconnect a layer of convolution block after upsampling.
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        # Output layer.
        self.conv4 = nn.Conv2d(self.channels, self.colors, (1, 1), (1, 1))

    # The model should be defined in the Torch.script method.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        out1 = self.conv1(x)
        out = self.trunk(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(F.interpolate(out, scale_factor=2, mode="nearest"))
        out = self.upsampling(F.interpolate(out, scale_factor=2, mode="nearest"))
        out = self.conv3(out)
        out = self.conv4(out)

        return out[:, :, 0:H*self.scale, 0:W*self.scale]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data *= 0.1
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        wsize = self.window_sizes[0]
        for i in range(1, len(self.window_sizes)):
            wsize = wsize*self.window_sizes[i] // math.gcd(wsize, self.window_sizes[i])
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x


