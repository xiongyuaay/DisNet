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
import torchvision.models as models
from torchvision import transforms
__all__ = [
    "ResidualDenseBlock", "ResidualResidualDenseBlock",
    "Discriminator", "Generator",
    "ContentLoss"
]


class ResidualDenseBlock(nn.Module):
    """Achieves densely connected convolutional layers.
    `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.

    Args:
        channels (int): The number of channels in the input image.
        growths (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growths: int) -> None:
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels + growths * 0, growths, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(channels + growths * 1, growths, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(channels + growths * 2, growths, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(channels + growths * 3, growths, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(channels + growths * 4, channels, (3, 3), (1, 1), (1, 1))

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



class Generator(nn.Module):
    def __init__(self, args) -> None:
        super(Generator, self).__init__()

        self.scale = args.scale
        self.colors = args.colors
        self.window_sizes = args.window_sizes
        self.Module2_depth  = args.Module2_depth
        self.channels  = args.channels
        self.r_expand = args.r_expand

        # The first layer of convolutional layer.
        self.conv1 = nn.Conv2d(self.colors, self.channels, (3, 3), (1, 1), (1, 1))

        # Feature extraction backbone network.
        trunk = []
        for _ in range(self.Module2_depth):
            trunk.append(ResidualResidualDenseBlock(self.channels, 30))
        self.trunk = nn.Sequential(*trunk)

        # After the feature extraction network, reconnect a layer of convolutional blocks.
        self.conv2 = nn.Conv2d(self.channels, self.channels, (3, 3), (1, 1), (1, 1))

        # Upsampling convolutional layer.
        self.upsampling = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        # Reconnect a layer of convolution block after upsampling.
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        # Output layer.
        self.conv4 = nn.Conv2d(self.channels, self.colors, (3, 3), (1, 1), (1, 1))

    # The model should be defined in the Torch.script method.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(x)
        out = self.trunk(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(F.interpolate(out, scale_factor=2, mode="nearest"))
        out = self.upsampling(F.interpolate(out, scale_factor=2, mode="nearest"))
        out = self.conv3(out)
        out = self.conv4(out)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data *= 0.1
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


# class ContentLoss(nn.Module):
#     """Constructs a content loss function based on the VGG19 network.
#     Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.
#
#     Paper reference list:
#         -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
#         -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
#         -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.
#
#      """
#
#     def __init__(self) -> None:
#         super(ContentLoss, self).__init__()
#         # Load the VGG19 model trained on the ImageNet dataset.
#         vgg19 = models.vgg19(pretrained=True).eval()
#         # Extract the thirty-sixth layer output in the VGG19 model as the content loss.
#         self.feature_extractor = nn.Sequential(*list(vgg19.features.children())[:35])
#         # Freeze model parameters.
#         for parameters in self.feature_extractor.parameters():
#             parameters.requires_grad = False
#
#         # The preprocessing method of the input data. This is the VGG model preprocessing method of the ImageNet dataset.
#         self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
#         self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
#
#     def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
#         # Standardized operations
#         sr = sr.sub(self.mean).div(self.std)
#         hr = hr.sub(self.mean).div(self.std)
#
#         # Find the feature map difference between the two images
#         loss = F.l1_loss(self.feature_extractor(sr), self.feature_extractor(hr))

        # return loss

class ContentLoss(nn.Module):
    """Constructs a content loss function based on the VGG19 network.
    Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.
    Paper reference list:
        -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.
     """

    def __init__(self, feature_extractor_node: str, normalize_mean: list, normalize_std: list) -> None:
        super(ContentLoss, self).__init__()
        # Get the name of the specified feature extraction node
        self.feature_extractor_node = feature_extractor_node
        # Load the VGG19 model trained on the ImageNet dataset.
        model = models.vgg19(True)
        # Extract the thirty-fifth layer output in the VGG19 model as the content loss.
        self.feature_extractor = nn.Sequential(*list(model.features.children())[:35])
        # set to validation mode
        self.feature_extractor.eval()

        # The preprocessing method of the input data.
        # This is the VGG model preprocessing method of the ImageNet dataset
        self.normalize = transforms.Normalize(normalize_mean, normalize_std)

        # Freeze model parameters.
        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False

    def forward(self, sr_tensor: torch.Tensor, hr_tensor: torch.Tensor) -> torch.Tensor:
        # Standardized operations
        sr_tensor = self.normalize(sr_tensor)
        hr_tensor = self.normalize(hr_tensor)

        sr_feature = self.feature_extractor(sr_tensor)
        hr_feature = self.feature_extractor(hr_tensor)

        # Find the feature map difference between the two images
        feature_loss = F.l1_loss(sr_feature, hr_feature)

        return feature_loss
