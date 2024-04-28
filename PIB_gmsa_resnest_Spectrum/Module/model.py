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
from einops import rearrange
import math
from Module.ResNest import SplitAttn

class GMSA(nn.Module):
    def __init__(self, channels, shifts=4, window_sizes=[4, 8, 12], calc_attn=True):
        super(GMSA, self).__init__()    
        self.channels = channels
        self.shifts   = shifts
        self.window_sizes = window_sizes
        self.calc_attn = calc_attn

        if self.calc_attn:
            self.split_chns  = [channels*2//3, channels*2//3, channels*2//3]
            self.project_inp = nn.Sequential(
                nn.Conv2d(self.channels, self.channels*2, kernel_size=1), 
                nn.BatchNorm2d(self.channels*2)
            )
            self.project_out = nn.Sequential(
                SplitAttn(in_channels=channels),
                nn.Conv2d(channels, channels, kernel_size=1)
            )
        else:
            self.split_chns  = [channels//3, channels//3,channels//3]
            self.project_inp = nn.Sequential(
                nn.Conv2d(self.channels, self.channels, kernel_size=1), 
                nn.BatchNorm2d(self.channels)
            )
            self.project_out = nn.Sequential(
                SplitAttn(in_channels=channels),
                nn.Conv2d(channels, channels, kernel_size=1)
            )

    def forward(self, x, prev_atns = None):
        b,c,h,w = x.shape
        x = self.project_inp(x)
        xs = torch.split(x, self.split_chns, dim=1)
        ys = []
        atns = []
        if prev_atns is None:
            for idx, x_ in enumerate(xs):
                wsize = self.window_sizes[idx]
                if self.shifts > 0:
                    x_ = torch.roll(x_, shifts=(-wsize//2, -wsize//2), dims=(2,3))
                q, v = rearrange(
                    x_, 'b (qv c) (h dh) (w dw) -> qv (b h w) (dh dw) c', 
                    qv=2, dh=wsize, dw=wsize
                )
                atn = (q @ q.transpose(-2, -1)) 
                atn = atn.softmax(dim=-1)
                y_ = (atn @ v)
                y_ = rearrange(
                    y_, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)', 
                    h=h//wsize, w=w//wsize, dh=wsize, dw=wsize
                )
                if self.shifts > 0:
                    y_ = torch.roll(y_, shifts=(wsize//2, wsize//2), dims=(2, 3))
                ys.append(y_)
                atns.append(atn)
            y = torch.cat(ys, dim=1)            
            y = self.project_out(y)
            return y, atns
        else:
            for idx, x_ in enumerate(xs):
                wsize = self.window_sizes[idx]
                if self.shifts > 0:
                    x_ = torch.roll(x_, shifts=(-wsize//2, -wsize//2), dims=(2,3))
                atn = prev_atns[idx]
                v = rearrange(
                    x_, 'b (c) (h dh) (w dw) -> (b h w) (dh dw) c', 
                    dh=wsize, dw=wsize
                )
                y_ = (atn @ v)
                y_ = rearrange(
                    y_, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)', 
                    h=h//wsize, w=w//wsize, dh=wsize, dw=wsize
                )
                if self.shifts > 0:
                    y_ = torch.roll(y_, shifts=(wsize//2, wsize//2), dims=(2, 3))
                ys.append(y_)
            y = torch.cat(ys, dim=1)            
            y = self.project_out(y)
            return y, prev_atns


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
        self.conv1 = nn.Conv2d(self.colors, self.channels, (3, 3), (1, 1), (1, 1))

        # Feature extraction backbone network.
        self.gmsa1 = GMSA(channels=self.channels, shifts=0, window_sizes=self.window_sizes, calc_attn=True)
        self.gmsa2 = GMSA(channels=self.channels, shifts=0, window_sizes=self.window_sizes, calc_attn=False)
        self.gmsa3 = GMSA(channels=self.channels, shifts=1, window_sizes=self.window_sizes, calc_attn=True)
        self.gmsa4 = GMSA(channels=self.channels, shifts=1, window_sizes=self.window_sizes, calc_attn=False) 
        
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
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        out1 = self.conv1(x)
        y, attn = self.gmsa1(out1, None)
        out1 = torch.add(out1, y)
        y, attn = self.gmsa2(out1, attn)
        out1 = torch.add(out1, y)
        y, attn = self.gmsa3(out1, None)
        out1 = torch.add(out1, y)
        y, attn = self.gmsa4(out1, attn)
        out1 = torch.add(out1, y)
        out2 = self.conv2(out1)
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
