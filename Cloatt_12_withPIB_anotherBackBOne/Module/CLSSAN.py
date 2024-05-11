import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from einops import rearrange
from efficientnet_pytorch.model import MemoryEfficientSwish


class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class ShiftConv2d(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d, self).__init__()    
        self.inp_channels = inp_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.zeros(inp_channels, 1, 3, 3), requires_grad=False)
        self.n_div = 5
        g = inp_channels // self.n_div
        self.weight[0*g:1*g, 0, 1, 2] = 1.0 ## left
        self.weight[1*g:2*g, 0, 1, 0] = 1.0 ## right
        self.weight[2*g:3*g, 0, 2, 1] = 1.0 ## up
        self.weight[3*g:4*g, 0, 0, 1] = 1.0 ## down
        self.weight[4*g:, 0, 1, 1] = 1.0 ## identity     

        self.conv1x1 = nn.Conv2d(inp_channels, out_channels, 1)

    def forward(self, x):
        y = F.conv2d(input=x, weight=self.weight, bias=None, stride=1, padding=1, groups=self.inp_channels)
        y = self.conv1x1(y) 
        return y
    


class AttnMap(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act_block = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            MemoryEfficientSwish(),
            nn.Conv2d(dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.act_block(x)


class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=10, group_split=[5, 5], kernel_sizes=[5], window_size=4,
                 attn_drop=0., proj_drop=0., qkv_bias=True):
        super().__init__()
        assert sum(group_split) == num_heads
        assert len(kernel_sizes) + 1 == len(group_split)
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scalor = self.dim_head ** -0.5
        self.kernel_sizes = kernel_sizes
        self.window_size = window_size
        self.group_split = group_split
        convs = []
        act_blocks = []
        qkvs = []
        # projs = []
        for i in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[i]
            group_head = group_split[i]
            if group_head == 0:
                continue
            convs.append(nn.Conv2d(3 * self.dim_head * group_head, 3 * self.dim_head * group_head, kernel_size,
                                   1, kernel_size // 2, groups=3 * self.dim_head * group_head))
            act_blocks.append(AttnMap(self.dim_head * group_head))
            qkvs.append(nn.Conv2d(dim, 3 * group_head * self.dim_head, 1, 1, 0, bias=qkv_bias))
            # projs.append(nn.Linear(group_head*self.dim_head, group_head*self.dim_head, bias=qkv_bias))
        if group_split[-1] != 0:
            self.global_q = nn.Conv2d(dim, group_split[-1] * self.dim_head, 1, 1, 0, bias=qkv_bias)
            self.global_kv = nn.Conv2d(dim, group_split[-1] * self.dim_head * 2, 1, 1, 0, bias=qkv_bias)
            # self.global_proj = nn.Linear(group_split[-1]*self.dim_head, group_split[-1]*self.dim_head, bias=qkv_bias)
            self.avgpool = nn.AvgPool2d(window_size, window_size) if window_size != 1 else nn.Identity()

        self.convs = nn.ModuleList(convs)
        self.act_blocks = nn.ModuleList(act_blocks)
        self.qkvs = nn.ModuleList(qkvs)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def high_fre_attntion(self, x: torch.Tensor, to_qkv: nn.Module, mixer: nn.Module, attn_block: nn.Module):
        '''
        x: (b c h w)
        '''
        b, c, h, w = x.size()
        qkv = to_qkv(x)  # (b (3 m d) h w)
        qkv = mixer(qkv).reshape(b, 3, -1, h, w).transpose(0, 1).contiguous()  # (3 b (m d) h w)
        q, k, v = qkv  # (b (m d) h w)
        attn = attn_block(q.mul(k)).mul(self.scalor)
        attn = self.attn_drop(torch.tanh(attn))
        res = attn.mul(v)  # (b (m d) h w)
        return res

    def low_fre_attention(self, x: torch.Tensor, to_q: nn.Module, to_kv: nn.Module, avgpool: nn.Module):
        '''
        x: (b c h w)
        '''
        b, c, h, w = x.size()

        q = to_q(x).reshape(b, -1, self.dim_head, h * w).transpose(-1, -2).contiguous()  # (b m (h w) d)
        kv = avgpool(x)  # (b c h w)
        kv = to_kv(kv).view(b, 2, -1, self.dim_head, (h * w) // (self.window_size ** 2)).permute(1, 0, 2, 4,
                                                                                                 3).contiguous()  # (2 b m (H W) d)
        k, v = kv  # (b m (H W) d)
        attn = self.scalor * q @ k.transpose(-1, -2)  # (b m (h w) (H W))
        attn = self.attn_drop(attn.softmax(dim=-1))
        res = attn @ v  # (b m (h w) d)
        res = res.transpose(2, 3).reshape(b, -1, h, w).contiguous()
        return res

    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        res = []
        for i in range(len(self.kernel_sizes)):
            if self.group_split[i] == 0:
                continue
            res.append(self.high_fre_attntion(x, self.qkvs[i], self.convs[i], self.act_blocks[i]))
        if self.group_split[-1] != 0:
            res.append(self.low_fre_attention(x, self.global_q, self.global_kv, self.avgpool))
        return self.proj_drop(self.proj(torch.cat(res, dim=1)))









class CLSSAB(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=2, window_sizes=[4, 8, 12]) -> None:
        super(CLSSAB, self).__init__()
        self.exp_ratio = exp_ratio
        self.window_sizes = window_sizes
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.Module = nn.Sequential(
            ShiftConv2d(self.inp_channels, self.out_channels*self.exp_ratio),
            nn.ReLU(inplace=True),
            ShiftConv2d(self.out_channels*self.exp_ratio, self.out_channels),  
        )
        self.EffA = EfficientAttention(dim=self.inp_channels).cuda()
        modules_gmsa = {}
        self.modules_gmsa = nn.ModuleDict(modules_gmsa)
        
    def forward(self, x):
        x = self.Module(x) + x
        x = self.EffA(x) + x
        return x
        


class CLSSAN(nn.Module):
    def __init__(self, args) -> None:
        super(CLSSAN, self).__init__()
        self.scale = args.scale
        self.colors = args.colors
        self.window_sizes = args.window_sizes
        self.Module2_depth  = args.Module2_depth
        self.channels  = args.channels
        self.r_expand = args.r_expand
        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)
        self.Module1 = nn.Conv2d(self.colors, self.channels, kernel_size=3, stride=1, padding=1)
        
        self.module2 = []
        for i in range(self.Module2_depth):
            self.module2.append(CLSSAB(self.channels, self.channels, exp_ratio=self.r_expand, window_sizes=self.window_sizes))
        self.Module2 = nn.Sequential(*self.module2)
        
        self.Module3 = nn.Sequential(
            nn.Conv2d(self.channels, self.colors*self.scale*self.scale, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(self.scale)
        )
        # 定义反卷积层
        self.Module3_deconv = nn.Sequential(
            nn.Conv2d(self.channels, self.colors * self.scale * self.scale, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(self.colors * self.scale * self.scale, self.colors, kernel_size=3, stride=self.scale, padding=1, output_padding=1)
        )

        
    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        x = self.sub_mean(x)
        x = self.Module1(x)
        result = self.Module2(x)
        result = result + x
        x = self.Module3_deconv(result)
        x = self.add_mean(x)
        return x[:, :, 0:H*self.scale, 0:W*self.scale]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        wsize = self.window_sizes[0]
        for i in range(1, len(self.window_sizes)):
            wsize = wsize*self.window_sizes[i] // math.gcd(wsize, self.window_sizes[i])
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x