from functools import partial
import torch
import torch.nn as nn
import torch
import torch.nn as nn
from spikingjelly.clock_driven import layer
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Block
from .util.pos_embed import get_2d_sincos_pos_embed

import copy
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
#timestep 1x4
T=4

class multispike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lens=T):
        ctx.save_for_backward(input)
        ctx.lens = lens
        return torch.floor(torch.clamp(input, 0, lens) + 0.5)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp1 = 0 < input
        temp2 = input < ctx.lens
        return grad_input * temp1.float() * temp2.float(), None


class Multispike(nn.Module):
    def __init__(self, spike=multispike,norm=T):
        super().__init__()
        self.lens = norm
        self.spike = spike
        self.norm=norm

    def forward(self, inputs):
        return self.spike.apply(inputs)/self.norm


def MS_conv_unit(in_channels, out_channels,kernel_size=1,padding=0,groups=1):
    return nn.Sequential(
        layer.SeqToANNContainer(
           nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=groups,bias=True),
           nn.BatchNorm2d(out_channels) 
        )
    )
class MS_ConvBlock(nn.Module):
    def __init__(self, dim,
        mlp_ratio=4.0):
        super().__init__()

        self.neuron1 = Multispike()
        self.conv1 = MS_conv_unit(dim, dim * mlp_ratio, 3, 1)

        self.neuron2 = Multispike()
        self.conv2 = MS_conv_unit(dim*mlp_ratio, dim, 3, 1)


    def forward(self, x, mask=None):
        short_cut = x
        x = self.neuron1(x)
        x = self.conv1(x)
        x = self.neuron2(x)
        x = self.conv2(x)
        x = x +short_cut
        return x

class MS_MLP(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, drop=0.0, layer=0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif =  Multispike()


        self.fc2_conv = nn.Conv1d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = Multispike()

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, N= x.shape

        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()

        x = self.fc2_lif(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, N).contiguous()

        return x

class RepConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        bias=False,
    ):
        super().__init__()
        # TODO in_channel-> 2*in_channel->in_channel
        self.conv1 = nn.Sequential(nn.Conv1d(in_channel, int(in_channel*1.5), kernel_size=1, stride=1,bias=False), nn.BatchNorm1d(int(in_channel*1.5)))
        self.conv2 = nn.Sequential(nn.Conv1d(int(in_channel*1.5), out_channel, kernel_size=1, stride=1,bias=False), nn.BatchNorm1d(out_channel))
    def forward(self, x):
        return self.conv2(self.conv1(x))
class RepConv2(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        bias=False,
    ):
        super().__init__()
        # TODO in_channel-> 2*in_channel->in_channel
        self.conv1 = nn.Sequential(nn.Conv1d(in_channel, int(in_channel), kernel_size=1, stride=1,bias=False), nn.BatchNorm1d(int(in_channel)))
        self.conv2 = nn.Sequential(nn.Conv1d(int(in_channel), out_channel, kernel_size=1, stride=1,bias=False), nn.BatchNorm1d(out_channel))
    def forward(self, x):
        return self.conv2(self.conv1(x))

class MS_Attention_Conv_qkv_id(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.sr_ratio=sr_ratio

        self.head_lif = Multispike()

        # track 1: split convs
        self.q_conv = nn.Sequential(RepConv(dim,dim), nn.BatchNorm1d(dim))
        self.k_conv = nn.Sequential(RepConv(dim,dim), nn.BatchNorm1d(dim))
        self.v_conv = nn.Sequential(RepConv(dim,dim*sr_ratio), nn.BatchNorm1d(dim*sr_ratio))

        # track 2: merge (prefer) NOTE: need `chunk` in forward
        # self.qkv_conv = nn.Sequential(RepConv(dim,dim * 3), nn.BatchNorm2d(dim * 3))

        self.q_lif = Multispike()

        self.k_lif = Multispike()

        self.v_lif = Multispike()

        self.attn_lif = Multispike()

        self.proj_conv = nn.Sequential(RepConv(sr_ratio*dim,dim), nn.BatchNorm1d(dim))

    def forward(self, x):
        T, B, C, N = x.shape

        x = self.head_lif(x)

        x_for_qkv = x.flatten(0, 1)
        q_conv_out = self.q_conv(x_for_qkv).reshape(T, B, C, N)

        q_conv_out = self.q_lif(q_conv_out)

        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2,
                                                                                                       4)

        k_conv_out = self.k_conv(x_for_qkv).reshape(T, B, C, N)

        k_conv_out = self.k_lif(k_conv_out)

        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2,
                                                                                                       4)

        v_conv_out = self.v_conv(x_for_qkv).reshape(T, B, self.sr_ratio*C, N)

        v_conv_out = self.v_lif(v_conv_out)

        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, self.sr_ratio*C // self.num_heads).permute(0, 1, 3, 2,
                                                                                                       4)

        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale
        x = x.transpose(3, 4).reshape(T, B, self.sr_ratio*C, N)
        x = self.attn_lif(x)

        x = self.proj_conv(x.flatten(0, 1)).reshape(T, B, C, N)
        return x




class MS_Block(nn.Module):
    def __init__(
            self,
            dim,
            choice,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            sr_ratio=1,init_values=1e-6,finetune=False,
    ):
        super().__init__()
        self.model=choice
        if self.model=="base":
            self.rep_conv=RepConv2(dim,dim) #if have param==83M
        self.lif = Multispike()
        self.attn = MS_Attention_Conv_qkv_id(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        self.finetune = finetune
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        if self.finetune:
            self.layer_scale1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.layer_scale2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        T, B, C, N = x.shape
        if self.model=="base":
            x= x + self.rep_conv(self.lif(x).flatten(0, 1)).reshape(T, B, C, N)
        # TODO: need channel-wise layer scale, init as 1e-6
        if self.finetune:
            x = x + self.drop_path(self.attn(x) * self.layer_scale1.unsqueeze(0).unsqueeze(0).unsqueeze(-1))
            x = x + self.drop_path(self.mlp(x) * self.layer_scale2.unsqueeze(0).unsqueeze(0).unsqueeze(-1))
        else:
            x = x + self.attn(x)
            x = x + self.mlp(x)
        return x


class MS_DownSampling(nn.Module):
    def __init__(
        self,
        in_channels=2,
        embed_dims=256,
        kernel_size=3,
        stride=2,
        padding=1,
        first_layer=True,
    ):
        super().__init__()

        self.encode_conv = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.encode_bn =nn.BatchNorm2d(embed_dims)
        if not first_layer:
            self.encode_lif = Multispike()

    def forward(self, x):
        T, B, _, _, _ = x.shape
        if hasattr(self, "encode_lif"):
            x = self.encode_lif(x)
        x = self.encode_conv(x.flatten(0, 1))
        _, _, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguous()
        return x


class Spikformer(nn.Module):
    def __init__(self, T=1,
    choice=None,
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[128, 256, 512, 640],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        num_classes=1000,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), #norm_layer=nn.LayerNorm shaokun
        depths=8,
        sr_ratios=1,
        mlp_ratio=4.,
        nb_classes=1000,
        kd=True):
        super().__init__()

        ### MAE encoder spikformer
        self.T = T
        self.patch_size = patch_size
        self.embed_dim =embed_dim
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]  # stochastic depth decay rule
        self.downsample1_1 = MS_DownSampling(
            in_channels=in_channels,
            embed_dims=embed_dim[0] // 2,
            kernel_size=7,
            stride=2,
            padding=3,
            first_layer=True,
        )

        self.ConvBlock1_1 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratios)]
        )

        self.downsample1_2 = MS_DownSampling(
            in_channels=embed_dim[0] // 2,
            embed_dims=embed_dim[0],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.ConvBlock1_2 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[0], mlp_ratio=mlp_ratios)]
        )

        self.downsample2 = MS_DownSampling(
            in_channels=embed_dim[0],
            embed_dims=embed_dim[1],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.ConvBlock2_1 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
        )

        self.ConvBlock2_2 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
        )

        self.downsample3 = MS_DownSampling(
            in_channels=embed_dim[1],
            embed_dims=embed_dim[2],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.block3 = nn.ModuleList(
            [
                MS_Block(
                    dim=embed_dim[2],
                    choice=choice,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                    finetune=True,
                )
                for j in range(depths)
            ]
        )
        self.head = nn.Linear(embed_dim[2], nb_classes)
        self.lif = Multispike(norm=1)
        self.kd = kd
        if self.kd:
            self.head_kd = (
                nn.Linear(embed_dim[-1], num_classes)
                if num_classes > 0
                else nn.Identity()
            )
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward_encoder(self, x ):
        x  = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)


        x = self.downsample1_1(x)

        for blk in self.ConvBlock1_1:
            x = blk(x)

        x = self.downsample1_2(x)
        for blk in self.ConvBlock1_2:
            x = blk(x)

        x = self.downsample2(x)

        for blk in self.ConvBlock2_1:
            x = blk(x)

        for blk in self.ConvBlock2_2:
            x = blk(x)
        x = self.downsample3(x)
        x = x.flatten(3)  # T,B,C,N

        for blk in self.block3:
            x = blk(x)

        return x


    def forward(self, imgs):
        x = self.forward_encoder(imgs)

        x = x.flatten(3).mean(3)
        x_lif = self.lif(x)
        x = self.head(x).mean(0)

        if self.kd:
            x_kd = self.head_kd(x_lif).mean(0)
            if self.training:
                return x, x_kd
            else:
                return (x + x_kd) / 2
        return x



def spikformer12_512(**kwargs):
    model = Spikformer(
        T=1,
        choice="base",
        img_size_h=32,
        img_size_w=32,
        patch_size=16,
        embed_dim=[128,256,512],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=1000,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=12,
        **kwargs)
    return model
def spikformer12_768(**kwargs):
    model = Spikformer(
        T=1,
        choice="large",
        img_size_h=32,
        img_size_w=32,
        patch_size=16,
        embed_dim=[196, 384, 768],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=1000,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=12,
        **kwargs)
    return model





if __name__ == "__main__":
    # from encoder import SparseEncoder,nn.Conv2d
    import torchinfo

    model = spikformer12_512()


    print(f"number of params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
