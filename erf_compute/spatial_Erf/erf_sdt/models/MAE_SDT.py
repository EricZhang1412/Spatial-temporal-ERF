from functools import partial
import torch
import torch.nn as nn
import torchinfo
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Block

from spikingjelly.clock_driven import layer
import copy
from torchvision import transforms
import matplotlib.pyplot as plt

import models.encoder as encoder
from .util.pos_embed import get_2d_sincos_pos_embed

import torch

#timestep
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
           encoder.SparseConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=groups,bias=True),
           encoder.SparseBatchNorm2d(out_channels) 
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
        self.conv1 = nn.Sequential(nn.Conv1d(in_channel, int(in_channel*1.5), kernel_size=1, stride=1,bias=False), nn.BatchNorm1d(int(in_channel*1.5)))
        self.conv2 = nn.Sequential(nn.Conv1d(int(in_channel*1.5), out_channel, kernel_size=1, stride=1,bias=False), nn.BatchNorm1d(out_channel))
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

        self.encode_conv = encoder.SparseConv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.encode_bn = encoder.SparseBatchNorm2d(embed_dims)
        self.first_layer = first_layer
        if not first_layer:
            self.encode_spike = Multispike()

    def forward(self, x):
        T, B, _, _, _ = x.shape
        if hasattr(self, "encode_spike"):
            x = self.encode_spike(x)
        x = self.encode_conv(x.flatten(0, 1))
        _, _, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W)

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
        # T, B, C, N = x.shape
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

class Spikmae(nn.Module):
    def __init__(self, T=1,choice=None,
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[128, 256, 512],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        num_classes=1000,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), #norm_layer=nn.LayerNorm shaokun
        depths=8,
        sr_ratios=1,
        decoder_embed_dim=768,
        decoder_depth=4,
        decoder_num_heads=16,
        mlp_ratio=4.,
        norm_pix_loss=False, nb_classes=1000):
        super().__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.T = 1

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
                    finetune=False,
                )
                for j in range(depths)
            ]
        )

        self.norm = nn.BatchNorm1d(embed_dim[-1])
        self.downsample_raito =16

        num_patches = 196

        self.pos_embed = nn.Parameter(torch.zeros(1,  embed_dim[-1],num_patches), requires_grad=False)

        ## MAE decoder vit
        self.decoder_embed = nn.Linear(embed_dim[-1], decoder_embed_dim,bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # Try  larned decoder
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=False, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_channels,bias=True)  # decoder to patch
        self.initialize_weights()

    def initialize_weights(self):
        num_patches=196
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[1], int(num_patches ** .5),
                                            cls_token=False)

        self.pos_embed.data.copy_(torch.from_numpy(pos_embed.transpose(1,0)).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(num_patches** .5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        num_patches=196
        T, N, _, _, _ = x.shape  # batch, length, dim
        L = num_patches
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # active is inverse mask
        active = torch.ones([N, L], device=x.device)
        active[:, len_keep:] = 0
        active = torch.gather(active, dim=1, index=ids_restore)

        return ids_keep, active, ids_restore

    def forward_encoder(self, x , mask_ratio=1.0):
        x  = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        # step1. Mask
        ids_keep, active, ids_restore = self.random_masking(x , mask_ratio)
        B,N=active.shape
        active_b1ff=active.reshape(B,1,14,14)


        encoder._cur_active = active_b1ff
        active_hw = active_b1ff.repeat_interleave(self.downsample_raito, 2).repeat_interleave(self.downsample_raito, 3)
        active_hw = active_hw.unsqueeze(0)
        masked_bchw = x * active_hw
        x = masked_bchw
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
        x = x.flatten(3)
        for blk in self.block3:
            x = blk(x)

        x = x.mean(0)
        x = self.norm(x).transpose(-1, -2).contiguous()
        return x, active,ids_restore,active_hw

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        B, N, C = x.shape
        x = self.decoder_embed(x)  # B, N, C
        # append mask tokens to sequence
        # ids_restore#1,196
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = x_
#
        # add pos embed
        x = x + self.decoder_pos_embed
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)

        return x

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = 16
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = 16
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """

        inp, rec = self.patchify(imgs), pred # inp and rec: (B, L = f*f, N = C*downsample_raito**2)
        mean = inp.mean(dim=-1, keepdim=True)
        var = (inp.var(dim=-1, keepdim=True) + 1e-6) ** .5
        inp = (inp - mean) / var
        l2_loss = ((rec - inp) ** 2).mean(dim=2, keepdim=False)  # (B, L, C) ==mean==> (B, L)
        non_active = mask.logical_not().int().view(mask.shape[0], -1)  # (B, 1, f, f) => (B, L)
        recon_loss = l2_loss.mul_(non_active).sum() / (non_active.sum() + 1e-8)  # loss only on masked (non-active) patches
        return recon_loss,mean,var

    def forward(self, imgs, mask_ratio=0.5,vis=False):

        latent, active, ids_restore,active_hw = self.forward_encoder(imgs, mask_ratio)
        rec = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        recon_loss,mean,var = self.forward_loss(imgs, rec, active)
        if vis:
            masked_bchw = imgs * active_hw.flatten(0,1)
            rec_bchw = self.unpatchify(rec * var + mean)
            rec_or_inp = torch.where(active_hw.flatten(0,1).bool(), imgs, rec_bchw)
            return imgs, masked_bchw, rec_or_inp
        else:
            return recon_loss


def spikmae_12_512(**kwargs):
    model = Spikmae(
        T=1,
        choice="base",
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[128,256,512],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=1000,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=12,
        sr_ratios=1, decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=4,
        **kwargs)
    return model
def spikmae_12_768(**kwargs):
    model = Spikmae(
        T=1,
        choice="large",
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[192,384,768],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=1000,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=12,
        sr_ratios=1, decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=4,
        **kwargs)
    return model




if __name__ == "__main__":
    model = spikmae_12_768()
    x=torch.randn(1,3,224,224)
    loss = model(x,mask_ratio=0.50)
    print('loss',loss)
    torchinfo.summary(model, (1, 3, 224, 224))
    print(f"number of params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
