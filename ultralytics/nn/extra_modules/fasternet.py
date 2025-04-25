# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch, yaml
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
from typing import List
from torch import Tensor
import copy
import os
import numpy as np
from pytorch_wavelets import DWTForward
from ..modules.conv import DWConv, Conv


__all__ = ['fasternet_t0', 'fasternet_t1', 'fasternet_t2', 'fasternet_s', 'fasternet_m', 'fasternet_l','fasternet_t0_dw','fasternet_t1_dw']


class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


class MLPBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 act_layer,
                 norm_layer,
                 pconv_fw_type,
                 groups
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1,groups=groups, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1,groups=groups, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class BasicStage(nn.Module):

    def __init__(self,
                 dim,
                 depth,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 norm_layer,
                 act_layer,
                 pconv_fw_type,
                 groups
                 ):
        super().__init__()

        blocks_list = [
            MLPBlock(
                dim=dim,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pconv_fw_type=pconv_fw_type,
                groups=groups
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x
class PatchMerging(nn.Module):

    def __init__(self, patch_size2, patch_stride2, dim, norm_layer):
        super().__init__()
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=patch_size2, stride=patch_stride2, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(2 * dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.reduction(x))
        return x

class PatchEmbed(nn.Module):

    def __init__(self, patch_size, patch_stride, in_chans, embed_dim, norm_layer):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.proj(x))
        return x

########  Wavelet module
class DW_down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.dw = DWTForward(J=1, mode='zero', wave='haar')
        #self.cv1 = DWConv(in_channel, in_channel, 3, 2, 1)
        # self.cv2 = Conv(in_channel*4, out_channel, 1,act=nn.ReLU())
        self.act = nn.GELU()

    def forward(self, x):
        yL, yH = self.dw(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x1 = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        #x2 = self.cv1(x)
        #return self.act(torch.cat((x1, x2), 1))
        return x1


class DW_down_reduce(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.dw = DWTForward(J=1, mode='zero', wave='haar')
        #self.cv1 = DWConv(in_channel, in_channel, 3, 2, 1)
        self.cv2 = Conv(in_channel*4, in_channel*2, 1,act=nn.ReLU())
        self.act = nn.GELU()

    def forward(self, x):
        yL, yH = self.dw(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x1 = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        #x2 = self.cv1(x)
        #return self.act(torch.cat((x1, , 1))
        return self.cv2(x1)

# class DW_down(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super().__init__()
#         self.dw = DWTForward(J=1, mode='zero', wave='haar')
#         self.reduce = nn.Conv2d(in_channel * 4, out_channel, kernel_size=1)  # 降维减少参数
#         self.local_conv = nn.Sequential(
#             nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, groups=out_channel),  # Depthwise
#             nn.Conv2d(out_channel, out_channel, kernel_size=1)  # Pointwise
#         )
#         self.act = nn.ReLU6()
#
#     def forward(self, x):
#         yL, yH = self.dw(x)
#         y_HL = yH[0][:, :, 0, ::]
#         y_LH = yH[0][:, :, 1, ::]
#         y_HH = yH[0][:, :, 2, ::]
#         x1 = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)  # 4 * in_channel
#         x1 = self.reduce(x1)  # 降维
#         return self.act(self.local_conv(x1))  # 加强局部信息

class DW_embedded(nn.Module):
    def __init__(self, patch_size, patch_stride, in_chans, embed_dim, norm_layer):
        super().__init__()
        #self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, bias=False)
        self.conv_path = Conv(in_chans, in_chans*4, k=3, s=2, p=1, act=nn.GELU)
        self.DW_path = DW_down(in_chans, 4*in_chans)
        self.combine_conv = nn.Conv2d(in_chans*8, embed_dim, kernel_size=3, stride=2, padding=1,groups=2)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self,x):
        x1 = self.conv_path(x)
        x2 = self.DW_path(x)
        return self.norm(self.combine_conv(torch.cat((x1,x2),1)))


# class DW_embedded(nn.Module):
    # def __init__(self, patch_size, patch_stride, in_chans, embed_dim, norm_layer):
    #     super().__init__()
    #     self.conv_path = nn.Conv2d(in_chans, in_chans * 2, kernel_size=3, stride=2,padding=1, bias=False)
    #     self.DW_path = DW_down(in_chans, in_chans * 2)  # 降低输出通道数
    #     self.combine_conv = nn.Sequential(
    #         nn.Conv2d(in_chans*4, in_chans*4, kernel_size=3, stride=2, padding=1, groups=in_chans*4),
    #         nn.Conv2d(in_chans*4, embed_dim, kernel_size=1, stride=1)
    #     )
    #     self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    #
    # def forward(self, x):
    #     x1 = self.conv_path(x)
    #     x2 = self.DW_path(x)
    #     return self.norm(self.combine_conv(torch.cat((x1, x2), dim=1)))


class PatchMerging_DW(nn.Module):

    def __init__(self, patch_size2, patch_stride2, dim, norm_layer):
        super().__init__()
        self.reduction = nn.Conv2d(int(dim/2), dim, kernel_size=patch_size2, stride=patch_stride2, bias=False)
        self.dw_reduction = DW_down_reduce(int(dim/2), dim)
        self.conv = DWConv(2*dim, 2*dim)
        if norm_layer is not None:
            self.norm = norm_layer(2 * dim)
        else:
            self.norm = nn.Identity()
        self.cmce = ImprovedCMCE(dim)

    def forward(self, x: Tensor) -> Tensor:
        x1,x2 = torch.chunk(x, 2, dim=1)
        x1 = self.reduction(x1)
        x2 = self.dw_reduction(x2)
        x = self.norm(self.cmce(self.conv(torch.cat((x1,x2), dim=1))))
        return x

class ImprovedCMCE(nn.Module):
    def __init__(self, channels, reduction_ratio=2):
        super(ImprovedCMCE, self).__init__()
        
        # 中间通道数，避免过度压缩
        mid_channels = max(channels // reduction_ratio, 8)
        
        # 特征提取分支
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 相互增强分支
        self.mutual_enhance = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 输出调整
        self.output_conv = nn.Conv2d(channels*2, channels*2, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        fa, fb = x.chunk(2, dim=1)
        
        # 通道注意力，更有效地计算特征重要性
        b, c, h, w = fa.size()
        
        # 计算互相关性
        fa_pool = fa.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
        fb_pool = fb.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
        
        # 计算通道注意力权重
        wa = self.channel_attention(fa_pool)
        wb = self.channel_attention(fb_pool)
        
        # 交叉增强
        fa_enhanced = fa + fb * wa
        fb_enhanced = fb + fa * wb
        
        # 特征融合（添加非线性交互）
        fused = self.mutual_enhance(torch.cat([fa, fb], dim=1))
        
        # 残差连接
        fa_new = fa_enhanced + fused
        fb_new = fb_enhanced + fused
        
        # 最终输出
        output = self.relu(self.output_conv(torch.cat([fa_new, fb_new], dim=1)))
        
        return output

class CMCE2(nn.Module):
    def __init__(self, in_channel=3):
        super(CMCE2, self).__init__()
        self.relu = nn.ReLU()

        self.l1 = nn.Linear(in_channel, in_channel // 2)
        #self.norm = nn.BatchNorm1d(in_channel // 2, track_running_stats=True)
    
        self.l2 = nn.Linear(in_channel // 2, in_channel)

    def forward(self, x):
        fa, fb = x.chunk(2, dim=1)
        (b1, c1, h1, w1), (b2, c2, h2, w2) = fa.size(), fb.size()
        assert c1 == c2

        s_cos_sim = F.cosine_similarity(fa.view(b1, c1, h1 * w1), fb.view(b2, c2, h2 * w2), dim=2).view(b1, -1)

        w = F.sigmoid(self.l2(self.relu((self.l1(s_cos_sim))))).view(b1, -1, 1, 1)


        cos_sim = F.cosine_similarity(fa, fb, dim=1)
        cos_sim = cos_sim.unsqueeze(1)
        fa_new = fa + fb * (cos_sim) * w
        fb_new = fb + fa * (cos_sim) * (1 - w)

        fa_new = self.relu(fa_new)
        fb_new = self.relu(fb_new)

        return torch.cat((fa_new, fb_new), dim=1)

########  Wavelet module

class FasterNet_DWave(nn.Module):
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=96,
                 depths=(1, 2, 8, 2),
                 mlp_ratio=2.,
                 n_div=4,
                 patch_size=4,
                 patch_stride=4,
                 patch_size2=2,  # for subsequent layers
                 patch_stride2=2,
                 patch_norm=True,
                 feature_dim=1280,
                 drop_path_rate=0.1,
                 layer_scale_init_value=0,
                 norm_layer='BN',
                 act_layer='RELU',
                 init_cfg=None,
                 pretrained=None,
                 pconv_fw_type='split_cat',
                 groups=2,
                 **kwargs):
        super().__init__()

        if norm_layer == 'BN':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError

        if act_layer == 'GELU':
            act_layer = nn.GELU
        elif act_layer == 'RELU':
            act_layer = partial(nn.ReLU, inplace=True)
        else:
            raise NotImplementedError

        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_stages - 1))
        self.mlp_ratio = mlp_ratio
        self.depths = depths

        # split image into non-overlapping patches
        # self.patch_embed = PatchEmbed(
        #     patch_size=patch_size,
        #     patch_stride=patch_stride,
        #     in_chans=in_chans,
        #     embed_dim=embed_dim,
        #     norm_layer=norm_layer if self.patch_norm else None
        # )

        self.patch_embed = DW_embedded(patch_size=patch_size,
                                       patch_stride=patch_stride,
                                       in_chans=in_chans,
                                       embed_dim=embed_dim,
                                       norm_layer=norm_layer if self.patch_norm else None
                                       )
        # stochastic depth decay rule
        dpr = [x.item()
               for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        stages_list = []
        for i_stage in range(self.num_stages):
            stage = BasicStage(dim=int(embed_dim * 2 ** i_stage),
                               n_div=n_div,
                               depth=depths[i_stage],
                               mlp_ratio=self.mlp_ratio,
                               drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                               layer_scale_init_value=layer_scale_init_value,
                               norm_layer=norm_layer,
                               act_layer=act_layer,
                               pconv_fw_type=pconv_fw_type,
                               groups=groups
                               )
            stages_list.append(stage)


            # if i_stage < self.num_stages - 1:
            #     stages_list.append(
            #         PatchMerging(patch_size2=patch_size2,
            #                      patch_stride2=patch_stride2,
            #                      dim=int(embed_dim * 2 ** i_stage),
            #                      norm_layer=norm_layer)
            #     )

            # patch merging layer
            if i_stage < self.num_stages - 1:
                if i_stage>=2:
                    stages_list.append(
                        PatchMerging(patch_size2=patch_size2,
                                     patch_stride2=patch_stride2,
                                     dim=int(embed_dim * 2 ** i_stage),
                                     norm_layer=norm_layer)
                    )
                else:
                    stages_list.append(
                        PatchMerging_DW(patch_size2=patch_size2,
                                        patch_stride2=patch_stride2,
                                        dim=int(embed_dim * 2 ** i_stage),
                                        norm_layer=norm_layer)
                    )

        self.stages = nn.Sequential(*stages_list)

        # add a norm layer for each output
        self.out_indices = [0, 2, 4, 6]
        for i_emb, i_layer in enumerate(self.out_indices):
            if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                raise NotImplementedError
            else:
                layer = norm_layer(int(embed_dim * 2 ** i_emb))
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.channel = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]

    def forward(self, x: Tensor) -> Tensor:
        # output the features of four stages for dense prediction
        x = self.patch_embed(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        return outs






class FasterNet(nn.Module):
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=96,
                 depths=(1, 2, 8, 2),
                 mlp_ratio=2.,
                 n_div=4,
                 patch_size=4,
                 patch_stride=4,
                 patch_size2=2,  # for subsequent layers
                 patch_stride2=2,
                 patch_norm=True,
                 feature_dim=1280,
                 drop_path_rate=0.1,
                 layer_scale_init_value=0,
                 norm_layer='BN',
                 act_layer='RELU',
                 init_cfg=None,
                 pretrained=None,
                 pconv_fw_type='split_cat',
                 groups=1,
                 **kwargs):
        super().__init__()

        if norm_layer == 'BN':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError

        if act_layer == 'GELU':
            act_layer = nn.GELU
        elif act_layer == 'RELU':
            act_layer = partial(nn.ReLU, inplace=True)
        else:
            raise NotImplementedError

        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_stages - 1))
        self.mlp_ratio = mlp_ratio
        self.depths = depths

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            patch_stride=patch_stride,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )

        # self.patch_embed = DW_embedded(patch_size=patch_size,
        #                                patch_stride=patch_stride,
        #                                in_chans=in_chans,
        #                                embed_dim=embed_dim,
        #                                norm_layer=norm_layer if self.patch_norm else None
        #                                )
        # stochastic depth decay rule
        dpr = [x.item()
               for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        stages_list = []
        for i_stage in range(self.num_stages):
            stage = BasicStage(dim=int(embed_dim * 2 ** i_stage),
                               n_div=n_div,
                               depth=depths[i_stage],
                               mlp_ratio=self.mlp_ratio,
                               drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                               layer_scale_init_value=layer_scale_init_value,
                               norm_layer=norm_layer,
                               act_layer=act_layer,
                               pconv_fw_type=pconv_fw_type,
                               groups=groups
                               )
            stages_list.append(stage)

            # patch merging layer
            if i_stage < self.num_stages - 1:
                stages_list.append(
                    PatchMerging(patch_size2=patch_size2,
                                 patch_stride2=patch_stride2,
                                 dim=int(embed_dim * 2 ** i_stage),
                                 norm_layer=norm_layer)
                )

        self.stages = nn.Sequential(*stages_list)

        # add a norm layer for each output
        self.out_indices = [0, 2, 4, 6]
        for i_emb, i_layer in enumerate(self.out_indices):
            if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                raise NotImplementedError
            else:
                layer = norm_layer(int(embed_dim * 2 ** i_emb))
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.channel = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]

    def forward(self, x: Tensor) -> Tensor:
        # output the features of four stages for dense prediction
        x = self.patch_embed(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        return outs




def update_weight(model_dict, weight_dict):
    idx, temp_dict = 0, {}
    for k, v in weight_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            idx += 1
    model_dict.update(temp_dict)
    print(f'loading weights... {idx}/{len(model_dict)} items')
    return model_dict


def fasternet_t0(weights=None, cfg='ultralytics/nn/extra_modules/cfg/fasternet_t0.yaml'):
    with open(cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    model = FasterNet(**cfg)
    if weights is not None:
        pretrain_weight = torch.load(weights, map_location='cpu')
        model.load_state_dict(update_weight(model.state_dict(), pretrain_weight))
    return model

def fasternet_t0_dw(weights=None, cfg='ultralytics/nn/extra_modules/cfg/fasternet_t0_dw.yaml'):
    with open(cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    model = FasterNet_DWave(**cfg)
    if weights is not None:
        pretrain_weight = torch.load(weights, map_location='cpu')
        model.load_state_dict(update_weight(model.state_dict(), pretrain_weight))
    return model

def fasternet_t1(weights=None, cfg='ultralytics/nn/extra_modules/cfg/fasternet_t1.yaml'):
    with open(cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    model = FasterNet(**cfg)
    if weights is not None:
        pretrain_weight = torch.load(weights, map_location='cpu')
        model.load_state_dict(update_weight(model.state_dict(), pretrain_weight))
    return model

def fasternet_t1_dw(weights=None, cfg='ultralytics/nn/extra_modules/cfg/fasternet_t1_dw.yaml'):
    with open(cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    model = FasterNet_DWave(**cfg)
    if weights is not None:
        pretrain_weight = torch.load(weights, map_location='cpu')
        model.load_state_dict(update_weight(model.state_dict(), pretrain_weight))
    return model

def fasternet_t2(weights=None, cfg='ultralytics/nn/extra_modules/cfg/fasternet_t2.yaml'):
    with open(cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    model = FasterNet(**cfg)
    if weights is not None:
        pretrain_weight = torch.load(weights, map_location='cpu')
        model.load_state_dict(update_weight(model.state_dict(), pretrain_weight))
    return model


def fasternet_s(weights=None, cfg='ultralytics/nn/extra_modules/cfg/fasternet_s.yaml'):
    with open(cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    model = FasterNet(**cfg)
    if weights is not None:
        pretrain_weight = torch.load(weights, map_location='cpu')
        model.load_state_dict(update_weight(model.state_dict(), pretrain_weight))
    return model


def fasternet_m(weights=None, cfg='ultralytics/nn/extra_modules/cfg/fasternet_m.yaml'):
    with open(cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    model = FasterNet(**cfg)
    if weights is not None:
        pretrain_weight = torch.load(weights, map_location='cpu')
        model.load_state_dict(update_weight(model.state_dict(), pretrain_weight))
    return model


def fasternet_l(weights=None, cfg='ultralytics/nn/extra_modules/cfg/fasternet_l.yaml'):
    with open(cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    model = FasterNet(**cfg)
    if weights is not None:
        pretrain_weight = torch.load(weights, map_location='cpu')
        model.load_state_dict(update_weight(model.state_dict(), pretrain_weight))
    return model


if __name__ == '__main__':
    import yaml

    model = fasternet_t0( cfg='cfg/fasternet_t0.yaml')
    print(model.channel)
    inputs = torch.randn((1, 3, 640, 640))
    for i in model(inputs):
        print(i.size())