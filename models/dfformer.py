# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MetaFormer baselines including IdentityFormer, RandFormer, PoolFormerV2,
ConvFormer and CAFormer.
Some implementations are modified from timm (https://github.com/rwightman/pytorch-image-models).
"""
import os
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers.helpers import to_2tuple


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 1.0, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'dfformer_s18': _cfg(url="https://github.com/okojoalg/dfformer/releases/download/weights/dfformer_s18.pth"),
    'dfformer_s36': _cfg(url="https://github.com/okojoalg/dfformer/releases/download/weights/dfformer_s36.pth"),
    'dfformer_m36': _cfg(url="https://github.com/okojoalg/dfformer/releases/download/weights/dfformer_m36.pth"),
    'dfformer_b36': _cfg(url="https://github.com/okojoalg/dfformer/releases/download/weights/dfformer_b36.pth"),
    'gfformer_s18': _cfg(),
    'cdfformer_s18': _cfg(url="https://github.com/okojoalg/dfformer/releases/download/weights/cdfformer_s18.pth"),
    'cdfformer_s36': _cfg(url="https://github.com/okojoalg/dfformer/releases/download/weights/cdfformer_s36.pth"),
    'cdfformer_m36': _cfg(url="https://github.com/okojoalg/dfformer/releases/download/weights/cdfformer_m36.pth"),
    'cdfformer_b36': _cfg(url="https://github.com/okojoalg/dfformer/releases/download/weights/cdfformer_b36.pth"),
    'dfformer_s18_k2': _cfg(),
    'dfformer_s18_d8': _cfg(),
    'dfformer_s18_gelu': _cfg(),
    'dfformer_s18_relu': _cfg(),
}


class Downsampling(nn.Module):
    """
    Downsampling implemented by a layer of convolution.
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0,
                 pre_norm=None, post_norm=None, pre_permute=False):
        super().__init__()
        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()
        self.pre_permute = pre_permute
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.post_norm = post_norm(out_channels) if post_norm else nn.Identity()

    def forward(self, x):
        x = self.pre_norm(x)
        if self.pre_permute:
            # if take [B, H, W, C] as input, permute it to [B, C, H, W]
            x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.post_norm(x)
        return x


class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """

    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class SquaredReLU(nn.Module):
    """
        Squared ReLU: https://arxiv.org/abs/2109.08668
    """

    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return torch.square(self.relu(x))


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class Attention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
    """

    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
                 attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0,
                                                                                  3, 1,
                                                                                  4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RandomMixing(nn.Module):
    def __init__(self, num_tokens=196, **kwargs):
        super().__init__()
        self.random_matrix = nn.parameter.Parameter(
            data=torch.softmax(torch.rand(num_tokens, num_tokens), dim=-1),
            requires_grad=False)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        x = torch.einsum('mn, bnc -> bmc', self.random_matrix, x)
        x = x.reshape(B, H, W, C)
        return x


class LayerNormGeneral(nn.Module):
    r""" General LayerNorm for different situations.
    Args:
        affine_shape (int, list or tuple): The shape of affine weight and bias.
            Usually the affine_shape=C, but in some implementation, like torch.nn.LayerNorm,
            the affine_shape is the same as normalized_dim by default. 
            To adapt to different situations, we offer this argument here.
        normalized_dim (tuple or list): Which dims to compute mean and variance. 
        scale (bool): Flag indicates whether to use scale or not.
        bias (bool): Flag indicates whether to use scale or not.
        We give several examples to show how to specify the arguments.
        LayerNorm (https://arxiv.org/abs/1607.06450):
            For input shape of (B, *, C) like (B, N, C) or (B, H, W, C),
                affine_shape=C, normalized_dim=(-1, ), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, ), scale=True, bias=True.
        Modified LayerNorm (https://arxiv.org/abs/2111.11418)
            that is idental to partial(torch.nn.GroupNorm, num_groups=1):
            For input shape of (B, N, C),
                affine_shape=C, normalized_dim=(1, 2), scale=True, bias=True;
            For input shape of (B, H, W, C),
                affine_shape=C, normalized_dim=(1, 2, 3), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, 2, 3), scale=True, bias=True.
        For the several metaformer baslines,
            IdentityFormer, RandFormer and PoolFormerV2 utilize Modified LayerNorm without bias (bias=False);
            ConvFormer and CAFormer utilizes LayerNorm without bias (bias=False).
    """

    def __init__(self, affine_shape=None, normalized_dim=(-1,), scale=True,
                 bias=True, eps=1e-5):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x


class GlobalFilter(nn.Module):
    def __init__(self, dim, expansion_ratio=2,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, size=14,
                 **kwargs, ):
        super().__init__()
        size = to_2tuple(size)
        self.size = size[0]
        self.filter_size = size[1] // 2 + 1
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()
        self.complex_weights = nn.Parameter(
            torch.randn(self.size, self.filter_size, self.med_channels, 2,
                        dtype=torch.float32) * 0.02)
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)

    def forward(self, x):
        B, H, W, _ = x.shape
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        complex_weights = torch.view_as_complex(self.complex_weights)
        x = x * complex_weights
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
        x = self.act2(x)
        x = self.pwconv2(x)
        return x


class DynamicFilter(nn.Module):
    def __init__(self, dim, expansion_ratio=2, reweight_expansion_ratio=.25,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, num_filters=4, size=14, weight_resize=False,
                 **kwargs):
        super().__init__()
        size = to_2tuple(size)
        self.size = size[0]
        self.filter_size = size[1] // 2 + 1
        self.num_filters = num_filters
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)
        self.weight_resize = weight_resize
        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()
        self.reweight = Mlp(dim, reweight_expansion_ratio, num_filters * self.med_channels)
        self.complex_weights = nn.Parameter(
            torch.randn(self.size, self.filter_size, num_filters, 2,
                        dtype=torch.float32) * 0.02)
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)

    def forward(self, x):
        B, H, W, _ = x.shape

        routeing = self.reweight(x.mean(dim=(1, 2))).view(B, self.num_filters,
                                                          -1).softmax(dim=1)
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        if self.weight_resize:
            complex_weights = resize_complex_weight(self.complex_weights, x.shape[1],
                                                    x.shape[2])
            complex_weights = torch.view_as_complex(complex_weights.contiguous())
        else:
            complex_weights = torch.view_as_complex(self.complex_weights)
        routeing = routeing.to(torch.complex64)
        weight = torch.einsum('bfc,hwf->bhwc', routeing, complex_weights)
        if self.weight_resize:
            weight = weight.view(-1, x.shape[1], x.shape[2], self.med_channels)
        else:
            weight = weight.view(-1, self.size, self.filter_size, self.med_channels)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')

        x = self.act2(x)
        x = self.pwconv2(x)
        return x


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(self, dim, expansion_ratio=2,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, kernel_size=7, padding=3,
                 **kwargs, ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=padding, groups=med_channels, bias=bias)  # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act2(x)
        x = self.pwconv2(x)
        return x


# ref https://github.com/NVlabs/AFNO-transformer/blob/master/afno/afno2d.py
class AFNO2D(nn.Module):
    def __init__(self, dim, expansion_ratio=2,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, size=14,
                 num_blocks=8, sparsity_threshold=0.01,
                 hard_thresholding_fraction=1, hidden_size_factor=1,
                 **kwargs):
        super().__init__()
        size = to_2tuple(size)
        self.size = size[0]
        self.filter_size = size[1] // 2 + 1
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)

        assert self.med_channels % num_blocks == 0, f"hidden_size {self.med_channels} should be divisble by num_blocks {num_blocks}"

        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.med_channels // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.to(torch.float32)
        bias = x
        B, H, W, C = x.shape
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        x = x.reshape(B, self.size, self.filter_size, self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, self.size, self.filter_size, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, self.size, self.filter_size, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        kept_modes = int(self.filter_size * self.hard_thresholding_fraction)

        o1_real[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, :, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, :, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, self.size, self.filter_size, C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
        x = x + bias
        x = self.act2(x)
        x = self.pwconv2(x)
        return x


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer: https://arxiv.org/abs/2111.11418
    Modfiled for [B, H, W, C] input
    """

    def __init__(self, pool_size=3, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        y = x.permute(0, 3, 1, 2)
        y = self.pool(y)
        y = y.permute(0, 2, 3, 1)
        return y - x


class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """

    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.,
                 bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MlpHead(nn.Module):
    """ MLP classification head
    """

    def __init__(self, dim, num_classes=1000, mlp_ratio=4, act_layer=SquaredReLU,
                 norm_layer=nn.LayerNorm, head_dropout=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)
        return x


class MetaFormerBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    """

    def __init__(self, dim,
                 token_mixer=nn.Identity, mlp=Mlp,
                 norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.,
                 layer_scale_init_value=None, res_scale_init_value=None,
                 size=14,
                 ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim, drop=drop, size=size)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

    def forward(self, x):
        x = self.res_scale1(x) + \
            self.layer_scale1(
                self.drop_path1(
                    self.token_mixer(self.norm1(x))
                )
            )
        x = self.res_scale2(x) + \
            self.layer_scale2(
                self.drop_path2(
                    self.mlp(self.norm2(x))
                )
            )
        return x


r"""
downsampling (stem) for the first stage is a layer of conv with k7, s4 and p2
downsamplings for the last 3 stages is a layer of conv with k3, s2 and p1
DOWNSAMPLE_LAYERS_FOUR_STAGES format: [Downsampling, Downsampling, Downsampling, Downsampling]
use `partial` to specify some arguments
"""
DOWNSAMPLE_LAYERS_FOUR_STAGES = [partial(Downsampling,
                                         kernel_size=7, stride=4, padding=2,
                                         post_norm=partial(LayerNormGeneral, bias=False,
                                                           eps=1e-6)
                                         )] + \
                                [partial(Downsampling,
                                         kernel_size=3, stride=2, padding=1,
                                         pre_norm=partial(LayerNormGeneral, bias=False,
                                                          eps=1e-6), pre_permute=True
                                         )] * 3


class MetaFormer(nn.Module):
    r""" MetaFormer
        A PyTorch impl of : `MetaFormer Baselines for Vision`  -
          https://arxiv.org/abs/2210.13452
    Args:
        in_chans (int): Number of input image channels. Default: 3.
        num_classes (int): Number of classes for classification head. Default: 1000.
        depths (list or tuple): Number of blocks at each stage. Default: [2, 2, 6, 2].
        dims (int): Feature dimension at each stage. Default: [64, 128, 320, 512].
        downsample_layers: (list or tuple): Downsampling layers before each stage.
        token_mixers (list, tuple or token_fcn): Token mixer for each stage. Default: nn.Identity.
        mlps (list, tuple or mlp_fcn): Mlp for each stage. Default: Mlp.
        norm_layers (list, tuple or norm_fcn): Norm layers for each stage. Default: partial(LayerNormGeneral, eps=1e-6, bias=False).
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_dropout (float): dropout for MLP classifier. Default: 0.
        layer_scale_init_values (list, tuple, float or None): Init value for Layer Scale. Default: None.
            None means not use the layer scale. Form: https://arxiv.org/abs/2103.17239.
        res_scale_init_values (list, tuple, float or None): Init value for Layer Scale. Default: [None, None, 1.0, 1.0].
            None means not use the layer scale. From: https://arxiv.org/abs/2110.09456.
        fork_feat (bool): whether output features of the 4 stages, for dense prediction
        output_norm: norm before classifier head. Default: partial(nn.LayerNorm, eps=1e-6).
        head_fn: classification head. Default: nn.Linear.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[2, 2, 6, 2],
                 dims=[64, 128, 320, 512],
                 downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
                 token_mixers=nn.Identity,
                 mlps=Mlp,
                 norm_layers=partial(LayerNormGeneral, eps=1e-6, bias=False),
                 drop_path_rate=0.,
                 head_dropout=0.0,
                 layer_scale_init_values=None,
                 res_scale_init_values=[None, None, 1.0, 1.0],
                 fork_feat=False,
                 output_norm=partial(nn.LayerNorm, eps=1e-6),
                 head_fn=nn.Linear,
                 input_size=(3, 224, 224),
                 **kwargs,
                 ):
        super().__init__()
        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        if not isinstance(depths, (list, tuple)):
            depths = [depths]  # it means the model has only one stage
        if not isinstance(dims, (list, tuple)):
            dims = [dims]

        num_stage = len(depths)
        self.num_stage = num_stage

        if not isinstance(downsample_layers, (list, tuple)):
            downsample_layers = [downsample_layers] * num_stage
        down_dims = [in_chans] + dims
        self.downsample_layers = nn.ModuleList(
            [downsample_layers[i](down_dims[i], down_dims[i + 1]) for i in
             range(num_stage)]
        )

        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * num_stage

        if not isinstance(mlps, (list, tuple)):
            mlps = [mlps] * num_stage

        if not isinstance(norm_layers, (list, tuple)):
            norm_layers = [norm_layers] * num_stage

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        if not isinstance(layer_scale_init_values, (list, tuple)):
            layer_scale_init_values = [layer_scale_init_values] * num_stage
        if not isinstance(res_scale_init_values, (list, tuple)):
            res_scale_init_values = [res_scale_init_values] * num_stage

        self.stages = nn.ModuleList()  # each stage consists of multiple metaformer blocks
        cur = 0
        for i in range(num_stage):
            stage = nn.Sequential(
                *[MetaFormerBlock(dim=dims[i],
                                  token_mixer=token_mixers[i],
                                  mlp=mlps[i],
                                  norm_layer=norm_layers[i],
                                  drop_path=dp_rates[cur + j],
                                  layer_scale_init_value=layer_scale_init_values[i],
                                  res_scale_init_value=res_scale_init_values[i],
                                  size=(input_size[1] // (2 ** (i + 2)),
                                        input_size[2] // (2 ** (i + 2))),
                                  ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        if self.fork_feat:
            # add a norm layer for each output
            for i in range(4):
                if i == 0 and os.environ.get('FORK_LAST3', None):
                    # TODO: more elegant way
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = output_norm(dims[i])
                layer_name = f'norm{i}'
                self.add_module(layer_name, layer)
        else:
            self.norm = output_norm(dims[-1])

            if head_dropout > 0.0:
                self.head = head_fn(dims[-1], num_classes, head_dropout=head_dropout)
            else:
                self.head = head_fn(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'norm'}

    def forward_features(self, x):
        outs = []
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if self.fork_feat:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out.permute(0, 3, 1, 2))
        if self.fork_feat:
            # output the features of four stages for dense prediction
            return outs
        return self.norm(x.mean([1, 2]))  # (B, H, W, C) -> (B, C)

    def forward(self, x):
        x = self.forward_features(x)
        if self.fork_feat:
            # output features of four stages for dense prediction
            return x
        x = self.head(x)
        return x


def resize_complex_weight(origin_weight, new_h, new_w):
    h, w, num_heads = origin_weight.shape[0:3]  # size, w, c, 2
    origin_weight = origin_weight.reshape(1, h, w, num_heads * 2).permute(0, 3, 1, 2)
    new_weight = torch.nn.functional.interpolate(
        origin_weight,
        size=(new_h, new_w),
        mode='bicubic',
        align_corners=True
    ).permute(0, 2, 3, 1).reshape(new_h, new_w, num_heads, 2)
    return new_weight


def load_weights(model, input_size):
    out_dict = {}
    state_dict = torch.hub.load_state_dict_from_url(
        url=model.default_cfg['url'], map_location="cpu", check_hash=True)
    for k, v in state_dict.items():
        if 'complex_weights' in k:
            if 'stages.0' in k:
                size = input_size[1] // 4
                filter_size = input_size[2] // 8 + 1
            elif 'stages.1' in k:
                size = input_size[1] // 8
                filter_size = input_size[2] // 16 + 1
            elif 'stages.2' in k:
                size = input_size[1] // 16
                filter_size = input_size[2] // 32 + 1
            elif 'stages.3' in k:
                size = input_size[1] // 32
                filter_size = input_size[2] // 64 + 1
            v = resize_complex_weight(v, size, filter_size)
        out_dict[k] = v
    model.load_state_dict(out_dict, strict=False)


@register_model
def dfformer_s18(pretrained=False, **kwargs):
    default_cfg = default_cfgs['dfformer_s18']
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=DynamicFilter,
        head_fn=MlpHead,
        input_size=default_cfg['input_size'],
        **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_weights(model, default_cfg['input_size'])
    return model


@register_model
def dfformer_s36(pretrained=False, **kwargs):
    default_cfg = default_cfgs['dfformer_s36']
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        token_mixers=DynamicFilter,
        head_fn=MlpHead,
        input_size=default_cfg['input_size'],
        **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_weights(model, default_cfg['input_size'])
    return model


@register_model
def dfformer_m36(pretrained=False, **kwargs):
    default_cfg = default_cfgs['dfformer_m36']
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[96, 192, 384, 576],
        token_mixers=DynamicFilter,
        head_fn=MlpHead,
        input_size=default_cfg['input_size'],
        **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_weights(model, default_cfg['input_size'])
    return model


@register_model
def dfformer_b36(pretrained=False, **kwargs):
    default_cfg = default_cfgs['dfformer_b36']
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=DynamicFilter,
        head_fn=MlpHead,
        input_size=default_cfg['input_size'],
        **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_weights(model, default_cfg['input_size'])
    return model


@register_model
def gfformer_s18(pretrained=False, **kwargs):
    default_cfg = default_cfgs['gfformer_s18']
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=GlobalFilter,
        head_fn=MlpHead,
        input_size=default_cfg['input_size'],
        **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_weights(model, default_cfg['input_size'])
    return model


@register_model
def cdfformer_s18(pretrained=False, **kwargs):
    default_cfg = default_cfgs['cdfformer_s18']
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, DynamicFilter, DynamicFilter],
        head_fn=MlpHead,
        input_size=default_cfg['input_size'],
        **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_weights(model, default_cfg['input_size'])
    return model


@register_model
def cdfformer_s36(pretrained=False, **kwargs):
    default_cfg = default_cfgs['cdfformer_s36']
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, DynamicFilter, DynamicFilter],
        head_fn=MlpHead,
        input_size=default_cfg['input_size'],
        **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_weights(model, default_cfg['input_size'])
    return model


@register_model
def cdfformer_m36(pretrained=False, **kwargs):
    default_cfg = default_cfgs['cdfformer_m36']
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[96, 192, 384, 576],
        token_mixers=[SepConv, SepConv, DynamicFilter, DynamicFilter],
        head_fn=MlpHead,
        input_size=default_cfg['input_size'],
        **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_weights(model, default_cfg['input_size'])
    return model


@register_model
def cdfformer_b36(pretrained=False, **kwargs):
    default_cfg = default_cfgs['cdfformer_b36']
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=[SepConv, SepConv, DynamicFilter, DynamicFilter],
        head_fn=MlpHead,
        input_size=default_cfg['input_size'],
        **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_weights(model, default_cfg['input_size'])
    return model


# ablation


@register_model
def dfformer_s18_gelu(pretrained=False, **kwargs):
    default_cfg = default_cfgs['dfformer_s18_gelu']
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        mlps=partial(Mlp, act_layer=nn.GELU),
        token_mixers=partial(DynamicFilter, act1_layer=nn.GELU),
        head_fn=MlpHead,
        input_size=default_cfg['input_size'],
        **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_weights(model, default_cfg['input_size'])
    return model


@register_model
def dfformer_s18_relu(pretrained=False, **kwargs):
    default_cfg = default_cfgs['dfformer_s18_relu']
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        mlps=partial(Mlp, act_layer=nn.ReLU),
        token_mixers=partial(DynamicFilter, act1_layer=nn.ReLU),
        head_fn=MlpHead,
        input_size=default_cfg['input_size'],
        **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_weights(model, default_cfg['input_size'])
    return model


@register_model
def dfformer_s18_k2(pretrained=False, **kwargs):
    default_cfg = default_cfgs['dfformer_s18_k2']
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=partial(DynamicFilter, num_filters=2),
        head_fn=MlpHead,
        input_size=default_cfg['input_size'],
        **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_weights(model, default_cfg['input_size'])
    return model


@register_model
def dfformer_s18_d8(pretrained=False, **kwargs):
    default_cfg = default_cfgs['dfformer_s18_d8']
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=partial(DynamicFilter, reweight_expansion_ratio=.125),
        head_fn=MlpHead,
        input_size=default_cfg['input_size'],
        **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_weights(model, default_cfg['input_size'])
    return model


@register_model
def dfformer_s18_afno(pretrained=False, **kwargs):
    default_cfg = default_cfgs['dfformer_s18_k2']
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=AFNO2D,
        head_fn=MlpHead,
        input_size=default_cfg['input_size'],
        **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_weights(model, default_cfg['input_size'])
    return model
