"""Network architectures for CUT and CycleGAN baselines.

Based on official implementations:
- ResnetGenerator: CycleGAN official (junyanz/pytorch-CycleGAN-and-pix2pix)
  with encode_only feature from CUT official (taesungp/contrastive-unpaired-translation)
- NLayerDiscriminator: CycleGAN official
- PatchSampleF: CUT official (MLP projection for PatchNCE)
"""

import functools

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init


class Identity(nn.Module):
    """Identity pass-through used as a no-op normalization layer."""

    def forward(self, x):
        return x


def get_norm_layer(norm_type="instance"):
    """Return a normalization layer.

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    """
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == "none":
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float) -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def init_net(net, init_type="normal", init_gain=0.02, device=None):
    """Initialize a network and optionally move to specified device."""
    if device is not None:
        net.to(device)
    init_weights(net, init_type, init_gain=init_gain)
    return net


# --------------------------------------------------------------------------- #
#  Normalization utility
# --------------------------------------------------------------------------- #

class Normalize(nn.Module):
    """L2 normalization along a given dimension."""

    def __init__(self, power=2):
        super().__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        return x.div(norm + 1e-7)


# --------------------------------------------------------------------------- #
#  Generator
# --------------------------------------------------------------------------- #

class ResnetGenerator(nn.Module):
    """Resnet-based generator with optional encode_only mode for CUT NCE loss.

    Architecture: c7s1-ngf, d-2ngf, d-4ngf, R×n_blocks, u-2ngf, u-ngf, c7s1-output, Tanh

    Based on CycleGAN official with encode_only from CUT official.

    Parameters:
        input_nc (int)      -- the number of channels in input images (default 1 for MRI)
        output_nc (int)     -- the number of channels in output images
        ngf (int)           -- the number of filters in the last conv layer
        norm_layer          -- normalization layer
        use_dropout (bool)  -- if use dropout layers
        n_blocks (int)      -- the number of ResNet blocks
        padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
    """

    def __init__(self, input_nc=1, output_nc=1, ngf=64,
                 norm_layer=nn.InstanceNorm2d, use_dropout=False,
                 n_blocks=9, padding_type="reflect"):
        assert n_blocks >= 0
        super().__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]

        # ResNet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                                  norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                   kernel_size=3, stride=2, padding=1, output_padding=1,
                                   bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, layers=[], encode_only=False):
        """Forward pass with optional intermediate feature extraction for CUT.

        Args:
            input: Input tensor.
            layers: List of layer indices to extract features from (for CUT NCE loss).
                   If empty, standard forward pass.
            encode_only: If True, return only intermediate features (stop after last requested layer).

        Returns:
            If layers is empty: output tensor.
            If layers is non-empty and encode_only: list of intermediate features.
            If layers is non-empty and not encode_only: (output, list of intermediate features).
        """
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in layers:
                    feats.append(feat)
                if layer_id == layers[-1] and encode_only:
                    return feats
            return feat, feats
        else:
            return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block with skip connections."""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super().__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# --------------------------------------------------------------------------- #
#  Discriminator
# --------------------------------------------------------------------------- #

class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator (70x70 receptive field with default n_layers=3).

    Based on CycleGAN official.

    Parameters:
        input_nc (int) -- the number of channels in input images (default 1 for MRI)
        ndf (int)      -- the number of filters in the first conv layer
        n_layers (int) -- the number of conv layers in the discriminator
        norm_layer     -- normalization layer
    """

    def __init__(self, input_nc=1, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


# --------------------------------------------------------------------------- #
#  PatchSampleF — MLP projection head for CUT NCE loss
# --------------------------------------------------------------------------- #

class PatchSampleF(nn.Module):
    """Feature sampling + MLP projection for PatchNCE loss.

    Based on CUT official. Samples spatial patches from feature maps and
    projects them through per-layer MLPs for contrastive learning.

    Parameters:
        use_mlp (bool) -- if True, use MLP projection heads
        nc (int)       -- output feature dimension (default 256)
    """

    def __init__(self, use_mlp=True, nc=256, init_type="normal", init_gain=0.02):
        super().__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain

    def create_mlp(self, feats):
        """Lazily create MLP heads based on feature dimensions."""
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(
                nn.Linear(input_nc, self.nc),
                nn.ReLU(),
                nn.Linear(self.nc, self.nc),
            )
            mlp.to(feat.device)
            setattr(self, "mlp_%d" % mlp_id, mlp)
        init_weights(self, self.init_type, self.init_gain)
        self.mlp_init = True

    def forward(self, feats, num_patches=256, patch_ids=None):
        """Sample patches from features and project through MLPs.

        Args:
            feats: List of feature maps from encoder layers.
            num_patches: Number of spatial patches to sample per layer.
            patch_ids: Pre-computed patch indices (for consistency between src/tgt).

        Returns:
            return_feats: List of projected feature vectors.
            return_ids: List of sampled patch indices.
        """
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = np.random.permutation(feat_reshape.shape[1])
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]
                patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, "mlp_%d" % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids
