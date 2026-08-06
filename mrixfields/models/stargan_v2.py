"""StarGAN v2 for multi-domain image translation (Task 3: Any→Any).

Based on the official StarGAN v2 implementation:
    clovaai/stargan-v2/core/model.py

Reference: Choi et al., "StarGAN v2: Diverse Image Synthesis for Multiple
Domains", CVPR 2020.

Adaptations from official code:
    - 3-channel RGB → 1-channel grayscale MRI
    - num_domains: 2 → 5 (0.1T, 1.5T, 3T, 5T, 7T)
    - Removed HighPass filter and FAN face detection (not relevant for MRI)
    - Removed nn.DataParallel wrapping (handled externally)

Usage for Task 3: Single model trained on all 5 field strengths simultaneously.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


_VALID_IMG_SIZES = {32, 64, 128, 256, 512}


def _validate_img_size(img_size: int) -> None:
    """Validate that img_size is a supported power of 2."""
    if img_size not in _VALID_IMG_SIZES:
        raise ValueError(
            f"img_size must be one of {sorted(_VALID_IMG_SIZES)}, got {img_size}"
        )


# --------------------------------------------------------------------------- #
#  Building blocks
# --------------------------------------------------------------------------- #

class ResBlk(nn.Module):
    """Residual block with optional downsampling and normalization.

    Based on StarGAN v2 official.
    """

    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)


class AdaIN(nn.Module):
    """Adaptive Instance Normalization.

    Modulates features using style code: (1 + gamma) * norm(x) + beta
    """

    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    """Residual block with AdaIN for style injection.

    Based on StarGAN v2 official.
    """

    def __init__(self, dim_in, dim_out, style_dim=64,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


# --------------------------------------------------------------------------- #
#  Generator
# --------------------------------------------------------------------------- #

class StarGANv2Generator(nn.Module):
    """StarGAN v2 generator with AdaIN style injection.

    Encoder-decoder architecture where decoder blocks use AdaIN
    to inject target style codes.

    Adapted: 3ch RGB → 1ch grayscale MRI, removed HighPass/FAN.

    Parameters:
        img_size (int) -- input image size (default 128 for MRI slices)
        style_dim (int) -- style code dimension
        max_conv_dim (int) -- maximum conv channels
        input_nc (int) -- input channels (1 for grayscale MRI)
    """

    def __init__(self, img_size=128, style_dim=64, max_conv_dim=512, input_nc=1):
        super().__init__()
        _validate_img_size(img_size)
        # Initial channel count scales inversely with image size (StarGAN v2 convention).
        dim_in = 2 ** 14 // img_size
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(input_nc, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, input_nc, 1, 1, 0),
            nn.Tanh(),
        )

        # Down/up-sampling blocks
        repeat_num = int(math.log2(img_size)) - 4
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim, upsample=True))
            dim_in = dim_out

        # Bottleneck blocks
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim))

    def forward(self, x, s):
        """Forward pass.

        Args:
            x: Input image (B, 1, H, W).
            s: Style code (B, style_dim).

        Returns:
            Generated image (B, 1, H, W).
        """
        x = self.from_rgb(x)
        for block in self.encode:
            x = block(x)
        for block in self.decode:
            x = block(x, s)
        return self.to_rgb(x)


# --------------------------------------------------------------------------- #
#  Mapping Network
# --------------------------------------------------------------------------- #

class MappingNetwork(nn.Module):
    """Maps random latent codes to domain-specific style codes.

    Shared MLP layers followed by per-domain branches.

    Parameters:
        latent_dim (int) -- input latent dimension
        style_dim (int) -- output style dimension
        num_domains (int) -- number of domains (5 for MRIxFields)
    """

    def __init__(self, latent_dim=16, style_dim=64, num_domains=5):
        super().__init__()
        layers = [nn.Linear(latent_dim, 512), nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512), nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared.append(nn.Sequential(
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, style_dim),
            ))

    def forward(self, z, y):
        """
        Args:
            z: Latent code (B, latent_dim).
            y: Target domain label (B,) as LongTensor.

        Returns:
            Style code (B, style_dim).
        """
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out.append(layer(h))
        out = torch.stack(out, dim=1)  # (B, num_domains, style_dim)
        idx = torch.arange(y.size(0), device=y.device)
        s = out[idx, y]  # (B, style_dim)
        return s


# --------------------------------------------------------------------------- #
#  Style Encoder
# --------------------------------------------------------------------------- #

class StyleEncoder(nn.Module):
    """Encodes a reference image into a domain-specific style code.

    Parameters:
        img_size (int) -- input image size
        style_dim (int) -- output style dimension
        num_domains (int) -- number of domains
        max_conv_dim (int) -- maximum conv channels
        input_nc (int) -- input channels (1 for grayscale MRI)
    """

    def __init__(self, img_size=128, style_dim=64, num_domains=5,
                 max_conv_dim=512, input_nc=1):
        super().__init__()
        _validate_img_size(img_size)
        # Initial channel count scales inversely with image size (StarGAN v2 convention).
        dim_in = 2 ** 14 // img_size
        blocks = [nn.Conv2d(input_nc, dim_in, 3, 1, 1)]

        repeat_num = int(math.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks.append(ResBlk(dim_in, dim_out, downsample=True))
            dim_in = dim_out

        blocks.append(nn.LeakyReLU(0.2))
        blocks.append(nn.Conv2d(dim_out, dim_out, 4, 1, 0))
        blocks.append(nn.LeakyReLU(0.2))
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared.append(nn.Linear(dim_out, style_dim))

    def forward(self, x, y):
        """
        Args:
            x: Reference image (B, 1, H, W).
            y: Domain label (B,) as LongTensor.

        Returns:
            Style code (B, style_dim).
        """
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out.append(layer(h))
        out = torch.stack(out, dim=1)  # (B, num_domains, style_dim)
        idx = torch.arange(y.size(0), device=y.device)
        s = out[idx, y]  # (B, style_dim)
        return s


# --------------------------------------------------------------------------- #
#  Discriminator
# --------------------------------------------------------------------------- #

class StarGANv2Discriminator(nn.Module):
    """Multi-domain discriminator for StarGAN v2.

    Outputs one real/fake score per domain.

    Parameters:
        img_size (int) -- input image size
        num_domains (int) -- number of domains
        max_conv_dim (int) -- maximum conv channels
        input_nc (int) -- input channels (1 for grayscale MRI)
    """

    def __init__(self, img_size=128, num_domains=5, max_conv_dim=512, input_nc=1):
        super().__init__()
        _validate_img_size(img_size)
        # Initial channel count scales inversely with image size (StarGAN v2 convention).
        dim_in = 2 ** 14 // img_size
        blocks = [nn.Conv2d(input_nc, dim_in, 3, 1, 1)]

        repeat_num = int(math.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks.append(ResBlk(dim_in, dim_out, downsample=True))
            dim_in = dim_out

        blocks.append(nn.LeakyReLU(0.2))
        blocks.append(nn.Conv2d(dim_out, dim_out, 4, 1, 0))
        blocks.append(nn.LeakyReLU(0.2))
        blocks.append(nn.Conv2d(dim_out, num_domains, 1, 1, 0))
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        """
        Args:
            x: Input image (B, 1, H, W).
            y: Domain label (B,) as LongTensor.

        Returns:
            Discriminator output (B,) — score for the specified domain.
        """
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (B, num_domains)
        idx = torch.arange(y.size(0), device=y.device)
        out = out[idx, y]  # (B,)
        return out


# --------------------------------------------------------------------------- #
#  Loss functions (from StarGAN v2 solver.py)
# --------------------------------------------------------------------------- #

def adv_loss(logits, target):
    """Binary cross-entropy adversarial loss."""
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    """R1 gradient penalty for real images."""
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert grad_dout2.size() == x_in.size()
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg


def compute_d_loss(nets, x_real, y_org, y_trg, z_trg=None, x_ref=None,
                   lambda_reg=1.0):
    """Compute discriminator loss (from StarGAN v2 official solver.py).

    Args:
        nets: Dict with generator, discriminator, mapping_network, style_encoder.
        x_real: Real images.
        y_org: Original domain labels.
        y_trg: Target domain labels.
        z_trg: Latent code (for latent-guided).
        x_ref: Reference image (for reference-guided).
        lambda_reg: R1 regularization weight.

    Returns:
        (loss, loss_dict)
    """
    assert (z_trg is None) != (x_ref is None)

    # Real images
    x_real.requires_grad_()
    out = nets["discriminator"](x_real, y_org)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real)

    # Fake images
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets["mapping_network"](z_trg, y_trg)
        else:
            s_trg = nets["style_encoder"](x_ref, y_trg)
        x_fake = nets["generator"](x_real, s_trg)
    out = nets["discriminator"](x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + lambda_reg * loss_reg
    return loss, dict(real=loss_real.item(), fake=loss_fake.item(), reg=loss_reg.item())


def compute_g_loss(nets, x_real, y_org, y_trg, z_trgs=None, x_refs=None,
                   lambda_sty=1.0, lambda_ds=1.0, lambda_cyc=1.0):
    """Compute generator loss (from StarGAN v2 official solver.py).

    Args:
        nets: Dict with generator, discriminator, mapping_network, style_encoder.
        x_real: Real images.
        y_org: Original domain labels.
        y_trg: Target domain labels.
        z_trgs: Pair of latent codes [z1, z2] (for latent-guided).
        x_refs: Pair of reference images [ref1, ref2] (for reference-guided).
        lambda_sty: Style reconstruction loss weight.
        lambda_ds: Diversity sensitive loss weight.
        lambda_cyc: Cycle consistency loss weight.

    Returns:
        (loss, loss_dict)
    """
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # Adversarial loss
    if z_trgs is not None:
        s_trg = nets["mapping_network"](z_trg, y_trg)
    else:
        s_trg = nets["style_encoder"](x_ref, y_trg)

    x_fake = nets["generator"](x_real, s_trg)
    out = nets["discriminator"](x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # Style reconstruction loss
    s_pred = nets["style_encoder"](x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # Diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = nets["mapping_network"](z_trg2, y_trg)
    else:
        s_trg2 = nets["style_encoder"](x_ref2, y_trg)
    x_fake2 = nets["generator"](x_real, s_trg2)
    x_fake2 = x_fake2.detach()
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    # Cycle consistency loss
    s_org = nets["style_encoder"](x_real, y_org)
    x_rec = nets["generator"](x_fake, s_org)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    loss = (loss_adv + lambda_sty * loss_sty
            - lambda_ds * loss_ds + lambda_cyc * loss_cyc)
    return loss, dict(adv=loss_adv.item(), sty=loss_sty.item(),
                      ds=loss_ds.item(), cyc=loss_cyc.item())


def moving_average(model, model_test, beta=0.999):
    """Exponential moving average of model parameters."""
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


# --------------------------------------------------------------------------- #
#  Build helper
# --------------------------------------------------------------------------- #

def build_stargan_v2(img_size=128, style_dim=64, latent_dim=16, num_domains=5,
                     max_conv_dim=512, input_nc=1):
    """Build StarGAN v2 networks.

    Returns:
        nets: Dict of training networks.
        nets_ema: Dict of EMA networks (for inference).
    """
    generator = StarGANv2Generator(img_size, style_dim, max_conv_dim, input_nc)
    mapping_network = MappingNetwork(latent_dim, style_dim, num_domains)
    style_encoder = StyleEncoder(img_size, style_dim, num_domains, max_conv_dim, input_nc)
    discriminator = StarGANv2Discriminator(img_size, num_domains, max_conv_dim, input_nc)

    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)
    style_encoder_ema = copy.deepcopy(style_encoder)

    nets = dict(
        generator=generator,
        mapping_network=mapping_network,
        style_encoder=style_encoder,
        discriminator=discriminator,
    )
    nets_ema = dict(
        generator=generator_ema,
        mapping_network=mapping_network_ema,
        style_encoder=style_encoder_ema,
    )

    return nets, nets_ema
