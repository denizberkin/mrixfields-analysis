"""Vendored from experiment-pipeline/components/models/conditional_unet.py.

Everything above ConditionalUNetFactory, with the two eval_pipeline imports dropped --
they are used only by the factory, which the container does not need.
scripts/verify_docker_model.py asserts this module and the pipeline one produce
bit-identical output from the shipped checkpoint, so the copy cannot drift silently.

Regenerated 2026-09-11 to pick up _SliceEmbedding: the shipped checkpoint is now a
slice-conditioned model, and the previous copy predated it -- it would have raised on
the slice_embedding.* keys, or, had they been dropped, silently ignored the conditioning.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F



class _ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(output_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(output_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )


class _SliceEmbedding(nn.Module):
    """Sinusoidal embedding of normalised axial position, added to the bottleneck vector.

    Where a slice sits along z decides which anatomy is in it -- orbits and temporal lobe low,
    ventricles mid, vertex high -- and the field transition is not the same mapping in each.
    The domain embeddings cannot carry that: every slice of a volume shares them, so the model
    currently has no way to know whether it is looking at cerebellum or corona radiata.

    Sinusoidal rather than an nn.Embedding over the 220 slice indices: position is continuous
    and the useful structure is smooth in it, so neighbouring slices should share a
    representation instead of each learning an independent row from 1/220th of the data.

    The output projection is zero-init, so a model built with slice_conditioning=True and
    loaded from a checkpoint trained without it computes exactly the same function at step 0.
    That is what lets this be seeded from mc_25d_e10 and scored as a pure delta.
    """

    def __init__(self, channels: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        self.channels = channels
        half = max(1, channels // 2)
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
        )
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.projection = nn.Sequential(
            nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, channels)
        )
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        # x1000 puts [0,1] on the same angular scale the timestep embeddings use, so the
        # low frequencies vary appreciably across the volume instead of being near-constant.
        angles = position.float().reshape(-1, 1) * 1000.0 * self.frequencies.reshape(1, -1)
        embedded = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        if embedded.shape[-1] < self.channels:
            embedded = F.pad(embedded, (0, self.channels - embedded.shape[-1]))
        return self.projection(embedded)


class _FiLM(nn.Module):
    """Scale and shift both normalisations of one decoder block.

    A conditioning vector *added* at the bottleneck is exactly the quantity
    InstanceNorm subtracts back out again, so little of it survives the trip to
    the output - the same leak that forced _ConditionalRefinement onto the Swin
    decoder. A scale and shift applied *after* a norm cannot be cancelled
    downstream, and the norm still bounds what the convolutions build up; the
    ViT's first run protected its embedding by dropping normalisation instead
    and its decoder features grew 12x.

    One projection covers both norms of the block - chunk(4) rather than two
    Linear layers, same parameter count - and is zero-initialised in weight
    *and* bias, so gamma = beta = 0 and the block computes exactly the plain
    _ConvBlock the existing checkpoints were trained as.
    """

    def __init__(self, embed_dim: int, channels: int) -> None:
        super().__init__()
        self.projection = nn.Linear(embed_dim, 4 * channels)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(
        self, block: _ConvBlock, x: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        # The block stays an nn.Sequential and is unpacked here rather than
        # rewritten as a FiLM-aware module: renaming its children would rename
        # every decoder tensor and break every existing checkpoint.
        conv1, norm1, activation1, conv2, norm2, activation2 = block
        gamma1, beta1, gamma2, beta2 = (
            part[:, :, None, None] for part in self.projection(conditioning).chunk(4, dim=1)
        )
        x = activation1(norm1(conv1(x)) * (1 + gamma1) + beta1)
        return activation2(norm2(conv2(x)) * (1 + gamma2) + beta2)


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        num_domains: int = 15,
        base_channels: int = 64,
        max_channels: int = 512,
        levels: int = 4,
        residual_output: bool = False,
        film_conditioning: bool = False,
        slice_conditioning: bool = False,
    ) -> None:
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be positive")
        if residual_output and input_channels != output_channels and output_channels != 1:
            raise ValueError(
                "residual_output predicts a correction to the source image, so "
                f"input_channels ({input_channels}) must equal output_channels "
                f"({output_channels}), or output a single channel that the residual is "
                "added to (multi-contrast input)"
            )
        self.input_channels = int(input_channels)
        channels = [min(base_channels * 2**level, max_channels) for level in range(levels)]
        bottleneck_channels = min(channels[-1] * 2, max_channels)

        self.encoders = nn.ModuleList()
        previous = input_channels
        for channel in channels:
            self.encoders.append(_ConvBlock(previous, channel))
            previous = channel
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _ConvBlock(channels[-1], bottleneck_channels)
        # 5*m + c -> (m \in {t1, t2, t2*}) and (c \in {0.1T, 1.5T, 3T, 5T, 7T}) -> {0...14}
        self.source_embedding = nn.Embedding(num_domains, bottleneck_channels)
        self.target_embedding = nn.Embedding(num_domains, bottleneck_channels)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        previous = bottleneck_channels
        for channel in reversed(channels):
            self.upconvs.append(nn.ConvTranspose2d(previous, channel, 2, stride=2))
            self.decoders.append(_ConvBlock(channel * 2, channel))
            previous = channel

        # Off by default: both flags add tensors or move them, and the
        # task3_unet_pro artifacts have to keep loading with strict=True.
        self.slice_embedding = (
            _SliceEmbedding(bottleneck_channels) if slice_conditioning else None
        )
        self.film_projections = (
            nn.ModuleList(_FiLM(bottleneck_channels, channel) for channel in reversed(channels))
            if film_conditioning
            else None
        )
        if residual_output:
            # Deliberately not named output.*: the head is what tells a loader
            # which of the two forwards a checkpoint was trained for, so the
            # submission script can read it off the tensors instead of being
            # handed a flag that may be wrong (same reason as derive_shape).
            self.residual_head = nn.Conv2d(channels[0], output_channels, 1)
            nn.init.zeros_(self.residual_head.weight)
            nn.init.zeros_(self.residual_head.bias)
            self.output = None
        else:
            self.residual_head = None
            self.output = nn.Sequential(nn.Conv2d(channels[0], output_channels, 1), nn.Tanh())

    def forward(
        self,
        image: torch.Tensor,
        target_domain: torch.Tensor,
        source_domain: torch.Tensor | None = None,
        slice_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        skips = []
        x = image
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        source_domain = target_domain if source_domain is None else source_domain
        conditioning = self.source_embedding(source_domain) + self.target_embedding(target_domain)
        # Added to the same vector the FiLM projections read, so axial position modulates the
        # decoder exactly the way the domain pair does, rather than only shifting the bottleneck.
        if self.slice_embedding is not None and slice_pos is not None:
            conditioning = conditioning + self.slice_embedding(slice_pos)
        # The bottleneck add survives even with FiLM on. Removing it would
        # change the function a loaded checkpoint computes, and the whole point
        # of the zero-init FiLM is that step 0 reproduces that checkpoint.
        x = x + conditioning.unsqueeze(-1).unsqueeze(-1)
        films = (
            [None] * len(self.decoders) if self.film_projections is None else self.film_projections
        )
        for upconv, decoder, film, skip in zip(
            self.upconvs, self.decoders, films, reversed(skips), strict=True
        ):
            x = upconv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            merged = torch.cat((x, skip), dim=1)
            x = decoder(merged) if film is None else film(decoder, merged, conditioning)

        if self.residual_head is None:
            return self.output(x)
        # The residual is added in the bounded range itself, not in tanh space
        # (the [-1, 1] analogue of the tubelet model's sigmoid(logit(x) + r)):
        # this model's I/O convention is [-1, 1] and ~85% of a slice is air at
        # -1, where atanh puts the base near -5 and the tanh derivative is
        # ~1e-4 - the saturation that froze the ViT's first run at a constant.
        # Zero-init head means step 0 is the exact identity, which already
        # scores SSIM 0.836 here. Clamping only outside training bounds the
        # prediction for inference without removing the gradient during it.
        # Multi-contrast input puts the contrast being predicted in channel 0 (see
        # CachedMultiContrastDataset), so the residual is added there. Single-channel input
        # takes the whole image, which is the same expression when there is one channel.
        base = image if image.shape[1] == self.residual_head.out_channels else image[:, :1]
        prediction = base + self.residual_head(x)
        return prediction if self.training else prediction.clamp(-1, 1)


def unet_from_state_dict(state: dict[str, "torch.Tensor"]) -> "ConditionalUNet":
    """Rebuild the exact architecture a checkpoint was trained with, then load it.

    Every consumer that hardcoded ``base_channels=32, max_channels=512, levels=4``
    was already one config edit away from a silent mismatch, and the residual head
    and per-scale FiLM add two more things to get wrong. All four are visible in
    the tensors, so read them rather than trust a caller: widths come from the
    first encoder conv and the bottleneck, and the two flags from the key names
    they move (``residual_head.*`` replaces ``output.0.*``; FiLM adds
    ``film_projections.*``). The load is strict, so a wrong guess still fails
    loudly instead of quietly computing something else.
    """
    base_channels = int(state["encoders.0.0.weight"].shape[0])
    input_channels = int(state["encoders.0.0.weight"].shape[1])
    levels = len({key.split(".")[1] for key in state if key.startswith("encoders.")})
    max_channels = int(state["bottleneck.0.weight"].shape[0])
    model = ConditionalUNet(
        input_channels=input_channels,
        base_channels=base_channels,
        max_channels=max_channels,
        levels=levels,
        num_domains=int(state["target_embedding.weight"].shape[0]),
        residual_output="residual_head.weight" in state,
        film_conditioning=any(key.startswith("film_projections.") for key in state),
        slice_conditioning=any(key.startswith("slice_embedding.") for key in state),
    )
    model.load_state_dict(state, strict=True)
    return model
