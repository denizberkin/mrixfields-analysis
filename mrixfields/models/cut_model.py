"""CUT (Contrastive Unpaired Translation) model.

Based on the official CUT implementation:
    taesungp/contrastive-unpaired-translation/models/cut_model.py

Reference: Park et al., "Contrastive Learning for Unpaired Image-to-Image
Translation", ECCV 2020.

Usage for Task 1 & 2: Each field-strength pair trains independently.
    e.g., CUT(0.1T → 7T), CUT(1.5T → 7T), etc.
"""

import torch
import torch.nn as nn

from .networks import ResnetGenerator, NLayerDiscriminator, PatchSampleF, init_net
from ..losses.patchnce import PatchNCELoss
from ..losses.adversarial import GANLoss


class CUTModel(nn.Module):
    """CUT model: single generator G (src→tgt), discriminator D, feature network F.

    Loss = lambda_GAN * GAN(G(X)) + lambda_NCE * NCE(G(X), X)
           + lambda_NCE * NCE_idt(G(Y), Y)  [optional identity NCE]

    Parameters:
        input_nc (int) -- input channels (default 1 for MRI)
        output_nc (int) -- output channels
        ngf (int) -- generator filters
        ndf (int) -- discriminator filters
        n_blocks (int) -- ResNet blocks in generator
        nce_layers (list) -- encoder layer indices for NCE loss
        nce_T (float) -- NCE temperature
        num_patches (int) -- patches to sample per layer
        lambda_GAN (float) -- weight for GAN loss
        lambda_NCE (float) -- weight for NCE loss
        nce_idt (bool) -- use identity NCE loss
        gan_mode (str) -- GAN loss type: 'lsgan' or 'vanilla'
        netF_nc (int) -- MLP projection dimension
    """

    def __init__(
        self,
        input_nc=1,
        output_nc=1,
        ngf=64,
        ndf=64,
        n_blocks=9,
        nce_layers=None,
        nce_T=0.07,
        num_patches=256,
        lambda_GAN=1.0,
        lambda_NCE=1.0,
        nce_idt=True,
        gan_mode="lsgan",
        netF_nc=256,
        lr=0.0002,
        beta1=0.5,
        beta2=0.999,
    ):
        super().__init__()

        if nce_layers is None:
            nce_layers = [0, 4, 8, 12, 16]
        self.nce_layers = nce_layers
        self.nce_T = nce_T
        self.num_patches = num_patches
        self.lambda_GAN = lambda_GAN
        self.lambda_NCE = lambda_NCE
        self.nce_idt = nce_idt
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        # Networks
        self.netG = ResnetGenerator(input_nc, output_nc, ngf, n_blocks=n_blocks)
        self.netD = NLayerDiscriminator(output_nc, ndf)
        self.netF = PatchSampleF(use_mlp=True, nc=netF_nc)

        # Losses
        self.criterionGAN = GANLoss(mode=gan_mode)
        self.criterionNCE = nn.ModuleList([
            PatchNCELoss(nce_T=nce_T) for _ in self.nce_layers
        ])

    def init_weights(self, init_type="normal", init_gain=0.02):
        """Initialize network weights."""
        init_net(self.netG, init_type, init_gain)
        init_net(self.netD, init_type, init_gain)
        # netF is lazily initialized on first forward pass

    def setup_optimizers(self):
        """Create optimizers. Call after init_weights and data_dependent_initialize."""
        self.optimizer_G = torch.optim.Adam(
            self.netG.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)
        )
        self.optimizer_D = torch.optim.Adam(
            self.netD.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)
        )
        # optimizer_F is created in data_dependent_initialize after netF MLPs are built

    def data_dependent_initialize(self, real_A, real_B):
        """Initialize netF MLPs based on actual feature dimensions.

        Must be called once with real data before training starts.
        """
        self.forward(real_A, real_B)
        self.compute_D_loss(real_A, real_B).backward()
        self.compute_G_loss(real_A, real_B).backward()
        self.optimizer_F = torch.optim.Adam(
            self.netF.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)
        )

    def forward(self, real_A, real_B):
        """Run forward pass.

        Args:
            real_A: Source domain image.
            real_B: Target domain image.

        Returns:
            fake_B: Generated target domain image.
        """
        if self.nce_idt and self.training:
            real = torch.cat((real_A, real_B), dim=0)
        else:
            real = real_A

        fake = self.netG(real)
        self.fake_B = fake[:real_A.size(0)]
        if self.nce_idt and self.training:
            self.idt_B = fake[real_A.size(0):]
        return self.fake_B

    def compute_D_loss(self, real_A, real_B):
        """Calculate GAN loss for discriminator."""
        fake = self.fake_B.detach()
        pred_fake = self.netD(fake)
        loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        pred_real = self.netD(real_B)
        loss_D_real = self.criterionGAN(pred_real, True).mean()
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D

    def compute_G_loss(self, real_A, real_B):
        """Calculate GAN + NCE loss for generator."""
        fake = self.fake_B

        # GAN loss
        if self.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.lambda_GAN
        else:
            loss_G_GAN = 0.0

        # NCE loss
        if self.lambda_NCE > 0.0:
            loss_NCE = self.calculate_NCE_loss(real_A, self.fake_B)
        else:
            loss_NCE = 0.0

        # Identity NCE loss
        if self.nce_idt and self.lambda_NCE > 0.0 and self.training:
            loss_NCE_Y = self.calculate_NCE_loss(real_B, self.idt_B)
            loss_NCE_both = (loss_NCE + loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = loss_NCE

        loss_G = loss_G_GAN + loss_NCE_both
        return loss_G

    def calculate_NCE_loss(self, src, tgt):
        """Calculate PatchNCE loss between source and target.

        Extracts encoder features from both images, samples patches,
        projects through MLPs, and computes InfoNCE per layer.
        """
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)
        feat_k = self.netG(src, self.nce_layers, encode_only=True)

        feat_k_pool, sample_ids = self.netF(feat_k, self.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit in zip(feat_q_pool, feat_k_pool, self.criterionNCE):
            loss = crit(f_q, f_k) * self.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def optimize_parameters(self, real_A, real_B):
        """Full training step: forward + D update + G update.

        Args:
            real_A: Source domain batch.
            real_B: Target domain batch.

        Returns:
            Dict of loss values for logging.
        """
        # Forward
        self.forward(real_A, real_B)

        # Update D
        for p in self.netD.parameters():
            p.requires_grad = True
        self.optimizer_D.zero_grad()
        loss_D = self.compute_D_loss(real_A, real_B)
        loss_D.backward()
        self.optimizer_D.step()

        # Update G and F
        for p in self.netD.parameters():
            p.requires_grad = False
        self.optimizer_G.zero_grad()
        self.optimizer_F.zero_grad()
        loss_G = self.compute_G_loss(real_A, real_B)
        loss_G.backward()
        self.optimizer_G.step()
        self.optimizer_F.step()

        return {"loss_D": loss_D.item(), "loss_G": loss_G.item()}
