"""CycleGAN model for unpaired image-to-image translation.

Based on the official CycleGAN implementation:
    junyanz/pytorch-CycleGAN-and-pix2pix/models/cycle_gan_model.py

Reference: Zhu et al., "Unpaired Image-to-Image Translation using
Cycle-Consistent Adversarial Networks", ICCV 2017.

Usage for Task 1 & 2: Each field-strength pair trains independently.
    e.g., CycleGAN(0.1T ↔ 7T), CycleGAN(1.5T ↔ 7T), etc.
"""

import itertools

import torch
import torch.nn as nn

from .networks import ResnetGenerator, NLayerDiscriminator, init_net
from ..losses.adversarial import GANLoss
from ..data.unpaired_loader import ImagePool


class CycleGANModel(nn.Module):
    """CycleGAN: dual generators + dual discriminators with cycle consistency.

    G_AB: domain A → domain B
    G_BA: domain B → domain A
    D_A: discriminate real A vs fake A
    D_B: discriminate real B vs fake B

    Loss = GAN(G_AB, D_B) + GAN(G_BA, D_A)
         + lambda_cycle * (||G_BA(G_AB(A)) - A|| + ||G_AB(G_BA(B)) - B||)
         + lambda_idt * (||G_AB(B) - B|| + ||G_BA(A) - A||)

    Parameters:
        input_nc (int) -- input channels (default 1 for MRI)
        output_nc (int) -- output channels
        ngf (int) -- generator filters
        ndf (int) -- discriminator filters
        n_blocks (int) -- ResNet blocks in generator
        lambda_cycle (float) -- weight for cycle consistency loss (default 10.0)
        lambda_idt (float) -- weight for identity loss relative to cycle (default 0.5)
        pool_size (int) -- size of image buffer (default 50)
        gan_mode (str) -- GAN loss type: 'lsgan' or 'vanilla'
    """

    def __init__(
        self,
        input_nc=1,
        output_nc=1,
        ngf=64,
        ndf=64,
        n_blocks=9,
        lambda_cycle=10.0,
        lambda_idt=0.5,
        pool_size=50,
        gan_mode="lsgan",
        lr=0.0002,
        beta1=0.5,
    ):
        super().__init__()

        self.lambda_cycle = lambda_cycle
        self.lambda_idt = lambda_idt
        self.lr = lr
        self.beta1 = beta1

        # Generators
        self.netG_AB = ResnetGenerator(input_nc, output_nc, ngf, n_blocks=n_blocks)
        self.netG_BA = ResnetGenerator(output_nc, input_nc, ngf, n_blocks=n_blocks)

        # Discriminators
        self.netD_A = NLayerDiscriminator(input_nc, ndf)
        self.netD_B = NLayerDiscriminator(output_nc, ndf)

        # Losses
        self.criterionGAN = GANLoss(mode=gan_mode)
        self.criterionCycle = nn.L1Loss()
        self.criterionIdt = nn.L1Loss()

        # Image buffers
        self.fake_A_pool = ImagePool(pool_size)
        self.fake_B_pool = ImagePool(pool_size)

    def init_weights(self, init_type="normal", init_gain=0.02):
        """Initialize network weights."""
        init_net(self.netG_AB, init_type, init_gain)
        init_net(self.netG_BA, init_type, init_gain)
        init_net(self.netD_A, init_type, init_gain)
        init_net(self.netD_B, init_type, init_gain)

    def setup_optimizers(self):
        """Create optimizers."""
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.netG_AB.parameters(), self.netG_BA.parameters()),
            lr=self.lr, betas=(self.beta1, 0.999),
        )
        self.optimizer_D = torch.optim.Adam(
            itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
            lr=self.lr, betas=(self.beta1, 0.999),
        )

    def forward(self, real_A, real_B):
        """Run forward pass: generate fakes and reconstructions.

        Args:
            real_A: Image from domain A.
            real_B: Image from domain B.

        Returns:
            fake_B: G_AB(A)
        """
        self.fake_B = self.netG_AB(real_A)       # G_AB(A)
        self.rec_A = self.netG_BA(self.fake_B)   # G_BA(G_AB(A))
        self.fake_A = self.netG_BA(real_B)       # G_BA(B)
        self.rec_B = self.netG_AB(self.fake_A)   # G_AB(G_BA(B))
        return self.fake_B

    def _backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for a discriminator."""
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D

    def compute_D_loss(self, real_A, real_B):
        """Calculate D_A and D_B losses."""
        fake_B = self.fake_B_pool.query(self.fake_B)
        loss_D_B = self._backward_D_basic(self.netD_B, real_B, fake_B)

        fake_A = self.fake_A_pool.query(self.fake_A)
        loss_D_A = self._backward_D_basic(self.netD_A, real_A, fake_A)

        return loss_D_A + loss_D_B

    def compute_G_loss(self, real_A, real_B):
        """Calculate generator losses: GAN + cycle + identity."""
        lambda_A = self.lambda_cycle
        lambda_B = self.lambda_cycle
        lambda_idt = self.lambda_idt

        # Identity loss
        if lambda_idt > 0:
            self.idt_A = self.netG_AB(real_B)
            loss_idt_A = self.criterionIdt(self.idt_A, real_B) * lambda_B * lambda_idt
            self.idt_B = self.netG_BA(real_A)
            loss_idt_B = self.criterionIdt(self.idt_B, real_A) * lambda_A * lambda_idt
        else:
            loss_idt_A = 0.0
            loss_idt_B = 0.0

        # GAN loss
        loss_G_AB = self.criterionGAN(self.netD_B(self.fake_B), True)
        loss_G_BA = self.criterionGAN(self.netD_A(self.fake_A), True)

        # Cycle consistency loss
        loss_cycle_A = self.criterionCycle(self.rec_A, real_A) * lambda_A
        loss_cycle_B = self.criterionCycle(self.rec_B, real_B) * lambda_B

        loss_G = (loss_G_AB + loss_G_BA + loss_cycle_A + loss_cycle_B
                  + loss_idt_A + loss_idt_B)
        return loss_G

    def optimize_parameters(self, real_A, real_B):
        """Full training step.

        Args:
            real_A: Domain A batch.
            real_B: Domain B batch.

        Returns:
            Dict of loss values for logging.
        """
        # Forward
        self.forward(real_A, real_B)

        # Update G
        for p in self.netD_A.parameters():
            p.requires_grad = False
        for p in self.netD_B.parameters():
            p.requires_grad = False
        self.optimizer_G.zero_grad()
        loss_G = self.compute_G_loss(real_A, real_B)
        loss_G.backward()
        torch.nn.utils.clip_grad_norm_(
            itertools.chain(self.netG_AB.parameters(), self.netG_BA.parameters()),
            max_norm=1.0,
        )
        self.optimizer_G.step()

        # Update D
        for p in self.netD_A.parameters():
            p.requires_grad = True
        for p in self.netD_B.parameters():
            p.requires_grad = True
        self.optimizer_D.zero_grad()
        loss_D = self.compute_D_loss(real_A, real_B)
        loss_D.backward()
        torch.nn.utils.clip_grad_norm_(
            itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
            max_norm=1.0,
        )
        self.optimizer_D.step()

        return {"loss_G": loss_G.item(), "loss_D": loss_D.item()}
