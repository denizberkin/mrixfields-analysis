"""PatchNCE contrastive loss for CUT (Contrastive Unpaired Translation).

Based on the official CUT implementation:
    taesungp/contrastive-unpaired-translation/models/patchnce.py

Reference: Park et al., "Contrastive Learning for Unpaired Image-to-Image
Translation", ECCV 2020.
"""

import torch
import torch.nn as nn


class PatchNCELoss(nn.Module):
    """Patch-wise InfoNCE loss for contrastive unpaired translation.

    Given query features (from generated image) and key features (from real
    source image) at corresponding spatial locations, treats same-position
    pairs as positives and all other positions as negatives.

    This is called once per encoder layer. Multiple instances are typically
    created (one per layer) in the CUT model.

    Parameters:
        nce_T (float) -- temperature for softmax (default 0.07)
        batch_size (int) -- batch size (used when not including all negatives)
        nce_includes_all_negatives_from_minibatch (bool) -- if True, negatives
            from other samples in the batch are included (for single-image translation)
    """

    def __init__(self, nce_T=0.07, batch_size=1,
                 nce_includes_all_negatives_from_minibatch=False):
        super().__init__()
        self.nce_T = nce_T
        self.batch_size = batch_size
        self.nce_includes_all_negatives_from_minibatch = nce_includes_all_negatives_from_minibatch
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        self.mask_dtype = torch.bool

    def forward(self, feat_q, feat_k):
        """Compute PatchNCE loss.

        Args:
            feat_q: Query features from generated image, shape (num_patches, dim).
                     Already L2-normalized and projected by PatchSampleF.
            feat_k: Key features from source image, shape (num_patches, dim).
                     Already L2-normalized and projected by PatchSampleF.

        Returns:
            Per-patch NCE loss, shape (num_patches,).
        """
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # Positive logit: dot product of corresponding positions
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1),
            feat_k.view(num_patches, -1, 1),
        )
        l_pos = l_pos.view(num_patches, 1)

        # Negative logits
        if self.nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.batch_size

        # Reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # Fill diagonal with very small number (same position = positive, not negative)
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.nce_T

        loss = self.cross_entropy_loss(
            out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device)
        )

        return loss
