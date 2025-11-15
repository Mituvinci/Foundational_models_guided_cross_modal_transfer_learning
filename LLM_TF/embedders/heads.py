# LLM_TF/heads.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor



def grl(x: Tensor, lambd: float) -> Tensor:
    """
    Gradient Reversal (stateless).
    Forward: identity; Backward: multiply gradient by -λ.
    """
    return x + (-lambd * x - x).detach()


class CrossModalHeads(nn.Module):
    """
    Multi-Level Domain Adaptation heads on top of frozen scGPT embeddings:

    - classifier: 3-way cell-cycle head (on scGPT embeddings)
    - raw_domain_discriminator: domain adaptation on raw gene expression (preserves 7x domain gap!)
    - dom_src/dom_tgt: domain discriminators on embeddings (via GRL)
    - reconstruction: MLM-style gene expression reconstruction head
    - optional CORAL alignment loss (computed in forward)
    """
    def __init__(
        self,
        in_dim: int,
        n_genes: int,  # Raw gene expression dimension (DYNAMIC - determined by data)
        vocab_size: int = 60697,  # scGPT vocabulary size
        cls_hidden: int = 512,
        dom_hidden: int = 512,
        rec_hidden: int = 512,
        p_drop: float = 0.10,
        use_coral: bool = True,
        use_reconstruction: bool = True,
        mask_ratio: float = 0.15,  # MLM masking ratio
        use_s_phase_head: bool = False,  # Two-stage S-phase specialization
        use_raw_domain_adaptation: bool = True,  # Multi-level DA
        use_spectral_norm: bool = False,  # Solution 4: Spectral normalization for discriminator stability
        use_weak_discriminator: bool = False,  # Solution 1: Use 128 hidden units instead of 512
    ):
        super().__init__()
        # Classification head (on scGPT embeddings)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, cls_hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(cls_hidden, 3),
        )

        # S-phase specialized binary classifier (G1/G2M vs S)
        self.s_phase_head = nn.Sequential(
            nn.Linear(in_dim, cls_hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(cls_hidden, 2),  # Binary: S-phase vs others
        ) if use_s_phase_head else None

        # RAW DOMAIN ADAPTATION: Operates on raw gene expression (preserves 7x domain gap!)
        self.raw_domain_discriminator = nn.Sequential(
            nn.Linear(n_genes, dom_hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(dom_hidden, 2),  # Source=0, Target=1
        ) if use_raw_domain_adaptation else None

        # Standard domain discriminators on embeddings (existing approach)
        # Solution 1: Reduce discriminator capacity (512 → 128) to prevent saturation
        # Solution 4: Add spectral normalization for stability
        actual_dom_hidden = 128 if use_weak_discriminator else dom_hidden

        if use_spectral_norm:
            from torch.nn.utils import spectral_norm
            self.dom_src = nn.Sequential(
                spectral_norm(nn.Linear(in_dim, actual_dom_hidden)), nn.ReLU(), nn.Dropout(p_drop),
                spectral_norm(nn.Linear(actual_dom_hidden, 1)),
            )
            self.dom_tgt = nn.Sequential(
                spectral_norm(nn.Linear(in_dim, actual_dom_hidden)), nn.ReLU(), nn.Dropout(p_drop),
                spectral_norm(nn.Linear(actual_dom_hidden, 1)),
            )
        else:
            self.dom_src = nn.Sequential(
                nn.Linear(in_dim, actual_dom_hidden), nn.ReLU(), nn.Dropout(p_drop),
                nn.Linear(actual_dom_hidden, 1),
            )
            self.dom_tgt = nn.Sequential(
                nn.Linear(in_dim, actual_dom_hidden), nn.ReLU(), nn.Dropout(p_drop),
                nn.Linear(actual_dom_hidden, 1),
            )

        self.use_s_phase_head = use_s_phase_head
        self.use_raw_domain_adaptation = use_raw_domain_adaptation
        self.use_spectral_norm = use_spectral_norm
        self.use_weak_discriminator = use_weak_discriminator

        # Reconstruction head for MLM-style pretraining
        self.reconstruction = nn.Sequential(
            nn.Linear(in_dim, rec_hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(rec_hidden, vocab_size),
        ) if use_reconstruction else None

        self.use_coral = use_coral
        self.use_reconstruction = use_reconstruction
        self.mask_ratio = mask_ratio
        self.vocab_size = vocab_size

        # Initialize weights using Xavier/He initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # He initialization for ReLU activations
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


    @staticmethod
    def _cov(x: Tensor) -> Tensor:
        x = x - x.mean(dim=0, keepdim=True)
        return (x.T @ x) / (max(x.size(0) - 1, 1))

    @staticmethod
    def coral(xs: Tensor, xt: Tensor) -> Tensor:
        """CORAL loss aligning covariances of two batches."""
        return ((CrossModalHeads._cov(xs) - CrossModalHeads._cov(xt)) ** 2).mean()

    def _create_mlm_mask(self, token_ids: Tensor, values: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Create MLM-style mask for reconstruction loss.
        Args:
            token_ids: [B, L] gene token IDs
            values: [B, L] expression values
        Returns:
            masked_values: [B, L] values with some set to 0
            mask: [B, L] boolean mask (True = masked position)
        """
        B, L = token_ids.shape
        device = token_ids.device

        # Create random mask (exclude special tokens like CLS, PAD)
        # Assume token_id=0 is PAD, token_id=1 is CLS
        valid_positions = (token_ids > 1)  # Only mask actual gene tokens

        # Random masking
        rand = torch.rand(B, L, device=device)
        mask = (rand < self.mask_ratio) & valid_positions

        # Create masked values (set masked positions to 0)
        masked_values = values.clone()
        masked_values[mask] = 0.0

        return masked_values, mask

    def forward(self, feats_src: Tensor, feats_tgt: Tensor, lambd: float = 1.0,
                raw_src: Optional[Tensor] = None, raw_tgt: Optional[Tensor] = None,
                src_tokens: Optional[Tensor] = None, src_values: Optional[Tensor] = None,
                tgt_tokens: Optional[Tensor] = None, tgt_values: Optional[Tensor] = None) -> dict:
        """
        Forward pass through all heads.

        Args:
            feats_src: [Bs, H] source domain features (scGPT embeddings)
            feats_tgt: [Bt, H] target domain features (scGPT embeddings)
            lambd: GRL lambda parameter
            raw_src: [Bs, n_genes] raw source gene expression (for raw domain adaptation)
            raw_tgt: [Bt, n_genes] raw target gene expression (for raw domain adaptation)
            src_tokens: [Bs, L] source token IDs (for reconstruction)
            src_values: [Bs, L] source expression values (for reconstruction)
            tgt_tokens: [Bt, L] target token IDs (for reconstruction)
            tgt_values: [Bt, L] target expression values (for reconstruction)
        """
        # Classification on scGPT embeddings
        logits_src = self.classifier(feats_src)

        # Standard domain discriminators on embeddings
        dom_src_log = self.dom_src(grl(feats_src, lambd))
        dom_tgt_log = self.dom_tgt(grl(feats_tgt, lambd))

        # CORAL alignment on embeddings
        coral = self.coral(feats_src, feats_tgt) if self.use_coral else None

        # RAW DOMAIN ADAPTATION: Domain discrimination on raw gene expression (preserves 7x gap!)
        raw_domain_logits_src = None
        raw_domain_logits_tgt = None
        if self.use_raw_domain_adaptation and self.raw_domain_discriminator is not None:
            if raw_src is not None and raw_tgt is not None:
                # Apply GRL to raw data for adversarial training
                raw_domain_logits_src = self.raw_domain_discriminator(grl(raw_src, lambd))  # [Bs, 2]
                raw_domain_logits_tgt = self.raw_domain_discriminator(grl(raw_tgt, lambd))  # [Bt, 2]

        # S-phase specialized head (binary classification: S vs others)
        s_phase_logits = None
        if self.use_s_phase_head and self.s_phase_head is not None:
            s_phase_logits = self.s_phase_head(feats_src)  # [Bs, 2]

        # Reconstruction outputs (if enabled and inputs provided)
        reconstruction_outputs = {}
        if self.use_reconstruction and self.reconstruction is not None:
            if src_tokens is not None and src_values is not None:
                # Source reconstruction
                reconstruction_outputs["src_rec_logits"] = self.reconstruction(feats_src)  # [Bs, vocab_size]
                masked_values, mask = self._create_mlm_mask(src_tokens, src_values)
                reconstruction_outputs["src_mask"] = mask
                reconstruction_outputs["src_targets"] = src_values
                reconstruction_outputs["src_masked_values"] = masked_values

            if tgt_tokens is not None and tgt_values is not None:
                # Target reconstruction
                reconstruction_outputs["tgt_rec_logits"] = self.reconstruction(feats_tgt)  # [Bt, vocab_size]
                masked_values, mask = self._create_mlm_mask(tgt_tokens, tgt_values)
                reconstruction_outputs["tgt_mask"] = mask
                reconstruction_outputs["tgt_targets"] = tgt_values
                reconstruction_outputs["tgt_masked_values"] = masked_values

        result = {
            "logits_src": logits_src,                        # [Bs, 3]
            "dom_src_logits": dom_src_log,                   # [Bs, 1]
            "dom_tgt_logits": dom_tgt_log,                   # [Bt, 1]
            "raw_domain_logits_src": raw_domain_logits_src,  # [Bs, 2] or None
            "raw_domain_logits_tgt": raw_domain_logits_tgt,  # [Bt, 2] or None
            "coral": coral,                                  # scalar or None
            "s_phase_logits": s_phase_logits,                # [Bs, 2] or None
        }
        result.update(reconstruction_outputs)
        return result

    @staticmethod
    def reconstruction_loss(rec_logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        """
        Compute MLM-style reconstruction loss.

        Args:
            rec_logits: [B, vocab_size] predicted gene expression logits
            targets: [B, L] original expression values
            mask: [B, L] boolean mask (True = masked position to predict)

        Returns:
            Scalar loss averaged over masked positions
        """
        if mask.sum() == 0:
            return torch.tensor(0.0, device=rec_logits.device, requires_grad=True)

        # For simplicity, we'll predict the mean expression value across all genes
        # In practice, you might want a more sophisticated approach
        B, L = targets.shape
        vocab_size = rec_logits.shape[1]

        # Extract masked target values and compute their mean as a single scalar target
        masked_targets = targets[mask]  # [num_masked_tokens]
        target_mean = masked_targets.mean().unsqueeze(0).repeat(B)  # [B]

        # Use the first logit as prediction (simplified approach)
        # In practice, you might use a learned linear layer to map vocab_size -> 1
        predicted_mean = rec_logits[:, 0]  # [B]

        # MSE loss between predicted and target mean expression
        loss = F.mse_loss(predicted_mean, target_mean)
        return loss


def warmstart_classifier(heads: CrossModalHeads, baseline_clf: nn.Sequential) -> None:
    """
    Copy weights from a baseline classifier (Linear-ReLU-Dropout-Linear)
    into heads.classifier. No gradients are touched.
    """
    with torch.no_grad():
        heads.classifier[0].weight.copy_(baseline_clf[0].weight)
        heads.classifier[0].bias.copy_(baseline_clf[0].bias)
        heads.classifier[3].weight.copy_(baseline_clf[3].weight)
        heads.classifier[3].bias.copy_(baseline_clf[3].bias)
