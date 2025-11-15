"""
Dual-Encoder DANN for Cross-Modality Transfer Learning

Architecture:
    Source Stream (scRNA-seq):
        Input: scRNA expression matrix → LLM Encoder → Source Latent [B, latent_dim]

    Target Stream (scATAC-seq):
        Input: scATAC peak matrix → Peak Encoder → Target Latent [B, latent_dim]

    DANN Framework:
        Both latents → Shared Projection → Domain-invariant space
        ├─ Domain Discriminator (adversarial): Classify source vs target
        └─ Label Classifier: Predict cell cycle phase

References:
    - Ganin et al., "Domain-Adversarial Training of Neural Networks" (2016)
    - scJoint (Lin et al., 2022) - Multi-modal single-cell integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


def gradient_reversal_layer(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Gradient Reversal Layer (GRL).

    Forward: Identity (output = input)
    Backward: Multiply gradient by -alpha

    Args:
        x: Input tensor
        alpha: Gradient reversal strength (lambda in DANN paper)

    Returns:
        Same tensor with reversed gradient during backprop
    """
    return x + (x * -alpha - x).detach()


class LatentProjection(nn.Module):
    """
    Projects source and target encodings to a shared latent space.

    This ensures both modalities have the same dimensionality before
    domain adaptation.
    """
    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        shared_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.source_proj = nn.Sequential(
            nn.Linear(source_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.target_proj = nn.Sequential(
            nn.Linear(target_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.shared_dim = shared_dim

    def forward(self, x: torch.Tensor, domain: str = 'source') -> torch.Tensor:
        """
        Project encoder output to shared latent space.

        Args:
            x: Encoder output [B, source_dim or target_dim]
            domain: 'source' or 'target'

        Returns:
            Projected latent features [B, shared_dim]
        """
        if domain == 'source':
            return self.source_proj(x)
        elif domain == 'target':
            return self.target_proj(x)
        else:
            raise ValueError(f"Invalid domain: {domain}. Expected 'source' or 'target'.")


class DomainDiscriminator(nn.Module):
    """
    Binary classifier to distinguish source vs target domain.

    During adversarial training:
    - Discriminator tries to maximize accuracy (classify domains correctly)
    - Encoder tries to minimize accuracy (fool the discriminator)

    This forces encoders to learn domain-invariant features.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_spectral_norm: bool = False
    ):
        super().__init__()

        layers = []
        curr_dim = input_dim

        for i in range(num_layers - 1):
            linear = nn.Linear(curr_dim, hidden_dim)
            if use_spectral_norm:
                linear = nn.utils.spectral_norm(linear)

            layers.extend([
                linear,
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            curr_dim = hidden_dim

        final_layer = nn.Linear(curr_dim, 1)
        if use_spectral_norm:
            final_layer = nn.utils.spectral_norm(final_layer)

        layers.append(final_layer)

        self.discriminator = nn.Sequential(*layers)
        self.use_spectral_norm = use_spectral_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict domain label (0=source, 1=target).

        Args:
            x: Latent features [B, input_dim]

        Returns:
            Domain logits [B, 1]
        """
        return self.discriminator(x)


class ClassConditionalDiscriminator(nn.Module):
    """
    Class-Conditional Domain Discriminator (CDANN).

    Conditions domain prediction on predicted class labels to enforce
    class-wise alignment instead of just global domain confusion.

    Reference: Long et al., "Conditional Adversarial Domain Adaptation" (2018)
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_spectral_norm: bool = False
    ):
        super().__init__()

        layers = []
        curr_dim = input_dim + num_classes

        for i in range(num_layers - 1):
            linear = nn.Linear(curr_dim, hidden_dim)
            if use_spectral_norm:
                linear = nn.utils.spectral_norm(linear)

            layers.extend([
                linear,
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            curr_dim = hidden_dim

        final_layer = nn.Linear(curr_dim, 1)
        if use_spectral_norm:
            final_layer = nn.utils.spectral_norm(final_layer)

        layers.append(final_layer)

        self.discriminator = nn.Sequential(*layers)
        self.use_spectral_norm = use_spectral_norm
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, class_probs: torch.Tensor) -> torch.Tensor:
        """
        Predict domain label conditioned on class probabilities.

        Args:
            x: Latent features [B, input_dim]
            class_probs: Class probabilities [B, num_classes] (softmax output)

        Returns:
            Domain logits [B, 1]
        """
        combined = torch.cat([x, class_probs], dim=1)
        return self.discriminator(combined)


class LabelClassifier(nn.Module):
    """
    Classifies cell cycle phase from latent features.

    Trained only on source data (labeled scRNA-seq).
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        layers = []
        curr_dim = input_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(curr_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            curr_dim = hidden_dim

        layers.append(nn.Linear(curr_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict cell cycle phase.

        Args:
            x: Latent features [B, input_dim]

        Returns:
            Class logits [B, num_classes]
        """
        return self.classifier(x)


class ContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss for cross-modality alignment.

    Pulls same-class cells together across source/target domains,
    pushes different-class cells apart in shared latent space.

    This helps prevent class collapse by enforcing semantic similarity
    across modalities (e.g., all G1 cells should cluster together).

    Reference: Khosla et al., "Supervised Contrastive Learning" (NeurIPS 2020)
    """
    def __init__(self, temperature: float = 0.1):
        """
        Initialize Contrastive Loss.

        Args:
            temperature: Scaling factor for similarity (lower = sharper clusters)
                        Typical range: 0.05-0.5
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        source_labels: torch.Tensor,
        target_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.

        Args:
            source_features: Source latent features [B_src, D]
            target_features: Target latent features [B_tgt, D]
            source_labels: Source class labels [B_src]
            target_labels: Target pseudo-labels [B_tgt] (predicted labels)

        Returns:
            Contrastive loss scalar
        """
        features = torch.cat([source_features, target_features], dim=0)
        labels = torch.cat([source_labels, target_labels], dim=0)

        batch_size = features.shape[0]

        features = F.normalize(features, p=2, dim=1)

        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask = mask.float()

        mask.fill_diagonal_(0)

        exp_sim = torch.exp(similarity_matrix)

        exp_sim = exp_sim * (1 - torch.eye(batch_size, device=features.device))

        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        loss = -mean_log_prob_pos
        loss = loss[mask.sum(dim=1) > 0].mean()

        return loss


class EnsembleSourceEncoder(nn.Module):
    """
    Ensemble of two LLM encoders for richer source representations.

    Combines embeddings from two pre-trained LLMs (e.g., geneformer + scfoundation)
    by concatenating their outputs.

    This provides complementary information from different model architectures
    and training objectives.
    """
    def __init__(self, encoder1: nn.Module, encoder2: nn.Module):
        """
        Initialize Ensemble Source Encoder.

        Args:
            encoder1: First LLM encoder (e.g., geneformer-104m)
            encoder2: Second LLM encoder (e.g., scfoundation)
        """
        super().__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through both encoders and concatenate outputs.

        Args:
            x: Input data [B, input_dim]

        Returns:
            Concatenated embeddings [B, dim1 + dim2]
        """
        emb1 = self.encoder1(x)
        emb2 = self.encoder2(x)
        return torch.cat([emb1, emb2], dim=1)


class DualEncoderDANN(nn.Module):
    """
    Dual-Encoder Domain-Adversarial Neural Network for cross-modality transfer.

    Components:
        1. Source Encoder (LLM or Ensemble): scRNA-seq → source latent
        2. Target Encoder (Peak model): scATAC-seq → target latent
        3. Latent Projection: Align both to shared space
        4. Domain Discriminator: Adversarial alignment
        5. Label Classifier: Cell cycle prediction

    Usage:
        model = DualEncoderDANN(
            source_encoder=llm_model,
            target_encoder=peak_model,
            source_dim=512,
            target_dim=256,
            shared_dim=256
        )

        outputs = model(
            source_data=rna_expr,
            target_data=peak_counts,
            alpha=1.0  # GRL strength
        )
    """
    def __init__(
        self,
        source_encoder: nn.Module,
        target_encoder: nn.Module,
        source_dim: int,
        target_dim: int,
        shared_dim: int = 512,
        num_classes: int = 3,
        disc_hidden: int = 256,
        disc_layers: int = 3,
        cls_hidden: int = 256,
        cls_layers: int = 2,
        dropout: float = 0.1,
        use_spectral_norm: bool = False,
        freeze_source_encoder: bool = True,
        use_cdann: bool = False
    ):
        """
        Initialize Dual-Encoder DANN.

        Args:
            source_encoder: Pre-trained LLM for scRNA-seq (e.g., scGPT, Geneformer)
            target_encoder: Peak embedder for scATAC-seq (e.g., MLP, VAE, DAE, GNN, CNN)
            source_dim: Source encoder output dimension
            target_dim: Target encoder output dimension
            shared_dim: Shared latent space dimension
            num_classes: Number of cell cycle phases (default: 3 = G1, S, G2M)
            disc_hidden: Discriminator hidden dimension
            disc_layers: Discriminator number of layers
            cls_hidden: Classifier hidden dimension
            cls_layers: Classifier number of layers
            dropout: Dropout probability
            use_spectral_norm: Use spectral normalization in discriminator
            freeze_source_encoder: Freeze LLM encoder weights (recommended for large LLMs)
            use_cdann: Use class-conditional discriminator (CDANN) instead of standard discriminator
        """
        super().__init__()

        self.source_encoder = source_encoder
        self.target_encoder = target_encoder

        if freeze_source_encoder:
            for param in self.source_encoder.parameters():
                param.requires_grad = False
            print("Source encoder (LLM) frozen - only fine-tuning target encoder")

        self.projection = LatentProjection(
            source_dim=source_dim,
            target_dim=target_dim,
            shared_dim=shared_dim,
            dropout=dropout
        )

        if use_cdann:
            self.domain_discriminator = ClassConditionalDiscriminator(
                input_dim=shared_dim,
                num_classes=num_classes,
                hidden_dim=disc_hidden,
                num_layers=disc_layers,
                dropout=dropout,
                use_spectral_norm=use_spectral_norm
            )
            print("Using Class-Conditional Discriminator (CDANN) for class-wise alignment")
        else:
            self.domain_discriminator = DomainDiscriminator(
                input_dim=shared_dim,
                hidden_dim=disc_hidden,
                num_layers=disc_layers,
                dropout=dropout,
                use_spectral_norm=use_spectral_norm
            )

        self.label_classifier = LabelClassifier(
            input_dim=shared_dim,
            num_classes=num_classes,
            hidden_dim=cls_hidden,
            num_layers=cls_layers,
            dropout=dropout
        )

        self.source_dim = source_dim
        self.target_dim = target_dim
        self.shared_dim = shared_dim
        self.num_classes = num_classes
        self.freeze_source_encoder = freeze_source_encoder
        self.use_cdann = use_cdann

    def forward(
        self,
        source_data: Optional[torch.Tensor] = None,
        target_data: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
        return_features: bool = False
    ) -> dict:
        """
        Forward pass through dual-encoder DANN.

        Args:
            source_data: Source domain input [B_src, source_input_dim]
            target_data: Target domain input [B_tgt, target_input_dim]
            alpha: Gradient reversal strength (0.0 = no adaptation, 1.0 = full)
            return_features: Return latent features for visualization

        Returns:
            dict with keys:
                - 'source_latent': Source latent features [B_src, shared_dim] (if source_data provided)
                - 'target_latent': Target latent features [B_tgt, shared_dim] (if target_data provided)
                - 'source_domain_pred': Domain predictions for source [B_src, 1]
                - 'target_domain_pred': Domain predictions for target [B_tgt, 1]
                - 'source_class_pred': Cell cycle predictions for source [B_src, num_classes]
                - 'target_class_pred': Cell cycle predictions for target [B_tgt, num_classes] (optional)
        """
        outputs = {}

        if source_data is not None:
            # Handle dict input for tokenized data (dynamic embeddings)
            if isinstance(source_data, dict):
                # Check if it's Geneformer (no values) or scGPT (with values)
                if 'values' in source_data:
                    # scGPT-style: (input_ids, values, attention_mask)
                    source_encoded = self.source_encoder(
                        source_data['input_ids'],
                        source_data['values'],
                        attention_mask=source_data['attention_mask']
                    )
                else:
                    # Geneformer-style: (input_ids, attention_mask)
                    # For BertForMaskedLM, access the underlying bert model to get embeddings
                    if hasattr(self.source_encoder, 'bert'):
                        bert_output = self.source_encoder.bert(
                            input_ids=source_data['input_ids'],
                            attention_mask=source_data['attention_mask'],
                            return_dict=True
                        )
                        source_encoded = bert_output.last_hidden_state[:, 0, :]
                    else:
                        source_encoded = self.source_encoder(
                            input_ids=source_data['input_ids'],
                            attention_mask=source_data['attention_mask'],
                            output_hidden_states=True
                        )
                # Extract embeddings from model output (if not already a tensor)
                if not isinstance(source_encoded, torch.Tensor):
                    if hasattr(source_encoded, 'last_hidden_state'):
                        # Transformer output with last_hidden_state
                        source_encoded = source_encoded.last_hidden_state[:, 0, :]
                    elif hasattr(source_encoded, 'hidden_states') and source_encoded.hidden_states is not None:
                        # Use last hidden state if available
                        source_encoded = source_encoded.hidden_states[-1][:, 0, :]
                    elif hasattr(source_encoded, 'pooler_output') and source_encoded.pooler_output is not None:
                        # Use pooler output if available
                        source_encoded = source_encoded.pooler_output
                    else:
                        raise TypeError(f"Unexpected source_encoded type: {type(source_encoded)}")
                elif source_encoded.ndim == 3:
                    # Tensor with sequence dimension - extract [CLS] token
                    source_encoded = source_encoded[:, 0, :]
            else:
                # Pre-computed embeddings
                source_encoded = self.source_encoder(source_data)
            source_latent = self.projection(source_encoded, domain='source')

            source_class_pred = self.label_classifier(source_latent)

            source_grl = gradient_reversal_layer(source_latent, alpha=alpha)

            if self.use_cdann:
                source_class_probs = F.softmax(source_class_pred, dim=1)
                source_domain_pred = self.domain_discriminator(source_grl, source_class_probs)
            else:
                source_domain_pred = self.domain_discriminator(source_grl)

            outputs['source_latent'] = source_latent
            outputs['source_domain_pred'] = source_domain_pred
            outputs['source_class_pred'] = source_class_pred

        if target_data is not None:
            target_encoded = self.target_encoder(target_data)
            target_latent = self.projection(target_encoded, domain='target')

            target_class_pred = self.label_classifier(target_latent)

            target_grl = gradient_reversal_layer(target_latent, alpha=alpha)

            if self.use_cdann:
                target_class_probs = F.softmax(target_class_pred, dim=1)
                target_domain_pred = self.domain_discriminator(target_grl, target_class_probs)
            else:
                target_domain_pred = self.domain_discriminator(target_grl)

            outputs['target_latent'] = target_latent
            outputs['target_domain_pred'] = target_domain_pred
            outputs['target_class_pred'] = target_class_pred

        return outputs

    @staticmethod
    def get_grl_lambda(current_epoch: int, total_epochs: int, gamma: float = 10.0) -> float:
        """
        Compute GRL lambda with progressive annealing schedule.

        Formula: λ(p) = 2/(1 + exp(-γ*p)) - 1
        where p = current_epoch / total_epochs ∈ [0, 1]

        Args:
            current_epoch: Current training epoch (0-indexed)
            total_epochs: Total number of training epochs
            gamma: Annealing rate (higher = faster ramp-up, default 10.0 from DANN paper)

        Returns:
            lambda value ∈ [0, 1]

        Behavior:
            - Epoch 0: λ ≈ 0.0 (no domain adaptation, focus on classification)
            - Epoch total/2: λ ≈ 0.76 (moderate adaptation)
            - Epoch total: λ ≈ 1.0 (full domain adaptation)
        """
        p = current_epoch / max(total_epochs, 1)
        lambda_val = 2.0 / (1.0 + np.exp(-gamma * p)) - 1.0
        return float(lambda_val)

    def get_latent(
        self,
        data: torch.Tensor,
        domain: str = 'source'
    ) -> torch.Tensor:
        """
        Get latent representation for input data.

        Args:
            data: Input data [B, input_dim]
            domain: 'source' or 'target'

        Returns:
            Latent features [B, shared_dim]
        """
        with torch.no_grad():
            if domain == 'source':
                encoded = self.source_encoder(data)
            else:
                encoded = self.target_encoder(data)

            latent = self.projection(encoded, domain=domain)

        return latent


def compute_dual_encoder_loss(
    outputs: dict,
    source_labels: Optional[torch.Tensor] = None,
    target_labels: Optional[torch.Tensor] = None,
    lambda_domain: float = 1.0,
    lambda_class: float = 1.0,
    lambda_target_class: float = 1.0,
    lambda_contrastive: float = 0.5,
    use_contrastive_loss: bool = False,
    contrastive_temperature: float = 0.1,
    lambda_balance: float = 0.5,
    use_balance_loss: bool = False,
    source_class_prior: Optional[torch.Tensor] = None,
    lambda_entropy: float = 0.1,
    use_entropy_loss: bool = False,
    confidence_threshold: float = 0.5,
    divergence_type: str = 'kl',
    device: str = 'cuda'
) -> Tuple[torch.Tensor, dict]:
    """
    Compute total DANN loss for dual-encoder architecture.

    Loss = λ_class * L_class + λ_domain * L_domain + λ_contrastive * L_contrastive + λ_balance * L_balance + λ_entropy * L_entropy

    Args:
        outputs: Dictionary from DualEncoderDANN.forward()
        source_labels: Ground truth labels for source data [B_src]
        lambda_domain: Weight for domain adaptation loss
        lambda_class: Weight for classification loss
        lambda_contrastive: Weight for contrastive loss (default 0.5)
        use_contrastive_loss: Enable contrastive loss for class semantics (default False)
        contrastive_temperature: Temperature for contrastive loss (default 0.1)
        lambda_balance: Weight for target class balance loss (default 0.5)
        use_balance_loss: Enable balance loss to prevent target collapse (default False)
        source_class_prior: Source class distribution [num_classes] (required if use_balance_loss=True)
        lambda_entropy: Weight for entropy minimization loss (default 0.1)
        use_entropy_loss: Enable entropy minimization for confident target predictions (default False)
        confidence_threshold: Only minimize entropy for predictions with max_prob > threshold (default 0.5)
        device: Device for tensors

    Returns:
        total_loss: Weighted sum of losses
        loss_dict: Individual loss components for logging
    """
    loss_dict = {}
    total_loss = torch.tensor(0.0, device=device)

    if 'source_class_pred' in outputs and source_labels is not None:
        class_loss = F.cross_entropy(outputs['source_class_pred'], source_labels)
        loss_dict['classification'] = class_loss.item()
        total_loss += lambda_class * class_loss

    # Target classification loss (for semi-supervised training with paired labels)
    if 'target_class_pred' in outputs and target_labels is not None:
        target_class_loss = F.cross_entropy(outputs['target_class_pred'], target_labels)
        loss_dict['target_classification'] = target_class_loss.item()
        total_loss += lambda_target_class * target_class_loss

    if 'source_domain_pred' in outputs:
        batch_size = outputs['source_domain_pred'].shape[0]
        source_domain_labels = torch.zeros(batch_size, device=device)
        domain_loss_source = F.binary_cross_entropy_with_logits(
            outputs['source_domain_pred'].squeeze(),
            source_domain_labels
        )
        loss_dict['domain_source'] = domain_loss_source.item()
        total_loss += lambda_domain * domain_loss_source

        # Track domain confusion accuracy (ChatGPT's recommendation #3)
        # Domain confusion accuracy: how well discriminator distinguishes domains
        # Ideal for DANN: ~50% (discriminator confused = good domain alignment)
        # If > 65% for many epochs: discriminator too strong, slow down GRL
        source_domain_probs = torch.sigmoid(outputs['source_domain_pred'].squeeze())
        source_correct = (source_domain_probs < 0.5).float().mean().item()  # Should predict 0
        loss_dict['domain_acc_source'] = source_correct

    if 'target_domain_pred' in outputs:
        batch_size = outputs['target_domain_pred'].shape[0]
        target_domain_labels = torch.ones(batch_size, device=device)
        domain_loss_target = F.binary_cross_entropy_with_logits(
            outputs['target_domain_pred'].squeeze(),
            target_domain_labels
        )
        loss_dict['domain_target'] = domain_loss_target.item()
        total_loss += lambda_domain * domain_loss_target

        # Track target domain accuracy
        target_domain_probs = torch.sigmoid(outputs['target_domain_pred'].squeeze())
        target_correct = (target_domain_probs > 0.5).float().mean().item()  # Should predict 1
        loss_dict['domain_acc_target'] = target_correct

        # Compute overall domain confusion accuracy
        # Average of source and target accuracy (0.5 = perfect confusion)
        if 'domain_acc_source' in loss_dict:
            domain_acc = (loss_dict['domain_acc_source'] + target_correct) / 2.0
            loss_dict['domain_acc'] = domain_acc

    if use_contrastive_loss:
        if ('source_latent' in outputs and 'target_latent' in outputs and
            'target_class_pred' in outputs and source_labels is not None):

            target_pseudo_labels = torch.argmax(outputs['target_class_pred'], dim=1)

            contrastive_criterion = ContrastiveLoss(temperature=contrastive_temperature)
            contrastive_loss = contrastive_criterion(
                source_features=outputs['source_latent'],
                target_features=outputs['target_latent'],
                source_labels=source_labels,
                target_labels=target_pseudo_labels
            )

            loss_dict['contrastive'] = contrastive_loss.item()
            total_loss += lambda_contrastive * contrastive_loss

    # Target class balance loss (prevents collapse)
    if use_balance_loss and 'target_class_pred' in outputs and source_class_prior is not None:
        # Choose divergence measure based on divergence_type
        if divergence_type == 'kl':
            balance_loss = compute_balance_loss(
                target_predictions=outputs['target_class_pred'],
                source_class_prior=source_class_prior,
                device=device
            )
        elif divergence_type == 'jensen':
            balance_loss = compute_balance_loss_jensen(
                target_predictions=outputs['target_class_pred'],
                source_class_prior=source_class_prior,
                device=device
            )
        elif divergence_type == 'coral':
            # CORAL uses features, not distributions
            if 'source_latent' in outputs and 'target_latent' in outputs:
                balance_loss = compute_balance_loss_coral(
                    source_features=outputs['source_latent'],
                    target_features=outputs['target_latent'],
                    device=device
                )
            else:
                # Fallback to KL if features not available
                print("WARNING: CORAL requires source_latent and target_latent, falling back to KL")
                balance_loss = compute_balance_loss(
                    target_predictions=outputs['target_class_pred'],
                    source_class_prior=source_class_prior,
                    device=device
                )
        elif divergence_type == 'mmd':
            # MMD uses features, not distributions
            if 'source_latent' in outputs and 'target_latent' in outputs:
                balance_loss = compute_balance_loss_mmd(
                    source_features=outputs['source_latent'],
                    target_features=outputs['target_latent'],
                    device=device
                )
            else:
                # Fallback to KL if features not available
                print("WARNING: MMD requires source_latent and target_latent, falling back to KL")
                balance_loss = compute_balance_loss(
                    target_predictions=outputs['target_class_pred'],
                    source_class_prior=source_class_prior,
                    device=device
                )
        else:
            raise ValueError(f"Unknown divergence_type: {divergence_type}. Choose from: kl, jensen, coral, mmd")

        loss_dict['balance'] = balance_loss.item()
        total_loss += lambda_balance * balance_loss

    # Target entropy minimization (encourages confident predictions, prevents early collapse)
    # ChatGPT's "Option 4" - most effective for DANN when target labels are missing
    if use_entropy_loss and 'target_class_pred' in outputs:
        entropy_loss = compute_entropy_loss(
            target_predictions=outputs['target_class_pred'],
            confidence_threshold=confidence_threshold,
            device=device
        )
        loss_dict['entropy'] = entropy_loss.item()
        total_loss += lambda_entropy * entropy_loss

    loss_dict['total'] = total_loss.item()

    return total_loss, loss_dict


def compute_balance_loss(target_predictions, source_class_prior, device='cuda'):
    """
    Compute KL divergence between target predictions distribution and source class distribution.

    Prevents target domain collapse by penalizing predictions that deviate from source class balance.

    Args:
        target_predictions: Tensor of shape (batch_size, num_classes) - target class logits/probs
        source_class_prior: Tensor of shape (num_classes,) - source class distribution [p(G1), p(S), p(G2M)]
        device: Device for computation

    Returns:
        balance_loss: KL divergence between target batch distribution and source prior
    """
    # Get predicted probabilities for target (batch_size, num_classes)
    target_probs = F.softmax(target_predictions, dim=1)

    # Compute empirical distribution over this batch (num_classes,)
    target_batch_dist = target_probs.mean(dim=0)  # Average over batch

    # Ensure source prior is on correct device
    source_prior = source_class_prior.to(device)

    # Add small epsilon to avoid log(0)
    eps = 1e-8
    target_batch_dist = torch.clamp(target_batch_dist, min=eps, max=1.0)
    source_prior = torch.clamp(source_prior, min=eps, max=1.0)

    # Compute KL divergence: KL(target_dist || source_dist)
    # KL(P||Q) = sum(P * log(P/Q))
    kl_loss = F.kl_div(
        torch.log(target_batch_dist),  # log(P)
        source_prior,                   # Q
        reduction='batchmean'
    )

    return kl_loss


def compute_balance_loss_jensen(target_predictions, source_class_prior, device='cuda'):
    """
    Compute Jensen-Shannon divergence between target predictions and source class distribution.

    Jensen-Shannon is symmetric (unlike KL) and bounded [0, log(2)].
    More stable when distributions are very different.

    JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = (P+Q)/2

    Args:
        target_predictions: Tensor of shape (batch_size, num_classes)
        source_class_prior: Tensor of shape (num_classes,)
        device: Device for computation

    Returns:
        js_divergence: Jensen-Shannon divergence
    """
    # Get predicted probabilities for target
    target_probs = F.softmax(target_predictions, dim=1)
    target_batch_dist = target_probs.mean(dim=0)

    # Ensure source prior is on correct device
    source_prior = source_class_prior.to(device)

    # Add small epsilon to avoid log(0)
    eps = 1e-8
    P = torch.clamp(target_batch_dist, min=eps, max=1.0)
    Q = torch.clamp(source_prior, min=eps, max=1.0)

    # Compute mixture distribution M = (P + Q) / 2
    M = (P + Q) / 2.0
    M = torch.clamp(M, min=eps, max=1.0)

    # JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    kl_pm = F.kl_div(torch.log(M), P, reduction='batchmean')
    kl_qm = F.kl_div(torch.log(M), Q, reduction='batchmean')
    js_div = 0.5 * kl_pm + 0.5 * kl_qm

    return js_div


def compute_balance_loss_coral(source_features, target_features, device='cuda'):
    """
    Compute CORAL (Correlation Alignment) loss between source and target features.

    CORAL aligns second-order statistics (covariance) of source and target domains.
    Performs "whitening + recoloring" to reduce domain shift in feature space.

    CORAL(S, T) = ||Cov(S) - Cov(T)||_F^2 / (4 * d^2)

    Args:
        source_features: Tensor of shape (batch_size, feature_dim) - source latent features
        target_features: Tensor of shape (batch_size, feature_dim) - target latent features
        device: Device for computation

    Returns:
        coral_loss: Squared Frobenius norm of covariance difference
    """
    # Ensure features are 2D
    if source_features.dim() > 2:
        source_features = source_features.view(source_features.size(0), -1)
    if target_features.dim() > 2:
        target_features = target_features.view(target_features.size(0), -1)

    d = source_features.size(1)  # Feature dimension

    # Center the features (subtract mean)
    source_centered = source_features - source_features.mean(dim=0, keepdim=True)
    target_centered = target_features - target_features.mean(dim=0, keepdim=True)

    # Compute covariance matrices
    n_source = source_features.size(0)
    n_target = target_features.size(0)

    cov_source = torch.mm(source_centered.t(), source_centered) / (n_source - 1)
    cov_target = torch.mm(target_centered.t(), target_centered) / (n_target - 1)

    # Compute Frobenius norm of difference
    coral_loss = torch.sum((cov_source - cov_target) ** 2) / (4 * d * d)

    return coral_loss


def compute_balance_loss_mmd(source_features, target_features, device='cuda', kernel='rbf', bandwidth=None):
    """
    Compute MMD (Maximum Mean Discrepancy) between source and target features.

    MMD is a kernel-based distance that captures higher-order statistics beyond mean.
    Used in many domain adaptation papers (DAN, Deep CORAL, etc.).

    MMD^2(S, T) = ||E[φ(x_s)] - E[φ(x_t)]||^2

    Args:
        source_features: Tensor of shape (batch_size, feature_dim)
        target_features: Tensor of shape (batch_size, feature_dim)
        device: Device for computation
        kernel: Kernel type ('rbf' or 'linear')
        bandwidth: RBF kernel bandwidth (auto-computed if None)

    Returns:
        mmd_loss: MMD^2 between source and target distributions
    """
    # Ensure features are 2D
    if source_features.dim() > 2:
        source_features = source_features.view(source_features.size(0), -1)
    if target_features.dim() > 2:
        target_features = target_features.view(target_features.size(0), -1)

    def rbf_kernel(X, Y, bandwidth):
        """Compute RBF kernel matrix between X and Y."""
        # X: (n, d), Y: (m, d)
        # Output: (n, m)
        n = X.size(0)
        m = Y.size(0)

        # Expand dimensions for pairwise distance computation
        X_expanded = X.unsqueeze(1).expand(n, m, -1)  # (n, m, d)
        Y_expanded = Y.unsqueeze(0).expand(n, m, -1)  # (n, m, d)

        # Compute squared Euclidean distances
        sq_dist = torch.sum((X_expanded - Y_expanded) ** 2, dim=2)

        # Apply RBF kernel
        return torch.exp(-sq_dist / (2 * bandwidth ** 2))

    def linear_kernel(X, Y):
        """Compute linear kernel matrix between X and Y."""
        return torch.mm(X, Y.t())

    # Auto-compute bandwidth using median heuristic
    if kernel == 'rbf' and bandwidth is None:
        # Compute pairwise distances for bandwidth estimation
        with torch.no_grad():
            combined = torch.cat([source_features, target_features], dim=0)
            n = combined.size(0)
            # Sample subset if too large (for efficiency)
            if n > 1000:
                indices = torch.randperm(n)[:1000]
                combined = combined[indices]

            dists = torch.cdist(combined, combined)
            bandwidth = torch.median(dists[dists > 0])

    # Compute kernel matrices
    if kernel == 'rbf':
        K_ss = rbf_kernel(source_features, source_features, bandwidth)
        K_tt = rbf_kernel(target_features, target_features, bandwidth)
        K_st = rbf_kernel(source_features, target_features, bandwidth)
    elif kernel == 'linear':
        K_ss = linear_kernel(source_features, source_features)
        K_tt = linear_kernel(target_features, target_features)
        K_st = linear_kernel(source_features, target_features)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # Compute MMD^2
    n_source = source_features.size(0)
    n_target = target_features.size(0)

    mmd_loss = (K_ss.sum() / (n_source * n_source) +
                K_tt.sum() / (n_target * n_target) -
                2 * K_st.sum() / (n_source * n_target))

    return mmd_loss


def compute_entropy_loss(target_predictions, confidence_threshold=0.5, device='cuda'):
    """
    Compute entropy minimization loss for target predictions.

    Encourages confident predictions on target domain while avoiding early collapse.
    Uses confidence threshold to only minimize entropy for high-confidence predictions.

    This is ChatGPT's "Option 4" - most effective for DANN when target labels missing.

    Args:
        target_predictions: Tensor of shape (batch_size, num_classes) - target class logits
        confidence_threshold: Only minimize entropy for predictions with max_prob > threshold
        device: Device for computation

    Returns:
        entropy_loss: Average entropy of confident predictions
    """
    # Get predicted probabilities (batch_size, num_classes)
    target_probs = F.softmax(target_predictions, dim=1)

    # Find max probability per sample
    max_probs, _ = torch.max(target_probs, dim=1)  # (batch_size,)

    # Only compute entropy for confident predictions (avoids early collapse)
    confident_mask = max_probs > confidence_threshold  # (batch_size,)

    if confident_mask.sum() == 0:
        # No confident predictions, return zero loss
        return torch.tensor(0.0, device=device)

    # Select only confident predictions
    confident_probs = target_probs[confident_mask]  # (n_confident, num_classes)

    # Compute entropy: H(p) = -sum(p * log(p))
    # Add epsilon to avoid log(0)
    eps = 1e-8
    confident_probs = torch.clamp(confident_probs, min=eps, max=1.0)

    # Entropy per sample
    entropy_per_sample = -torch.sum(confident_probs * torch.log(confident_probs), dim=1)

    # Average entropy (we want to MINIMIZE this)
    entropy_loss = entropy_per_sample.mean()

    return entropy_loss
