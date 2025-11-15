"""
Peak Count Embedder for Raw ATAC-seq Data

Directly embeds ATAC peak counts without gene conversion.
Supports LoRA for parameter-efficient fine-tuning.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class PeriodicActivation(nn.Module):
    """
    Periodic activation using sin/cos for modeling cell cycle periodicity.

    Maps x → [sin(x), cos(x)] then projects back to original dimension.
    Useful for cell cycle classification (G1 → S → G2M → G1 is circular).

    Reference: CCAN uses periodic activations for cell cycle modeling.
    """
    def __init__(self, dim: int):
        super().__init__()
        # Linear projection to compress [sin(x), cos(x)] back to original dim
        self.post_linear = nn.Linear(dim * 2, dim)
        # Xavier initialization (better for periodic functions)
        nn.init.xavier_uniform_(self.post_linear.weight)
        nn.init.zeros_(self.post_linear.bias)

    def forward(self, x):
        # Apply sin/cos transformation
        x_sin = torch.sin(x)
        x_cos = torch.cos(x)
        # Concatenate [sin(x), cos(x)] along feature dimension
        x_periodic = torch.cat([x_sin, x_cos], dim=-1)
        # Project back to original dimension
        return self.post_linear(x_periodic)


def get_activation(activation: str, dim: int = None) -> nn.Module:
    """
    Get activation function by name.

    Supports: relu, leaky_relu, elu, prelu, gelu, selu, swish, mish, periodic

    Args:
        activation: Name of activation function
        dim: Dimension (required for 'periodic' activation)
    """
    if activation == 'gelu':
        return nn.GELU()
    elif activation == 'selu':
        return nn.SELU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.01)
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'prelu':
        return nn.PReLU()
    elif activation == 'swish':
        return nn.SiLU()  # Swish = SiLU in PyTorch
    elif activation == 'mish':
        return nn.Mish()
    elif activation == 'periodic':
        if dim is None:
            raise ValueError("dim must be specified for periodic activation")
        return PeriodicActivation(dim)
    else:
        return nn.ReLU()


class PeakEmbedder(nn.Module):
    """
    Trainable embedder for ATAC-seq peak counts.

    Maps raw peak counts [n_cells, n_peaks] → embeddings [n_cells, hidden_dim]
    without requiring peak→gene conversion.

    Architecture:
        - Multi-layer MLP with residual connections
        - Batch normalization for stability
        - Dropout for regularization
        - Optional LoRA for fine-tuning
    """

    def __init__(
        self,
        n_peaks: int,
        hidden_dim: int = 512,
        intermediate_dim: int = 2048,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = 'relu',
        use_batch_norm: bool = True,
        use_residual: bool = True
    ):
        """
        Initialize peak embedder.

        Args:
            n_peaks: Number of ATAC peaks
            hidden_dim: Output embedding dimension (should match RNA embedder)
            intermediate_dim: Hidden layer dimension
            num_layers: Number of transformation layers
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', 'selu')
            use_batch_norm: Use batch normalization
            use_residual: Use residual connections
        """
        super().__init__()
        self.n_peaks = n_peaks
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual

        # Input projection: peaks → intermediate
        self.input_proj = nn.Sequential(
            nn.Linear(n_peaks, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim) if use_batch_norm else nn.Identity(),
            get_activation(activation, dim=intermediate_dim),
            nn.Dropout(dropout)
        )

        # Middle layers with residual connections
        self.layers = nn.ModuleList()
        for i in range(num_layers - 2):
            layer = nn.Sequential(
                nn.Linear(intermediate_dim, intermediate_dim),
                nn.BatchNorm1d(intermediate_dim) if use_batch_norm else nn.Identity(),
                get_activation(activation, dim=intermediate_dim),
                nn.Dropout(dropout)
            )
            self.layers.append(layer)

        # Output projection: intermediate → hidden_dim
        self.output_proj = nn.Sequential(
            nn.Linear(intermediate_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
        )

        # Layer normalization for final output (like scGPT)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, peak_counts: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            peak_counts: [batch_size, n_peaks] ATAC peak counts

        Returns:
            embeddings: [batch_size, hidden_dim]
        """
        # Input projection
        x = self.input_proj(peak_counts)

        # Middle layers with optional residual connections
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)  # Residual connection
            else:
                x = layer(x)

        # Output projection
        x = self.output_proj(x)

        # Layer normalization (consistent with scGPT/Geneformer)
        x = self.layer_norm(x)

        return x

    def get_embeddings_batched(
        self,
        peak_counts: np.ndarray,
        batch_size: int = 32,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Generate embeddings for peak count matrix with batching.

        Args:
            peak_counts: [n_cells, n_peaks] numpy array
            batch_size: Batch size for inference
            device: Device to use

        Returns:
            embeddings: [n_cells, hidden_dim] tensor
        """
        self.eval()
        self.to(device)

        n_cells = peak_counts.shape[0]
        embeddings = []

        with torch.no_grad():
            for i in range(0, n_cells, batch_size):
                batch = peak_counts[i:i+batch_size]
                batch_tensor = torch.tensor(batch, dtype=torch.float32, device=device)

                batch_emb = self(batch_tensor)
                embeddings.append(batch_emb.cpu())

        embeddings = torch.cat(embeddings, dim=0)
        return embeddings



class PeakVAE(nn.Module):
    """
    Variational Autoencoder for ATAC-seq peak counts.

    Maps peak counts → latent embeddings via probabilistic encoder.
    Includes reconstruction decoder for regularization.

    Architecture:
        Encoder: peaks → intermediate → [mu, logvar] → latent (via reparameterization)
        Decoder: latent → intermediate → peaks (reconstruction)

    Benefits for DANN:
        - KL divergence creates smooth, Gaussian-like latent space (better for CORAL)
        - Reconstruction loss prevents mode collapse
        - More robust to sparse ATAC data
    """

    def __init__(
        self,
        n_peaks: int,
        hidden_dim: int = 512,
        intermediate_dim: int = 2048,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = 'relu',
        beta_vae: float = 1.0,
        use_batch_norm: bool = True
    ):
        """
        Initialize Peak VAE.

        Args:
            n_peaks: Number of ATAC peaks
            hidden_dim: Latent embedding dimension (output)
            intermediate_dim: Hidden layer dimension
            num_layers: Number of layers in encoder/decoder
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', 'selu', etc.)
            beta_vae: Weight for KL divergence (beta-VAE)
            use_batch_norm: Use batch normalization
        """
        super().__init__()
        self.n_peaks = n_peaks
        self.hidden_dim = hidden_dim
        self.beta_vae = beta_vae

        # Encoder: peaks → intermediate
        encoder_layers = []
        encoder_layers.append(nn.Linear(n_peaks, intermediate_dim))
        if use_batch_norm:
            encoder_layers.append(nn.BatchNorm1d(intermediate_dim))
        encoder_layers.append(get_activation(activation, dim=intermediate_dim))
        encoder_layers.append(nn.Dropout(dropout))

        # Additional encoder layers
        for _ in range(num_layers - 2):
            encoder_layers.append(nn.Linear(intermediate_dim, intermediate_dim))
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(intermediate_dim))
            encoder_layers.append(get_activation(activation, dim=intermediate_dim))
            encoder_layers.append(nn.Dropout(dropout))

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space projections: intermediate → [mu, logvar]
        self.fc_mu = nn.Linear(intermediate_dim, hidden_dim)
        self.fc_logvar = nn.Linear(intermediate_dim, hidden_dim)

        # Decoder: latent → intermediate
        decoder_layers = []
        decoder_layers.append(nn.Linear(hidden_dim, intermediate_dim))
        if use_batch_norm:
            decoder_layers.append(nn.BatchNorm1d(intermediate_dim))
        decoder_layers.append(get_activation(activation, dim=intermediate_dim))
        decoder_layers.append(nn.Dropout(dropout))

        # Additional decoder layers
        for _ in range(num_layers - 2):
            decoder_layers.append(nn.Linear(intermediate_dim, intermediate_dim))
            if use_batch_norm:
                decoder_layers.append(nn.BatchNorm1d(intermediate_dim))
            decoder_layers.append(get_activation(activation, dim=intermediate_dim))
            decoder_layers.append(nn.Dropout(dropout))

        # Final reconstruction layer
        decoder_layers.append(nn.Linear(intermediate_dim, n_peaks))

        self.decoder = nn.Sequential(*decoder_layers)

        # Layer normalization for embeddings (like scGPT)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization for VAE."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def encode(self, peak_counts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode peak counts to latent distribution parameters.

        Args:
            peak_counts: [batch_size, n_peaks]

        Returns:
            mu: [batch_size, hidden_dim] - mean of latent distribution
            logvar: [batch_size, hidden_dim] - log variance of latent distribution
        """
        h = self.encoder(peak_counts)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * epsilon

        Args:
            mu: [batch_size, hidden_dim]
            logvar: [batch_size, hidden_dim]

        Returns:
            z: [batch_size, hidden_dim] - sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstructed peaks.

        Args:
            z: [batch_size, hidden_dim]

        Returns:
            reconstruction: [batch_size, n_peaks]
        """
        return self.decoder(z)

    def forward(self, peak_counts: torch.Tensor, return_reconstruction: bool = False):
        """
        Forward pass through VAE.

        Args:
            peak_counts: [batch_size, n_peaks]
            return_reconstruction: If True, return (embedding, reconstruction, mu, logvar)

        Returns:
            If return_reconstruction=False: embedding [batch_size, hidden_dim]
            If return_reconstruction=True: (embedding, reconstruction, mu, logvar)
        """
        # Encode
        mu, logvar = self.encode(peak_counts)

        # Reparameterize (sample from latent distribution)
        z = self.reparameterize(mu, logvar)

        # Normalize embeddings (consistent with scGPT/Geneformer)
        embedding = self.layer_norm(z)

        if return_reconstruction:
            # Decode
            reconstruction = self.decode(z)
            return embedding, reconstruction, mu, logvar
        else:
            return embedding

    def compute_loss(
        self,
        peak_counts: torch.Tensor,
        reconstruction: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss: reconstruction + KL divergence.

        Args:
            peak_counts: Original input [batch_size, n_peaks]
            reconstruction: Reconstructed output [batch_size, n_peaks]
            mu: Latent mean [batch_size, hidden_dim]
            logvar: Latent log variance [batch_size, hidden_dim]

        Returns:
            total_loss: Combined loss
            recon_loss: MSE reconstruction loss
            kl_loss: KL divergence
        """
        # Reconstruction loss (MSE for continuous values)
        recon_loss = nn.functional.mse_loss(reconstruction, peak_counts, reduction='mean')

        # KL divergence: D_KL(N(mu, sigma) || N(0, 1))
        # = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = kl_loss.mean()

        # Total loss with beta weighting
        total_loss = recon_loss + self.beta_vae * kl_loss

        return total_loss, recon_loss, kl_loss

    def get_embeddings_batched(
        self,
        peak_counts: np.ndarray,
        batch_size: int = 32,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Generate embeddings for peak count matrix with batching.

        Args:
            peak_counts: [n_cells, n_peaks] numpy array
            batch_size: Batch size for inference
            device: Device to use

        Returns:
            embeddings: [n_cells, hidden_dim] tensor
        """
        self.eval()
        self.to(device)

        n_cells = peak_counts.shape[0]
        embeddings = []

        with torch.no_grad():
            for i in range(0, n_cells, batch_size):
                batch = peak_counts[i:i+batch_size]
                batch_tensor = torch.tensor(batch, dtype=torch.float32, device=device)

                # Just get embeddings (no reconstruction during inference)
                batch_emb = self(batch_tensor, return_reconstruction=False)
                embeddings.append(batch_emb.cpu())

        embeddings = torch.cat(embeddings, dim=0)
        return embeddings


class PeakDAE(nn.Module):
    """
    Denoising Autoencoder for ATAC-seq peak counts.

    Adds noise to input peaks, forces model to reconstruct clean signal.
    Simpler than VAE (no KL divergence), focuses on robustness.

    Architecture:
        Encoder: peaks + noise → intermediate → latent
        Decoder: latent → intermediate → peaks (clean reconstruction)

    Benefits for DANN:
        - Learns robust features invariant to noise (good for sparse ATAC)
        - No probabilistic overhead (simpler than VAE)
        - Reconstruction loss prevents overfitting
        - Good for noisy/dropout-heavy data
    """

    def __init__(
        self,
        n_peaks: int,
        hidden_dim: int = 512,
        intermediate_dim: int = 2048,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = 'relu',
        noise_level: float = 0.1,
        use_batch_norm: bool = True
    ):
        """
        Initialize Peak DAE.

        Args:
            n_peaks: Number of ATAC peaks
            hidden_dim: Latent embedding dimension (output)
            intermediate_dim: Hidden layer dimension
            num_layers: Number of layers in encoder/decoder
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', 'selu', etc.)
            noise_level: Std dev of Gaussian noise added to input (0.0 = no noise)
            use_batch_norm: Use batch normalization
        """
        super().__init__()
        self.n_peaks = n_peaks
        self.hidden_dim = hidden_dim
        self.noise_level = noise_level

        # Encoder: peaks + noise → intermediate → latent
        encoder_layers = []
        encoder_layers.append(nn.Linear(n_peaks, intermediate_dim))
        if use_batch_norm:
            encoder_layers.append(nn.BatchNorm1d(intermediate_dim))
        encoder_layers.append(get_activation(activation, dim=intermediate_dim))
        encoder_layers.append(nn.Dropout(dropout))

        # Additional encoder layers
        for _ in range(num_layers - 2):
            encoder_layers.append(nn.Linear(intermediate_dim, intermediate_dim))
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(intermediate_dim))
            encoder_layers.append(get_activation(activation, dim=intermediate_dim))
            encoder_layers.append(nn.Dropout(dropout))

        # Final encoder layer to latent space
        encoder_layers.append(nn.Linear(intermediate_dim, hidden_dim))

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder: latent → intermediate → peaks
        decoder_layers = []
        decoder_layers.append(nn.Linear(hidden_dim, intermediate_dim))
        if use_batch_norm:
            decoder_layers.append(nn.BatchNorm1d(intermediate_dim))
        decoder_layers.append(get_activation(activation, dim=intermediate_dim))
        decoder_layers.append(nn.Dropout(dropout))

        # Additional decoder layers
        for _ in range(num_layers - 2):
            decoder_layers.append(nn.Linear(intermediate_dim, intermediate_dim))
            if use_batch_norm:
                decoder_layers.append(nn.BatchNorm1d(intermediate_dim))
            decoder_layers.append(get_activation(activation, dim=intermediate_dim))
            decoder_layers.append(nn.Dropout(dropout))

        # Final reconstruction layer
        decoder_layers.append(nn.Linear(intermediate_dim, n_peaks))

        self.decoder = nn.Sequential(*decoder_layers)

        # Layer normalization for embeddings (like scGPT)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming initialization for DAE."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def add_noise(self, peak_counts: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to peak counts.

        Args:
            peak_counts: [batch_size, n_peaks]

        Returns:
            noisy_peaks: [batch_size, n_peaks]
        """
        if self.training and self.noise_level > 0:
            noise = torch.randn_like(peak_counts) * self.noise_level
            return peak_counts + noise
        else:
            return peak_counts

    def encode(self, peak_counts: torch.Tensor) -> torch.Tensor:
        """
        Encode peak counts to latent representation.

        Args:
            peak_counts: [batch_size, n_peaks]

        Returns:
            latent: [batch_size, hidden_dim]
        """
        # Add noise during training (denoising objective)
        noisy_peaks = self.add_noise(peak_counts)

        # Encode to latent space
        latent = self.encoder(noisy_peaks)

        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstructed peaks.

        Args:
            latent: [batch_size, hidden_dim]

        Returns:
            reconstruction: [batch_size, n_peaks]
        """
        return self.decoder(latent)

    def forward(self, peak_counts: torch.Tensor, return_reconstruction: bool = False):
        """
        Forward pass through DAE.

        Args:
            peak_counts: [batch_size, n_peaks]
            return_reconstruction: If True, return (embedding, reconstruction)

        Returns:
            If return_reconstruction=False: embedding [batch_size, hidden_dim]
            If return_reconstruction=True: (embedding, reconstruction)
        """
        # Encode (with noise added internally)
        latent = self.encode(peak_counts)

        # Normalize embeddings (consistent with scGPT/Geneformer)
        embedding = self.layer_norm(latent)

        if return_reconstruction:
            # Decode
            reconstruction = self.decode(latent)
            return embedding, reconstruction
        else:
            return embedding

    def compute_loss(
        self,
        peak_counts: torch.Tensor,
        reconstruction: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute DAE loss: MSE reconstruction loss.

        Args:
            peak_counts: Original CLEAN input [batch_size, n_peaks]
            reconstruction: Reconstructed output [batch_size, n_peaks]

        Returns:
            loss: MSE reconstruction loss
        """
        # Reconstruction loss (MSE between clean input and reconstruction)
        loss = nn.functional.mse_loss(reconstruction, peak_counts, reduction='mean')

        return loss

    def get_embeddings_batched(
        self,
        peak_counts: np.ndarray,
        batch_size: int = 32,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Generate embeddings for peak count matrix with batching.

        Args:
            peak_counts: [n_cells, n_peaks] numpy array
            batch_size: Batch size for inference
            device: Device to use

        Returns:
            embeddings: [n_cells, hidden_dim] tensor
        """
        self.eval()
        self.to(device)

        n_cells = peak_counts.shape[0]
        embeddings = []

        with torch.no_grad():
            for i in range(0, n_cells, batch_size):
                batch = peak_counts[i:i+batch_size]
                batch_tensor = torch.tensor(batch, dtype=torch.float32, device=device)

                # Just get embeddings (no noise or reconstruction during inference)
                batch_emb = self(batch_tensor, return_reconstruction=False)
                embeddings.append(batch_emb.cpu())

        embeddings = torch.cat(embeddings, dim=0)
        return embeddings


class PeakHybridMLP(nn.Module):
    """
    Hybrid MLP with Residual Connections for PEAK embedding.

    Improvements over basic MLP:
    - Residual/skip connections for better gradient flow
    - Batch normalization for training stability
    - GELU/SELU activations (smoother than ReLU)
    - Optional residual types: add, concat, or none
    """

    def __init__(
        self,
        n_peaks: int,
        hidden_dim: int = 512,
        intermediate_dim: int = 2048,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = 'gelu',
        use_batch_norm: bool = True,
        residual_type: str = 'add',
        use_batch_norm_input: bool = True
    ):
        super().__init__()
        self.n_peaks = n_peaks
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.activation_type = activation
        self.use_batch_norm = use_batch_norm
        self.residual_type = residual_type
        self.activation = get_activation(activation, dim=intermediate_dim)

        # Input projection: peaks → intermediate
        layers = []
        if use_batch_norm_input:
            layers.append(nn.BatchNorm1d(n_peaks))
        layers.extend([
            nn.Linear(n_peaks, intermediate_dim),
            get_activation(activation, dim=intermediate_dim),
            nn.Dropout(dropout)
        ])
        self.input_proj = nn.Sequential(*layers)

        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.residual_blocks.append(
                self._make_residual_block(intermediate_dim, dropout)
            )

        # Output projection: intermediate → hidden_dim
        self.output_proj = nn.Sequential(
            nn.Linear(intermediate_dim, hidden_dim),
            get_activation(activation, dim=hidden_dim)
        )

    def _make_residual_block(self, dim: int, dropout: float) -> nn.Module:
        """Create a single residual block."""
        layers = []

        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(dim))

        layers.extend([
            nn.Linear(dim, dim),
            self.activation,
            nn.Dropout(dropout)
        ])

        return nn.Sequential(*layers)

    def forward(self, peak_counts: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections.

        Args:
            peak_counts: [batch_size, n_peaks] ATAC peak counts

        Returns:
            embeddings: [batch_size, hidden_dim]
        """
        # Input projection
        x = self.input_proj(peak_counts)  # [B, intermediate_dim]

        # Apply residual blocks
        for block in self.residual_blocks:
            identity = x
            x = block(x)

            # Apply residual connection
            if self.residual_type == 'add':
                x = x + identity
            elif self.residual_type == 'concat':
                # Concatenate and project back to intermediate_dim
                x = torch.cat([x, identity], dim=-1)
                # Need a projection layer (add dynamically)
                if not hasattr(self, 'concat_proj'):
                    self.concat_proj = nn.Linear(2 * self.intermediate_dim, self.intermediate_dim).to(x.device)
                x = self.concat_proj(x)
            # else: no residual (just x = block(x))

        # Output projection
        embeddings = self.output_proj(x)  # [B, hidden_dim]

        return embeddings

    def get_embeddings(
        self,
        peak_counts: np.ndarray,
        genes: Optional[np.ndarray] = None,
        batch_size: int = 128,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Batch inference for large datasets.

        Args:
            peak_counts: [n_cells, n_peaks] numpy array
            genes: Ignored (for compatibility)
            batch_size: Batch size for inference
            device: Device to use

        Returns:
            embeddings: [n_cells, hidden_dim] tensor
        """
        self.eval()
        self.to(device)

        n_cells = peak_counts.shape[0]
        embeddings = []

        with torch.no_grad():
            for i in range(0, n_cells, batch_size):
                batch = peak_counts[i:i+batch_size]
                batch_tensor = torch.tensor(batch, dtype=torch.float32, device=device)

                batch_emb = self(batch_tensor)
                embeddings.append(batch_emb.cpu())

        embeddings = torch.cat(embeddings, dim=0)
        return embeddings


class PeakPeriodicMLP(nn.Module):
    """
    Periodic MLP for PEAK embedding with sin/cos activations.

    Motivation: Cell cycle is inherently periodic (G1 → S → G2M → G1).
    Uses periodic activations (sin/cos) instead of monotonic activations (ReLU/GELU)
    to better capture circular/cyclic patterns in cell cycle progression.

    Inspired by CCAN (Cell Cycle Analysis Network) which used periodic activations
    for cell cycle phase prediction.
    """

    def __init__(
        self,
        n_peaks: int,
        hidden_dim: int = 512,
        intermediate_dim: int = 2048,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = 'relu',  # Base activation before periodic transform
        use_batch_norm: bool = True,
        use_residual: bool = True
    ):
        """
        Initialize periodic MLP embedder.

        Args:
            n_peaks: Number of ATAC peaks
            hidden_dim: Output embedding dimension
            intermediate_dim: Hidden layer dimension
            num_layers: Number of transformation layers
            dropout: Dropout probability
            activation: Base activation before periodic transform
            use_batch_norm: Use batch normalization
            use_residual: Use residual connections
        """
        super().__init__()
        self.n_peaks = n_peaks
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual
        self.intermediate_dim = intermediate_dim

        # Input projection: peaks → intermediate (with base activation)
        self.input_proj = nn.Sequential(
            nn.Linear(n_peaks, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim) if use_batch_norm else nn.Identity(),
            get_activation(activation, dim=intermediate_dim),
            nn.Dropout(dropout)
        )

        # Periodic transformation layers
        # Each layer: linear → sin/cos split → concatenate → linear
        self.periodic_layers = nn.ModuleList()
        for i in range(num_layers - 2):
            layer = nn.ModuleDict({
                'pre_linear': nn.Linear(intermediate_dim, intermediate_dim),
                'pre_norm': nn.BatchNorm1d(intermediate_dim) if use_batch_norm else nn.Identity(),
                # After sin/cos split, dimension doubles (sin + cos)
                'post_linear': nn.Linear(intermediate_dim * 2, intermediate_dim),
                'post_norm': nn.BatchNorm1d(intermediate_dim) if use_batch_norm else nn.Identity(),
                'dropout': nn.Dropout(dropout)
            })
            self.periodic_layers.append(layer)

        # Output projection: intermediate → hidden_dim
        self.output_proj = nn.Sequential(
            nn.Linear(intermediate_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
        )

        # Layer normalization for final output
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization (good for sin/cos)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform is better for periodic activations
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, peak_counts: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with periodic activations.

        Args:
            peak_counts: [batch_size, n_peaks] ATAC peak counts

        Returns:
            embeddings: [batch_size, hidden_dim]
        """
        # Input projection with base activation
        x = self.input_proj(peak_counts)

        # Periodic transformation layers
        for layer_dict in self.periodic_layers:
            # Store for residual
            residual = x

            # Pre-transformation
            x = layer_dict['pre_linear'](x)
            x = layer_dict['pre_norm'](x)

            # Apply periodic transformation (sin/cos)
            # This captures circular/cyclic patterns in cell cycle
            x_sin = torch.sin(x)
            x_cos = torch.cos(x)
            x_periodic = torch.cat([x_sin, x_cos], dim=-1)  # [batch, intermediate_dim * 2]

            # Post-transformation (reduce back to intermediate_dim)
            x = layer_dict['post_linear'](x_periodic)
            x = layer_dict['post_norm'](x)
            x = layer_dict['dropout'](x)

            # Residual connection
            if self.use_residual:
                x = x + residual

        # Output projection
        x = self.output_proj(x)
        x = self.layer_norm(x)

        return x


class PeakContrastive(nn.Module):
    """
    Contrastive Learning for PEAK embedding (SimCLR-style).

    Learn embeddings by contrasting augmented views of the same cell.
    Uses NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
    """

    def __init__(
        self,
        n_peaks: int,
        hidden_dim: int = 512,
        intermediate_dim: int = 2048,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = 'relu',
        temperature: float = 0.1,
        aug_dropout_prob: float = 0.2,
        projection_dim: int = 128
    ):
        super().__init__()
        self.n_peaks = n_peaks
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.temperature = temperature
        self.aug_dropout_prob = aug_dropout_prob
        self.projection_dim = projection_dim

        # Encoder: peaks → hidden_dim
        layers = []
        layers.append(nn.Linear(n_peaks, intermediate_dim))
        layers.append(get_activation(activation, dim=intermediate_dim))
        layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(intermediate_dim, intermediate_dim))
            layers.append(get_activation(activation, dim=intermediate_dim))
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(intermediate_dim, hidden_dim))
        self.encoder = nn.Sequential(*layers)

        # Projection head for contrastive learning (maps to lower-dim space for contrastive loss)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            get_activation(activation, dim=projection_dim),
            nn.Linear(projection_dim, projection_dim)
        )

    def augment_peaks(self, peak_counts: torch.Tensor) -> torch.Tensor:
        """
        Augment PEAK data by randomly dropping peaks.

        Args:
            peak_counts: [batch_size, n_peaks]

        Returns:
            augmented: [batch_size, n_peaks]
        """
        if self.training and self.aug_dropout_prob > 0:
            # Random dropout mask
            mask = torch.bernoulli(torch.full_like(peak_counts, 1 - self.aug_dropout_prob))
            return peak_counts * mask
        return peak_counts

    def forward(self, peak_counts: torch.Tensor, return_projection: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Args:
            peak_counts: [batch_size, n_peaks]
            return_projection: If True, return projection for contrastive loss

        Returns:
            embeddings: [batch_size, hidden_dim] if not return_projection
                       [batch_size, projection_dim] if return_projection
        """
        # Encode
        embeddings = self.encoder(peak_counts)  # [B, hidden_dim]

        if return_projection:
            # Project to contrastive space
            projections = self.projection_head(embeddings)  # [B, projection_dim]
            return projections
        else:
            return embeddings

    def compute_contrastive_loss(
        self,
        peak_counts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute NT-Xent contrastive loss.

        Args:
            peak_counts: [batch_size, n_peaks]

        Returns:
            loss: Scalar contrastive loss
            embeddings_1: [batch_size, hidden_dim] from view 1
            embeddings_2: [batch_size, hidden_dim] from view 2
        """
        batch_size = peak_counts.shape[0]

        # Create two augmented views
        view1 = self.augment_peaks(peak_counts)
        view2 = self.augment_peaks(peak_counts)

        # Get projections for contrastive loss
        z1 = self(view1, return_projection=True)  # [B, projection_dim]
        z2 = self(view2, return_projection=True)  # [B, projection_dim]

        # Normalize projections
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Concatenate to form [2B, projection_dim]
        z = torch.cat([z1, z2], dim=0)  # [2B, projection_dim]

        # Compute similarity matrix [2B, 2B]
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # [2B, 2B]

        # Create labels: for each sample i, positive is i+B (or i-B)
        labels = torch.arange(batch_size, device=peak_counts.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)  # [2B]

        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=peak_counts.device)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

        # NT-Xent loss (cross-entropy)
        loss = F.cross_entropy(sim_matrix, labels)

        # Also return embeddings (not projections) for downstream tasks
        with torch.no_grad():
            emb1 = self(view1, return_projection=False)
            emb2 = self(view2, return_projection=False)

        return loss, emb1, emb2

    def get_embeddings(
        self,
        peak_counts: np.ndarray,
        genes: Optional[np.ndarray] = None,
        batch_size: int = 128,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Batch inference for large datasets.

        Args:
            peak_counts: [n_cells, n_peaks] numpy array
            genes: Ignored (for compatibility)
            batch_size: Batch size for inference
            device: Device to use

        Returns:
            embeddings: [n_cells, hidden_dim] tensor
        """
        self.eval()
        self.to(device)

        n_cells = peak_counts.shape[0]
        embeddings = []

        with torch.no_grad():
            for i in range(0, n_cells, batch_size):
                batch = peak_counts[i:i+batch_size]
                batch_tensor = torch.tensor(batch, dtype=torch.float32, device=device)

                # Get embeddings (not projections)
                batch_emb = self(batch_tensor, return_projection=False)
                embeddings.append(batch_emb.cpu())

        embeddings = torch.cat(embeddings, dim=0)
        return embeddings


class PeakEmbedderWithLoRA(nn.Module):
    """
    Wrapper to add LoRA to a pretrained PeakEmbedder.

    Useful for fine-tuning on new datasets without full retraining.
    """

    def __init__(
        self,
        base_embedder: PeakEmbedder,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05
    ):
        """
        Add LoRA adapters to base embedder.

        Args:
            base_embedder: Pretrained PeakEmbedder
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha scaling
            lora_dropout: LoRA dropout
        """
        super().__init__()
        self.base_embedder = base_embedder

        # Freeze base embedder
        for param in self.base_embedder.parameters():
            param.requires_grad = False

        # Add LoRA to linear layers
        from LLM_TF.manual_analysis.manual_lora import inject_lora_to_scgpt

        target_modules = [
            "input_proj.0",
            "output_proj.0"
        ]

        self.base_embedder, trainable = inject_lora_to_scgpt(
            self.base_embedder,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=target_modules
        )

        print(f"  LoRA applied to PeakEmbedder: {trainable:,} trainable params")

    def forward(self, peak_counts: torch.Tensor) -> torch.Tensor:
        return self.base_embedder(peak_counts)

    def get_embeddings_batched(
        self,
        peak_counts: np.ndarray,
        batch_size: int = 32,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Generate embeddings for peak count matrix with batching.
        Delegates to base embedder's method.

        Args:
            peak_counts: [n_cells, n_peaks] numpy array
            batch_size: Batch size for inference
            device: Device to use

        Returns:
            embeddings: [n_cells, hidden_dim] tensor
        """
        self.eval()
        self.to(device)

        n_cells = peak_counts.shape[0]
        embeddings = []

        with torch.no_grad():
            for i in range(0, n_cells, batch_size):
                batch = peak_counts[i:i+batch_size]
                batch_tensor = torch.tensor(batch, dtype=torch.float32, device=device)

                batch_emb = self(batch_tensor)
                embeddings.append(batch_emb.cpu())

        embeddings = torch.cat(embeddings, dim=0)
        return embeddings


def create_peak_embedder(
    n_peaks: int,
    hidden_dim: int = 512,
    intermediate_dim: Optional[int] = None,
    num_layers: Optional[int] = None,
    dropout: Optional[float] = None,
    peak_arch_type: str = 'mlp',
    beta_vae: float = 1.0,
    noise_level: float = 0.1,
    temperature: float = 0.1,
    aug_dropout_prob: float = 0.2,
    projection_dim: int = 128,
    activation: str = 'gelu',
    use_batch_norm: bool = True,
    residual_type: str = 'add',
    gnn_hidden_dim: int = 256,
    num_gnn_layers: int = 3,
    num_conv_layers: int = 3,
    num_filters: int = 128,
    kernel_size: int = 5,
    use_lora: bool = False,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    device: str = 'cuda'
) -> Tuple[nn.Module, str]:
    """
    Factory function to create peak embedder with selectable architecture.

    Args:
        n_peaks: Number of ATAC peaks
        hidden_dim: Output embedding dimension
        intermediate_dim: Intermediate layer dimension (defaults to 2048)
        num_layers: Number of layers (defaults to 3)
        dropout: Dropout rate (defaults to 0.1)
        peak_arch_type: Architecture type
            Basic: 'mlp', 'hybrid'
            Autoencoder: 'vae', 'dae'
            Contrastive: 'contrastive'
            GNN: 'gnn' (GAT), 'gnn_gcn' (GCN)
            CNN: 'cnn', 'cnn_multiscale', 'cnn_dilated'
        beta_vae: Beta weight for VAE KL divergence (only for 'vae')
        noise_level: Noise std dev for DAE (only for 'dae')
        temperature: Contrastive temperature (only for 'contrastive')
        aug_dropout_prob: Augmentation dropout (only for 'contrastive')
        projection_dim: Projection dimension (only for 'contrastive')
        use_lora: Apply LoRA for fine-tuning
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        device: Device to use

    Returns:
        model: PeakEmbedder (or variant)
        device: Device string
    """
    # Set defaults
    if intermediate_dim is None:
        intermediate_dim = 2048
    if num_layers is None:
        num_layers = 3
    if dropout is None:
        dropout = 0.1

    print(f"\n{'='*60}")
    print(f"CREATING PEAK EMBEDDER")
    print(f"{'='*60}")
    print(f"  Architecture type: {peak_arch_type.upper()}")
    print(f"  Input peaks: {n_peaks:,}")
    print(f"  Output dimension: {hidden_dim}")
    print(f"  Intermediate dimension: {intermediate_dim}")
    print(f"  Number of layers: {num_layers}")
    print(f"  Dropout: {dropout}")
    if peak_arch_type == 'vae':
        print(f"  Beta-VAE weight: {beta_vae}")
    elif peak_arch_type == 'dae':
        print(f"  Noise level: {noise_level}")
    print(f"  LoRA: {'Enabled' if use_lora else 'Disabled'}")

    # Create embedder based on architecture type
    if peak_arch_type == 'vae':
        embedder = PeakVAE(
            n_peaks=n_peaks,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            beta_vae=beta_vae,
            use_batch_norm=use_batch_norm
        )
    elif peak_arch_type == 'dae':
        embedder = PeakDAE(
            n_peaks=n_peaks,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            noise_level=noise_level,
            use_batch_norm=use_batch_norm
        )
    elif peak_arch_type == 'hybrid':
        # Hybrid MLP with residual connections
        embedder = PeakHybridMLP(
            n_peaks=n_peaks,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            use_batch_norm=use_batch_norm,
            residual_type=residual_type
        )
    elif peak_arch_type == 'contrastive':
        # Contrastive learning (SimCLR-style)
        embedder = PeakContrastive(
            n_peaks=n_peaks,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            temperature=temperature,
            aug_dropout_prob=aug_dropout_prob,
            projection_dim=projection_dim
        )
    elif peak_arch_type == 'mlp':
        embedder = PeakEmbedder(
            n_peaks=n_peaks,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            use_batch_norm=use_batch_norm
        )
    elif peak_arch_type == 'periodic_mlp':
        # Periodic MLP with sin/cos activations for cell cycle periodicity
        embedder = PeakPeriodicMLP(
            n_peaks=n_peaks,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            use_batch_norm=use_batch_norm
        )
    elif peak_arch_type == 'gnn':
        from LLM_TF.embedders.peak_gnn_embedder import PeakGNNEmbedder
        embedder = PeakGNNEmbedder(
            n_peaks=n_peaks,
            hidden_dim=hidden_dim,
            node_dim=gnn_hidden_dim,
            num_gnn_layers=num_gnn_layers,
            gnn_type='gat',
            k_neighbors=10,
            dropout=dropout or 0.1,
            pooling='attention'
        )
    elif peak_arch_type == 'gnn_gcn':
        from LLM_TF.embedders.peak_gnn_embedder import PeakGNNEmbedder
        embedder = PeakGNNEmbedder(
            n_peaks=n_peaks,
            hidden_dim=hidden_dim,
            node_dim=gnn_hidden_dim,
            num_gnn_layers=num_gnn_layers,
            gnn_type='gcn',
            k_neighbors=10,
            dropout=dropout or 0.1,
            pooling='mean'
        )
    elif peak_arch_type == 'cnn':
        from LLM_TF.embedders.peak_cnn_embedder import PeakCNNEmbedder
        n_conv = num_conv_layers or 3
        embedder = PeakCNNEmbedder(
            n_peaks=n_peaks,
            hidden_dim=hidden_dim,
            conv_channels=[num_filters] * n_conv,
            kernel_sizes=[kernel_size] * n_conv,
            pool_sizes=[2] * n_conv,
            dropout=dropout or 0.1,
            use_residual=True,
            global_pool='max',
            activation=activation
        )
    elif peak_arch_type == 'cnn_multiscale':
        from LLM_TF.embedders.peak_cnn_embedder import PeakMultiScaleCNN
        embedder = PeakMultiScaleCNN(
            n_peaks=n_peaks,
            hidden_dim=hidden_dim,
            base_channels=num_filters,
            num_layers=num_conv_layers or 3,
            kernel_sizes=[3, 5, 7],
            dropout=dropout or 0.1,
            global_pool='both',
            activation=activation
        )
    elif peak_arch_type == 'cnn_dilated':
        from LLM_TF.embedders.peak_cnn_embedder import PeakDilatedCNN
        embedder = PeakDilatedCNN(
            n_peaks=n_peaks,
            hidden_dim=hidden_dim,
            channels=num_filters,
            num_layers=num_conv_layers or 4,
            kernel_size=kernel_size,
            dropout=dropout or 0.1,
            global_pool='max',
            activation=activation
        )
    else:
        raise ValueError(f"Unknown peak_arch_type: {peak_arch_type}. Currently supported: 'mlp', 'vae', 'dae', 'hybrid', 'contrastive', 'gnn', 'gnn_gcn', 'cnn', 'cnn_multiscale', 'cnn_dilated'")

    # Count parameters
    total_params = sum(p.numel() for p in embedder.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Apply LoRA if requested
    if use_lora:
        embedder = PeakEmbedderWithLoRA(
            base_embedder=embedder,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha
        )

    # Move to device
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    embedder.to(device_obj)

    print(f"  Device: {device_obj}")
    print(f"✓ Peak embedder ready!")
    print(f"{'='*60}\n")

    return embedder, str(device_obj)


def test_peak_embedder():
    """Test peak embedder with dummy data."""
    print("="*60)
    print("Testing Peak Embedder")
    print("="*60)

    # Dummy data
    n_cells = 100
    n_peaks = 50000  # Typical ATAC peak count
    peak_counts = np.random.rand(n_cells, n_peaks).astype(np.float32)

    print(f"\nTest data: {n_cells} cells × {n_peaks} peaks")

    # Create embedder
    embedder, device = create_peak_embedder(
        n_peaks=n_peaks,
        hidden_dim=512,
        use_lora=False,
        device='cuda'
    )

    # Generate embeddings
    embeddings = embedder.get_embeddings_batched(
        peak_counts,
        batch_size=32,
        device=device
    )

    print(f"\n✓ Embeddings generated: {embeddings.shape}")
    print(f"  Expected: [{n_cells}, 512]")
    print(f"  Mean: {embeddings.mean().item():.4f}")
    print(f"  Std: {embeddings.std().item():.4f}")

    if embeddings.shape == (n_cells, 512):
        print(f"\n🎉 Peak embedder working correctly!")
        return True
    else:
        print(f"\n❌ Shape mismatch!")
        return False


if __name__ == "__main__":
    test_peak_embedder()
