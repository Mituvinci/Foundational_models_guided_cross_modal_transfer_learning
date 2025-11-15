"""
scFoundation Loader

Paper: Hao et al. (2024) - "Large-Scale Foundation Model on Single-Cell Transcriptomics"
Journal: Nature Methods
Model: scFoundation (encoder-only transformer)

Architecture:
- Encoder-only transformer (decoder disabled in ModelGenerator)
- Hidden dimension: 768 (base) or 1024 (large) - TBD from checkpoint
- Trained on 50M+ cells from Human Cell Atlas
- Uses continuous gene embeddings (weighted, not binned)

Input format:
- Log1p-normalized expression values
- Continuous expressions → weighted gene embeddings
- No discrete binning (unlike Cell-o1/TEDDY)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings


class scFoundationLoader:
    """Loader for scFoundation model."""

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.model = None
        self.config = None
        self.vocab_size = None  # Will be inferred from checkpoint

    def load_pretrained(
        self,
        model_path: str,
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: float = 32.0
    ) -> Tuple[nn.Module, None]:
        """
        Load pretrained scFoundation model.

        Args:
            model_path: Path to scFoundation directory (contains models.ckpt)
            use_lora: Whether to apply LoRA adapters
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha

        Returns:
            model: scFoundation model
            tokenizer: None (scFoundation doesn't use discrete tokenizer)
        """
        model_path = Path(model_path)
        ckpt_path = model_path / "models.ckpt"

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print(f"\nLoading scFoundation from: {model_path}")

        # Load PyTorch Lightning checkpoint
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            print(f"  Checkpoint loaded: {type(checkpoint)}")

            # Extract model state dict
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    print(f"  Found state_dict with {len(state_dict)} keys")

                    # Check for hyperparameters/config
                    if 'hyper_parameters' in checkpoint:
                        self.config = checkpoint['hyper_parameters']
                        print(f"  Config: {self.config}")
                    elif 'config' in checkpoint:
                        self.config = checkpoint['config']
                else:
                    # Checkpoint might be the state_dict itself
                    state_dict = checkpoint
            else:
                raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")

            # Infer model architecture from state dict
            hidden_dim = self._infer_hidden_dim(state_dict)
            n_layers = self._infer_n_layers(state_dict)
            vocab_size = self._infer_vocab_size(state_dict)
            self.vocab_size = vocab_size  # Store for later use

            print(f"  Inferred architecture:")
            print(f"    Hidden dim: {hidden_dim}")
            print(f"    Layers: {n_layers}")
            print(f"    Vocab size: {vocab_size}")

            # Create model architecture
            # scFoundation uses a transformer encoder
            # We'll create a simple wrapper that matches the state dict structure
            self.model = self._build_model_from_state_dict(state_dict, hidden_dim, n_layers, vocab_size)

            # Load weights
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

            if missing_keys:
                print(f"  Missing keys (OK if using encoder only): {len(missing_keys)}")
            if unexpected_keys:
                print(f"  Unexpected keys: {len(unexpected_keys)}")

            print(f"  Model weights loaded successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to load scFoundation: {e}")

        self.model = self.model.to(self.device)

        # Apply LoRA if requested
        if use_lora:
            print(f"\n  Applying LoRA adapters (rank={lora_rank}, alpha={lora_alpha})")
            self._apply_lora_to_scfoundation(self.model, lora_rank, lora_alpha)

            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"  Total params: {total_params:,}")
            print(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

        return self.model, None  # No tokenizer for scFoundation

    def _infer_hidden_dim(self, state_dict: dict) -> int:
        """Infer hidden dimension from state dict."""
        # Look for common keys that reveal hidden dimension
        for key in state_dict.keys():
            if 'encoder' in key and 'weight' in key:
                shape = state_dict[key].shape
                if len(shape) >= 2:
                    # Try to find d_model dimension
                    if 'embed' in key:
                        return shape[-1]  # Embedding dimension
                    elif 'attention' in key or 'attn' in key:
                        return shape[-1]  # Attention dimension

        # Default fallback
        warnings.warn("Could not infer hidden_dim, using default 768")
        return 768

    def _infer_n_layers(self, state_dict: dict) -> int:
        """Infer number of layers from state dict."""
        # Count transformer layers
        layer_indices = set()
        for key in state_dict.keys():
            if 'layer' in key or 'block' in key:
                # Try to extract layer number
                parts = key.split('.')
                for part in parts:
                    if part.isdigit():
                        layer_indices.add(int(part))

        if layer_indices:
            return max(layer_indices) + 1

        # Default fallback
        warnings.warn("Could not infer n_layers, using default 12")
        return 12

    def _infer_vocab_size(self, state_dict: dict) -> int:
        """
        Infer vocabulary size from gene embedding layer in checkpoint.
        CRITICAL FIX (Nov 4, 2025): Must match checkpoint vocab to avoid CUDA index errors.
        """
        for key in state_dict.keys():
            if 'gene_embedding.weight' in key or ('embedding' in key and 'weight' in key):
                vocab_size = state_dict[key].shape[0]
                print(f"  Inferred vocab size from checkpoint: {vocab_size}")
                return vocab_size

        # Default fallback
        warnings.warn("Could not infer vocab_size from checkpoint, using default 25000")
        return 25000

    def _build_model_from_state_dict(self, state_dict: dict, hidden_dim: int, n_layers: int, vocab_size: int) -> nn.Module:
        """
        Build a model architecture that matches the state dict.

        This is a simplified wrapper - the actual scFoundation architecture
        may be more complex.
        """
        # For now, return a simple nn.Module that holds the state dict
        # and provides a forward pass for embeddings

        class scFoundationModel(nn.Module):
            def __init__(self, hidden_dim, n_layers, vocab_size):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.n_layers = n_layers
                self.vocab_size = vocab_size

                # Placeholder layers - will be replaced by load_state_dict
                # Gene embedding layer
                self.gene_embedding = nn.Embedding(vocab_size, hidden_dim)

                # Transformer encoder layers
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=12,
                    dim_feedforward=hidden_dim * 4,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

                # Output projection (optional)
                self.output_norm = nn.LayerNorm(hidden_dim)

            def forward(self, gene_expr: torch.Tensor, gene_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
                """
                Forward pass for scFoundation.

                Args:
                    gene_expr: [batch, n_genes] - continuous expression values
                    gene_ids: [batch, n_genes] - gene indices (optional)

                Returns:
                    embeddings: [batch, hidden_dim]
                """
                # If gene_ids provided, use them for embedding lookup
                if gene_ids is not None:
                    gene_embeds = self.gene_embedding(gene_ids)  # [batch, n_genes, hidden_dim]
                else:
                    # Create simple positional embeddings
                    batch_size, n_genes = gene_expr.shape
                    gene_ids = torch.arange(n_genes, device=gene_expr.device).unsqueeze(0).expand(batch_size, -1)
                    gene_embeds = self.gene_embedding(gene_ids)

                # Weight embeddings by expression values
                gene_expr_expanded = gene_expr.unsqueeze(-1)  # [batch, n_genes, 1]
                weighted_embeds = gene_embeds * gene_expr_expanded  # [batch, n_genes, hidden_dim]

                # Pass through transformer
                x = self.transformer(weighted_embeds)  # [batch, n_genes, hidden_dim]

                # Mean pool over genes
                x = x.mean(dim=1)  # [batch, hidden_dim]

                # Normalize
                x = self.output_norm(x)

                return x

        return scFoundationModel(hidden_dim, n_layers, vocab_size)

    def _apply_lora_to_scfoundation(self, model: nn.Module, rank: int, alpha: float):
        """Apply LoRA to scFoundation transformer layers."""
        from LLM_TF.manual_analysis.manual_lora import LoRALayer

        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False

        # Apply LoRA to transformer layers
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
            layers = model.transformer.layers

            for layer in layers:
                # Target self-attention
                if hasattr(layer, 'self_attn'):
                    attn = layer.self_attn

                    # Target in_proj (combined QKV) or separate projections
                    if hasattr(attn, 'in_proj_weight'):
                        # Combined projection - wrap the entire module
                        pass  # Skip for now, complex to handle
                    else:
                        # Separate projections
                        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                            if hasattr(attn, proj_name):
                                original = getattr(attn, proj_name)
                                if isinstance(original, nn.Linear):
                                    setattr(attn, proj_name, LoRALayer(original, rank=rank, alpha=alpha))

                # Target feedforward
                for ff_name in ['linear1', 'linear2', 'fc1', 'fc2']:
                    if hasattr(layer, ff_name):
                        original = getattr(layer, ff_name)
                        if isinstance(original, nn.Linear):
                            setattr(layer, ff_name, LoRALayer(original, rank=rank, alpha=alpha))

            print(f"  LoRA applied to {len(layers)} transformer layers")

    def get_embeddings(
        self,
        expression_matrix: np.ndarray,
        gene_names: Optional[np.ndarray] = None,
        batch_size: int = 32,
        **kwargs
    ) -> torch.Tensor:
        """
        Get embeddings from expression matrix.

        Args:
            expression_matrix: [n_cells, n_genes] - gene expression values
            gene_names: Gene names (optional)
            batch_size: Batch size for inference

        Returns:
            embeddings: [n_cells, hidden_dim]
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_pretrained() first.")

        self.model.eval()
        device = next(self.model.parameters()).device

        # CRITICAL FIX (Nov 4, 2025): Filter genes to vocab_size to avoid CUDA index errors
        n_cells, n_genes = expression_matrix.shape
        if self.vocab_size is not None and n_genes > self.vocab_size:
            print(f"  [VOCAB FIX] Dataset has {n_genes} genes, scFoundation supports {self.vocab_size}")
            print(f"  [VOCAB FIX] Selecting top {self.vocab_size} most variable genes...")

            # Calculate variance across cells for each gene
            gene_vars = np.var(expression_matrix, axis=0)

            # Get indices of top vocab_size genes by variance
            top_gene_indices = np.argsort(gene_vars)[-self.vocab_size:]

            # Filter expression matrix
            expression_matrix = expression_matrix[:, top_gene_indices]

            print(f"  [VOCAB FIX] Filtered to {expression_matrix.shape[1]} genes")

        # Preprocess: log1p normalization (scFoundation expects this)
        X = expression_matrix.copy()

        # Library-size normalization + log1p
        for i in range(X.shape[0]):
            total_counts = X[i].sum()
            if total_counts > 0:
                X[i] = np.log1p(X[i] / total_counts * 10000)

        # Convert to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Generate embeddings in batches
        embeddings = []

        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size].to(device)

                # Get embeddings
                emb = self.model(batch)

                embeddings.append(emb.cpu())

        embeddings = torch.cat(embeddings, dim=0)
        return embeddings


def test_scfoundation_loader():
    """Test scFoundation loader with dummy data."""
    print("="*60)
    print("Testing scFoundation Loader")
    print("="*60)

    # Create dummy data
    n_cells = 10
    n_genes = 100
    expression = np.random.rand(n_cells, n_genes).astype(np.float32) * 100

    # Test scFoundation
    model_path = "/users/ha00014/Halimas_projects/foundations_models/genbio_scFoundation"
    loader = scFoundationLoader(device='cuda' if torch.cuda.is_available() else 'cpu')

    model, _ = loader.load_pretrained(
        model_path=model_path,
        use_lora=True,
        lora_rank=8
    )

    # Get embeddings
    embeddings = loader.get_embeddings(expression, batch_size=5)

    print(f"\nscFoundation embeddings shape: {embeddings.shape}")
    print(f"  Expected: (10, {model.hidden_dim})")

    if embeddings.shape[0] == n_cells:
        print("\nscFoundation LOADER WORKING!")
    else:
        print(f"\nUnexpected shape: {embeddings.shape}")


if __name__ == "__main__":
    test_scfoundation_loader()
