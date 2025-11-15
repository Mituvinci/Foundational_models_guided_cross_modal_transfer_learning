"""
Unified Embedder Interface for Multiple Foundation Models

Supports:
- scGPT (TDC, 10M cells) - for RNA/GeneActivity data
- Geneformer (10M/104M/316M variants) - for RNA/GeneActivity data
- PeakEmbedder - for raw ATAC peak counts (NO gene conversion needed!)

Usage:
    # For RNA/GeneActivity data
    embedder = UnifiedEmbedder(model_name='scgpt', use_lora=True)
    model, device = embedder.load_model()
    embeddings = embedder.get_embeddings(expression_matrix, gene_names)

    # For raw ATAC peak counts (hybrid mode)
    embedder = UnifiedEmbedder(model_name='peak', use_lora=True)
    model, device = embedder.load_model(n_peaks=50000)  # Specify peak count
    embeddings = embedder.get_embeddings(peak_counts, peak_names=None)
"""
import torch
import numpy as np
from typing import Optional


class UnifiedEmbedder:
    """
    Unified interface for loading and using different foundation models.
    """

    SUPPORTED_MODELS = {
        'scgpt': 'scGPT from TDC (10M cells, 51M params) - RNA/GeneActivity',
        'geneformer-10m': 'Geneformer V1 (10M cells, 10M params) - RNA/GeneActivity',
        'geneformer-104m': 'Geneformer V2 (104M params) - RNA/GeneActivity',
        'geneformer-104m-clcancer': 'Geneformer V2 (104M params, CL cancer fine-tuned) - RNA/GeneActivity',
        'geneformer-316m': 'Geneformer V2 (316M params) - RNA/GeneActivity',
        'uce-100m': 'UCE (Universal Cell Embeddings, 36M cells, 100M params) - RNA/GeneActivity',
        'teddy-70m': 'TEDDY-70M (70M params, 12 layers, 512-dim) - RNA/GeneActivity',
        'teddy-160m': 'TEDDY-160M (160M params, 12 layers, 768-dim) - RNA/GeneActivity',
        'teddy-400m': 'TEDDY-400M (400M params, 24 layers, 1024-dim) - RNA/GeneActivity',
        'cell-o1': 'Cell-o1 (7B+ params, Qwen2-based, 3584-dim) - RNA/ATAC cross-modal',
        'scfoundation': 'scFoundation (50M+ cells, encoder-only) - RNA/GeneActivity',
        'peak': 'PeakEmbedder (trainable) - Raw ATAC peak counts',
    }

    def __init__(
        self,
        model_name: str = 'scgpt',
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        device: str = 'cuda'
    ):
        """
        Initialize unified embedder.

        Args:
            model_name: Model to use ('scgpt', 'geneformer-10m', etc.)
            use_lora: Apply LoRA for parameter-efficient fine-tuning
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha
            device: Device to use
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model_name}\n"
                f"Supported: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.model_name = model_name
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.device = device

        self.model = None
        self.tokenizer = None
        self.loader = None

    def load_model(self, n_peaks: Optional[int] = None, hidden_dim: Optional[int] = None,
                   intermediate_dim: Optional[int] = None, num_layers: Optional[int] = None,
                   dropout: Optional[float] = None, peak_arch_type: str = 'mlp',
                   beta_vae: float = 1.0, noise_level: float = 0.1,
                   temperature: float = 0.1, aug_dropout_prob: float = 0.2, projection_dim: int = 128,
                   activation: str = 'gelu', use_batch_norm: bool = True, residual_type: str = 'add',
                   gnn_hidden_dim: int = 256, num_gnn_layers: int = 3, num_conv_layers: int = 3, num_filters: int = 128, kernel_size: int = 5):
        """
        Load the specified foundation model.

        Args:
            n_peaks: Number of ATAC peaks (required if model_name='peak')
            hidden_dim: Output embedding dimension (for peak embedder, should match source model)
            intermediate_dim: Intermediate layer dimension (for peak embedder, from Optuna)
            num_layers: Number of layers (for peak embedder, from Optuna)
            dropout: Dropout rate (for peak embedder, from Optuna)
            peak_arch_type: Peak architecture type - 'mlp', 'vae', 'dae', 'contrastive', 'hybrid'
            beta_vae: Beta weight for VAE KL divergence (only for 'vae')
            noise_level: Noise std dev for DAE (only for 'dae')

        Returns:
            model: Loaded model with optional LoRA
            device: torch device
        """
        print(f"\n{'='*60}")
        print(f"LOADING FOUNDATION MODEL: {self.model_name}")
        print(f"{'='*60}")
        print(f"  Description: {self.SUPPORTED_MODELS[self.model_name]}")
        print(f"  LoRA: {'Enabled' if self.use_lora else 'Disabled'}")
        if self.use_lora:
            print(f"  LoRA rank: {self.lora_rank}")
            print(f"  LoRA alpha: {self.lora_alpha}")

        if self.model_name == 'scgpt':
            return self._load_scgpt()
        elif self.model_name.startswith('geneformer'):
            return self._load_geneformer()
        elif self.model_name == 'uce-100m':
            return self._load_uce()
        elif self.model_name.startswith('teddy'):
            return self._load_teddy()
        elif self.model_name == 'cell-o1':
            return self._load_cello1()
        elif self.model_name == 'scfoundation':
            return self._load_scfoundation()
        elif self.model_name == 'peak':
            if n_peaks is None:
                raise ValueError("n_peaks required for peak embedder. Call load_model(n_peaks=...)")
            return self._load_peak_embedder(n_peaks, hidden_dim, intermediate_dim, num_layers, dropout,
                                           peak_arch_type, beta_vae, noise_level,
                                           temperature, aug_dropout_prob, projection_dim,
                                           activation, use_batch_norm, residual_type,
                                           gnn_hidden_dim, num_gnn_layers, num_conv_layers, num_filters, kernel_size)
        else:
            raise NotImplementedError(f"Loader for {self.model_name} not implemented")

    def _load_scgpt(self):
        """Load scGPT model."""
        from LLM_TF.embedders.embedder import load_scgpt_with_lora

        if self.use_lora:
            model, tokenizer, device = load_scgpt_with_lora(
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=0.05
            )
        else:
            from LLM_TF.embedders.embedder import load_scgpt_backbone
            model, tokenizer, device = load_scgpt_backbone()

        self.model = model
        self.tokenizer = tokenizer
        self.loader = 'scgpt'

        return model, device

    def _load_geneformer(self):
        """Load Geneformer model."""
        from LLM_TF.loaders.geneformer_loader import GeneformerLoader

        # Map model name to local path
        models_root = "/users/ha00014/Halimas_projects/foundations_models/Geneformer"
        model_paths = {
            'geneformer-10m': f"{models_root}/Geneformer-V1-10M",
            'geneformer-104m': f"{models_root}/Geneformer-V2-104M",
            'geneformer-104m-clcancer': f"{models_root}/Geneformer-V2-104M_CLcancer",
            'geneformer-316m': f"{models_root}/Geneformer-V2-316M",
        }

        loader = GeneformerLoader(device=self.device)
        model, tokenizer = loader.load_pretrained(
            model_name=model_paths[self.model_name],
            use_lora=self.use_lora,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha
        )

        self.model = model
        self.tokenizer = tokenizer
        self.loader = loader

        return model, torch.device(self.device)

    def _load_uce(self):
        """Load UCE (Universal Cell Embeddings) model."""
        from LLM_TF.loaders.uce_loader import UCELoader

        model_path = "/users/ha00014/Halimas_projects/foundations_models/minwoosun_uce-100m"

        loader = UCELoader(device=self.device)
        model, tokenizer = loader.load_pretrained(
            model_path=model_path,
            use_lora=self.use_lora,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha
        )

        self.model = model
        self.tokenizer = None  # UCE doesn't use tokenizer
        self.loader = loader

        return model, torch.device(self.device)

    def _load_peak_embedder(self, n_peaks: int, hidden_dim: Optional[int] = None,
                            intermediate_dim: Optional[int] = None, num_layers: Optional[int] = None,
                            dropout: Optional[float] = None, peak_arch_type: str = 'mlp',
                            beta_vae: float = 1.0, noise_level: float = 0.1,
                            temperature: float = 0.1, aug_dropout_prob: float = 0.2, projection_dim: int = 128,
                            activation: str = 'gelu', use_batch_norm: bool = True, residual_type: str = 'add',
                            gnn_hidden_dim: int = 256, num_gnn_layers: int = 3, num_conv_layers: int = 3, num_filters: int = 128, kernel_size: int = 5):
        """Load peak embedder for raw ATAC counts with Optuna-selected architecture."""
        from LLM_TF.embedders.peak_embedder import create_peak_embedder

        # Use provided hidden_dim, or default to 512 (scGPT default)
        if hidden_dim is None:
            hidden_dim = 512
            print(f"  Warning: hidden_dim not provided, defaulting to 512")
        else:
            print(f"  Using hidden_dim={hidden_dim} to match source model")

        model, device = create_peak_embedder(
            n_peaks=n_peaks,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            dropout=dropout,
            peak_arch_type=peak_arch_type,
            beta_vae=beta_vae,
            noise_level=noise_level,
            temperature=temperature,
            aug_dropout_prob=aug_dropout_prob,
            projection_dim=projection_dim,
            activation=activation,
            use_batch_norm=use_batch_norm,
            residual_type=residual_type,
            gnn_hidden_dim=gnn_hidden_dim,
            num_gnn_layers=num_gnn_layers,
            num_conv_layers=num_conv_layers,
            num_filters=num_filters,
            kernel_size=kernel_size,
            use_lora=self.use_lora,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            device=self.device
        )

        self.model = model
        self.tokenizer = None  # Peak embedder doesn't use tokenizer
        self.loader = 'peak'

        return model, torch.device(device)

    def _load_teddy(self):
        """Load TEDDY model (70M, 160M, or 400M variant)."""
        from LLM_TF.loaders.teddy_loader import TEDDYLoader

        model_paths = {
            'teddy-70m': "/users/ha00014/Halimas_projects/foundations_models/TEDDY/teddy/models/teddy_g/70M",
            'teddy-160m': "/users/ha00014/Halimas_projects/foundations_models/TEDDY/teddy/models/teddy_g/160M",
            'teddy-400m': "/users/ha00014/Halimas_projects/foundations_models/TEDDY/teddy/models/teddy_g/400M",
        }

        loader = TEDDYLoader(device=self.device)
        model, tokenizer = loader.load_pretrained(
            model_path=model_paths[self.model_name],
            use_lora=self.use_lora,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha
        )

        self.model = model
        self.tokenizer = tokenizer
        self.loader = loader
        self.embedding_dim = 512 if '70m' in self.model_name else (768 if '160m' in self.model_name else 1024)

        return model, torch.device(self.device)

    def _load_cello1(self):
        """Load Cell-o1 model."""
        from LLM_TF.loaders.cello1_loader import CellO1Loader

        model_path = "/users/ha00014/Halimas_projects/foundations_models/Cell-o1"

        loader = CellO1Loader(device=self.device)
        model, tokenizer = loader.load_pretrained(
            model_path=model_path,
            use_lora=self.use_lora,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha
        )

        self.model = model
        self.tokenizer = tokenizer
        self.loader = loader
        self.embedding_dim = 3584

        return model, torch.device(self.device)

    def _load_scfoundation(self):
        """Load scFoundation model."""
        from LLM_TF.loaders.scfoundation_loader import scFoundationLoader

        model_path = "/users/ha00014/Halimas_projects/foundations_models/genbio_scFoundation"

        loader = scFoundationLoader(device=self.device)
        model, _ = loader.load_pretrained(
            model_path=model_path,
            use_lora=self.use_lora,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha
        )

        self.model = model
        self.tokenizer = None
        self.loader = loader
        self.embedding_dim = model.hidden_dim if hasattr(model, 'hidden_dim') else 768

        return model, torch.device(self.device)

    def get_embeddings(
        self,
        expression_matrix: np.ndarray,
        gene_names: Optional[np.ndarray] = None,
        batch_size: int = 32,
        use_global_ranking: bool = False,
        normalize: bool = False
    ) -> torch.Tensor:
        """
        Get embeddings from expression/peak matrix.

        Args:
            expression_matrix: [n_cells, n_features]
                - For RNA/GeneActivity: [n_cells, n_genes]
                - For peaks: [n_cells, n_peaks]
            gene_names: Gene/peak names (optional for peak embedder)
            batch_size: Batch size
            use_global_ranking: (Geneformer only) Use global gene ranking for consistent vocabulary
            normalize: (Geneformer only) Apply log1p normalization

        Returns:
            embeddings: [n_cells, hidden_dim]
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        if self.model_name == 'scgpt':
            return self._get_scgpt_embeddings(expression_matrix, gene_names, batch_size)
        elif self.model_name.startswith('geneformer'):
            return self._get_geneformer_embeddings(
                expression_matrix, gene_names, batch_size,
                use_global_ranking=use_global_ranking,
                normalize=normalize
            )
        elif self.model_name == 'uce-100m':
            return self._get_uce_embeddings(expression_matrix, gene_names, batch_size)
        elif self.model_name.startswith('teddy'):
            return self._get_teddy_embeddings(expression_matrix, gene_names, batch_size)
        elif self.model_name == 'cell-o1':
            return self._get_cello1_embeddings(expression_matrix, gene_names, batch_size)
        elif self.model_name == 'scfoundation':
            return self._get_scfoundation_embeddings(expression_matrix, gene_names, batch_size)
        elif self.model_name == 'peak':
            return self._get_peak_embeddings(expression_matrix, batch_size)
        else:
            raise NotImplementedError()

    def _get_scgpt_embeddings(self, expression_matrix, gene_names, batch_size):
        """Get scGPT embeddings."""
        from LLM_TF.embedders.embedder import tokenize_matrix, embed_tokens_then_norm

        tokens = tokenize_matrix(expression_matrix, gene_names, self.tokenizer)
        device = next(self.model.parameters()).device
        embeddings = embed_tokens_then_norm(self.model, tokens, device, batch_size=batch_size)

        return embeddings

    def _get_geneformer_embeddings(
        self,
        expression_matrix,
        gene_names,
        batch_size,
        use_global_ranking=False,
        normalize=False
    ):
        """Get Geneformer embeddings."""
        embeddings = self.loader.get_embeddings(
            expression_matrix, gene_names,
            batch_size=batch_size,
            top_k=2048,
            use_global_ranking=use_global_ranking,
            normalize=normalize
        )
        return embeddings

    def _get_uce_embeddings(self, expression_matrix, gene_names, batch_size):
        """Get UCE embeddings."""
        embeddings = self.loader.get_embeddings(
            expression_matrix, gene_names,
            batch_size=batch_size
        )
        return embeddings

    def _get_peak_embeddings(self, peak_counts, batch_size):
        """Get peak embeddings from raw ATAC counts."""
        embeddings = self.model.get_embeddings_batched(
            peak_counts,
            batch_size=batch_size,
            device=self.device
        )
        return embeddings

    def _get_teddy_embeddings(self, expression_matrix, gene_names, batch_size):
        """Get TEDDY embeddings."""
        return self.loader.get_embeddings(
            expression_matrix,
            gene_names=gene_names,
            batch_size=batch_size
        )

    def _get_cello1_embeddings(self, expression_matrix, gene_names, batch_size):
        """Get Cell-o1 embeddings."""
        return self.loader.get_embeddings(
            expression_matrix,
            gene_names=gene_names,
            batch_size=batch_size
        )

    def _get_scfoundation_embeddings(self, expression_matrix, gene_names, batch_size):
        """Get scFoundation embeddings."""
        return self.loader.get_embeddings(
            expression_matrix,
            gene_names=gene_names,
            batch_size=batch_size
        )

    @classmethod
    def list_models(cls):
        """List all supported models."""
        print("\nSupported Foundation Models:")
        print("=" * 60)
        for name, desc in cls.SUPPORTED_MODELS.items():
            print(f"  {name:20s} - {desc}")
        print()


def test_unified_embedder():
    """Test unified embedder with both scGPT and Geneformer."""
    print("=" * 60)
    print("Testing Unified Embedder")
    print("=" * 60)

    # List available models
    UnifiedEmbedder.list_models()

    # Create dummy data
    n_cells = 10
    n_genes = 100
    expression = np.random.rand(n_cells, n_genes).astype(np.float32)
    gene_names = np.array([f"GENE{i}" for i in range(n_genes)])

    # Test scGPT
    print("\n" + "=" * 60)
    print("Testing scGPT")
    print("=" * 60)
    embedder_scgpt = UnifiedEmbedder(model_name='scgpt', use_lora=True, lora_rank=8)
    model_scgpt, device = embedder_scgpt.load_model()
    emb_scgpt = embedder_scgpt.get_embeddings(expression, gene_names, batch_size=5)
    print(f"✓ scGPT embeddings: {emb_scgpt.shape}")

    # Test Geneformer
    print("\n" + "=" * 60)
    print("Testing Geneformer")
    print("=" * 60)
    embedder_gf = UnifiedEmbedder(model_name='geneformer-10m', use_lora=True, lora_rank=8)
    model_gf, device = embedder_gf.load_model()
    emb_gf = embedder_gf.get_embeddings(expression, gene_names, batch_size=5)
    print(f"✓ Geneformer embeddings: {emb_gf.shape}")

    print("\n" + "=" * 60)
    print("✅ UNIFIED EMBEDDER WORKING!")
    print("=" * 60)
    print(f"\nYou can now train with:")
    print(f"  --model_name scgpt")
    print(f"  --model_name geneformer-10m")
    print(f"  --model_name geneformer-104m")
    print(f"  --model_name geneformer-316m")


if __name__ == "__main__":
    test_unified_embedder()
