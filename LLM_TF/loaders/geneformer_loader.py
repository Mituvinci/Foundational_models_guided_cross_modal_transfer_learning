"""
Geneformer Model Loader

Geneformer: Foundation model pretrained on 30M single cells
Paper: "Transfer learning enables predictions in network biology"
HuggingFace: https://huggingface.co/ctheodoris/Geneformer
"""
import torch
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict


class GeneformerLoader:
    """
    Loader for Geneformer pretrained model from HuggingFace.

    Geneformer uses rank-based gene expression encoding:
    - Genes are ranked by expression level in each cell
    - Top N expressed genes are tokenized
    - BERT-style transformer processes ranked gene tokens
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize Geneformer loader.

        Args:
            device: Device to load model on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.config = None
        self.global_gene_ranking = None  # For use_global_ranking mode

    def load_pretrained(
        self,
        model_name: str = "/users/ha00014/Halimas_projects/foundations_models/Geneformer/Geneformer-V1-10M",  # Local path or HuggingFace ID
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0
    ):
        """
        Load pretrained Geneformer from HuggingFace.

        Args:
            model_name: HuggingFace model ID
            use_lora: Apply LoRA for parameter-efficient fine-tuning
            lora_rank: LoRA rank
            lora_alpha: LoRA scaling factor

        Returns:
            model: Geneformer model
            tokenizer: Gene tokenizer
        """
        try:
            from transformers import BertForMaskedLM, BertTokenizer
        except ImportError:
            raise ImportError("transformers not installed. Run: pip install transformers")

        print(f"Loading Geneformer from HuggingFace: {model_name}")


        # Load model (Geneformer uses custom tokenization, not BERT tokenizer)
        self.model = BertForMaskedLM.from_pretrained(model_name)

        # Geneformer doesn't have a traditional tokenizer
        # It uses gene rank-based encoding (see their token_dictionary.pkl)
        # For now, we'll use a simple gene-to-index mapping
        self.tokenizer = None  # Will create custom gene vocabulary
        print(f"  Note: Geneformer uses custom gene vocabulary, not BERT tokenizer")

        # Move to device
        self.model.to(self.device)
        self.model.eval()

        # Freeze base model
        for param in self.model.parameters():
            param.requires_grad = False

        # Apply LoRA if requested
        if use_lora:
            print(f"\nApplying LoRA (rank={lora_rank}, alpha={lora_alpha})...")
            self._apply_lora(lora_rank, lora_alpha)

        self.config = self.model.config

        print(f"[OK] Geneformer loaded successfully")
        print(f"  Hidden size: {self.config.hidden_size}")
        print(f"  Layers: {self.config.num_hidden_layers}")
        print(f"  Vocab size: {self.config.vocab_size}")
        print(f"  Device: {self.device}")

        return self.model, self.tokenizer

    def _apply_lora(self, rank: int, alpha: float):
        """
        Apply LoRA adapters to Geneformer transformer.

        Uses our manual LoRA implementation for consistency.
        """
        try:
            from LLM_TF.manual_analysis.manual_lora import inject_lora_to_scgpt
        except ImportError:
            # Fallback if running as script
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from LLM_TF.manual_analysis.manual_lora import inject_lora_to_scgpt

        # Target attention and FFN layers
        target_modules = [
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "attention.output.dense",
            "intermediate.dense",
            "output.dense"
        ]

        self.model, trainable = inject_lora_to_scgpt(
            self.model,
            rank=rank,
            alpha=alpha,
            dropout=0.05,
            target_modules=target_modules
        )

    def build_gene_vocab(self, gene_names: np.ndarray) -> Dict[str, int]:
        """
        Build gene vocabulary mapping gene SYMBOLS to token IDs.

        IMPORTANT FIX (Nov 4, 2025): Load Geneformer's PRETRAINED vocabulary.
        - Geneformer uses Ensembl IDs internally (ENSG00000...)
        - Dataset uses gene symbols (TP53, GAPDH, etc.)
        - Need to map: gene_symbol -> Ensembl_ID -> token_ID

        This prevents index out of bounds errors when genes exceed vocab_size.
        """
        base_dir = Path("/users/ha00014/Halimas_projects/foundations_models/Geneformer/geneformer")

        # Paths to pretrained dictionaries
        token_dict_paths = [
            base_dir / "token_dictionary_gc104M.pkl",  # Ensembl_ID -> token_ID
            base_dir / "gene_dictionaries_30m" / "token_dictionary_gc30M.pkl",
        ]
        gene_name_dict_paths = [
            base_dir / "gene_name_id_dict_gc104M.pkl",  # gene_symbol -> Ensembl_ID
            base_dir / "gene_dictionaries_30m" / "gene_name_id_dict_gc30M.pkl",
        ]

        # Load token dictionary (Ensembl -> token_ID)
        token_dict = None
        for path in token_dict_paths:
            if path.exists():
                try:
                    with open(path, 'rb') as f:
                        token_dict = pickle.load(f)
                    print(f"  [VOCAB FIX] Loaded token dict: {path.name} ({len(token_dict)} tokens)")
                    break
                except Exception as e:
                    continue

        # Load gene name mapping (symbol -> Ensembl)
        gene_name_dict = None
        for path in gene_name_dict_paths:
            if path.exists():
                try:
                    with open(path, 'rb') as f:
                        gene_name_dict = pickle.load(f)
                    print(f"  [VOCAB FIX] Loaded gene name dict: {path.name} ({len(gene_name_dict)} symbols)")
                    break
                except Exception as e:
                    continue

        # Build vocab: gene_symbol -> token_ID
        if token_dict is not None and gene_name_dict is not None:
            vocab = {}

            # Add special tokens (Geneformer uses lowercase + angle brackets)
            vocab["[PAD]"] = token_dict.get("<pad>", 0)
            vocab["[CLS]"] = token_dict.get("<cls>", 1)
            vocab["[SEP]"] = token_dict.get("<eos>", 2)
            vocab["[MASK]"] = token_dict.get("<mask>", 3)
            vocab["[UNK]"] = token_dict.get("<pad>", 0)  # Use <pad> as unknown

            # Map gene symbols to token IDs
            genes_found = 0
            genes_missing = 0

            for gene_symbol in gene_names:
                # Step 1: gene_symbol -> Ensembl_ID
                ensembl_id = gene_name_dict.get(gene_symbol)

                if ensembl_id is not None:
                    # Step 2: Ensembl_ID -> token_ID
                    token_id = token_dict.get(ensembl_id)

                    if token_id is not None:
                        vocab[gene_symbol] = token_id
                        genes_found += 1
                    else:
                        genes_missing += 1
                else:
                    genes_missing += 1

            print(f"  [VOCAB FIX] Dataset genes: {len(gene_names)}")
            print(f"  [VOCAB FIX] Genes mapped: {genes_found} ({100*genes_found/len(gene_names):.1f}%)")
            print(f"  [VOCAB FIX] Genes missing (will use [UNK]): {genes_missing} ({100*genes_missing/len(gene_names):.1f}%)")

            return vocab

        # Fallback (should not reach here)
        print(f"  [VOCAB ERROR] Could not load pretrained vocabularies!")
        print(f"  [VOCAB ERROR] Creating limited vocab to prevent crashes...")

        vocab = {
            "[PAD]": 0,
            "[CLS]": 1,
            "[SEP]": 2,
            "[MASK]": 3,
            "[UNK]": 4
        }

        # Add genes (LIMITED to prevent index errors)
        max_genes = min(len(gene_names), 20000)
        for i, gene in enumerate(gene_names[:max_genes]):
            vocab[gene] = i + 5

        return vocab

    def tokenize_expression(
        self,
        expression_matrix: np.ndarray,
        gene_names: np.ndarray,
        top_k: int = 2048,
        use_global_ranking: bool = False,
        normalize: bool = False
    ) -> List[List[int]]:
        """
        Tokenize gene expression matrix using Geneformer's rank-based encoding.

        Args:
            expression_matrix: Gene expression [n_cells, n_genes]
            gene_names: Gene names [n_genes]
            top_k: Number of top expressed genes to keep per cell (2048 is Geneformer default)
            use_global_ranking: If True, use global gene ranking from training set (recommended for fine-tuning).
                               If False, rank genes per-cell independently (original Geneformer behavior, used by DANN).
            normalize: If True, apply log1p normalization before ranking

        Returns:
            tokenized_cells: List of token ID lists
        """
        # Build gene vocabulary if not exists
        if not hasattr(self, 'gene_vocab'):
            self.gene_vocab = self.build_gene_vocab(gene_names)

        # Apply log1p normalization if requested
        if normalize:
            expression_matrix = np.log1p(expression_matrix)

        # GLOBAL RANKING MODE: Use fixed gene set from training
        if use_global_ranking:
            if self.global_gene_ranking is None:
                raise ValueError(
                    "use_global_ranking=True but global ranking not set! "
                    "Call compute_global_ranking() on training data first."
                )
            tokenized_cells = self._tokenize_with_global_ranking(
                expression_matrix, gene_names, top_k
            )
        # PER-CELL RANKING MODE (ORIGINAL): Rank genes independently per cell
        else:
            tokenized_cells = self._tokenize_per_cell_ranking(
                expression_matrix, gene_names, top_k
            )

        return tokenized_cells

    def _tokenize_per_cell_ranking(
        self,
        expression_matrix: np.ndarray,
        gene_names: np.ndarray,
        top_k: int
    ) -> List[List[int]]:
        """
        Original Geneformer tokenization: rank genes independently per cell.

        This is the ORIGINAL behavior used by DANN and domain adaptation.
        Each cell gets different genes based on its own expression profile.
        """
        tokenized_cells = []

        for i in range(expression_matrix.shape[0]):
            # Get expression values for this cell
            cell_expr = expression_matrix[i, :]

            # Rank genes by expression (descending) - PER CELL
            ranked_indices = np.argsort(-cell_expr)[:top_k]

            # Get top expressed genes
            top_genes = gene_names[ranked_indices]
            top_expr = cell_expr[ranked_indices]

            # Filter out zero/low expression
            mask = top_expr > 0
            top_genes = top_genes[mask]

            # Convert to token IDs: [CLS] + gene_tokens + [SEP]
            tokens = [self.gene_vocab["[CLS]"]]
            for gene in top_genes:
                token_id = self.gene_vocab.get(gene, self.gene_vocab["[UNK]"])
                tokens.append(token_id)
            tokens.append(self.gene_vocab["[SEP]"])

            tokenized_cells.append(tokens)

        return tokenized_cells

    def _tokenize_with_global_ranking(
        self,
        expression_matrix: np.ndarray,
        gene_names: np.ndarray,
        top_k: int
    ) -> List[List[int]]:
        """
        NEW: Use global gene ranking (same genes across all cells).

        This ensures consistent vocabulary between training and test sets,
        solving the batch effect problem in fine-tuning.
        """
        # Use pre-computed global ranking
        top_gene_names = self.global_gene_ranking[:top_k]

        # Find indices of top genes in current gene_names array
        gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
        top_gene_indices = [gene_to_idx[g] for g in top_gene_names if g in gene_to_idx]

        tokenized_cells = []

        for i in range(expression_matrix.shape[0]):
            cell_expr = expression_matrix[i, :]

            # Extract expression for globally-ranked genes only
            selected_expr = cell_expr[top_gene_indices]
            selected_genes = np.array([gene_names[idx] for idx in top_gene_indices])

            # Filter out zero/low expression (but maintain global gene order)
            mask = selected_expr > 0
            final_genes = selected_genes[mask]

            # Convert to token IDs: [CLS] + gene_tokens + [SEP]
            tokens = [self.gene_vocab["[CLS]"]]
            for gene in final_genes:
                token_id = self.gene_vocab.get(gene, self.gene_vocab["[UNK]"])
                tokens.append(token_id)
            tokens.append(self.gene_vocab["[SEP]"])

            tokenized_cells.append(tokens)

        return tokenized_cells

    def compute_global_ranking(
        self,
        expression_matrix: np.ndarray,
        gene_names: np.ndarray,
        normalize: bool = False
    ):
        """
        Compute global gene ranking based on mean expression across all cells.

        Call this ONCE on training data, then use use_global_ranking=True during
        tokenization to ensure consistent vocabulary.

        Args:
            expression_matrix: Training data [n_cells, n_genes]
            gene_names: Gene names [n_genes]
            normalize: If True, apply log1p before computing mean
        """
        # Apply normalization if requested
        if normalize:
            expression_matrix = np.log1p(expression_matrix)

        # Compute mean expression per gene across all training cells
        mean_expression = np.mean(expression_matrix, axis=0)

        # Rank genes by mean expression (descending)
        ranked_indices = np.argsort(-mean_expression)

        # Store ranked gene names
        self.global_gene_ranking = gene_names[ranked_indices]

        print(f"[OK] Computed global gene ranking from {expression_matrix.shape[0]} cells")
        print(f"  Top 10 genes: {list(self.global_gene_ranking[:10])}")

        return self.global_gene_ranking

    @torch.no_grad()
    def get_embeddings(
        self,
        expression_matrix: np.ndarray,
        gene_names: np.ndarray,
        batch_size: int = 32,
        top_k: int = 2048,
        use_global_ranking: bool = False,
        normalize: bool = False
    ) -> torch.Tensor:
        """
        Get Geneformer embeddings for gene expression data.

        Args:
            expression_matrix: [n_cells, n_genes]
            gene_names: Gene names [n_genes]
            batch_size: Batch size for inference
            top_k: Top K genes to use per cell
            use_global_ranking: Use global gene ranking (for fine-tuning consistency)
            normalize: Apply log1p normalization

        Returns:
            embeddings: Cell embeddings [n_cells, hidden_size]
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_pretrained() first.")

        print(f"Generating Geneformer embeddings...")
        print(f"  Cells: {expression_matrix.shape[0]}")
        print(f"  Genes: {expression_matrix.shape[1]}")
        print(f"  Top K per cell: {top_k}")
        print(f"  Global ranking: {use_global_ranking}")
        print(f"  Normalize (log1p): {normalize}")

        # Tokenize
        tokenized = self.tokenize_expression(
            expression_matrix, gene_names, top_k,
            use_global_ranking=use_global_ranking,
            normalize=normalize
        )

        # Pad sequences
        max_len = max(len(t) for t in tokenized)
        max_len = min(max_len, 512)  # BERT max length

        embeddings = []
        self.model.eval()

        for i in range(0, len(tokenized), batch_size):
            batch_tokens = tokenized[i:i+batch_size]

            # Pad batch
            input_ids = torch.zeros(len(batch_tokens), max_len, dtype=torch.long)
            attention_mask = torch.zeros(len(batch_tokens), max_len, dtype=torch.long)

            for j, tokens in enumerate(batch_tokens):
                length = min(len(tokens), max_len)
                input_ids[j, :length] = torch.tensor(tokens[:length])
                attention_mask[j, :length] = 1

            # Move to device
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            # Forward pass
            outputs = self.model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )

            # Use [CLS] token embedding
            batch_emb = outputs.last_hidden_state[:, 0, :]
            embeddings.append(batch_emb.cpu())

        embeddings = torch.cat(embeddings, dim=0)
        print(f"[OK] Embeddings generated: {embeddings.shape}")

        return embeddings


def test_geneformer_loading():
    """
    Test Geneformer loading with dummy data.
    """
    print("=" * 60)
    print("Testing Geneformer Loader")
    print("=" * 60)

    try:
        # Initialize loader
        loader = GeneformerLoader(device="cuda")

        # Load model
        model, tokenizer = loader.load_pretrained(use_lora=True, lora_rank=16)

        # Test with dummy expression data
        print("\nTesting with dummy expression data...")
        n_cells = 10
        n_genes = 100
        expression = np.random.rand(n_cells, n_genes).astype(np.float32)
        gene_names = np.array([f"GENE{i}" for i in range(n_genes)])

        embeddings = loader.get_embeddings(
            expression, gene_names, batch_size=5, top_k=50
        )

        print(f"\n✅ Geneformer loading successful!")
        print(f"  Embeddings shape: {embeddings.shape}")
        print(f"  Expected: [{n_cells}, {model.config.hidden_size}]")

        if embeddings.shape[0] == n_cells:
            print("\n🎉 Geneformer is ready for use!")
            print("\nYou can now train with: --model_name geneformer")
            return True
        else:
            print(f"\n[WARNING] Shape mismatch!")
            return False

    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print("\nPlease install: pip install transformers")
        return False
    except Exception as e:
        print(f"\n❌ Error loading Geneformer: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_geneformer_loading()
