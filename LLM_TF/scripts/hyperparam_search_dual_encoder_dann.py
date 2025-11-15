"""
Hyperparameter search for Dual-Encoder DANN using Optuna.
Searches optimal hyperparameters for dual encoder architecture with source LLM and target peak embedder.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import optuna
from optuna.pruners import MedianPruner, NopPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Add TDC to path for scGPT tokenizer
tdc_path = "/users/ha00014/Halimas_projects/foundations_models/TDC-main"
if tdc_path not in sys.path:
    sys.path.insert(0, tdc_path)

from LLM_TF.dual_encoder_dann import DualEncoderDANN, compute_dual_encoder_loss
from LLM_TF.embedders.unified_embedder import UnifiedEmbedder
from LLM_TF.helpers.io_utils import ensure_dir, seed_everything
import hashlib
import pickle


class IdentityEncoder(nn.Module):
    """Pass-through encoder for pre-computed embeddings."""
    def forward(self, x):
        return x


def get_cached_embeddings(model_name, data_csv, data_array, gene_names, embedder, batch_size=16, cache_dir="DANN_output/embedding_cache"):
    """
    Get embeddings with caching to speed up preprocessing.

    Args:
        model_name: Foundation model name (e.g., 'geneformer-316m')
        data_csv: Path to source CSV file (for cache key)
        data_array: Expression data array
        gene_names: Gene names
        embedder: UnifiedEmbedder instance
        batch_size: Batch size for embedding generation
        cache_dir: Directory to store cached embeddings

    Returns:
        embeddings: numpy array or torch tensor
    """
    # Create cache directory
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Generate cache key from model + data file + shape
    cache_key_str = f"{model_name}_{data_csv}_{data_array.shape[0]}_{data_array.shape[1]}"
    cache_key = hashlib.md5(cache_key_str.encode()).hexdigest()
    cache_file = Path(cache_dir) / f"{cache_key}.pkl"

    # Check if cache exists
    if cache_file.exists():
        print(f"  ⚡ Loading cached embeddings from: {cache_file.name}")
        print(f"     Cache key: {model_name} + {data_array.shape[0]} cells × {data_array.shape[1]} genes")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        print(f"  ✓ Loaded embeddings: {cached_data['embeddings'].shape}")
        print(f"  ✓ Time saved: ~30-60 minutes!")
        return cached_data['embeddings']

    # Cache doesn't exist - compute embeddings
    print(f"  💾 Cache not found - computing embeddings (this will be cached for future runs)")
    print(f"     Cache file: {cache_file.name}")
    embeddings = embedder.get_embeddings(data_array, np.array(gene_names), batch_size=batch_size)

    # Save to cache
    cache_data = {
        'embeddings': embeddings,
        'model_name': model_name,
        'shape': embeddings.shape,
        'data_shape': data_array.shape
    }
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"  ✓ Embeddings cached - future runs will be 50-100x faster!")

    return embeddings


def filter_peaks_by_frequency(X_tgt, peak_names, threshold=0.10):
    """
    Filter PEAK features by frequency (remove peaks that appear in <threshold% of cells).
    This removes noisy/uninformative peaks common in sparse ATAC-seq data.

    Args:
        X_tgt: numpy array (n_cells, n_peaks)
        peak_names: list of peak names
        threshold: minimum fraction of cells a peak must appear in (default 0.10 = 10%)

    Returns:
        X_filtered: numpy array with filtered peaks
        filtered_peak_names: list of remaining peak names
    """
    n_cells = X_tgt.shape[0]
    peaks_per_cell = (X_tgt > 0).sum(axis=0)
    peak_frequency = peaks_per_cell / n_cells

    keep_peaks = peak_frequency >= threshold
    X_filtered = X_tgt[:, keep_peaks]
    filtered_peak_names = [peak_names[i] for i in range(len(peak_names)) if keep_peaks[i]]

    print(f"\n  PEAK Filtering (threshold={threshold*100:.0f}% of cells):")
    print(f"    Original peaks: {len(peak_names):,}")
    print(f"    Filtered peaks: {len(filtered_peak_names):,} ({len(filtered_peak_names)/len(peak_names)*100:.1f}% retained)")
    print(f"    Removed peaks: {len(peak_names) - len(filtered_peak_names):,} (low frequency noise)")

    return X_filtered, filtered_peak_names


def normalize_counts(X, modality_name="data"):
    """
    Normalize count data with consistent CPM + log1p + L2 normalization.
    Auto-detects if data is already normalized and skips if needed.

    Args:
        X: numpy array of shape (n_cells, n_features)
        modality_name: string identifier for logging

    Returns:
        Normalized numpy array (float32)
    """
    from sklearn.preprocessing import normalize as sk_normalize

    max_val = np.max(X)
    has_decimals = np.any(X != X.astype(int))
    mean_val = np.mean(X[X > 0]) if np.any(X > 0) else 0

    # Detection logic: if max < 20 and has decimals, likely already normalized
    if max_val < 20 and has_decimals:
        print(f"  {modality_name}: Already normalized detected (max={max_val:.2f}, mean_nonzero={mean_val:.2f})")
        print(f"  {modality_name}: Skipping normalization, using data as-is")
        return X.astype(np.float32)

    # Raw counts detected - apply full normalization pipeline
    print(f"  {modality_name}: RAW counts detected (max={max_val:.0f}, integers={not has_decimals})")
    print(f"  {modality_name}: Applying CPM + log1p + L2 normalization...")

    # Step 1: CPM normalization (counts per million)
    cell_totals = X.sum(axis=1, keepdims=True)
    cell_totals[cell_totals == 0] = 1  # Avoid division by zero
    X_cpm = (X / cell_totals) * 1e6

    # Step 2: log1p transformation
    X_log = np.log1p(X_cpm)

    # Step 3: L2 normalization for stability
    X_norm = sk_normalize(X_log, norm='l2', copy=False).astype(np.float32)

    print(f"  {modality_name}: After normalization - min={X_norm.min():.4f}, max={X_norm.max():.4f}, mean={X_norm.mean():.4f}")
    print(f"  {modality_name}: Sparsity: {(X_norm == 0).sum() / X_norm.size * 100:.1f}%")

    return X_norm


# Global variables to avoid reloading data
GLOBAL_DATA = {}


def prepare_data_once(args):
    """Load and preprocess data once, store globally."""
    global GLOBAL_DATA

    print("="*60)
    print("LOADING AND PREPROCESSING DATA")
    print("="*60)

    # Load source and target CSVs
    src_df = pd.read_csv(args.src_csv, index_col=0)
    tgt_df = pd.read_csv(args.tgt_csv, index_col=0)

    # Find label column
    label_candidates = [col for col in src_df.columns if 'phase' in col.lower() or 'label' in col.lower()]
    label_col = label_candidates[0] if label_candidates else 'phase'
    print(f"Label column: '{label_col}'")

    # Get genes/features
    src_genes = [col for col in src_df.columns if col != label_col]
    tgt_genes = list(tgt_df.columns)

    print(f"Source genes: {len(src_genes)}")
    print(f"Target peaks: {len(tgt_genes)}")

    # Extract data
    y_src = src_df[label_col].values
    X_src = src_df[src_genes].values.astype(np.float32)
    X_tgt = tgt_df[tgt_genes].values.astype(np.float32)

    # Filter PEAK features by frequency (if threshold > 0)
    if args.peak_filter_threshold > 0:
        print(f"\nFiltering PEAK features (threshold={args.peak_filter_threshold*100:.0f}% of cells)...")
        X_tgt, tgt_genes = filter_peaks_by_frequency(X_tgt, tgt_genes, threshold=args.peak_filter_threshold)

    # Apply consistent normalization to both modalities
    print("\nNormalizing source RNA data (GEX)...")
    X_src = normalize_counts(X_src, modality_name="Source RNA (GEX)")

    print("\nNormalizing target PEAK data (ATAC)...")
    X_tgt = normalize_counts(X_tgt, modality_name="Target PEAK (ATAC)")

    # Map labels
    label_map = {label: idx for idx, label in enumerate(sorted(set(y_src)))}
    y_src_encoded = np.array([label_map[label] for label in y_src])

    print(f"\nLabel mapping: {label_map}")
    print(f"Source class distribution: {np.bincount(y_src_encoded)}")

    # Load target labels from paired GEX file if provided
    y_tgt_encoded = None
    tgt_cell_ids = tgt_df.index.tolist()
    if args.tgt_label_csv:
        print(f"\n🎯 Loading target labels from paired GEX file: {args.tgt_label_csv}")
        tgt_label_df = pd.read_csv(args.tgt_label_csv, index_col=0)
        label_candidates = [col for col in tgt_label_df.columns if 'phase' in col.lower() or 'label' in col.lower()]
        tgt_label_col = label_candidates[0] if label_candidates else 'phase'
        print(f"  Label column: '{tgt_label_col}'")

        # Match cell_ids between target PEAK and GEX labels
        tgt_label_df_matched = tgt_label_df.loc[tgt_cell_ids]
        y_tgt = tgt_label_df_matched[tgt_label_col].values
        y_tgt_encoded = np.array([label_map[label] for label in y_tgt])
        print(f"Target class distribution: {np.bincount(y_tgt_encoded)}")
        print(f"✓ Matched {len(y_tgt_encoded)} target labels to PEAK cells via cell_id")

        # Split target into train/validation (semi-supervised mode)
        if args.tgt_validation_split > 0:
            print(f"\n{'='*60}")
            print(f" TARGET TRAIN/VAL SPLIT (Semi-Supervised)")
            print(f"{'='*60}")
            print(f"  Splitting target: {1-args.tgt_validation_split:.0%} train, {args.tgt_validation_split:.0%} val")

            from sklearn.model_selection import train_test_split
            indices = np.arange(len(y_tgt_encoded))
            idx_train, idx_val = train_test_split(
                indices, test_size=args.tgt_validation_split, random_state=42, stratify=y_tgt_encoded
            )

            # Split target data
            X_tgt_train = X_tgt[idx_train]
            X_tgt_val = X_tgt[idx_val]
            y_tgt_train = y_tgt_encoded[idx_train]
            y_tgt_val = y_tgt_encoded[idx_val]
            tgt_cell_ids_train = [tgt_cell_ids[i] for i in idx_train]
            tgt_cell_ids_val = [tgt_cell_ids[i] for i in idx_val]

            print(f"  Train: {len(idx_train)} cells (class dist: {np.bincount(y_tgt_train)})")
            print(f"  Val:   {len(idx_val)} cells (class dist: {np.bincount(y_tgt_val)})")
            print(f"  ⚠️  IMPORTANT: Validation cells NEVER seen by optimizer!")
            print(f"{'='*60}\n")

            # Replace full target with training split (val stored in GLOBAL_DATA later)
            X_tgt = X_tgt_train
            y_tgt_encoded = y_tgt_train
            tgt_cell_ids = tgt_cell_ids_train
        else:
            # No validation split
            X_tgt_val = None
            y_tgt_val = None
            tgt_cell_ids_val = None
    else:
        # No target labels - unsupervised mode
        X_tgt_val = None
        y_tgt_val = None
        tgt_cell_ids_val = None

    # Limit target cells if specified
    if args.max_target_cells and X_tgt.shape[0] > args.max_target_cells:
        print(f"Limiting target cells: {X_tgt.shape[0]} → {args.max_target_cells}")
        indices = np.random.choice(X_tgt.shape[0], args.max_target_cells, replace=False)
        X_tgt = X_tgt[indices]
        if y_tgt_encoded is not None:
            y_tgt_encoded = y_tgt_encoded[indices]

    print(f"\nSource: {X_src.shape[0]} cells × {X_src.shape[1]} features")
    print(f"Target: {X_tgt.shape[0]} cells × {X_tgt.shape[1]} features")

    # ANTI-OVERFITTING: Split source into train/validation
    if args.use_validation_split:
        X_train, X_val, y_train, y_val = train_test_split(
            X_src, y_src_encoded, test_size=args.validation_split, stratify=y_src_encoded, random_state=42
        )
        print(f"\n[ANTI-OVERFITTING] Train/Val Split: {X_train.shape[0]} train, {X_val.shape[0]} val")
        print(f"Train distribution: {np.bincount(y_train)}")
        print(f"Val distribution: {np.bincount(y_val)}")
    else:
        X_train = X_src
        X_val = None
        y_train = y_src_encoded
        y_val = None
        print(f"\n[WARNING] No validation split - may overfit!")

    # Get source embedding dimension map
    source_dim_map = {
        'scgpt': 512,
        'geneformer-10m': 256,
        'geneformer-104m': 768,
        'geneformer-104m-clcancer': 768,
        'geneformer-316m': 1152,
        'uce-100m': 768,
        'teddy-70m': 512,
        'teddy-160m': 768,
        'teddy-400m': 1024,
        'scfoundation': 768
    }

    # Initialize tokenizer and loader variables
    tokenizer_for_dynamic = None
    loader_for_dynamic = None

    # Load source encoder(s) and generate embeddings
    if args.dynamic_embeddings:
        print(f"\nDynamic Embeddings Mode: LLM will run on-the-fly during training")
        print(f"  Model: {args.source_model}")
        print(f"  Unfreeze last {args.unfreeze_last_n_layers} layers" if args.unfreeze_last_n_layers > 0 else "  LLM fully frozen")

        source_embedder = UnifiedEmbedder(
            model_name=args.source_model,
            use_lora=args.use_lora,
            lora_rank=args.lora_rank,
            device="cuda"
        )

        source_model, _ = source_embedder.load_model()

        if args.unfreeze_last_n_layers > 0:
            print(f"\nUnfreezing last {args.unfreeze_last_n_layers} transformer layers...")

            if hasattr(source_model, 'transformer') and hasattr(source_model.transformer, 'h'):
                total_layers = len(source_model.transformer.h)
                for i in range(total_layers - args.unfreeze_last_n_layers, total_layers):
                    for param in source_model.transformer.h[i].parameters():
                        param.requires_grad = True
                print(f"  Unfrozen layers {total_layers - args.unfreeze_last_n_layers} to {total_layers - 1}")
            else:
                print(f"  WARNING: Model architecture not recognized, keeping all layers frozen")

        source_dim = source_dim_map.get(args.source_model, 768)
        # Store RAW data for tokenization during training
        emb_train = X_train  # Will be tokenized in DataLoader
        emb_val = X_val if X_val is not None else None

        print(f"  Will tokenize RAW data on-the-fly: {emb_train.shape}")
        print(f"  Input dimension: {X_train.shape[1]} genes → {source_dim} embedding dim")

        # Store tokenizer and loader for dynamic mode
        tokenizer_for_dynamic = source_embedder.tokenizer
        loader_for_dynamic = source_embedder.loader

    elif args.use_ensemble:
        print(f"\nEnsemble Mode: Combining {args.ensemble_model1} + {args.ensemble_model2}")

        embedder1 = UnifiedEmbedder(
            model_name=args.ensemble_model1,
            use_lora=args.use_lora,
            lora_rank=args.lora_rank,
            device="cuda"
        )
        embedder2 = UnifiedEmbedder(
            model_name=args.ensemble_model2,
            use_lora=args.use_lora,
            lora_rank=args.lora_rank,
            device="cuda"
        )

        model1, _ = embedder1.load_model()
        model2, _ = embedder2.load_model()

        dim1 = source_dim_map.get(args.ensemble_model1, 512)
        dim2 = source_dim_map.get(args.ensemble_model2, 512)
        source_dim = dim1 + dim2

        print(f"  Model 1: {args.ensemble_model1} (dim: {dim1})")
        print(f"  Model 2: {args.ensemble_model2} (dim: {dim2})")
        print(f"  Combined dim: {source_dim}")

        src_batch_size = 16
        print(f"\nGenerating embeddings from Model 1 (with caching)...")
        emb1_train = get_cached_embeddings(args.ensemble_model1, args.src_csv, X_train, src_genes, embedder1, batch_size=src_batch_size)
        emb1_val = get_cached_embeddings(args.ensemble_model1, args.src_csv, X_val, src_genes, embedder1, batch_size=src_batch_size) if X_val is not None else None

        print(f"Generating embeddings from Model 2 (with caching)...")
        emb2_train = get_cached_embeddings(args.ensemble_model2, args.src_csv, X_train, src_genes, embedder2, batch_size=src_batch_size)
        emb2_val = get_cached_embeddings(args.ensemble_model2, args.src_csv, X_val, src_genes, embedder2, batch_size=src_batch_size) if X_val is not None else None

        emb_train = np.concatenate([emb1_train, emb2_train], axis=1)
        emb_val = np.concatenate([emb1_val, emb2_val], axis=1) if emb1_val is not None and emb2_val is not None else None

        print(f"  Train embeddings: {emb_train.shape}")
        if emb_val is not None:
            print(f"  Val embeddings: {emb_val.shape}")

        source_model = None

    else:
        print(f"\nLoading source encoder: {args.source_model}...")
        source_embedder = UnifiedEmbedder(
            model_name=args.source_model,
            use_lora=args.use_lora,
            lora_rank=args.lora_rank,
            device="cuda"
        )
        source_model, _ = source_embedder.load_model()

        source_dim = source_dim_map.get(args.source_model, 512)
        print(f"  Source embedding dimension: {source_dim}")

        # CRITICAL: Generate embeddings from raw data (with caching)
        print(f"\nGenerating source embeddings (with caching)...")
        src_batch_size = 16
        emb_train = get_cached_embeddings(args.source_model, args.src_csv, X_train, src_genes, source_embedder, batch_size=src_batch_size)
        emb_val = get_cached_embeddings(args.source_model, args.src_csv, X_val, src_genes, source_embedder, batch_size=src_batch_size) if X_val is not None else None
        print(f"  Train embeddings: {emb_train.shape}")
        if emb_val is not None:
            print(f"  Val embeddings: {emb_val.shape}")

    # Store globally
    if args.dynamic_embeddings:
        source_model_name = f"{args.source_model}_dynamic"
    elif args.use_ensemble:
        source_model_name = f"ensemble_{args.ensemble_model1}_{args.ensemble_model2}"
    else:
        source_model_name = args.source_model

    GLOBAL_DATA = {
        "emb_train": emb_train,  # Pre-computed embeddings OR raw data if dynamic
        "emb_val": emb_val,
        "y_train": y_train,
        "y_val": y_val,
        "X_tgt": X_tgt,
        "y_tgt": y_tgt_encoded,  # Target labels (paired GEX) if available
        "X_tgt_val": X_tgt_val,  # Target validation split (for semi-supervised evaluation)
        "y_tgt_val": y_tgt_val,  # Target validation labels
        "src_genes": np.array(src_genes),
        "tgt_genes": np.array(tgt_genes),
        "n_peaks": X_tgt.shape[1],
        "source_model": source_model,
        "source_dim": source_dim,
        "num_classes": len(label_map),
        "device": "cuda",
        "epochs": args.epochs,
        "use_validation_split": args.use_validation_split,
        "source_model_name": source_model_name,
        "peak_arch_type": args.peak_arch_type,
        "freeze_source": args.freeze_source,
        "use_lora": args.use_lora,
        "lora_rank": args.lora_rank,
        "use_ensemble": args.use_ensemble,
        "ensemble_model1": args.ensemble_model1 if args.use_ensemble else None,
        "ensemble_model2": args.ensemble_model2 if args.use_ensemble else None,
        "dynamic_embeddings": args.dynamic_embeddings,
        "unfreeze_last_n_layers": args.unfreeze_last_n_layers,
        "tokenizer": tokenizer_for_dynamic if args.dynamic_embeddings else None,
        "loader": loader_for_dynamic if args.dynamic_embeddings else None,
        # DANN improvements
        "shared_dim": args.shared_dim,
        "use_cdann": args.use_cdann,
        "use_contrastive_loss": args.use_contrastive_loss,
        "contrastive_temperature": args.contrastive_temperature,
        "lambda_contrastive": args.lambda_contrastive,
        "use_balance_loss": args.use_balance_loss,
        "lambda_balance": args.lambda_balance,
        "divergence_type": args.divergence_type,
        "use_entropy_loss": args.use_entropy_loss,
        "lambda_entropy": args.lambda_entropy,
        "lambda_target_class": args.lambda_target_class,
        "confidence_threshold": args.confidence_threshold,
        "use_grl_annealing": args.use_grl_annealing,
        "grl_gamma": args.grl_gamma,
        "tgt_label_csv": args.tgt_label_csv,  # Store for semi-supervised mode detection
    }

    # Compute source class distribution for balance loss
    if args.use_balance_loss:
        import torch
        y_train_full = np.concatenate([y_train, y_val]) if emb_val is not None else y_train
        class_counts = np.bincount(y_train_full, minlength=len(label_map))
        source_class_prior = torch.tensor(class_counts / class_counts.sum(), dtype=torch.float32)
        GLOBAL_DATA["source_class_prior"] = source_class_prior
        print(f"Source class distribution: {source_class_prior.numpy()}")
        print(f"  Class 0 (G1): {source_class_prior[0]:.1%}")
        print(f"  Class 1 (S):  {source_class_prior[1]:.1%}")
        print(f"  Class 2 (G2M): {source_class_prior[2]:.1%}\n")
    else:
        GLOBAL_DATA["source_class_prior"] = None

    print("Data preparation complete!\n")


class TokenizedSourceDataset(torch.utils.data.Dataset):
    """Dataset that tokenizes source data on-the-fly for dynamic embeddings."""
    def __init__(self, X_src, y_src, gene_names, tokenizer, loader=None):
        self.X_src = X_src
        self.y_src = y_src
        self.gene_names = gene_names
        self.tokenizer = tokenizer
        self.loader = loader

        # Detect model type and tokenize accordingly
        if loader is not None and hasattr(loader, 'tokenize_expression'):
            # Geneformer-style tokenization
            self.is_geneformer = True
            token_ids_list = loader.tokenize_expression(
                X_src, gene_names, top_k=2048, use_global_ranking=False, normalize=False
            )
            # Convert to tensors and pad
            max_len = max(len(ids) for ids in token_ids_list)
            self.tokens = []
            for ids in token_ids_list:
                ids_tensor = torch.tensor(ids, dtype=torch.long)
                # Pad to max_len
                if len(ids_tensor) < max_len:
                    pad_len = max_len - len(ids_tensor)
                    ids_tensor = torch.cat([ids_tensor, torch.zeros(pad_len, dtype=torch.long)])
                attention_mask = torch.cat([
                    torch.ones(len(ids), dtype=torch.long),
                    torch.zeros(max_len - len(ids), dtype=torch.long)
                ])
                self.tokens.append((ids_tensor, attention_mask))
        else:
            # scGPT-style tokenization
            self.is_geneformer = False
            from LLM_TF.embedders.embedder import tokenize_matrix
            self.tokens = tokenize_matrix(X_src, gene_names, tokenizer)

    def __len__(self):
        return len(self.y_src)

    def __getitem__(self, idx):
        label = torch.tensor(self.y_src[idx], dtype=torch.long)
        if self.is_geneformer:
            token_ids, attention_mask = self.tokens[idx]
            return token_ids, attention_mask, label
        else:
            token_ids, token_vals, attention_mask = self.tokens[idx]
            return token_ids, token_vals, attention_mask, label


def check_target_diversity(model, X_tgt, batch_size, device):
    """
    Check if target predictions have collapsed to single class.

    Returns dict with:
        - distribution: class counts [G1, G2M, S]
        - max_ratio: fraction of predictions in most common class
        - is_collapsed: True if >80% predictions in single class
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        tgt_tensor = X_tgt if isinstance(X_tgt, torch.Tensor) else torch.from_numpy(X_tgt).float()
        tgt_tensor = tgt_tensor.to(device)
        tgt_dataset = TensorDataset(tgt_tensor)
        tgt_loader = DataLoader(tgt_dataset, batch_size=batch_size, shuffle=False)

        for (batch_X,) in tgt_loader:
            outputs = model(target_data=batch_X, alpha=1.0)
            cls_out = outputs['target_class_pred']
            preds = cls_out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    all_preds = np.array(all_preds)
    class_counts = np.bincount(all_preds, minlength=3)
    max_class_ratio = class_counts.max() / len(all_preds)

    model.train()

    return {
        'distribution': class_counts,
        'max_ratio': max_class_ratio,
        'is_collapsed': max_class_ratio > 0.80  # 80% threshold for collapse detection
    }


def train_dual_encoder_trial(model, emb_src_train, y_src_train, X_tgt, epochs, batch_size,
                             lambda_domain, lambda_class, optimizer, device, use_grl_annealing=False, grl_gamma=10.0,
                             use_contrastive_loss=False, lambda_contrastive=0.5, contrastive_temperature=0.1,
                             lambda_balance=0.5, use_balance_loss=False, source_class_prior=None,
                             lambda_entropy=0.1, use_entropy_loss=False, confidence_threshold=0.5,
                             divergence_type='kl',
                             tokenizer=None, src_genes=None, loader=None, trial=None, emb_src_val=None, y_src_val=None,
                             patience=100, y_tgt_train=None, lambda_target_class=1.0, lambda_target_class_warmup=0,
                             X_tgt_val=None, y_tgt_val=None):
    """Train dual encoder for one trial with early stopping and Optuna pruning."""

    # Check if we need tokenized data (check loader instead of tokenizer - Geneformer/scFoundation/UCE have tokenizer=None but loader is set)
    use_tokenized = loader is not None and src_genes is not None

    if use_tokenized:
        # Dynamic embeddings mode: use tokenized dataset
        src_dataset = TokenizedSourceDataset(
            emb_src_train, y_src_train, src_genes, tokenizer, loader=loader
        )
        is_geneformer = src_dataset.is_geneformer
    else:
        # Pre-computed embeddings mode: use TensorDataset
        is_geneformer = False
        emb_src = emb_src_train if isinstance(emb_src_train, torch.Tensor) else torch.from_numpy(emb_src_train)
        y_src = y_src_train if isinstance(y_src_train, torch.Tensor) else torch.from_numpy(y_src_train)
        src_dataset = TensorDataset(
            emb_src.float(),
            y_src.long()
        )

    # Target dataset (always raw peaks, with labels if available)
    X_tgt_t = X_tgt if isinstance(X_tgt, torch.Tensor) else torch.from_numpy(X_tgt)
    if y_tgt_train is not None:
        y_tgt_t = y_tgt_train if isinstance(y_tgt_train, torch.Tensor) else torch.from_numpy(y_tgt_train)
        tgt_dataset = TensorDataset(
            X_tgt_t.float(),
            y_tgt_t.long()
        )
    else:
        tgt_dataset = TensorDataset(
            X_tgt_t.float()
        )

    src_loader = DataLoader(src_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    tgt_loader = DataLoader(tgt_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Early stopping variables
    best_val_acc = 0.0
    epochs_no_improve = 0
    use_early_stopping = (emb_src_val is not None and y_src_val is not None)

    # Collapse tracking variables (require persistence before pruning)
    collapse_counter = 0
    collapse_persistence_threshold = 5  # Require 5 consecutive collapsed epochs

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0

        # GRL lambda schedule
        if use_grl_annealing:
            from LLM_TF.dual_encoder_dann import DualEncoderDANN
            alpha = DualEncoderDANN.get_grl_lambda(epoch, epochs, gamma=grl_gamma)
        else:
            alpha = 1.0

        # Curriculum learning for lambda_target_class (warmup schedule)
        if lambda_target_class_warmup > 0 and epoch < lambda_target_class_warmup:
            # Linear warmup: 0 → lambda_target_class over warmup epochs
            current_lambda_target = lambda_target_class * (epoch / lambda_target_class_warmup)
        else:
            # After warmup, use full weight
            current_lambda_target = lambda_target_class

        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)

        for batch_idx in range(min(len(src_loader), len(tgt_loader))):
            try:
                src_batch = next(src_iter)
            except StopIteration:
                src_iter = iter(src_loader)
                src_batch = next(src_iter)

            try:
                tgt_batch = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tgt_batch = next(tgt_iter)

            # Unpack target batch (with or without labels)
            if len(tgt_batch) == 2:
                # Semi-supervised: (data, labels)
                tgt_data, tgt_labels = tgt_batch
                tgt_labels = tgt_labels.to(device)
            else:
                # Unsupervised: (data,)
                tgt_data, = tgt_batch
                tgt_labels = None

            # Handle tokenized vs pre-computed embeddings
            if use_tokenized:
                if is_geneformer:
                    # Geneformer: (token_ids, attention_mask, label)
                    token_ids, attention_mask, src_labels = src_batch
                    token_ids = token_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    src_labels = src_labels.to(device)
                    # Pack tokens into a dict for the model
                    src_data = {
                        'input_ids': token_ids,
                        'attention_mask': attention_mask
                    }
                else:
                    # scGPT: (token_ids, token_vals, attention_mask, label)
                    token_ids, token_vals, attention_mask, src_labels = src_batch
                    token_ids = token_ids.to(device)
                    token_vals = token_vals.to(device)
                    attention_mask = attention_mask.to(device)
                    src_labels = src_labels.to(device)
                    # Pack tokens into a dict for the model
                    src_data = {
                        'input_ids': token_ids,
                        'values': token_vals,
                        'attention_mask': attention_mask
                    }
            else:
                src_data, src_labels = src_batch
                src_data = src_data.to(device)
                src_labels = src_labels.to(device)

            tgt_data = tgt_data.to(device)

            optimizer.zero_grad()

            outputs = model(
                source_data=src_data,
                target_data=tgt_data,
                alpha=alpha
            )

            loss, loss_dict = compute_dual_encoder_loss(
                outputs=outputs,
                source_labels=src_labels,
                target_labels=tgt_labels,
                lambda_domain=lambda_domain,
                lambda_class=lambda_class,
                lambda_target_class=current_lambda_target if tgt_labels is not None else 0.0,  # Use warmup schedule
                lambda_contrastive=lambda_contrastive,
                use_contrastive_loss=use_contrastive_loss,
                contrastive_temperature=contrastive_temperature,
                lambda_balance=lambda_balance,
                use_balance_loss=use_balance_loss,
                source_class_prior=source_class_prior,
                lambda_entropy=lambda_entropy,
                use_entropy_loss=use_entropy_loss,
                confidence_threshold=confidence_threshold,
                divergence_type=divergence_type,
                device=device
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches if n_batches > 0 else 0

        # Evaluate on validation set for early stopping and Optuna pruning
        if use_early_stopping:
            val_acc = evaluate_dual_encoder(
                model=model,
                emb_data=emb_src_val,
                y_data=y_src_val,
                batch_size=batch_size,
                device=device,
                domain='source',
                tokenizer=tokenizer,
                src_genes=src_genes,
                loader=loader
            )

            # Check target diversity to detect class collapse
            # GRACE PERIOD: Don't check diversity for first 10 epochs (model needs time to learn)
            diversity = check_target_diversity(model, X_tgt, batch_size, device)

            # Report to Optuna for pruning
            if trial is not None:
                trial.report(val_acc, epoch)
                # Handle pruning (only after grace period of 20 epochs)
                if epoch >= 20 and trial.should_prune():
                    print(f"\n  🔍 PRUNING DIAGNOSTICS:")
                    print(f"     Epoch: {epoch+1}/{epochs}")
                    print(f"     Val Acc: {val_acc:.4f} (best: {best_val_acc:.4f})")
                    print(f"     Alpha (GRL): {alpha:.3f}")
                    print(f"     Target diversity: {diversity['distribution']} (max: {diversity['max_ratio']:.1%})")
                    print(f"     Reason: Optuna MedianPruner (performance below median)")
                    print(f"  [PRUNED by Optuna]\n")
                    raise optuna.TrialPruned()

            # Prune if target predictions collapsed to single class (after grace period + persistence)
            if epoch >= 25:  # Increased from 10 to allow GRL warmup
                if diversity['is_collapsed']:
                    collapse_counter += 1
                    if collapse_counter >= collapse_persistence_threshold:
                        print(f"\n  🔍 COLLAPSE PRUNING DIAGNOSTICS:")
                        print(f"     Epoch: {epoch+1}/{epochs}")
                        print(f"     Val Acc: {val_acc:.4f} (best: {best_val_acc:.4f})")
                        print(f"     Alpha (GRL): {alpha:.3f}")
                        print(f"     Target distribution: {diversity['distribution']} (max: {diversity['max_ratio']:.1%})")
                        print(f"     Collapsed epochs: {collapse_counter} consecutive (threshold: {collapse_persistence_threshold})")
                        print(f"     Reason: Persistent target class collapse (>{diversity['max_ratio']:.1%} in single class)")
                        print(f"  [PRUNED for persistent collapse]\n")
                        raise optuna.TrialPruned()
                else:
                    # Reset counter if diversity recovered
                    collapse_counter = 0

            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}, alpha={alpha:.3f}, no_improve={epochs_no_improve}")
                print(f"  Target diversity: {diversity['distribution']} (max: {diversity['max_ratio']:.1%})")

            # Early stopping
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
                break
        else:
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, alpha={alpha:.3f}")

    # Final diversity check after training completes
    final_diversity = check_target_diversity(model, X_tgt, batch_size, device)
    print(f"\n  ✓ Training complete!")
    print(f"  Final target diversity: {final_diversity['distribution']} (max: {final_diversity['max_ratio']:.1%})")
    if final_diversity['is_collapsed']:
        print(f"  ⚠️  WARNING: Final model has target class collapse!")

    return model


def evaluate_dual_encoder(model, emb_data, y_data, batch_size, device, domain='source', tokenizer=None, src_genes=None, loader=None):
    """Evaluate dual encoder model."""
    model.eval()

    # Check loader instead of tokenizer (Geneformer has tokenizer=None but loader is set)
    use_tokenized = loader is not None and src_genes is not None

    if use_tokenized:
        # Dynamic embeddings mode
        dataset = TokenizedSourceDataset(emb_data, y_data, src_genes, tokenizer, loader=loader)
        is_geneformer = dataset.is_geneformer
    else:
        is_geneformer = False
        # Pre-computed embeddings mode
        emb_tensor = emb_data if isinstance(emb_data, torch.Tensor) else torch.from_numpy(emb_data)
        y_tensor = y_data if isinstance(y_data, torch.Tensor) else torch.from_numpy(y_data)
        dataset = TensorDataset(
            emb_tensor.float(),
            y_tensor.long()
        )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            if use_tokenized:
                if is_geneformer:
                    # Geneformer: (token_ids, attention_mask, label)
                    token_ids, attention_mask, batch_labels = batch
                    token_ids = token_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    batch_data = {
                        'input_ids': token_ids,
                        'attention_mask': attention_mask
                    }
                else:
                    # scGPT: (token_ids, token_vals, attention_mask, label)
                    token_ids, token_vals, attention_mask, batch_labels = batch
                    token_ids = token_ids.to(device)
                    token_vals = token_vals.to(device)
                    attention_mask = attention_mask.to(device)
                    batch_data = {
                        'input_ids': token_ids,
                        'values': token_vals,
                        'attention_mask': attention_mask
                    }
            else:
                batch_data, batch_labels = batch
                batch_data = batch_data.to(device)

            if domain == 'source':
                outputs = model(source_data=batch_data, alpha=0.0)
                logits = outputs['source_class_pred']
            else:
                outputs = model(target_data=batch_data, alpha=0.0)
                logits = outputs['target_class_pred']

            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(batch_labels)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    accuracy = accuracy_score(all_labels, all_preds)

    return accuracy


def objective(trial):
    """Optuna objective function - returns validation accuracy to maximize."""

    # Hyperparameter ranges (similar to single encoder)
    lr = trial.suggest_float("lr", 5e-6, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    # Domain adaptation weights (increased ranges for stronger adaptation)
    lambda_domain = trial.suggest_float("lambda_domain", 0.5, 5.0, step=0.5)
    lambda_class = trial.suggest_float("lambda_class", 0.5, 2.0, step=0.1)

    # Batch size (smaller for GNN due to memory constraints with large graphs)
    if GLOBAL_DATA["peak_arch_type"] in ["gnn", "gnn_gcn"]:
        batch_size = trial.suggest_categorical("batch_size", [2, 4])
    else:
        batch_size = trial.suggest_categorical("batch_size", [64, 128])

    # Dual encoder architecture params
    # Use fixed value from args if provided, otherwise let Optuna tune
    shared_dim = GLOBAL_DATA["shared_dim"]  # Fixed from args (default 256)
    disc_hidden = trial.suggest_categorical("disc_hidden", [128, 256, 512])
    disc_layers = trial.suggest_int("disc_layers", 2, 4)
    cls_hidden = trial.suggest_categorical("cls_hidden", [128, 256, 512])
    cls_layers = trial.suggest_int("cls_layers", 2, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.05)
    use_spectral_norm = trial.suggest_categorical("use_spectral_norm", [True, False])

    # GRL lambda annealing - use fixed value from args if provided
    use_grl_annealing = GLOBAL_DATA["use_grl_annealing"]
    if use_grl_annealing:
        # If enabled via args, optionally tune gamma (or use fixed value)
        grl_gamma = trial.suggest_float("grl_gamma", 5.0, 15.0, step=1.0)
    else:
        grl_gamma = GLOBAL_DATA["grl_gamma"]  # Default 10.0

    # Class-conditional discriminator (CDANN) - use fixed value from args
    use_cdann = GLOBAL_DATA["use_cdann"]

    # Contrastive loss for cross-modality class semantics - use fixed value from args
    use_contrastive_loss = GLOBAL_DATA["use_contrastive_loss"]
    if use_contrastive_loss:
        # If enabled via args, optionally tune these hyperparams (increased range)
        lambda_contrastive = trial.suggest_float("lambda_contrastive", 0.5, 2.0, step=0.5)
        contrastive_temperature = trial.suggest_float("contrastive_temperature", 0.05, 0.5, step=0.05)
    else:
        lambda_contrastive = GLOBAL_DATA["lambda_contrastive"]  # Default 0.5
        contrastive_temperature = GLOBAL_DATA["contrastive_temperature"]  # Default 0.1

    # Target class balance loss - prevents collapse
    use_balance_loss = GLOBAL_DATA["use_balance_loss"]
    source_class_prior = GLOBAL_DATA["source_class_prior"]
    divergence_type = GLOBAL_DATA["divergence_type"]  # kl, jensen, coral, or mmd
    if use_balance_loss:
        # If enabled via args, tune lambda_balance with STRONGER weight (2.0-15.0)
        # Increased range to prevent target collapse more aggressively
        lambda_balance = trial.suggest_float("lambda_balance", 2.0, 15.0, step=1.0)
    else:
        lambda_balance = GLOBAL_DATA["lambda_balance"]  # Default 5.0

    # Entropy minimization - ChatGPT's "Option 4" (most effective for DANN)
    use_entropy_loss = GLOBAL_DATA["use_entropy_loss"]
    if use_entropy_loss:
        # Tune entropy loss weight and confidence threshold
        lambda_entropy = trial.suggest_float("lambda_entropy", 0.05, 0.5, step=0.05)
        confidence_threshold = trial.suggest_float("confidence_threshold", 0.3, 0.7, step=0.1)
    else:
        lambda_entropy = GLOBAL_DATA["lambda_entropy"]  # Default 0.1
        confidence_threshold = GLOBAL_DATA["confidence_threshold"]  # Default 0.5

    # Semi-supervised target classification - tune supervision weight and warmup
    # Check if we have target labels (semi-supervised mode)
    has_target_labels = GLOBAL_DATA.get("tgt_label_csv") is not None
    if has_target_labels:
        # Tune target classification weight (log scale: explores 1.0 → 15.0)
        lambda_target_class = trial.suggest_float("lambda_target_class", 1.0, 15.0, log=True)
        # Tune warmup schedule (50-200 epochs, step=50)
        lambda_target_class_warmup = trial.suggest_int("lambda_target_class_warmup", 50, 200, step=50)
    else:
        # Unsupervised mode - use default from args (typically 1.0, but won't be used)
        lambda_target_class = GLOBAL_DATA["lambda_target_class"]
        lambda_target_class_warmup = 0  # No warmup in unsupervised mode

    # Peak embedder architecture params
    peak_hidden_dim = trial.suggest_categorical("peak_hidden_dim", [128, 256, 512])
    peak_intermediate_dim = trial.suggest_categorical("peak_intermediate_dim", [256, 512, 1024])
    peak_num_layers = trial.suggest_int("peak_num_layers", 2, 5)
    peak_dropout = trial.suggest_float("peak_dropout", 0.0, 0.6, step=0.05)

    # Architecture-specific parameters
    # IMPORTANT: ALL architectures now tune activation function (9 variants, including periodic for cell cycle)
    activation = trial.suggest_categorical("activation", ["relu", "leaky_relu", "elu", "prelu", "gelu", "selu", "swish", "mish", "periodic"])

    if GLOBAL_DATA["peak_arch_type"] == "vae":
        beta_vae = trial.suggest_float("beta_vae", 0.5, 2.0, step=0.1)
        noise_level = 0.1
        use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
        residual_type = "add"
    elif GLOBAL_DATA["peak_arch_type"] == "dae":
        beta_vae = 1.0
        noise_level = trial.suggest_float("noise_level", 0.05, 0.3, step=0.05)
        use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
        residual_type = "add"
    elif GLOBAL_DATA["peak_arch_type"] == "contrastive":
        beta_vae = 1.0
        noise_level = 0.1
        temperature = trial.suggest_float("temperature", 0.05, 0.5, step=0.05)
        aug_dropout_prob = trial.suggest_float("aug_dropout_prob", 0.1, 0.5, step=0.05)
        projection_dim = trial.suggest_categorical("projection_dim", [64, 128, 256])
        use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
        residual_type = "add"
    elif GLOBAL_DATA["peak_arch_type"] == "hybrid":
        beta_vae = 1.0
        noise_level = 0.1
        use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
        residual_type = trial.suggest_categorical("residual_type", ["add", "concat"])
    elif GLOBAL_DATA["peak_arch_type"] in ["gnn", "gnn_gcn"]:
        beta_vae = 1.0
        noise_level = 0.1
        num_gnn_layers = trial.suggest_int("num_gnn_layers", 2, 4)
        gnn_hidden_dim = trial.suggest_categorical("gnn_hidden_dim", [128, 256, 512])
        use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
        residual_type = "add"
    elif GLOBAL_DATA["peak_arch_type"] in ["cnn", "cnn_multiscale", "cnn_dilated"]:
        beta_vae = 1.0
        noise_level = 0.1
        num_conv_layers = trial.suggest_int("num_conv_layers", 2, 4)
        num_filters = trial.suggest_categorical("num_filters", [64, 128, 256])
        kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
        use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
        residual_type = "add"
    else:  # mlp
        beta_vae = 1.0
        noise_level = 0.1
        use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
        residual_type = "add"

    epochs = GLOBAL_DATA["epochs"]

    # Set seed
    seed_everything(42 + trial.number)

    try:
        # Create target encoder with trial-specific parameters
        target_embedder = UnifiedEmbedder(model_name='peak', device=GLOBAL_DATA["device"])

        target_params = {
            "n_peaks": GLOBAL_DATA["n_peaks"],
            "hidden_dim": peak_hidden_dim,
            "intermediate_dim": peak_intermediate_dim,
            "num_layers": peak_num_layers,
            "dropout": peak_dropout,
            "peak_arch_type": GLOBAL_DATA["peak_arch_type"],
            "beta_vae": beta_vae,
            "noise_level": noise_level,
        }

        # Add architecture-specific params
        if GLOBAL_DATA["peak_arch_type"] == "contrastive":
            target_params.update({
                "temperature": temperature,
                "aug_dropout_prob": aug_dropout_prob,
                "projection_dim": projection_dim
            })
        elif GLOBAL_DATA["peak_arch_type"] == "hybrid":
            target_params.update({
                "activation": activation,
                "use_batch_norm": use_batch_norm,
                "residual_type": residual_type
            })
        elif GLOBAL_DATA["peak_arch_type"] in ["gnn", "gnn_gcn"]:
            target_params.update({
                "num_gnn_layers": num_gnn_layers,
                "gnn_hidden_dim": gnn_hidden_dim
            })
        elif GLOBAL_DATA["peak_arch_type"] in ["cnn", "cnn_multiscale", "cnn_dilated"]:
            target_params.update({
                "num_conv_layers": num_conv_layers,
                "num_filters": num_filters,
                "kernel_size": kernel_size
            })

        target_model, _ = target_embedder.load_model(**target_params)

        print(f"\n[Trial {trial.number}] Creating Dual-Encoder DANN")
        print(f"  Source: {GLOBAL_DATA['source_model_name']} (dim: {GLOBAL_DATA['source_dim']}) [pre-computed embeddings]")
        print(f"  Target: {GLOBAL_DATA['peak_arch_type'].upper()} (dim: {peak_hidden_dim})")
        print(f"  Shared: {shared_dim}, lr={lr:.2e}, wd={weight_decay:.2e}")
        print(f"  λ_domain={lambda_domain:.2f}, λ_class={lambda_class:.2f}, batch={batch_size}")

        # Create dual encoder model
        if GLOBAL_DATA["dynamic_embeddings"]:
            actual_source_encoder = GLOBAL_DATA["source_model"]

            source_dim_map = {
                'scgpt': 512,
                'geneformer-10m': 256,
                'geneformer-104m': 768,
                'geneformer-104m-clcancer': 768,
                'geneformer-316m': 1152,
                'uce-100m': 768,
                'teddy-70m': 512,
                'teddy-160m': 768,
                'teddy-400m': 1024,
                'scfoundation': 768
            }
            actual_source_dim = source_dim_map.get(GLOBAL_DATA["source_model_name"].replace("_dynamic", ""), 512)
            freeze_source = (GLOBAL_DATA["unfreeze_last_n_layers"] == 0)
        else:
            actual_source_encoder = IdentityEncoder()
            actual_source_dim = GLOBAL_DATA["source_dim"]
            freeze_source = False

        model = DualEncoderDANN(
            source_encoder=actual_source_encoder,
            target_encoder=target_model,
            source_dim=actual_source_dim,
            target_dim=peak_hidden_dim,
            shared_dim=shared_dim,
            num_classes=GLOBAL_DATA["num_classes"],
            disc_hidden=disc_hidden,
            disc_layers=disc_layers,
            cls_hidden=cls_hidden,
            cls_layers=cls_layers,
            dropout=dropout,
            use_spectral_norm=use_spectral_norm,
            freeze_source_encoder=freeze_source,
            use_cdann=use_cdann
        ).to(GLOBAL_DATA["device"])

        # Create optimizer
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )

        # Train
        model = train_dual_encoder_trial(
            model=model,
            emb_src_train=GLOBAL_DATA["emb_train"],
            y_src_train=GLOBAL_DATA["y_train"],
            X_tgt=GLOBAL_DATA["X_tgt"],
            y_tgt_train=GLOBAL_DATA["y_tgt"],
            epochs=epochs,
            batch_size=batch_size,
            lambda_domain=lambda_domain,
            lambda_class=lambda_class,
            lambda_target_class=lambda_target_class,  # Use Optuna-tuned value
            lambda_target_class_warmup=lambda_target_class_warmup,  # Use Optuna-tuned warmup
            optimizer=optimizer,
            device=GLOBAL_DATA["device"],
            use_grl_annealing=use_grl_annealing,
            grl_gamma=grl_gamma,
            use_contrastive_loss=use_contrastive_loss,
            lambda_contrastive=lambda_contrastive,
            contrastive_temperature=contrastive_temperature,
            lambda_balance=lambda_balance,
            use_balance_loss=use_balance_loss,
            source_class_prior=source_class_prior,
            lambda_entropy=lambda_entropy,
            use_entropy_loss=use_entropy_loss,
            confidence_threshold=confidence_threshold,
            divergence_type=divergence_type,
            tokenizer=GLOBAL_DATA.get("tokenizer"),
            src_genes=GLOBAL_DATA["src_genes"] if GLOBAL_DATA["dynamic_embeddings"] else None,
            loader=GLOBAL_DATA.get("loader"),
            trial=trial,
            emb_src_val=GLOBAL_DATA.get("emb_val"),
            y_src_val=GLOBAL_DATA.get("y_val"),
            patience=100,
            X_tgt_val=GLOBAL_DATA.get("X_tgt_val"),
            y_tgt_val=GLOBAL_DATA.get("y_tgt_val")
        )

        # Evaluate on validation set
        if GLOBAL_DATA["use_validation_split"]:
            val_acc_source = evaluate_dual_encoder(
                model=model,
                emb_data=GLOBAL_DATA["emb_val"],
                y_data=GLOBAL_DATA["y_val"],
                batch_size=batch_size,
                device=GLOBAL_DATA["device"],
                domain='source',
                tokenizer=GLOBAL_DATA.get("tokenizer"),
                src_genes=GLOBAL_DATA["src_genes"] if GLOBAL_DATA["dynamic_embeddings"] else None,
                loader=GLOBAL_DATA.get("loader")
            )

            # Also evaluate on train to check overfitting
            train_acc = evaluate_dual_encoder(
                model=model,
                emb_data=GLOBAL_DATA["emb_train"],
                y_data=GLOBAL_DATA["y_train"],
                batch_size=batch_size,
                device=GLOBAL_DATA["device"],
                domain='source',
                tokenizer=GLOBAL_DATA.get("tokenizer"),
                src_genes=GLOBAL_DATA["src_genes"] if GLOBAL_DATA["dynamic_embeddings"] else None,
                loader=GLOBAL_DATA.get("loader")
            )

            # CRITICAL: Evaluate on TARGET VALIDATION split (held-out, never seen during training!)
            val_acc_target = None
            if GLOBAL_DATA.get("y_tgt_val") is not None:
                # Use held-out target validation split (proper evaluation)
                val_acc_target = evaluate_dual_encoder(
                    model=model,
                    emb_data=GLOBAL_DATA["X_tgt_val"],
                    y_data=GLOBAL_DATA["y_tgt_val"],
                    batch_size=batch_size,
                    device=GLOBAL_DATA["device"],
                    domain='target'
                )
                # Use TARGET accuracy as primary metric (this is what we care about!)
                val_acc = val_acc_target  # Optimize for target PEAK accuracy, not source
                print(f"[Trial {trial.number}] Train: {train_acc:.4f}, Val(src): {val_acc_source:.4f}, Val(tgt): {val_acc_target:.4f} ← PRIMARY")
            else:
                # No target validation labels - use source only (fallback)
                val_acc = val_acc_source
                print(f"[Trial {trial.number}] Train: {train_acc:.4f}, Val: {val_acc:.4f}")

            return val_acc
        else:
            # Fallback to training accuracy
            train_acc = evaluate_dual_encoder(
                model=model,
                emb_data=GLOBAL_DATA["emb_train"],
                y_data=GLOBAL_DATA["y_train"],
                batch_size=batch_size,
                device=GLOBAL_DATA["device"],
                domain='source'
            )
            print(f"[Trial {trial.number}] Train: {train_acc:.4f} (no validation)")
            return train_acc

    except Exception as e:
        print(f"[Trial {trial.number}] FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="Dual-Encoder DANN Hyperparameter Search with Optuna")

    # Data
    parser.add_argument("--src_csv", type=str, required=True, help="Source CSV (labeled)")
    parser.add_argument("--tgt_csv", type=str, required=True, help="Target CSV (unlabeled)")
    parser.add_argument("--tgt_label_csv", type=str, default=None, help="Target labels CSV (paired GEX for semi-supervised training)")
    parser.add_argument("--peak_filter_threshold", type=float, default=0.10,
                       help="Filter peaks appearing in <X fraction of cells (0.10 = keep peaks in >10%% cells). Set 0 to disable. Reduces noise in sparse ATAC data.")
    parser.add_argument("--source_model", type=str, required=True,
                       choices=['scgpt', 'geneformer-10m', 'geneformer-104m', 'geneformer-104m-clcancer', 'geneformer-316m',
                                'uce-100m', 'teddy-70m', 'teddy-160m', 'teddy-400m', 'scfoundation'],
                       help="Source domain LLM model")

    # Peak architecture settings
    parser.add_argument("--peak_arch_type", type=str, default="mlp",
                       choices=["mlp", "periodic_mlp", "vae", "dae", "contrastive", "hybrid",
                                "gnn", "gnn_gcn", "cnn", "cnn_multiscale", "cnn_dilated"],
                       help="Peak embedder architecture type")

    # Model settings
    parser.add_argument("--freeze_source", action="store_true", help="Freeze source encoder")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for source model")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")

    # Ensemble settings
    parser.add_argument("--use_ensemble", action="store_true", help="Enable ensemble source encoder (combines two LLM models)")
    parser.add_argument("--ensemble_model1", type=str, default="geneformer-104m",
                       choices=['scgpt', 'geneformer-10m', 'geneformer-104m', 'geneformer-104m-clcancer', 'geneformer-316m',
                                'uce-100m', 'teddy-70m', 'teddy-160m', 'teddy-400m', 'scfoundation'],
                       help="First LLM model for ensemble (only used if --use_ensemble)")
    parser.add_argument("--ensemble_model2", type=str, default="scfoundation",
                       choices=['scgpt', 'geneformer-10m', 'geneformer-104m', 'geneformer-104m-clcancer', 'geneformer-316m',
                                'uce-100m', 'teddy-70m', 'teddy-160m', 'teddy-400m', 'scfoundation'],
                       help="Second LLM model for ensemble (only used if --use_ensemble)")

    # Dynamic embeddings and fine-tuning
    parser.add_argument("--dynamic_embeddings", action="store_true", help="Use dynamic embeddings (compute on-the-fly during training). Enables fine-tuning but slower and uses more memory.")
    parser.add_argument("--unfreeze_last_n_layers", type=int, default=0, help="Unfreeze last N transformer layers for fine-tuning (only works with --dynamic_embeddings). 0=fully frozen, 1-2 recommended.")

    # DANN improvements
    parser.add_argument("--shared_dim", type=int, default=768, help="Shared latent space dimension (default: 768 for rich alignment)")
    parser.add_argument("--use_cdann", action="store_true", default=True, help="Use Class-Conditional DANN (CDANN) for class-wise domain alignment (default: True)")
    parser.add_argument("--use_contrastive_loss", action="store_true", default=True, help="Use contrastive loss for cross-modality class semantics (default: True)")
    parser.add_argument("--contrastive_temperature", type=float, default=0.1, help="Temperature for contrastive loss (default: 0.1)")
    parser.add_argument("--lambda_contrastive", type=float, default=0.5, help="Weight for contrastive loss (default: 0.5)")
    parser.add_argument("--use_balance_loss", action="store_true", default=True, help="Use target class balance loss to prevent collapse (default: True)")
    parser.add_argument("--lambda_balance", type=float, default=5.0, help="Weight for target class balance loss (default: 5.0, increased from 0.5 which was too weak)")
    parser.add_argument("--divergence_type", type=str, default='kl', choices=['kl', 'jensen', 'coral', 'mmd'], help="Divergence measure for balance loss: kl=KL divergence, jensen=Jensen-Shannon, coral=CORAL (feature-level), mmd=MMD (feature-level) (default: kl)")
    parser.add_argument("--use_entropy_loss", action="store_true", default=True, help="Use entropy minimization for confident target predictions (ChatGPT Option 4, default: True)")
    parser.add_argument("--lambda_entropy", type=float, default=0.1, help="Weight for entropy minimization loss (default: 0.1)")
    parser.add_argument("--lambda_target_class", type=float, default=1.0, help="Weight for target classification loss when using paired labels (default: 1.0)")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="Only minimize entropy for predictions with max_prob > threshold (default: 0.5)")
    parser.add_argument("--use_grl_annealing", action="store_true", default=True, help="Use GRL lambda annealing for gradual adversarial adaptation (default: True)")
    parser.add_argument("--grl_gamma", type=float, default=15.0, help="Gamma for GRL annealing schedule (default: 15.0, slower warmup reduces early collapse)")

    # Optuna settings
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per trial")
    parser.add_argument("--max_target_cells", type=int, default=None, help="Max target cells to use")
    parser.add_argument("--no_prune", action="store_true", help="Disable Optuna pruning (use NopPruner for sanity check - all trials run to completion)")

    # ANTI-OVERFITTING: Validation split
    parser.add_argument("--use_validation_split", action="store_true", help="Use train/val split (anti-overfitting)")
    parser.add_argument("--validation_split", type=float, default=0.2, help="Validation split ratio (default 0.2)")
    parser.add_argument("--tgt_validation_split", type=float, default=0.2, help="Target validation split ratio for semi-supervised training (default 0.2)")

    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")

    args = parser.parse_args()

    if args.unfreeze_last_n_layers > 0 and not args.dynamic_embeddings:
        raise ValueError("--unfreeze_last_n_layers requires --dynamic_embeddings to be enabled")

    if args.use_ensemble and args.dynamic_embeddings:
        raise ValueError("--use_ensemble is not compatible with --dynamic_embeddings (ensemble requires pre-computed embeddings)")

    # Prepare data once
    prepare_data_once(args)

    # Create output directory
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    # Create Optuna study with appropriate pruner
    if args.no_prune:
        print("⚠️  PRUNING DISABLED: Using NopPruner (all trials run to completion)")
        pruner = NopPruner()
    else:
        pruner = MedianPruner(
            n_startup_trials=30,    # Keep first 30 trials
            n_warmup_steps=25,      # Increased from 15 - more room within each trial
            interval_steps=5        # Prune every 5 steps after warmup
        )

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=pruner
    )

    print("\n" + "="*60)
    print(f"STARTING DUAL-ENCODER HYPERPARAMETER SEARCH ({args.n_trials} trials)")
    print("="*60 + "\n")

    # Run optimization
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # Save results
    print("\n" + "="*60)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("="*60)

    best_params = study.best_params
    best_value = study.best_value

    print(f"\nBest accuracy: {best_value:.4f}")
    print("Best hyperparameters:")
    for key, val in best_params.items():
        print(f"  {key}: {val}")

    # Save to JSON
    results = {
        "best_params": best_params,
        "best_value": float(best_value),
        "n_trials": args.n_trials,
        "source_model": args.source_model,
        "peak_arch_type": args.peak_arch_type,
    }

    output_file = output_dir / "best_hyperparams.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
