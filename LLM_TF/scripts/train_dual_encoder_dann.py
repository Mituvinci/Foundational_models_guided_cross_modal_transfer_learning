"""
Training script for Dual-Encoder DANN architecture.

This integrates the dual-encoder framework with existing foundation models and peak embedders.

Usage:
    python -m LLM_TF.scripts.train_dual_encoder_dann \
        --source_model geneformer-10m \
        --peak_arch_type gnn \
        --src_csv data/source.csv \
        --tgt_csv data/target.csv \
        --test_csv data/test.csv \
        --shared_dim 512 \
        --epochs 100 \
        --freeze_source
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, average_precision_score
from scipy.stats import pearsonr, spearmanr
import json
from typing import Optional
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from LLM_TF.dual_encoder_dann import DualEncoderDANN, compute_dual_encoder_loss
from LLM_TF.embedders.unified_embedder import UnifiedEmbedder
import hashlib
import pickle


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


class IdentityEncoder(nn.Module):
    """Pass-through encoder for pre-computed embeddings."""
    def forward(self, x):
        return x


class TokenizedSourceDataset(torch.utils.data.Dataset):
    """Dataset that tokenizes gene expression data for dynamic embeddings."""

    def __init__(self, X_src, y_src, gene_names, tokenizer, loader=None):
        """
        Args:
            X_src: numpy array (n_cells, n_genes) of gene expression
            y_src: numpy array (n_cells,) of labels
            gene_names: list or array of gene names
            tokenizer: tokenizer from embedder
            loader: GeneformerLoader or similar (for Geneformer tokenization)
        """
        self.y_src = torch.from_numpy(y_src).long()

        # Convert gene_names to numpy array if it's a list
        if isinstance(gene_names, list):
            gene_names = np.array(gene_names)

        # Detect if we're using Geneformer or scGPT-style tokenization
        if loader is not None and hasattr(loader, 'tokenize_expression'):
            # Geneformer-style tokenization (rank-based, returns token_ids only)
            self.is_geneformer = True
            token_ids_list = loader.tokenize_expression(X_src, gene_names, top_k=2048)

            # Get pad_token_id (default to 0 if not set)
            pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None else 0

            # Pad to max length
            max_len = max(len(ids) for ids in token_ids_list)
            self.token_ids = []
            self.attention_masks = []

            for ids in token_ids_list:
                pad_len = max_len - len(ids)
                padded_ids = ids + [pad_token_id] * pad_len
                mask = [1] * len(ids) + [0] * pad_len

                self.token_ids.append(torch.tensor(padded_ids, dtype=torch.long))
                self.attention_masks.append(torch.tensor(mask, dtype=torch.long))

            self.token_ids = torch.stack(self.token_ids)
            self.attention_masks = torch.stack(self.attention_masks)

        else:
            # scGPT-style tokenization (expression values + token IDs)
            self.is_geneformer = False
            from LLM_TF.embedders.embedder import tokenize_matrix
            self.tokens = tokenize_matrix(X_src, gene_names, tokenizer)

    def __len__(self):
        return len(self.y_src)

    def __getitem__(self, idx):
        label = self.y_src[idx]

        if self.is_geneformer:
            return self.token_ids[idx], self.attention_masks[idx], label
        else:
            return (
                self.tokens['input_ids'][idx],
                self.tokens['values'][idx],
                self.tokens['attention_mask'][idx],
                label
            )


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


def load_and_preprocess_data(src_csv, tgt_csv, test_csv=None, peak_filter_threshold=0.0, tgt_label_csv=None, sup_peak_csv=None, sup_peak_label_csv=None, use_peak_mapper=False, tgt_validation_split=0.2):
    """
    Load and preprocess source, target, optional test data, and optional SUP PEAK data.

    Args:
        tgt_validation_split: Fraction of target data to hold out for validation (default: 0.2 = 20%)
                              If > 0 and tgt_label_csv provided, splits REH PEAK into train/val to prevent data leakage
    """
    print("\n" + "="*60)
    print("DATA LOADING")
    print("="*60)

    src_df = pd.read_csv(src_csv, index_col=0)
    tgt_df = pd.read_csv(tgt_csv, index_col=0)

    src_cell_ids = src_df.index.tolist()
    tgt_cell_ids = tgt_df.index.tolist()

    print(f"Loaded {len(src_cell_ids)} source cells, {len(tgt_cell_ids)} target cells")

    # Load target labels from paired GEX file if provided
    y_tgt_encoded = None
    if tgt_label_csv:
        print(f"\n🎯 Loading target labels from paired GEX file: {tgt_label_csv}")
        tgt_label_df = pd.read_csv(tgt_label_csv, index_col=0)
        label_candidates = [col for col in tgt_label_df.columns if 'phase' in col.lower() or 'label' in col.lower()]
        tgt_label_col = label_candidates[0] if label_candidates else 'phase'
        print(f"  Label column: '{tgt_label_col}'")

    label_candidates = [col for col in src_df.columns if 'phase' in col.lower() or 'label' in col.lower()]
    label_col = label_candidates[0] if label_candidates else 'phase'

    print(f"Label column: '{label_col}'")
    print(f"Unique labels: {src_df[label_col].unique()}")

    src_genes = [col for col in src_df.columns if col != label_col]
    tgt_genes = list(tgt_df.columns)

    y_src = src_df[label_col].values
    X_src = src_df[src_genes].values.astype(np.float32)
    X_tgt = tgt_df[tgt_genes].values.astype(np.float32)

    # Filter PEAK features by frequency (if threshold > 0)
    if peak_filter_threshold > 0:
        print(f"\nFiltering PEAK features (threshold={peak_filter_threshold*100:.0f}% of cells)...")
        X_tgt, tgt_genes = filter_peaks_by_frequency(X_tgt, tgt_genes, threshold=peak_filter_threshold)

    # Apply consistent normalization to both modalities
    print(f"\nNormalizing source RNA data (GEX)...")
    X_src = normalize_counts(X_src, modality_name="Source RNA (GEX)")

    print(f"\nNormalizing target PEAK data (ATAC)...")
    X_tgt = normalize_counts(X_tgt, modality_name="Target PEAK (ATAC)")

    label_map = {label: idx for idx, label in enumerate(sorted(set(y_src)))}
    y_src_encoded = np.array([label_map[label] for label in y_src])

    print(f"\nLabel mapping: {label_map}")
    print(f"Source class distribution: {np.bincount(y_src_encoded)}")

    # Match target labels to target cell_ids
    if tgt_label_csv and y_tgt_encoded is None:
        # Match cell_ids between target PEAK and GEX labels
        tgt_label_df_matched = tgt_label_df.loc[tgt_cell_ids]
        y_tgt = tgt_label_df_matched[tgt_label_col].values
        y_tgt_encoded = np.array([label_map[label] for label in y_tgt])
        print(f"Target class distribution: {np.bincount(y_tgt_encoded)}")
        print(f"✓ Matched {len(y_tgt_encoded)} target labels to PEAK cells via cell_id")

    # 80/20 split for target validation (prevent data leakage)
    tgt_val_data = None
    if tgt_label_csv and y_tgt_encoded is not None and tgt_validation_split > 0:
        print(f"\n{'='*60}")
        print(f" 80/20 TARGET SPLIT (Preventing Data Leakage)")
        print(f"{'='*60}")
        print(f"  Splitting REH PEAK: {1-tgt_validation_split:.0%} train, {tgt_validation_split:.0%} held-out validation")

        # Split target data
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(tgt_cell_ids))
        idx_train, idx_val = train_test_split(
            indices, test_size=tgt_validation_split, random_state=42, stratify=y_tgt_encoded
        )

        # Split matrices and labels
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

        # Replace full target data with training split
        X_tgt = X_tgt_train
        y_tgt_encoded = y_tgt_train
        tgt_cell_ids = tgt_cell_ids_train

        # Store validation split
        tgt_val_data = (X_tgt_val, y_tgt_val, tgt_genes, tgt_cell_ids_val)

    test_data = None
    if test_csv:
        test_df = pd.read_csv(test_csv, index_col=0)
        test_cell_ids = test_df.index.tolist()
        test_genes = [col for col in test_df.columns if col != label_col]
        y_test = test_df[label_col].values
        X_test = test_df[test_genes].values.astype(np.float32)

        print(f"\nNormalizing test RNA data (GEX)...")
        X_test = normalize_counts(X_test, modality_name="Test RNA (GEX)")

        y_test_encoded = np.array([label_map[label] for label in y_test])
        test_data = (X_test, y_test_encoded, test_genes, test_cell_ids)
        print(f"Loaded {len(test_cell_ids)} test cells")

    # Load SUP PEAK data for additional target evaluation
    sup_peak_data = None
    if sup_peak_csv:
        print(f"\n🎯 Loading SUP PEAK data: {sup_peak_csv}")
        sup_peak_df = pd.read_csv(sup_peak_csv, index_col=0)
        sup_peak_cell_ids = sup_peak_df.index.tolist()
        sup_peak_genes = list(sup_peak_df.columns)
        X_sup_peak = sup_peak_df[sup_peak_genes].values.astype(np.float32)

        # Filter PEAK features by frequency (same as target)
        if peak_filter_threshold > 0:
            print(f"  Filtering SUP PEAK features (threshold={peak_filter_threshold*100:.0f}% of cells)...")
            X_sup_peak, sup_peak_genes = filter_peaks_by_frequency(X_sup_peak, sup_peak_genes, threshold=peak_filter_threshold)

        # Align SUP PEAK features to match training PEAK features
        if use_peak_mapper:
            print(f"  🗺️ Using coordinate-based peak mapper for alignment...")
            try:
                from LLM_TF.peak_mapper.coordinate_mapper import PeakCoordinateMapper
                from LLM_TF.peak_mapper.imputer import PeakImputer

                # Parse peak coordinates (format: chr1:12345-67890)
                mapper = PeakCoordinateMapper(method='overlap_50pct')
                mapping = mapper.map_peaks(sup_peak_genes, tgt_genes)

                # Align matrix based on mapping
                aligned_X = np.zeros((X_sup_peak.shape[0], len(tgt_genes)), dtype=np.float32)
                for tgt_idx, src_indices in mapping.items():
                    if src_indices:
                        # Average values from overlapping source peaks
                        aligned_X[:, tgt_idx] = X_sup_peak[:, src_indices].mean(axis=1)

                # Impute missing values (peaks with no overlap)
                imputer = PeakImputer(strategy='zero')  # Use zero imputation for now
                X_sup_peak = imputer.transform(aligned_X, tgt_genes)

                overlap_count = sum(1 for v in mapping.values() if v)
                print(f"  ✓ Peak mapper: {len(tgt_genes)} peaks ({overlap_count} mapped, {len(tgt_genes)-overlap_count} imputed)")

            except Exception as e:
                print(f"  ⚠️ Peak mapper failed: {e}")
                print(f"  Falling back to zero-filling...")
                use_peak_mapper = False  # Fall back to zero-filling

        if not use_peak_mapper:
            # Zero-filling (fast, simple)
            print(f"  Aligning SUP PEAK to training peaks ({len(tgt_genes)} peaks)...")
            sup_peak_dict = {peak: idx for idx, peak in enumerate(sup_peak_genes)}
            aligned_X = np.zeros((X_sup_peak.shape[0], len(tgt_genes)), dtype=np.float32)

            overlap_count = 0
            for i, peak in enumerate(tgt_genes):
                if peak in sup_peak_dict:
                    aligned_X[:, i] = X_sup_peak[:, sup_peak_dict[peak]]
                    overlap_count += 1
                # else: leave as zeros (already initialized)

            X_sup_peak = aligned_X
            print(f"  ✓ Aligned: {len(tgt_genes)} peaks ({overlap_count} matched, {len(tgt_genes)-overlap_count} filled with zeros)")

        sup_peak_genes = tgt_genes

        # Normalize SUP PEAK data
        print(f"  Normalizing SUP PEAK data (ATAC)...")
        X_sup_peak = normalize_counts(X_sup_peak, modality_name="SUP PEAK (ATAC)")

        # Load paired GEX labels for SUP PEAK
        y_sup_peak_encoded = None
        if sup_peak_label_csv:
            print(f"  Loading SUP PEAK labels from paired GEX: {sup_peak_label_csv}")
            sup_peak_label_df = pd.read_csv(sup_peak_label_csv, index_col=0)
            label_candidates_sup = [col for col in sup_peak_label_df.columns if 'phase' in col.lower() or 'label' in col.lower()]
            sup_peak_label_col = label_candidates_sup[0] if label_candidates_sup else 'phase'

            # Match cell_ids between SUP PEAK and SUP GEX labels
            sup_peak_label_df_matched = sup_peak_label_df.loc[sup_peak_cell_ids]
            y_sup_peak = sup_peak_label_df_matched[sup_peak_label_col].values
            y_sup_peak_encoded = np.array([label_map[label] for label in y_sup_peak])
            print(f"  SUP PEAK class distribution: {np.bincount(y_sup_peak_encoded)}")
            print(f"  ✓ Matched {len(y_sup_peak_encoded)} SUP PEAK labels to PEAK cells via cell_id")

        sup_peak_data = (X_sup_peak, y_sup_peak_encoded, sup_peak_genes, sup_peak_cell_ids)
        print(f"✓ Loaded {len(sup_peak_cell_ids)} SUP PEAK cells")

    return (X_src, y_src_encoded, src_genes, src_cell_ids,
            X_tgt, y_tgt_encoded, tgt_genes, tgt_cell_ids,
            label_map, test_data, sup_peak_data, tgt_val_data)


def create_dataloaders(emb_src_train, y_src_train, X_tgt, batch_size, device, y_tgt_train=None):
    """Create PyTorch DataLoaders for training (source uses pre-computed embeddings)."""
    emb_src = emb_src_train if isinstance(emb_src_train, torch.Tensor) else torch.from_numpy(emb_src_train)
    y_src = y_src_train if isinstance(y_src_train, torch.Tensor) else torch.from_numpy(y_src_train)
    X_tgt_t = X_tgt if isinstance(X_tgt, torch.Tensor) else torch.from_numpy(X_tgt)

    src_dataset = TensorDataset(
        emb_src.float(),
        y_src.long()
    )

    # Include target labels if available (for semi-supervised training)
    if y_tgt_train is not None:
        y_tgt = y_tgt_train if isinstance(y_tgt_train, torch.Tensor) else torch.from_numpy(y_tgt_train)
        tgt_dataset = TensorDataset(
            X_tgt_t.float(),
            y_tgt.long()
        )
    else:
        tgt_dataset = TensorDataset(
            X_tgt_t.float()
        )

    src_loader = DataLoader(src_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    tgt_loader = DataLoader(tgt_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return src_loader, tgt_loader


def train_epoch(model, src_loader, tgt_loader, optimizer, epoch, total_epochs, lambda_domain, lambda_class, device, use_grl_annealing=False, grl_gamma=10.0, use_contrastive_loss=False, lambda_contrastive=0.5, contrastive_temperature=0.1, lambda_balance=0.5, use_balance_loss=False, source_class_prior=None, lambda_entropy=0.1, use_entropy_loss=False, confidence_threshold=0.5, divergence_type='kl', lambda_target_class=1.0):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_class_loss = 0
    total_target_class_loss = 0
    total_domain_loss = 0
    n_batches = 0

    if use_grl_annealing:
        alpha = model.get_grl_lambda(epoch, total_epochs, gamma=grl_gamma)
    else:
        alpha = 1.0

    src_iter = iter(src_loader)
    tgt_iter = iter(tgt_loader)

    pbar = tqdm(range(min(len(src_loader), len(tgt_loader))), desc=f"Epoch {epoch+1}/{total_epochs}")

    for batch_idx in pbar:
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

        # Handle tokenized data (dynamic embeddings) vs pre-computed embeddings
        if len(src_batch) == 3:
            # Geneformer-style: (token_ids, attention_mask, label)
            token_ids, attention_mask, src_labels = src_batch
            src_data = {
                'input_ids': token_ids.to(device),
                'attention_mask': attention_mask.to(device)
            }
            src_labels = src_labels.to(device)
        elif len(src_batch) == 4:
            # scGPT-style: (token_ids, values, attention_mask, label)
            token_ids, values, attention_mask, src_labels = src_batch
            src_data = {
                'input_ids': token_ids.to(device),
                'values': values.to(device),
                'attention_mask': attention_mask.to(device)
            }
            src_labels = src_labels.to(device)
        else:
            # Pre-computed embeddings: (embeddings, label)
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
            lambda_target_class=lambda_target_class if tgt_labels is not None else 0.0,
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
        total_class_loss += loss_dict.get('classification', 0)
        total_target_class_loss += loss_dict.get('target_classification', 0)
        total_domain_loss += (loss_dict.get('domain_source', 0) + loss_dict.get('domain_target', 0)) / 2
        n_batches += 1

        postfix_dict = {
            'loss': f"{loss.item():.4f}",
            'cls': f"{loss_dict.get('classification', 0):.4f}",
            'dom': f"{(loss_dict.get('domain_source', 0) + loss_dict.get('domain_target', 0))/2:.4f}",
            'alpha': f"{alpha:.3f}"
        }
        if tgt_labels is not None:
            postfix_dict['tgt_cls'] = f"{loss_dict.get('target_classification', 0):.4f}"
        pbar.set_postfix(postfix_dict)

    avg_loss = total_loss / n_batches if n_batches > 0 else 0
    avg_class_loss = total_class_loss / n_batches if n_batches > 0 else 0
    avg_domain_loss = total_domain_loss / n_batches if n_batches > 0 else 0

    return avg_loss, avg_class_loss, avg_domain_loss, alpha


def evaluate(model, emb_data, y_data=None, batch_size=32, device='cuda', domain='source', tokenizer=None, loader=None, gene_names=None):
    """
    Evaluate model on given data (expects embeddings for source, raw peaks for target).

    Args:
        y_data: Optional labels. If None, only predictions are returned without metrics.
    """
    model.eval()

    has_labels = y_data is not None

    # If loader is provided (for dynamic embeddings), use TokenizedSourceDataset
    # Note: Geneformer has tokenizer=None but loader is set
    if loader is not None and domain == 'source':
        dataset = TokenizedSourceDataset(
            X_src=emb_data,
            y_src=y_data,
            gene_names=gene_names,
            tokenizer=tokenizer,
            loader=loader
        )
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        is_tokenized = True
    else:
        # Use regular tensor dataset for pre-computed embeddings
        emb_tensor = emb_data if isinstance(emb_data, torch.Tensor) else torch.from_numpy(emb_data)
        if has_labels:
            y_tensor = y_data if isinstance(y_data, torch.Tensor) else torch.from_numpy(y_data)
            dataset = TensorDataset(
                emb_tensor.float(),
                y_tensor.long()
            )
        else:
            dataset = TensorDataset(emb_tensor.float())
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        is_tokenized = False

    all_preds = []
    all_probs = []
    all_labels = [] if has_labels else None

    with torch.no_grad():
        for batch in data_loader:
            if is_tokenized:
                # Handle tokenized data
                if len(batch) == 3:
                    # Geneformer-style: (token_ids, attention_mask, label)
                    token_ids, attention_mask, batch_labels = batch
                    batch_data = {
                        'input_ids': token_ids.to(device),
                        'attention_mask': attention_mask.to(device)
                    }
                else:
                    # scGPT-style: (token_ids, values, attention_mask, label)
                    token_ids, values, attention_mask, batch_labels = batch
                    batch_data = {
                        'input_ids': token_ids.to(device),
                        'values': values.to(device),
                        'attention_mask': attention_mask.to(device)
                    }
            else:
                # Pre-computed embeddings
                if has_labels:
                    batch_data, batch_labels = batch
                    batch_data = batch_data.to(device)
                    batch_labels = batch_labels.to(device)
                else:
                    (batch_data,) = batch
                    batch_data = batch_data.to(device)

            if domain == 'source':
                outputs = model(source_data=batch_data, alpha=0.0)
                logits = outputs['source_class_pred']
            else:
                outputs = model(target_data=batch_data, alpha=0.0)
                logits = outputs['target_class_pred']

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            if has_labels:
                all_labels.append(batch_labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()

    results = {
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': None,
        'accuracy': None,
        'balanced_accuracy': None,
        'kappa': None
    }

    if has_labels:
        labels = torch.cat(all_labels).numpy()
        results['labels'] = labels
        results['accuracy'] = accuracy_score(labels, all_preds)
        results['balanced_accuracy'] = balanced_accuracy_score(labels, all_preds)
        results['kappa'] = cohen_kappa_score(labels, all_preds)

    return results


def evaluate_peak_reconstruction(model, peak_data, labels=None, batch_size=32, device='cuda'):
    """
    Evaluate peak reconstruction quality using scATAC-seq recommended metrics.

    Args:
        model: Trained DANN model
        peak_data: Peak accessibility matrix (cells x peaks)
        labels: Optional cell cycle labels (0=G1, 1=S, 2=G2M) for per-phase AUROC
        batch_size: Batch size for evaluation
        device: Device to use

    Computes:
        - AUROC: Overall + per-phase (if labels provided)
        - Precision/Recall/F1: Peak calling quality
        - Pearson/Spearman: Peak count correlation

    Returns dict with reconstruction metrics or None if architecture lacks decoder.
    """
    model.eval()

    # Peak encoder lives inside DualEncoderDANN as target_encoder
    peak_encoder = getattr(model, 'peak_encoder', None)
    if peak_encoder is None and hasattr(model, 'target_encoder'):
        peak_encoder = model.target_encoder

    # Check for decoder in base_embedder (DAE/VAE wrap the real encoder)
    if peak_encoder is None:
        return None

    # The actual encoder with decode() is in base_embedder
    actual_encoder = getattr(peak_encoder, 'base_embedder', peak_encoder)
    if not hasattr(actual_encoder, 'decode'):
        return None

    arch_type = actual_encoder.__class__.__name__
    all_reconstructed = []

    with torch.no_grad():
        peak_tensor = torch.from_numpy(peak_data).float().to(device)
        peak_dataset = torch.utils.data.TensorDataset(peak_tensor)
        peak_loader = torch.utils.data.DataLoader(peak_dataset, batch_size=batch_size, shuffle=False)

        for (batch_X,) in peak_loader:
            # Encode to latent space
            if arch_type == 'VAEPeakEncoder':
                encoded, _, _ = actual_encoder(batch_X)  # VAE returns (z, mu, logvar)
            else:
                encoded = actual_encoder(batch_X)  # DAE/others return z directly

            # Decode back to peak space
            reconstructed = actual_encoder.decode(encoded)
            all_reconstructed.append(reconstructed.cpu())

    # Concatenate batches
    reconstructed = torch.cat(all_reconstructed, dim=0).numpy()
    true_peaks = peak_data

    # Binarize for AUROC/Precision/Recall
    true_peaks_binary = (true_peaks > 0).astype(int)

    # Normalize reconstructed peaks to [0, 1] for probability scores
    reconstructed_normalized = np.zeros_like(reconstructed)
    for j in range(reconstructed.shape[1]):
        recon_peak = reconstructed[:, j]
        if recon_peak.max() > recon_peak.min():
            reconstructed_normalized[:, j] = (recon_peak - recon_peak.min()) / (recon_peak.max() - recon_peak.min())
        else:
            reconstructed_normalized[:, j] = 0.5

    # Compute overall AUROC
    overall_auroc = None
    try:
        true_flat = true_peaks_binary.flatten()
        recon_flat = reconstructed_normalized.flatten()
        overall_auroc = roc_auc_score(true_flat, recon_flat)
    except:
        pass

    # Compute Precision/Recall/F1
    overall_precision = None
    overall_recall = None
    overall_f1 = None
    overall_avg_precision = None

    try:
        true_flat = true_peaks_binary.flatten()
        pred_flat = (reconstructed_normalized.flatten() > 0.5).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            true_flat, pred_flat, average='binary', zero_division=0
        )
        overall_precision = precision
        overall_recall = recall
        overall_f1 = f1

        # Average precision (AUPRC)
        overall_avg_precision = average_precision_score(true_flat, reconstructed_normalized.flatten())
    except:
        pass

    # Compute per-phase AUROC (if labels provided)
    auroc_g1 = None
    auroc_s = None
    auroc_g2m = None

    if labels is not None:
        try:
            # Compute AUROC for each cell cycle phase separately
            for phase_idx, phase_name in [(0, 'G1'), (1, 'S'), (2, 'G2M')]:
                phase_mask = labels == phase_idx
                if phase_mask.sum() > 0:
                    true_phase = true_peaks_binary[phase_mask].flatten()
                    recon_phase = reconstructed_normalized[phase_mask].flatten()

                    # Skip if all same class
                    if true_phase.sum() > 0 and true_phase.sum() < len(true_phase):
                        phase_auroc = roc_auc_score(true_phase, recon_phase)
                        if phase_idx == 0:
                            auroc_g1 = phase_auroc
                        elif phase_idx == 1:
                            auroc_s = phase_auroc
                        elif phase_idx == 2:
                            auroc_g2m = phase_auroc
        except:
            pass

    # Compute per-cell Pearson & Spearman correlation (sample subset to avoid slowdown)
    cell_pearson_mean = None
    cell_spearman_mean = None

    try:
        n_samples = min(500, len(true_peaks))  # Sample up to 500 cells
        sample_indices = np.random.choice(len(true_peaks), n_samples, replace=False)
        cell_pearson_corrs = []
        cell_spearman_corrs = []

        for i in sample_indices:
            if true_peaks[i].sum() > 0 and reconstructed[i].sum() > 0:
                try:
                    # Pearson (linear correlation)
                    p_corr, _ = pearsonr(true_peaks[i], reconstructed[i])
                    if not np.isnan(p_corr):
                        cell_pearson_corrs.append(p_corr)

                    # Spearman (rank-based correlation)
                    s_corr, _ = spearmanr(true_peaks[i], reconstructed[i])
                    if not np.isnan(s_corr):
                        cell_spearman_corrs.append(s_corr)
                except:
                    pass

        if len(cell_pearson_corrs) > 0:
            cell_pearson_mean = np.mean(cell_pearson_corrs)
        if len(cell_spearman_corrs) > 0:
            cell_spearman_mean = np.mean(cell_spearman_corrs)
    except:
        pass

    results = {
        'auroc': overall_auroc,
        'auroc_g1': auroc_g1,
        'auroc_s': auroc_s,
        'auroc_g2m': auroc_g2m,
        'avg_precision': overall_avg_precision,
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1,
        'cell_pearson_mean': cell_pearson_mean,
        'cell_spearman_mean': cell_spearman_mean
    }

    model.train()
    return results


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
        tgt_tensor = torch.from_numpy(X_tgt).float().to(device)
        tgt_dataset = torch.utils.data.TensorDataset(tgt_tensor)
        tgt_loader = torch.utils.data.DataLoader(tgt_dataset, batch_size=batch_size, shuffle=False)

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


def save_predictions(results, cell_ids, label_map, output_dir, prefix):
    """Save predictions to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inverse_label_map = {v: k for k, v in label_map.items()}

    has_labels = results['labels'] is not None

    # Build DataFrame with predictions
    pred_data = {
        'cell_id': cell_ids[:len(results['predictions'])],
        'predicted_label': [inverse_label_map[p] for p in results['predictions']],
        'prob_class_0': results['probabilities'][:, 0],
        'prob_class_1': results['probabilities'][:, 1],
        'prob_class_2': results['probabilities'][:, 2],
        'max_prob': results['probabilities'].max(axis=1),
    }

    # Add ground truth and correctness if labels are available
    if has_labels:
        pred_data['true_label'] = [inverse_label_map[l] for l in results['labels']]
        pred_data['correct'] = results['predictions'] == results['labels']
    else:
        pred_data['true_label'] = ['N/A'] * len(results['predictions'])

    pred_df = pd.DataFrame(pred_data)
    pred_df.to_csv(output_dir / f"{prefix}_predictions.csv", index=False)

    # Save metrics only if labels are available
    if has_labels:
        with open(output_dir / f"{prefix}_metrics.txt", 'w') as f:
            f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}\n")
            f.write(f"Kappa: {results['kappa']:.4f}\n\n")
            f.write("Classification Report:\n")
            sorted_labels = sorted(label_map.values())
            f.write(classification_report(results['labels'], results['predictions'],
                                         labels=sorted_labels,
                                         target_names=[inverse_label_map[i] for i in sorted_labels],
                                         zero_division=0))

    print(f"\nSaved {prefix} predictions to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train Dual-Encoder DANN')

    parser.add_argument('--src_csv', required=True, help='Source training CSV')
    parser.add_argument('--tgt_csv', required=True, help='Target training CSV')
    parser.add_argument('--tgt_label_csv', default=None, help='Target labels CSV (paired GEX for semi-supervised training)')
    parser.add_argument('--test_csv', default=None, help='Test CSV (optional)')
    parser.add_argument('--sup_peak_csv', default=None, help='SUP PEAK CSV for additional target evaluation (optional)')
    parser.add_argument('--sup_peak_label_csv', default=None, help='SUP PEAK labels CSV (paired GEX labels, optional)')
    parser.add_argument('--use_peak_mapper', action='store_true', default=False, help='Use coordinate-based peak mapper for SUP PEAK alignment (default: zero-filling)')
    parser.add_argument('--output_dir', required=True, help='Output directory')

    parser.add_argument('--peak_filter_threshold', type=float, default=0.10,
                       help='Filter peaks appearing in <X fraction of cells (0.10 = keep peaks in >10%% cells). Set 0 to disable. Reduces noise in sparse ATAC data.')

    parser.add_argument('--source_model', type=str, default='geneformer-10m',
                       choices=['scgpt', 'geneformer-10m', 'geneformer-104m', 'geneformer-104m-clcancer', 'geneformer-316m',
                                'uce-100m', 'teddy-70m', 'teddy-160m', 'teddy-400m', 'scfoundation'],
                       help='Source domain LLM model')

    parser.add_argument('--peak_arch_type', type=str, default='mlp',
                       choices=['mlp', 'periodic_mlp', 'vae', 'dae', 'contrastive', 'hybrid',
                                'gnn', 'gnn_gcn', 'cnn', 'cnn_multiscale', 'cnn_dilated'],
                       help='Target domain peak embedder architecture')

    parser.add_argument('--n_peaks', type=int, default=50000, help='Number of ATAC peaks')
    parser.add_argument('--peak_hidden_dim', type=int, default=256, help='Peak embedder output dimension')
    parser.add_argument('--peak_intermediate_dim', type=int, default=512, help='Peak embedder intermediate dimension')
    parser.add_argument('--peak_num_layers', type=int, default=3, help='Peak embedder number of layers')
    parser.add_argument('--peak_dropout', type=float, default=0.2, help='Peak embedder dropout rate')

    parser.add_argument('--shared_dim', type=int, default=768, help='Shared latent space dimension (default: 768 for rich alignment)')
    parser.add_argument('--disc_hidden', type=int, default=256, help='Domain discriminator hidden dimension')
    parser.add_argument('--disc_layers', type=int, default=3, help='Domain discriminator number of layers')
    parser.add_argument('--cls_hidden', type=int, default=256, help='Label classifier hidden dimension')
    parser.add_argument('--cls_layers', type=int, default=2, help='Label classifier number of layers')

    parser.add_argument('--dropout', type=float, default=0.1, help='General dropout rate')
    parser.add_argument('--use_spectral_norm', action='store_true', help='Use spectral normalization in discriminator')
    parser.add_argument('--freeze_source', action='store_true', help='Freeze source encoder')
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA for source model')
    parser.add_argument('--lora_rank', type=int, default=16, help='LoRA rank')

    parser.add_argument('--beta_vae', type=float, default=1.0, help='VAE beta parameter')
    parser.add_argument('--noise_level', type=float, default=0.1, help='DAE noise level')
    parser.add_argument('--temperature', type=float, default=0.1, help='Contrastive temperature')
    parser.add_argument('--aug_dropout_prob', type=float, default=0.2, help='Contrastive augmentation dropout')
    parser.add_argument('--projection_dim', type=int, default=128, help='Contrastive projection dimension')

    # NEW: Activation function and batch normalization (ALL architectures)
    parser.add_argument('--activation', type=str, default='relu',
                       choices=['relu', 'leaky_relu', 'elu', 'prelu', 'gelu', 'selu', 'swish', 'mish', 'periodic'],
                       help='Activation function for peak embedder (periodic for cell cycle periodicity)')
    parser.add_argument('--use_batch_norm', type=str, default='True', choices=['True', 'False'],
                       help='Use batch normalization in peak embedder')
    parser.add_argument('--residual_type', type=str, default='add', choices=['add', 'concat'],
                       help='Residual connection type (for hybrid architecture)')

    # CNN-specific parameters
    parser.add_argument('--num_conv_layers', type=int, default=3, help='Number of convolutional layers (CNN architectures)')
    parser.add_argument('--num_filters', type=int, default=128, help='Number of filters per conv layer (CNN architectures)')
    parser.add_argument('--kernel_size', type=int, default=5, help='Kernel size for convolutions (CNN architectures)')

    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--lambda_domain', type=float, default=1.0, help='Domain adaptation loss weight')
    parser.add_argument('--lambda_class', type=float, default=1.0, help='Classification loss weight')
    parser.add_argument('--lambda_target_class', type=float, default=1.0, help='Target classification loss weight (for semi-supervised training)')
    parser.add_argument('--lambda_target_class_warmup', type=int, default=100, help='Epochs to warmup target classification weight from 0 to lambda_target_class (default: 100, 0=no warmup)')
    parser.add_argument('--tgt_early_stop_patience', type=int, default=50, help='Stop if target accuracy does not improve for N evaluations (default: 50, 0=disabled)')

    parser.add_argument('--use_grl_annealing', action='store_true', default=True, help='Use GRL lambda annealing schedule (default: True, prevents early collapse)')
    parser.add_argument('--grl_gamma', type=float, default=15.0, help='GRL annealing rate (default: 15.0, slower warmup reduces early collapse)')
    parser.add_argument('--use_cdann', action='store_true', default=True, help='Use class-conditional discriminator (CDANN) for class-wise alignment (default: True)')

    parser.add_argument('--use_contrastive_loss', action='store_true', default=True, help='Enable contrastive loss for cross-modality class semantics (default: True)')
    parser.add_argument('--contrastive_temperature', type=float, default=0.1, help='Temperature for contrastive loss (lower=sharper clusters, range 0.05-0.5)')
    parser.add_argument('--lambda_contrastive', type=float, default=0.5, help='Contrastive loss weight')
    parser.add_argument('--use_balance_loss', action='store_true', default=True, help='Use target class balance loss to prevent collapse (default: True)')
    parser.add_argument('--lambda_balance', type=float, default=5.0, help='Weight for target class balance loss (default: 5.0, increased from 0.5 which was too weak)')
    parser.add_argument('--divergence_type', type=str, default='kl', choices=['kl', 'jensen', 'coral', 'mmd'], help='Divergence measure for balance loss: kl=KL divergence, jensen=Jensen-Shannon, coral=CORAL (feature-level), mmd=MMD (feature-level) (default: kl)')
    parser.add_argument('--use_entropy_loss', action='store_true', default=True, help='Use entropy minimization for confident target predictions (ChatGPT Option 4, default: True)')
    parser.add_argument('--lambda_entropy', type=float, default=0.1, help='Weight for entropy minimization loss (default: 0.1)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='Only minimize entropy for predictions with max_prob > threshold (default: 0.5)')

    parser.add_argument('--use_ensemble', action='store_true', help='Enable ensemble source encoder (combines two LLM models)')
    parser.add_argument('--ensemble_model1', type=str, default='geneformer-104m',
                       choices=['scgpt', 'geneformer-10m', 'geneformer-104m', 'geneformer-104m-clcancer', 'geneformer-316m',
                                'uce-100m', 'teddy-70m', 'teddy-160m', 'teddy-400m', 'scfoundation'],
                       help='First LLM model for ensemble (only used if --use_ensemble)')
    parser.add_argument('--ensemble_model2', type=str, default='scfoundation',
                       choices=['scgpt', 'geneformer-10m', 'geneformer-104m', 'geneformer-104m-clcancer', 'geneformer-316m',
                                'uce-100m', 'teddy-70m', 'teddy-160m', 'teddy-400m', 'scfoundation'],
                       help='Second LLM model for ensemble (only used if --use_ensemble)')

    parser.add_argument('--dynamic_embeddings', action='store_true', help='Use dynamic embeddings (compute on-the-fly during training instead of pre-computing). Enables fine-tuning but slower and uses more memory.')
    parser.add_argument('--unfreeze_last_n_layers', type=int, default=0, help='Unfreeze last N transformer layers for fine-tuning (only works with --dynamic_embeddings). 0=fully frozen, 1-2 recommended.')

    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio (for source GEX)')
    parser.add_argument('--tgt_validation_split', type=float, default=0.2, help='Target validation split ratio (for REH PEAK held-out set to prevent data leakage, default: 0.2 = 20%)')
    parser.add_argument('--eval_every', type=int, default=5, help='Evaluate every N epochs')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')

    args = parser.parse_args()

    if args.unfreeze_last_n_layers > 0 and not args.dynamic_embeddings:
        raise ValueError("--unfreeze_last_n_layers requires --dynamic_embeddings to be enabled")

    if args.use_ensemble and args.dynamic_embeddings:
        raise ValueError("--use_ensemble is not compatible with --dynamic_embeddings (ensemble requires pre-computed embeddings)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    (X_src, y_src, src_genes, src_cell_ids,
     X_tgt, y_tgt, tgt_genes, tgt_cell_ids,
     label_map, test_data, sup_peak_data, tgt_val_data) = load_and_preprocess_data(
        args.src_csv, args.tgt_csv, args.test_csv,
        peak_filter_threshold=args.peak_filter_threshold,
        tgt_label_csv=args.tgt_label_csv,
        sup_peak_csv=args.sup_peak_csv,
        sup_peak_label_csv=args.sup_peak_label_csv,
        use_peak_mapper=args.use_peak_mapper,
        tgt_validation_split=args.tgt_validation_split
    )

    X_src_train, X_src_val, y_src_train, y_src_val, idx_train, idx_val = train_test_split(
        X_src, y_src, np.arange(len(X_src)),
        test_size=args.val_split, random_state=42, stratify=y_src
    )

    src_cell_ids_train = [src_cell_ids[i] for i in idx_train]
    src_cell_ids_val = [src_cell_ids[i] for i in idx_val]

    print(f"\nSplit: {len(X_src_train)} train, {len(X_src_val)} val")
    print(f"Train distribution: {np.bincount(y_src_train)}")
    print(f"Val distribution: {np.bincount(y_src_val)}")

    # Compute source class distribution for balance loss
    if args.use_balance_loss:
        y_train_all = np.concatenate([y_src_train, y_src_val])
        class_counts = np.bincount(y_train_all, minlength=len(label_map))
        source_class_prior = torch.tensor(class_counts / class_counts.sum(), dtype=torch.float32)
        print(f"\nSource class distribution for balance loss:")
        print(f"  {source_class_prior.numpy()}")
        for label, label_idx in sorted(label_map.items(), key=lambda x: x[1]):
            print(f"  Class {label_idx} ({label}): {source_class_prior[label_idx]:.1%}")
    else:
        source_class_prior = None

    # Use ALL target cells (no subsampling) - allows full dataset validation
    print(f"\nUsing all {len(X_tgt)} target cells (no subsampling)")

    args.n_peaks = X_tgt.shape[1]

    print("\n" + "="*60)
    if args.dynamic_embeddings:
        print("LOADING SOURCE ENCODER (DYNAMIC MODE)")
    else:
        print("GENERATING SOURCE EMBEDDINGS (PRE-COMPUTED MODE)")
    print("="*60)

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

    if args.dynamic_embeddings:
        print(f"\nDynamic Embeddings Mode: LLM will run on-the-fly during training")
        print(f"  Model: {args.source_model}")
        print(f"  Unfreeze last {args.unfreeze_last_n_layers} layers" if args.unfreeze_last_n_layers > 0 else "  LLM fully frozen")

        source_embedder = UnifiedEmbedder(
            model_name=args.source_model,
            use_lora=args.use_lora,
            lora_rank=args.lora_rank,
            device=device
        )

        source_model, _ = source_embedder.load_model()
        source_dim = source_dim_map.get(args.source_model, 512)

        # Get tokenizer and loader for tokenization
        source_tokenizer = source_embedder.tokenizer
        source_loader = source_embedder.loader if hasattr(source_embedder, 'loader') else None

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

        emb_src_train = X_src_train
        emb_src_val = X_src_val
        if test_data is not None:
            X_test, y_test, test_genes, test_cell_ids = test_data
            emb_test = X_test
        else:
            emb_test = None

        source_dim = X_src_train.shape[1]
        print(f"  Will use RAW data: {emb_src_train.shape}")
        print(f"  Input dimension: {source_dim} genes")

    elif args.use_ensemble:
        print(f"\nEnsemble Mode: Combining {args.ensemble_model1} + {args.ensemble_model2}")

        embedder1 = UnifiedEmbedder(
            model_name=args.ensemble_model1,
            use_lora=args.use_lora,
            lora_rank=args.lora_rank,
            device=device
        )
        embedder2 = UnifiedEmbedder(
            model_name=args.ensemble_model2,
            use_lora=args.use_lora,
            lora_rank=args.lora_rank,
            device=device
        )

        model1, _ = embedder1.load_model()
        model2, _ = embedder2.load_model()

        dim1 = source_dim_map.get(args.ensemble_model1, 512)
        dim2 = source_dim_map.get(args.ensemble_model2, 512)
        source_dim = dim1 + dim2

        print(f"\n  Model 1: {args.ensemble_model1} (dim: {dim1})")
        print(f"  Model 2: {args.ensemble_model2} (dim: {dim2})")
        print(f"  Combined dim: {source_dim}")

        src_batch_size = 16

        print(f"\nPre-computing embeddings from Model 1 (with caching)...")
        emb1_train = get_cached_embeddings(args.ensemble_model1, args.src_csv, X_src_train, src_genes, embedder1, batch_size=src_batch_size)
        emb1_val = get_cached_embeddings(args.ensemble_model1, args.src_csv, X_src_val, src_genes, embedder1, batch_size=src_batch_size)

        print(f"Pre-computing embeddings from Model 2 (with caching)...")
        emb2_train = get_cached_embeddings(args.ensemble_model2, args.src_csv, X_src_train, src_genes, embedder2, batch_size=src_batch_size)
        emb2_val = get_cached_embeddings(args.ensemble_model2, args.src_csv, X_src_val, src_genes, embedder2, batch_size=src_batch_size)

        emb_src_train = np.concatenate([emb1_train, emb2_train], axis=1)
        emb_src_val = np.concatenate([emb1_val, emb2_val], axis=1)

        print(f"  Train embeddings: {emb_src_train.shape}")
        print(f"  Val embeddings: {emb_src_val.shape}")

        if test_data is not None:
            X_test, y_test, test_genes, test_cell_ids = test_data
            emb1_test = get_cached_embeddings(args.ensemble_model1, args.test_csv if args.test_csv else args.src_csv, X_test, test_genes, embedder1, batch_size=src_batch_size)
            emb2_test = get_cached_embeddings(args.ensemble_model2, args.test_csv if args.test_csv else args.src_csv, X_test, test_genes, embedder2, batch_size=src_batch_size)
            emb_test = np.concatenate([emb1_test, emb2_test], axis=1)
            print(f"  Test embeddings: {emb_test.shape}")
        else:
            emb_test = None

        source_tokenizer = None
        source_loader = None

    else:
        source_embedder = UnifiedEmbedder(
            model_name=args.source_model,
            use_lora=args.use_lora,
            lora_rank=args.lora_rank,
            device=device
        )

        source_model, _ = source_embedder.load_model()
        source_dim = source_dim_map.get(args.source_model, 512)

        print(f"\nPre-computing embeddings for source data...")
        print(f"  Model: {args.source_model}")
        print(f"  Embedding dim: {source_dim}")

        src_batch_size = 16
        emb_src_train = get_cached_embeddings(args.source_model, args.src_csv, X_src_train, src_genes, source_embedder, batch_size=src_batch_size)
        emb_src_val = get_cached_embeddings(args.source_model, args.src_csv, X_src_val, src_genes, source_embedder, batch_size=src_batch_size)

        print(f"  Train embeddings: {emb_src_train.shape}")
        print(f"  Val embeddings: {emb_src_val.shape}")

        if test_data is not None:
            X_test, y_test, test_genes, test_cell_ids = test_data
            emb_test = get_cached_embeddings(args.source_model, args.test_csv if args.test_csv else args.src_csv, X_test, test_genes, source_embedder, batch_size=src_batch_size)
            print(f"  Test embeddings: {emb_test.shape}")
        else:
            emb_test = None

        source_tokenizer = None
        source_loader = None

    print("\n" + "="*60)
    print("BUILDING DUAL-ENCODER DANN")
    print("="*60)

    target_embedder = UnifiedEmbedder(model_name='peak')
    # Convert string to bool for use_batch_norm
    use_batch_norm_bool = (args.use_batch_norm == 'True')

    target_model, _ = target_embedder.load_model(
        n_peaks=args.n_peaks,
        hidden_dim=args.peak_hidden_dim,
        intermediate_dim=args.peak_intermediate_dim,
        num_layers=args.peak_num_layers,
        dropout=args.peak_dropout,
        peak_arch_type=args.peak_arch_type,
        activation=args.activation,
        use_batch_norm=use_batch_norm_bool,
        residual_type=args.residual_type,
        beta_vae=args.beta_vae,
        noise_level=args.noise_level,
        temperature=args.temperature,
        aug_dropout_prob=args.aug_dropout_prob,
        projection_dim=args.projection_dim,
        num_conv_layers=args.num_conv_layers,
        num_filters=args.num_filters,
        kernel_size=args.kernel_size
    )

    if args.dynamic_embeddings:
        print(f"\nSource Encoder: {args.source_model} (input: {source_dim} genes) [DYNAMIC MODE]")
        if args.unfreeze_last_n_layers > 0:
            print(f"  Fine-tuning: Last {args.unfreeze_last_n_layers} layers unfrozen")
    elif args.use_ensemble:
        print(f"\nSource Encoder: ENSEMBLE [{args.ensemble_model1} + {args.ensemble_model2}] (dim: {source_dim}) [pre-computed embeddings]")
    else:
        print(f"\nSource Encoder: {args.source_model} (dim: {source_dim}) [pre-computed embeddings]")

    print(f"Target Encoder: {args.peak_arch_type.upper()} (dim: {args.peak_hidden_dim})")
    print(f"Shared Dimension: {args.shared_dim}")
    print(f"GRL Lambda: {'Annealing (gamma=' + str(args.grl_gamma) + ')' if args.use_grl_annealing else 'Constant (1.0)'}")
    print(f"Discriminator: {'CDANN (Class-Conditional)' if args.use_cdann else 'Standard'}")
    print(f"Contrastive Loss: {'Enabled (λ=' + str(args.lambda_contrastive) + ', T=' + str(args.contrastive_temperature) + ')' if args.use_contrastive_loss else 'Disabled'}")

    if args.dynamic_embeddings:
        actual_source_encoder = source_model
        actual_source_dim = source_dim_map.get(args.source_model, 512)
    else:
        actual_source_encoder = IdentityEncoder()
        actual_source_dim = source_dim

    model = DualEncoderDANN(
        source_encoder=actual_source_encoder,
        target_encoder=target_model,
        source_dim=actual_source_dim,
        target_dim=args.peak_hidden_dim,
        shared_dim=args.shared_dim,
        num_classes=3,
        disc_hidden=args.disc_hidden,
        disc_layers=args.disc_layers,
        cls_hidden=args.cls_hidden,
        cls_layers=args.cls_layers,
        dropout=args.dropout,
        use_spectral_norm=args.use_spectral_norm,
        freeze_source_encoder=(not args.dynamic_embeddings or args.unfreeze_last_n_layers == 0),
        use_cdann=args.use_cdann
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen: {total_params - trainable_params:,}")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Create dataloaders
    if args.dynamic_embeddings:
        # Use TokenizedSourceDataset for dynamic embeddings
        print("\nCreating tokenized source dataset...")
        src_dataset = TokenizedSourceDataset(
            X_src=emb_src_train,
            y_src=y_src_train,
            gene_names=src_genes,
            tokenizer=source_tokenizer,
            loader=source_loader
        )
        src_loader = DataLoader(src_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        # Target loader (with labels if available)
        tgt_tensor = torch.from_numpy(X_tgt).float()
        if y_tgt is not None:
            y_tgt_tensor = torch.from_numpy(y_tgt).long()
            tgt_dataset = TensorDataset(tgt_tensor, y_tgt_tensor)
        else:
            tgt_dataset = TensorDataset(tgt_tensor)
        tgt_loader = DataLoader(tgt_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    else:
        # Use regular dataloaders for pre-computed embeddings
        src_loader, tgt_loader = create_dataloaders(emb_src_train, y_src_train, X_tgt, args.batch_size, device, y_tgt_train=y_tgt)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0
    best_tgt_acc = 0.0  # Track best target validation accuracy (for semi-supervised)
    epochs_without_tgt_improvement = 0  # For target-aware early stopping
    history = {'train_loss': [], 'val_acc': [], 'test_acc': []}

    # Collapse tracking variables (require persistence before stopping)
    collapse_counter = 0
    collapse_persistence_threshold = 5  # Require 5 consecutive collapsed epochs

    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)

    for epoch in range(args.epochs):
        # Curriculum learning: Warmup lambda_target_class from 0 → final value
        if args.lambda_target_class_warmup > 0 and epoch < args.lambda_target_class_warmup:
            current_lambda_target = args.lambda_target_class * (epoch / args.lambda_target_class_warmup)
        else:
            current_lambda_target = args.lambda_target_class

        # Log warmup progress every 10 epochs
        if args.lambda_target_class_warmup > 0 and (epoch % 10 == 0 or epoch == args.lambda_target_class_warmup):
            warmup_pct = min(100, (epoch / args.lambda_target_class_warmup) * 100)
            if epoch < args.lambda_target_class_warmup:
                print(f"  [Warmup] lambda_target_class = {current_lambda_target:.3f} ({warmup_pct:.0f}% of {args.lambda_target_class})")
            elif epoch == args.lambda_target_class_warmup:
                print(f"  [Warmup Complete] lambda_target_class = {current_lambda_target:.3f} (100%)")

        train_loss, class_loss, domain_loss, alpha = train_epoch(
            model, src_loader, tgt_loader, optimizer, epoch, args.epochs,
            args.lambda_domain, args.lambda_class, device,
            use_grl_annealing=args.use_grl_annealing, grl_gamma=args.grl_gamma,
            use_contrastive_loss=args.use_contrastive_loss,
            lambda_contrastive=args.lambda_contrastive,
            contrastive_temperature=args.contrastive_temperature,
            lambda_balance=args.lambda_balance,
            use_balance_loss=args.use_balance_loss,
            source_class_prior=source_class_prior,
            lambda_entropy=args.lambda_entropy,
            use_entropy_loss=args.use_entropy_loss,
            confidence_threshold=args.confidence_threshold,
            divergence_type=args.divergence_type,
            lambda_target_class=current_lambda_target  # Use warmed-up value
        )

        history['train_loss'].append(train_loss)

        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f} (cls: {class_loss:.4f}, dom: {domain_loss:.4f}, alpha: {alpha:.3f})")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            val_results = evaluate(
                model, emb_src_val, y_src_val, args.batch_size, device, domain='source',
                tokenizer=source_tokenizer, loader=source_loader, gene_names=src_genes
            )
            val_acc = val_results['accuracy']
            history['val_acc'].append(val_acc)

            print(f"  Val Accuracy: {val_acc:.4f} (balanced: {val_results['balanced_accuracy']:.4f})")

            # Check target diversity to detect class collapse
            diversity = check_target_diversity(model, X_tgt, args.batch_size, device)
            print(f"  Target diversity: {diversity['distribution']} (max: {diversity['max_ratio']:.1%})")

            # Evaluate on target validation set (using real paired labels)
            if tgt_val_data is not None:
                X_tgt_val, y_tgt_val, tgt_genes_val, tgt_cell_ids_val = tgt_val_data
                print(f"\n  {'='*50}")
                print(f"  TARGET Validation (Real Labels, {len(y_tgt_val)} cells)")
                print(f"  {'='*50}")

                tgt_val_results = evaluate(
                    model, X_tgt_val, y_tgt_val, args.batch_size, device,
                    domain='target', tokenizer=None, loader=None, gene_names=tgt_genes_val
                )

                tgt_val_acc = tgt_val_results['accuracy']
                tgt_bal_acc = tgt_val_results['balanced_accuracy']
                tgt_kappa = tgt_val_results.get('kappa', 0.0)

                print(f"  Target Val Accuracy:  {tgt_val_acc:.4f}")
                print(f"  Target Bal Accuracy:  {tgt_bal_acc:.4f}")
                print(f"  Target Kappa:         {tgt_kappa:.4f}")

                # Evaluate peak reconstruction quality (AUROC, Precision/Recall, Correlation)
                peak_recon_metrics = evaluate_peak_reconstruction(
                    model, X_tgt_val, labels=y_tgt_val, batch_size=args.batch_size, device=device
                )
                if peak_recon_metrics is not None:
                    print(f"  Peak Reconstruction Metrics:")
                    if peak_recon_metrics['auroc'] is not None:
                        print(f"    AUROC (Overall):    {peak_recon_metrics['auroc']:.4f}")
                    if peak_recon_metrics['auroc_g1'] is not None:
                        print(f"    AUROC (G1):         {peak_recon_metrics['auroc_g1']:.4f}")
                    if peak_recon_metrics['auroc_s'] is not None:
                        print(f"    AUROC (S):          {peak_recon_metrics['auroc_s']:.4f}")
                    if peak_recon_metrics['auroc_g2m'] is not None:
                        print(f"    AUROC (G2M):        {peak_recon_metrics['auroc_g2m']:.4f}")
                    if peak_recon_metrics['avg_precision'] is not None:
                        print(f"    Avg Precision:      {peak_recon_metrics['avg_precision']:.4f}")
                    if peak_recon_metrics['precision'] is not None:
                        print(f"    Precision:          {peak_recon_metrics['precision']:.4f}")
                    if peak_recon_metrics['recall'] is not None:
                        print(f"    Recall:             {peak_recon_metrics['recall']:.4f}")
                    if peak_recon_metrics['f1'] is not None:
                        print(f"    F1-Score:           {peak_recon_metrics['f1']:.4f}")
                    if peak_recon_metrics['cell_pearson_mean'] is not None:
                        print(f"    Cell Corr (Pearson):  {peak_recon_metrics['cell_pearson_mean']:.4f}")
                    if peak_recon_metrics['cell_spearman_mean'] is not None:
                        print(f"    Cell Corr (Spearman): {peak_recon_metrics['cell_spearman_mean']:.4f}")

                # Track best target model (PRIMARY metric for semi-supervised!)
                if tgt_val_acc > best_tgt_acc:
                    best_tgt_acc = tgt_val_acc
                    epochs_without_tgt_improvement = 0

                    # Save best target model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'tgt_val_acc': tgt_val_acc,
                        'tgt_bal_acc': tgt_bal_acc,
                        'tgt_kappa': tgt_kappa,
                        'args': vars(args)
                    }, output_dir / 'best_model_target_acc.pt')
                    print(f"  ✓ New best target accuracy: {tgt_val_acc:.4f} (saved to best_model_target_acc.pt)")
                else:
                    epochs_without_tgt_improvement += 1
                    print(f"  No improvement for {epochs_without_tgt_improvement} eval(s) (best: {best_tgt_acc:.4f})")

                # Target-aware early stopping (more reliable than diversity heuristics)
                if args.tgt_early_stop_patience > 0 and epochs_without_tgt_improvement >= args.tgt_early_stop_patience:
                    print(f"\n  {'='*60}")
                    print(f"  TARGET EARLY STOPPING")
                    print(f"  {'='*60}")
                    print(f"  Best target accuracy: {best_tgt_acc:.4f}")
                    print(f"  No improvement for {epochs_without_tgt_improvement} evaluations (patience: {args.tgt_early_stop_patience})")
                    print(f"  Stopping early to save compute time.")
                    print(f"  {'='*60}\n")
                    break  # Exit training loop
                print(f"  {'='*50}\n")

            # Stop if target predictions collapsed to single class (after grace period + persistence)
            if epoch >= 25:  # Grace period: Allow GRL warmup
                if diversity['is_collapsed']:
                    collapse_counter += 1
                    if collapse_counter >= collapse_persistence_threshold:
                        print(f"\n  🔍 COLLAPSE EARLY STOPPING DIAGNOSTICS:")
                        print(f"     Epoch: {epoch+1}/{args.epochs}")
                        print(f"     Val Acc: {val_acc:.4f} (best: {best_val_acc:.4f})")
                        print(f"     Alpha (GRL): {alpha:.3f}")
                        print(f"     Target distribution: {diversity['distribution']} (max: {diversity['max_ratio']:.1%})")
                        print(f"     Collapsed epochs: {collapse_counter} consecutive (threshold: {collapse_persistence_threshold})")
                        print(f"     Reason: Persistent target class collapse (>{diversity['max_ratio']:.1%} in single class)")
                        print(f"  ⚠️  Stopping training early to prevent wasting GPU time on collapsed model.\n")
                        break  # Stop training immediately
                else:
                    # Reset counter if diversity recovered
                    collapse_counter = 0

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'args': vars(args)
                }, output_dir / 'best_model.pt')
                print(f"  → Saved best model (val_acc: {val_acc:.4f})")

            if test_data is not None:
                test_results = evaluate(
                    model, emb_test, y_test, args.batch_size, device, domain='source',
                    tokenizer=source_tokenizer, loader=source_loader, gene_names=test_genes
                )
                test_acc = test_results['accuracy']
                history['test_acc'].append(test_acc)
                print(f"  Test Accuracy: {test_acc:.4f} (balanced: {test_results['balanced_accuracy']:.4f})")

        scheduler.step()

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

    try:
        checkpoint = torch.load(output_dir / 'best_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("  Loaded best checkpoint for final evaluation")
    except Exception as e:
        print(f"  Warning: Could not reload checkpoint (using current model state): {e}")
        print("  This is expected when using LoRA/finetune - current model already has best weights")

    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)

    val_results = evaluate(
        model, emb_src_val, y_src_val, args.batch_size, device, domain='source',
        tokenizer=source_tokenizer, loader=source_loader, gene_names=src_genes
    )
    print(f"\nValidation Results:")
    print(f"  Accuracy: {val_results['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {val_results['balanced_accuracy']:.4f}")
    print(f"  Kappa: {val_results['kappa']:.4f}")

    save_predictions(val_results, src_cell_ids_val, label_map, output_dir, 'validation')

    if test_data is not None:
        X_test, y_test, _, test_cell_ids = test_data
        test_results = evaluate(
            model, emb_test, y_test, args.batch_size, device, domain='source',
            tokenizer=source_tokenizer, loader=source_loader, gene_names=test_genes
        )
        print(f"\nTest Results:")
        print(f"  Accuracy: {test_results['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {test_results['balanced_accuracy']:.4f}")
        print(f"  Kappa: {test_results['kappa']:.4f}")
        save_predictions(test_results, test_cell_ids, label_map, output_dir, 'test')

    # Final target diversity check
    final_diversity = check_target_diversity(model, X_tgt, args.batch_size, device)
    print(f"\nREH PEAK Target Predictions:")
    print(f"  Predicted distribution: {final_diversity['distribution']}")
    print(f"  Max class ratio: {final_diversity['max_ratio']:.1%}")

    if final_diversity['is_collapsed']:
        print(f"  ⚠️  WARNING: FINAL MODEL HAS TARGET CLASS COLLAPSE!")
        print(f"  This model predicts {final_diversity['max_ratio']:.1%} of target cells as a single class.")
        print(f"  Consider retraining with different hyperparameters.")
    else:
        print(f"  ✓ Target predictions are well-balanced across classes")

    # Predict on HELD-OUT REH PEAK validation set (if exists) - TRUE DANN TEST!
    if tgt_val_data is not None:
        X_tgt_val, y_tgt_val, _, tgt_val_cell_ids = tgt_val_data
        print(f"\n🎯 HELD-OUT REH PEAK VALIDATION (True DANN Test - Never Seen by Optimizer!)")

        # Check diversity
        tgt_val_diversity = check_target_diversity(model, X_tgt_val, args.batch_size, device)
        print(f"  Predicted distribution: {tgt_val_diversity['distribution']}")
        print(f"  Max class ratio: {tgt_val_diversity['max_ratio']:.1%}")

        # Evaluate
        tgt_val_results = evaluate(model, X_tgt_val, y_tgt_val, args.batch_size, device, domain='target')
        save_predictions(tgt_val_results, tgt_val_cell_ids, label_map, output_dir, 'reh_peak_val_HELDOUT')
        print(f"  ✅ HELD-OUT REH PEAK Accuracy: {tgt_val_results['accuracy']:.4f}")
        print(f"  ✅ HELD-OUT REH PEAK Balanced Accuracy: {tgt_val_results['balanced_accuracy']:.4f}")
        print(f"  ✅ HELD-OUT REH PEAK Kappa: {tgt_val_results['kappa']:.4f}")
        print(f"  ⚠️  This is the TRUE DANN performance (not trained on these cells!)")

    # Predict on REH PEAK TRAINING SET (for comparison - these cells were used in training!)
    if y_tgt is not None:
        print(f"\n📊 REH PEAK TRAINING SET (Used in Semi-Supervised Training)")
        tgt_results = evaluate(model, X_tgt, y_tgt, args.batch_size, device, domain='target')
        save_predictions(tgt_results, tgt_cell_ids, label_map, output_dir, 'reh_peak_train')
        print(f"  REH PEAK Train Accuracy: {tgt_results['accuracy']:.4f}")
        print(f"  REH PEAK Train Balanced Accuracy: {tgt_results['balanced_accuracy']:.4f}")
        print(f"  REH PEAK Train Kappa: {tgt_results['kappa']:.4f}")
        if tgt_val_data is not None:
            print(f"  ⚠️  This is NOT true DANN (these cells were trained with labels!)")
    else:
        # No labels - just predict
        tgt_results = evaluate(model, X_tgt, y_tgt, args.batch_size, device, domain='target')
        save_predictions(tgt_results, tgt_cell_ids, label_map, output_dir, 'reh_peak')
        if y_tgt is not None:
            print(f"  REH PEAK Accuracy: {tgt_results['accuracy']:.4f}")
            print(f"  REH PEAK Balanced Accuracy: {tgt_results['balanced_accuracy']:.4f}")
            print(f"  REH PEAK Kappa: {tgt_results['kappa']:.4f}")

    # Predict on SUP PEAK if provided
    if sup_peak_data is not None:
        X_sup_peak, y_sup_peak, sup_peak_genes, sup_peak_cell_ids = sup_peak_data
        print(f"\n🎯 SUP PEAK Target Predictions:")

        # Check diversity
        sup_diversity = check_target_diversity(model, X_sup_peak, args.batch_size, device)
        print(f"  Predicted distribution: {sup_diversity['distribution']}")
        print(f"  Max class ratio: {sup_diversity['max_ratio']:.1%}")

        # Predict with paired GEX labels
        sup_results = evaluate(model, X_sup_peak, y_sup_peak, args.batch_size, device, domain='target')
        save_predictions(sup_results, sup_peak_cell_ids, label_map, output_dir, 'sup_peak')

        if y_sup_peak is not None:
            print(f"  SUP PEAK Accuracy: {sup_results['accuracy']:.4f}")
            print(f"  SUP PEAK Balanced Accuracy: {sup_results['balanced_accuracy']:.4f}")
            print(f"  SUP PEAK Kappa: {sup_results['kappa']:.4f}")

        # Evaluate peak reconstruction quality (AUROC, Precision/Recall, Correlation)
        print(f"\n📊 SUP PEAK Reconstruction Metrics:")
        peak_recon_metrics = evaluate_peak_reconstruction(
            model, X_sup_peak, labels=y_sup_peak, batch_size=args.batch_size, device=device
        )
        if peak_recon_metrics is not None:
            if peak_recon_metrics['auroc'] is not None:
                print(f"  AUROC (Overall):        {peak_recon_metrics['auroc']:.4f}")
            if peak_recon_metrics['auroc_g1'] is not None:
                print(f"  AUROC (G1):             {peak_recon_metrics['auroc_g1']:.4f}")
            if peak_recon_metrics['auroc_s'] is not None:
                print(f"  AUROC (S):              {peak_recon_metrics['auroc_s']:.4f}")
            if peak_recon_metrics['auroc_g2m'] is not None:
                print(f"  AUROC (G2M):            {peak_recon_metrics['auroc_g2m']:.4f}")
            if peak_recon_metrics['avg_precision'] is not None:
                print(f"  Avg Precision (AUPRC):  {peak_recon_metrics['avg_precision']:.4f}")
            if peak_recon_metrics['precision'] is not None:
                print(f"  Precision:              {peak_recon_metrics['precision']:.4f}")
            if peak_recon_metrics['recall'] is not None:
                print(f"  Recall:                 {peak_recon_metrics['recall']:.4f}")
            if peak_recon_metrics['f1'] is not None:
                print(f"  F1-Score:               {peak_recon_metrics['f1']:.4f}")
            if peak_recon_metrics['cell_pearson_mean'] is not None:
                print(f"  Cell Corr (Pearson):    {peak_recon_metrics['cell_pearson_mean']:.4f}")
            if peak_recon_metrics['cell_spearman_mean'] is not None:
                print(f"  Cell Corr (Spearman):   {peak_recon_metrics['cell_spearman_mean']:.4f}")

            # Save reconstruction metrics to JSON
            recon_metrics_dict = {k: float(v) if v is not None else None for k, v in peak_recon_metrics.items()}
            with open(output_dir / 'peak_reconstruction_metrics.json', 'w') as f:
                json.dump(recon_metrics_dict, f, indent=2)
            print(f"  ✓ Saved to: {output_dir / 'peak_reconstruction_metrics.json'}")
        else:
            print(f"  ⚠️  Architecture does not support peak reconstruction (no decoder)")

    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
