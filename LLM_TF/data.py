#!/usr/bin/env python3
"""
Data Processing Module
=====================

Clean, professional data processing functions for transfer learning pipeline.
Contains the complete preprocessing pipeline in one convenient function.

Main Functions:
- run_complete_preprocessing: One-stop preprocessing pipeline
- make_class_weights: Compute class weights for imbalanced data
- load_multi_source_target: Load multiple sources with target (legacy)

Author: Transfer Learning Pipeline
"""

import json
import re
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Import modular components
# NOTE: feature_selection_mechanism_may_delete was removed - these imports commented out
# from LLM_TF.feature_selection_mechanism_may_delete.feature_selection import select_features_leak_free
# from LLM_TF.feature_selection_mechanism_may_delete.feature_scaling import scale_features_leak_free


# Constants
_LABEL_REGEX = re.compile(r"(phase|label|predicted)", re.IGNORECASE)
_META_COLS = {"cell", "cell_id", "barcode", "gex_barcode"}


def _find_label_col(df: pd.DataFrame) -> str:
    """Find label column in DataFrame."""
    candidates = [c for c in df.columns if _LABEL_REGEX.search(c)]
    if not candidates:
        raise ValueError(
            "Could not find a label column (tried /phase|label|Predicted/i). "
            f"Available columns: {list(df.columns)[:8]}..."
        )
    return candidates[0]


def load_and_merge_sources(source_paths: List[str]) -> pd.DataFrame:
    """Load and merge multiple source datasets."""
    print(f"\nLOADING SOURCE DATASETS:")

    source_dfs = []
    for i, path in enumerate(source_paths):
        df = pd.read_csv(path)
        df['source_dataset'] = f'source_{i+1}'
        source_dfs.append(df)
        print(f"   Source {i+1}: {df.shape} from {Path(path).name}")

    merged_df = pd.concat(source_dfs, axis=0, ignore_index=True)
    merged_df = merged_df.drop('source_dataset', axis=1, errors='ignore')

    print(f"Merged sources: {merged_df.shape}")
    return merged_df


def load_target_data(target_path: str) -> pd.DataFrame:
    """Load target dataset from file or directory."""
    print(f"\nLOADING TARGET DATASET:")

    if Path(target_path).is_dir():
        # Load from directory (multiple CSV files)
        target_files = list(Path(target_path).glob("*gene_activity*.csv")) + \
                      list(Path(target_path).glob("*predicted*.csv"))
        if not target_files:
            raise FileNotFoundError(f"No target files found in {target_path}")

        target_dfs = []
        for file in target_files:
            df = pd.read_csv(file)
            target_dfs.append(df)
            print(f"   Loaded: {df.shape} from {file.name}")

        target_df = pd.concat(target_dfs, axis=0, ignore_index=True)
    else:
        # Single CSV file
        target_df = pd.read_csv(target_path)
        print(f"   Loaded: {target_df.shape} from {Path(target_path).name}")

    print(f"Target data: {target_df.shape}")
    return target_df


def normalize_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Find and normalize cell cycle labels."""
    # Find label column
    label_col = _find_label_col(df)
    print(f"Found label column: {label_col}")

    # Normalize labels to G1/S/G2M
    df[label_col] = (
        df[label_col].astype(str)
        .str.replace(r"^G1.*", "G1", regex=True)
        .str.replace(r"^S.*", "S", regex=True)
        .str.replace(r"^G2M.*", "G2M", regex=True)
        .str.replace(r"^G2.*", "G2M", regex=True)
    )

    # Filter valid labels
    df = df[df[label_col].isin(["G1", "S", "G2M"])].copy()

    # Show distribution
    distribution = df[label_col].value_counts()
    print(f"Label distribution: {distribution.to_dict()}")

    return df, label_col


def create_splits(df: pd.DataFrame, label_col: str,
                 test_size: float = 0.2, val_size: float = 0.2,
                 random_state: int = 42) -> Dict[str, pd.DataFrame]:
    """Create stratified train/validation/test splits."""
    print(f"\nCREATING LEAK-FREE SPLITS:")

    # Split 1: 80% train+val, 20% test
    train_val_df, test_df = train_test_split(
        df, test_size=test_size,
        stratify=df[label_col],
        random_state=random_state
    )

    # Split 2: Further split train+val
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size,
        stratify=train_val_df[label_col],
        random_state=random_state
    )

    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }

    print(f"Splits created:")
    for split_name, split_df in splits.items():
        pct = len(split_df) / len(df) * 100
        dist = split_df[label_col].value_counts(normalize=True).round(3).to_dict()
        print(f"   {split_name.capitalize()}: {len(split_df):,} cells ({pct:.1f}%) - {dist}")

    return splits


def get_common_genes(splits: Dict[str, pd.DataFrame], target_df: pd.DataFrame,
                    label_col: str) -> List[str]:
    """Get common genes between source and target datasets."""
    print(f"\nFINDING COMMON GENES:")

    # Get gene columns (exclude metadata)
    train_genes = [c for c in splits['train'].columns if c not in _META_COLS and c != label_col]
    target_genes = [c for c in target_df.columns if c not in _META_COLS]
    common_genes = sorted(set(train_genes) & set(target_genes))

    print(f"   Source genes: {len(train_genes):,}")
    print(f"   Target genes: {len(target_genes):,}")
    print(f"   Common genes: {len(common_genes):,}")

    if not common_genes:
        raise ValueError("No common genes between source and target!")

    return common_genes


def clean_data(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Remove NaN values from selected features only."""
    print(f"\nCLEANING DATA (REMOVE NaN):")

    cleaned_data = {}
    for name, df in data_dict.items():
        original_shape = df.shape

        # Only check for NaN in the actual feature columns (not unselected features)
        # Target data only has selected features, source data has selected features + label
        if name == 'target':
            # Target has only selected features
            cleaned_df = df.dropna()
        else:
            # Source splits have selected features + label column
            # Find label column
            feature_cols = [col for col in df.columns if not col.lower().endswith(('phase', 'label', 'predicted'))]
            label_cols = [col for col in df.columns if col.lower().endswith(('phase', 'label', 'predicted'))]

            # Only drop rows with NaN in feature columns (keep label column intact)
            if feature_cols:
                feature_subset = df[feature_cols].dropna()
                cleaned_df = df.loc[feature_subset.index]
            else:
                cleaned_df = df.dropna()

        cleaned_data[name] = cleaned_df

        if original_shape != cleaned_df.shape:
            print(f"   {name.capitalize()}: {original_shape} -> {cleaned_df.shape} (removed NaN)")
        else:
            print(f"   {name.capitalize()}: {cleaned_df.shape} (no NaN found)")

    return cleaned_data


def save_processed_data(processed_data: Dict[str, pd.DataFrame],
                      output_dir: str, selected_genes: List[str],
                      preprocessing_params: dict,
                      prefix: str = "leak_free") -> Dict[str, str]:
    """Save processed datasets with metadata."""
    print(f"\nSAVING PROCESSED DATA:")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_paths = {}
    for name, df in processed_data.items():
        # Determine data type for filename
        if name == 'target':
            data_type = 'atac'  # Target is usually ATAC-seq
        else:
            data_type = 'rna'   # Source splits are usually RNA-seq

        filename = f"{prefix}_{name}_{data_type}.csv"
        filepath = output_path / filename

        df.to_csv(filepath, index=False)
        file_paths[name] = str(filepath)

        if name == 'trainval':
            print(f"   {name.capitalize()}: {filepath} ({df.shape}) [for hyperparameter search]")
        else:
            print(f"   {name.capitalize()}: {filepath} ({df.shape})")

    # Save metadata
    metadata = {
        'gene_selection_method': preprocessing_params.get('gene_selection_method'),
        'n_genes_selected': len(selected_genes),
        'feature_scaling': preprocessing_params.get('feature_scaling'),
        'test_size': preprocessing_params.get('test_size'),
        'val_size': preprocessing_params.get('val_size'),
        'random_state': preprocessing_params.get('random_state'),
        'selected_genes': selected_genes,
        'data_shapes': {name: df.shape for name, df in processed_data.items()},
        'file_paths': file_paths,
        'methodology': 'leak_free_preprocessing',
        'timestamp': pd.Timestamp.now().isoformat()
    }

    metadata_path = output_path / f"{prefix}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"   Metadata: {metadata_path}")
    print(f"All data saved successfully!")

    return file_paths


def run_complete_preprocessing(source_paths: List[str], target_path: str, output_dir: str,
                             gene_selection_method: str = 'hybrid', n_genes: int = 2000,
                             feature_scaling: str = 'standard', test_size: float = 0.2,
                             val_size: float = 0.2, random_state: int = 42,
                             prefix: str = "leak_free") -> Dict[str, str]:
    """
    ONE-STOP COMPLETE PREPROCESSING PIPELINE

    This is the main function that run_multi_source_pipeline.py should call.
    It does ALL preprocessing steps (1-8) and returns file paths for ML pipeline.

    Args:
        source_paths: List of source CSV file paths
        target_path: Target data path (file or directory)
        output_dir: Output directory for processed data
        gene_selection_method: 'variance', 'differential', 'hybrid', 'mutual_info'
        n_genes: Number of genes to select
        feature_scaling: 'standard', 'minmax', 'robust', 'none'
        test_size: Fraction for test set (0.0-1.0)
        val_size: Fraction of remaining data for validation
        random_state: Random seed for reproducibility
        prefix: Prefix for output files

    Returns:
        Dictionary mapping dataset names to file paths for ML pipeline
    """

    print("="*80)
    print("COMPLETE PREPROCESSING PIPELINE - NO DATA LEAKAGE")
    print("="*80)
    print("This function does ALL preprocessing steps (1-8):")
    print("  Step 1-2: Load and merge sources + target")
    print("  Step 3: Normalize labels")
    print("  Step 4: Create proper train/val/test splits")
    print("  Step 5: Find common genes")
    print("  Step 6: Gene selection (TRAINING DATA ONLY)")
    print("  Step 7: Feature scaling (TRAINING DATA ONLY)")
    print("  Step 8: Clean and save processed data")
    print()

    try:
        # Step 1-2: Load data
        source_df = load_and_merge_sources(source_paths)
        target_df = load_target_data(target_path)

        # Step 3: Normalize labels
        source_df, label_col = normalize_labels(source_df)

        # Step 4: Create splits
        splits = create_splits(
            source_df, label_col,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state
        )

        # Step 5: Get common genes
        common_genes = get_common_genes(splits, target_df, label_col)

        # Step 5.1: RESTRICT ALL DATA to common genes only (remove non-overlapping genes)
        print(f"\nSTEP 5.1: RESTRICTING TO COMMON GENES ONLY")
        restricted_splits = {}
        for name, df in splits.items():
            original_cols = len(df.columns)
            restricted_df = df[common_genes + [label_col]]  # Only common genes + label
            restricted_splits[name] = restricted_df
            print(f"   {name.capitalize()}: {original_cols} -> {len(restricted_df.columns)} columns (kept common genes + label)")

        # Restrict target to common genes only
        original_target_cols = len(target_df.columns)
        restricted_target_df = target_df[common_genes]  # Only common genes
        print(f"   Target: {original_target_cols} -> {len(restricted_target_df.columns)} columns (kept common genes only)")

        # Step 6: LEAK-FREE GENE SELECTION (on restricted common genes only)
        print(f"\nSTEP 6: LEAK-FREE GENE SELECTION")
        X_train = restricted_splits['train'][common_genes]
        y_train = restricted_splits['train'][label_col].map({"G1": 0, "S": 1, "G2M": 2})

        selected_features, feature_scores = select_features_leak_free(
            X_train=X_train,
            y_train=y_train,
            available_features=common_genes,
            method=gene_selection_method,
            n_features=min(n_genes, len(common_genes)),
            random_state=random_state
        )

        # Step 7: LEAK-FREE FEATURE SCALING (on restricted data with selected features only)
        print(f"\nSTEP 7: LEAK-FREE FEATURE SCALING")
        scaled_data = scale_features_leak_free(
            train_data=restricted_splits,
            target_data=restricted_target_df,
            selected_features=selected_features,
            method=feature_scaling
        )

        # Step 8: Clean and save (handle NaN only after selection and scaling)
        processed_data = clean_data(scaled_data)

        # Step 8.5: Create merged train+val for hyperparameter search (Géron's methodology)
        print(f"\nSTEP 8.5: CREATING MERGED TRAIN+VAL FOR HYPERPARAMETER SEARCH")
        train_val_df = pd.concat([processed_data['train'], processed_data['val']], ignore_index=True)
        processed_data['trainval'] = train_val_df
        print(f"   Train+Val merged: {len(processed_data['train'])} + {len(processed_data['val'])} = {len(train_val_df)} samples")
        print(f"   Hyperparameter search will use 80% of data, test set isolated")

        preprocessing_params = {
            'gene_selection_method': gene_selection_method,
            'feature_scaling': feature_scaling,
            'test_size': test_size,
            'val_size': val_size,
            'random_state': random_state
        }

        file_paths = save_processed_data(
            processed_data=processed_data,
            output_dir=output_dir,
            selected_genes=selected_features,
            preprocessing_params=preprocessing_params,
            prefix=prefix
        )

        print("\n" + "="*80)
        print("PREPROCESSING COMPLETE - READY FOR ML PIPELINE!")
        print("="*80)
        print(f"No data leakage: Gene selection & scaling fitted on training data only")
        print(f"Proper methodology: Train/val/test splits from source domain")
        print(f"Files ready for hyperparameter search, CV, and training")
        print()

        return file_paths

    except Exception as e:
        print(f"PREPROCESSING FAILED: {e}")
        raise


def make_class_weights(y_src: np.ndarray) -> torch.Tensor:
    """Compute inverse-frequency class weights (normalized) for 3 classes."""
    counts = np.bincount(y_src, minlength=3).astype(np.float64)
    weights = counts.sum() / (counts + 1e-8)  # inverse freq
    weights = weights / weights.mean()        # normalize around 1.0
    return torch.tensor(weights, dtype=torch.float32)


# Legacy function for compatibility with existing scripts
def load_multi_source_target(src_csv_list: List[str], ga_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Legacy function: Load multiple sources and target."""
    print("Using legacy load_multi_source_target function")

    # Load and merge sources
    merged_src = load_and_merge_sources(src_csv_list)

    # Load target
    target_df = load_target_data(ga_dir)

    # Normalize labels
    merged_src, label_col = normalize_labels(merged_src)

    return merged_src, target_df, label_col


# Export the functions we actually use
__all__ = [
    "run_complete_preprocessing",  # MAIN FUNCTION FOR PIPELINE
    "make_class_weights",
    "load_multi_source_target"     # Legacy
]