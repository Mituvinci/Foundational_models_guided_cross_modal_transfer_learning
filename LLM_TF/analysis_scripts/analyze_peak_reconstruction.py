"""
Analyze Peak Reconstruction Quality for DANN Models

Evaluates peak-level correlation metrics to assess whether low classification
accuracy (40%) reflects poor peak learning or just hard classification boundaries.

Usage:
    python analyze_peak_reconstruction.py --model_dir DANN_output/output_dae_mmd_mapper_SMALL_TEST/dual_encoder/train_baseline_geneformer-316m_dae
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, roc_auc_score,
    precision_recall_fscore_support, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')  # Ignore warnings for AUROC on constant columns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from LLM_TF.dual_encoder_dann import DualEncoderDANN
from LLM_TF.embedders.unified_embedder import UnifiedEmbedder
from LLM_TF.peak_mapper.coordinate_mapper import PeakCoordinateMapper
from LLM_TF.peak_mapper.imputer import PeakImputer


class IdentityEncoder(torch.nn.Module):
    """Simple identity encoder for pre-computed embeddings."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    def parameters(self):
        return []


class PeakMapperWrapper:
    """Adapter around PeakCoordinateMapper that aligns DataFrames."""

    def __init__(self, mapper: PeakCoordinateMapper, source_peaks, imputer: Optional[PeakImputer] = None,
                 training_matrix: Optional[np.ndarray] = None):
        self.mapper = mapper
        self.source_peaks = list(source_peaks)
        self.imputer = imputer

        if self.imputer is not None and training_matrix is not None:
            try:
                self.imputer.fit(training_matrix, self.source_peaks)
            except Exception as exc:
                print(f"  Warning: Peak imputer fit failed ({exc}) - continuing without fit.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        target_peaks = list(df.columns)
        mapping = self.mapper.map_peaks(self.source_peaks, target_peaks)

        aligned = self.mapper.align_matrix(
            df.values.astype(np.float32),
            target_peaks=target_peaks,
            source_peaks=self.source_peaks,
            mapping=mapping
        )

        if self.imputer is not None:
            aligned = self.imputer.transform(aligned, self.source_peaks)

        return pd.DataFrame(aligned, index=df.index, columns=self.source_peaks)


def load_model_and_data(model_dir, device='cuda'):
    """Load trained model and test data."""
    model_dir = Path(model_dir)

    # Load config
    with open(model_dir / 'config.json') as f:
        config = json.load(f)

    print(f"\n📂 Loading model from: {model_dir}")
    print(f"  Architecture: {config['peak_arch_type']}")
    print(f"  Source model: {config['source_model']}")
    print(f"  Peak filter: {config.get('peak_filter_threshold', 0.2) * 100}%")
    print(f"  Use mapper: {config.get('use_peak_mapper', False)}")

    use_dynamic = config.get('dynamic_embeddings', False)
    if use_dynamic:
        print("  Loading source encoder (dynamic embeddings)...")
        embedder = UnifiedEmbedder(
            model_name=config['source_model'],
            use_lora=config.get('use_lora', False),
            lora_rank=config.get('lora_rank', 16),
            device=device
        )
        embedder.load_model()
        source_encoder = embedder.model
        if hasattr(embedder, 'embedding_dim'):
            source_dim = embedder.embedding_dim
        elif hasattr(source_encoder, 'hidden_size'):
            source_dim = source_encoder.hidden_size
        elif hasattr(source_encoder, 'config') and hasattr(source_encoder.config, 'hidden_size'):
            source_dim = source_encoder.config.hidden_size
        else:
            source_dim = 768
        source_encoder.eval()
        for param in source_encoder.parameters():
            param.requires_grad = False
    else:
        print("  Using Identity source encoder (pre-computed embeddings)")
        source_encoder = IdentityEncoder()
        dim_map = {
            'geneformer-10m': 256,
            'geneformer-104m': 768,
            'geneformer-316m': 1152,
            'scfoundation': 768,
            'scgpt': 512,
            'uce-100m': 768
        }
        source_dim = config.get('source_embedding_dim',
                                config.get('source_dim',
                                           dim_map.get(config['source_model'], 768)))

    print(f"    Source dimension: {source_dim}")

    # Load peak mapper if used
    peak_mapper = None
    if config.get('use_peak_mapper', False):
        print("\n  Loading peak mapper...")
        sup_peak_df = pd.read_csv(config['sup_peak_csv'], index_col=0)
        reh_peak_df = pd.read_csv(config['tgt_csv'], index_col=0)

        threshold = config.get('peak_filter_threshold', 0.2)
        source_mask = (reh_peak_df > 0).sum(axis=0) / len(reh_peak_df) >= threshold
        reh_peak_filtered = reh_peak_df.loc[:, source_mask]
        source_peaks = list(reh_peak_filtered.columns)

        mapper_method = config.get('mapper_method', 'overlap_50pct')
        mapper = PeakCoordinateMapper(method=mapper_method)
        imputer_strategy = config.get('peak_imputer', 'zero')
        imputer = PeakImputer(strategy=imputer_strategy)
        peak_mapper = PeakMapperWrapper(mapper, source_peaks, imputer, training_matrix=reh_peak_filtered.values)

        n_mapped_peaks = len(source_peaks)
        print(f"    Mapper method: {mapper_method}, peaks retained: {n_mapped_peaks}")

    # Get actual peak count from training (already saved in config)
    # CRITICAL: Use config['n_peaks'] NOT raw file dimension!
    # Config saves the FILTERED peak count actually used during training
    n_training_peaks = config.get('n_peaks', config.get('target_dim', 18464))
    print(f"    Training peaks (from config): {n_training_peaks}")

    peak_embedder = UnifiedEmbedder(model_name='peak', device=device)
    peak_encoder, _ = peak_embedder.load_model(
        n_peaks=n_training_peaks,
        hidden_dim=config['peak_hidden_dim'],
        intermediate_dim=config.get('peak_intermediate_dim', 1024),
        num_layers=config.get('peak_num_layers', 2),
        dropout=config.get('peak_dropout', 0.3),
        peak_arch_type=config['peak_arch_type'],
        activation=config.get('activation', 'relu'),
        use_batch_norm=(config.get('use_batch_norm', 'False') == 'True' or config.get('use_batch_norm', False)),
        residual_type=config.get('residual_type', None),
        beta_vae=config.get('beta_vae', 1.0),
        noise_level=config.get('noise_level', 0.1),
        temperature=config.get('temperature', 0.1),
        aug_dropout_prob=config.get('aug_dropout_prob', 0.1),
        projection_dim=config.get('projection_dim', 128),
        num_conv_layers=config.get('num_conv_layers', 3),
        num_filters=config.get('num_filters', 128),
        kernel_size=config.get('kernel_size', 5)
    )
    peak_encoder = peak_encoder.to(device)

    # Create full model
    model = DualEncoderDANN(
        source_encoder=source_encoder,
        target_encoder=peak_encoder,
        source_dim=source_dim,
        target_dim=config['peak_hidden_dim'],
        shared_dim=config['shared_dim'],
        num_classes=3,
        disc_hidden=config['disc_hidden'],
        disc_layers=config['disc_layers'],
        cls_hidden=config['cls_hidden'],
        cls_layers=config['cls_layers'],
        dropout=config['dropout'],
        use_spectral_norm=config['use_spectral_norm'],
        freeze_source_encoder=True,
        use_cdann=config.get('use_cdann', False)
    ).to(device)

    # Load checkpoint
    checkpoint_path = model_dir / 'best_model_target_acc.pt'
    if not checkpoint_path.exists():
        checkpoint_path = model_dir / 'best_model.pt'

    print(f"\n  Loading checkpoint: {checkpoint_path.name}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load state dict with error handling for shape mismatches
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    except RuntimeError as e:
        print(f"    Warning: Some parameters have size mismatches, loading compatible ones only...")
        model_state = model.state_dict()
        checkpoint_state = checkpoint['model_state_dict']

        # Load only parameters with matching shapes
        compatible_state = {}
        for key in checkpoint_state:
            if key in model_state:
                if model_state[key].shape == checkpoint_state[key].shape:
                    compatible_state[key] = checkpoint_state[key]
                else:
                    print(f"      Skipping {key}: shape mismatch {checkpoint_state[key].shape} vs {model_state[key].shape}")

        model.load_state_dict(compatible_state, strict=False)
        print(f"    Loaded {len(compatible_state)}/{len(checkpoint_state)} parameters")

    print(f"    Best epoch: {checkpoint.get('epoch', 'N/A')}")
    target_acc = checkpoint.get('best_target_acc', None)
    if target_acc is not None:
        print(f"    Target val acc: {target_acc:.4f}")
    else:
        print(f"    Target val acc: N/A")

    # Load SUP PEAK test data
    print("\n  Loading SUP PEAK test data...")
    sup_peak_df = pd.read_csv(config['sup_peak_csv'], index_col=0)
    sup_peak_labels = pd.read_csv(config['sup_peak_label_csv'], index_col=0)

    # Check if model can handle this dataset
    if not config.get('use_peak_mapper', False):
        # Models without peak mapper can't be tested on different datasets
        # (peak dimensions would mismatch: SUP has 74K, REH has 92K → filtered to ~48K)
        print("\n⚠️  WARNING: Model trained WITHOUT peak mapper!")
        print("     Cannot test on SUP PEAK (different peak set than training)")
        print("     Classification and reconstruction will be SKIPPED for this model.\n")
        return None, None, None, None, config, None

    # Apply peak mapper if used
    if peak_mapper:
        print(f"    Original peaks: {sup_peak_df.shape[1]}")
        sup_peak_mapped_df = peak_mapper.transform(sup_peak_df)
        print(f"    Mapped peaks: {sup_peak_mapped_df.shape[1]}")
        sup_peak_data = torch.FloatTensor(sup_peak_mapped_df.values).to(device)
        sup_peak_original = torch.FloatTensor(sup_peak_df.values).to(device)
    else:
        sup_peak_tensor = torch.FloatTensor(sup_peak_df.values).to(device)
        sup_peak_data = sup_peak_tensor
        sup_peak_original = sup_peak_tensor

    # Get labels
    label_map = {'G1': 0, 'S': 1, 'G2M': 2}
    labels = torch.LongTensor([label_map[x] for x in sup_peak_labels['phase']]).to(device)

    print(f"    Cells: {sup_peak_data.shape[0]}")
    print(f"    Peaks (input): {sup_peak_data.shape[1]}")

    return model, sup_peak_data, sup_peak_original, labels, config, peak_mapper


def evaluate_peak_reconstruction(model, peak_data, device='cuda', batch_size=256):
    """
    Evaluate peak reconstruction quality.

    Returns dict with:
        - reconstructed: Reconstructed peak matrix
        - per_cell_corr: Per-cell Pearson correlations
        - per_peak_corr: Per-peak Pearson correlations
        - mse: Mean squared error
        - mae: Mean absolute error
    """
    model.eval()

    # Check if architecture has decoder (it's in base_embedder for wrapped encoders)
    peak_encoder = model.target_encoder
    actual_encoder = getattr(peak_encoder, 'base_embedder', peak_encoder)
    has_decoder = hasattr(actual_encoder, 'decode')
    arch_type = actual_encoder.__class__.__name__

    if not has_decoder:
        print(f"\n⚠️  {arch_type} doesn't have decoder - cannot compute reconstruction!")
        return None

    print(f"\n🔬 Evaluating peak reconstruction ({arch_type})...")

    all_reconstructed = []

    with torch.no_grad():
        # Process in batches to avoid memory issues
        for i in range(0, len(peak_data), batch_size):
            batch = peak_data[i:i+batch_size]

            # Encode to latent space
            if arch_type == 'VAEPeakEncoder':
                encoded, _, _ = actual_encoder(batch)  # VAE returns (z, mu, logvar)
            else:
                encoded = actual_encoder(batch)  # DAE/others return z directly

            # Decode back to peak space
            reconstructed = actual_encoder.decode(encoded)
            all_reconstructed.append(reconstructed.cpu())

    # Concatenate batches
    reconstructed = torch.cat(all_reconstructed, dim=0).numpy()
    true_peaks = peak_data.cpu().numpy()

    print(f"  Reconstructed shape: {reconstructed.shape}")
    print(f"  True peaks shape: {true_peaks.shape}")

    # Compute overall metrics
    mse = mean_squared_error(true_peaks.flatten(), reconstructed.flatten())
    mae = mean_absolute_error(true_peaks.flatten(), reconstructed.flatten())

    # Per-cell correlations (how well each cell is reconstructed)
    print("\n  Computing per-cell correlations...")
    cell_pearson = []
    cell_spearman = []

    for i in range(len(true_peaks)):
        # Skip if all zeros (avoid division by zero)
        if true_peaks[i].sum() == 0 or reconstructed[i].sum() == 0:
            continue

        try:
            p_corr, _ = pearsonr(true_peaks[i], reconstructed[i])
            s_corr, _ = spearmanr(true_peaks[i], reconstructed[i])
            cell_pearson.append(p_corr)
            cell_spearman.append(s_corr)
        except:
            continue

    # Per-peak correlations (how well each peak is predicted across cells)
    print("  Computing per-peak correlations...")
    peak_pearson = []
    peak_spearman = []

    for j in range(true_peaks.shape[1]):
        # Skip if all zeros
        if true_peaks[:, j].sum() == 0 or reconstructed[:, j].sum() == 0:
            continue

        try:
            p_corr, _ = pearsonr(true_peaks[:, j], reconstructed[:, j])
            s_corr, _ = spearmanr(true_peaks[:, j], reconstructed[:, j])
            peak_pearson.append(p_corr)
            peak_spearman.append(s_corr)
        except:
            continue

    # AUROC for peak presence/absence (NEW METRIC!)
    print("  Computing per-peak AUROC (presence/absence)...")
    peak_auroc = []

    # Binarize true peaks: 0 = absent, 1 = present
    true_peaks_binary = (true_peaks > 0).astype(int)

    # Normalize reconstructed peaks to [0, 1] for probability scores
    # Use sigmoid-like normalization or min-max scaling per peak
    reconstructed_normalized = np.zeros_like(reconstructed)
    for j in range(reconstructed.shape[1]):
        recon_peak = reconstructed[:, j]
        # Min-max normalization to [0, 1]
        if recon_peak.max() > recon_peak.min():
            reconstructed_normalized[:, j] = (recon_peak - recon_peak.min()) / (recon_peak.max() - recon_peak.min())
        else:
            reconstructed_normalized[:, j] = 0.5  # All same value

    for j in range(true_peaks.shape[1]):
        true_binary = true_peaks_binary[:, j]

        # Skip if all 0s or all 1s (AUROC undefined)
        if true_binary.sum() == 0 or true_binary.sum() == len(true_binary):
            continue

        try:
            auroc = roc_auc_score(true_binary, reconstructed_normalized[:, j])
            peak_auroc.append(auroc)
        except:
            continue

    # Overall AUROC (flatten all peaks)
    print("  Computing overall AUROC...")
    overall_auroc = None
    try:
        # Flatten and compute overall AUROC across all peaks
        true_flat = true_peaks_binary.flatten()
        recon_flat = reconstructed_normalized.flatten()

        # Remove positions where all cells have same value
        valid_mask = ~((true_flat == 0).all() | (true_flat == 1).all())
        if valid_mask.sum() > 0:
            overall_auroc = roc_auc_score(true_flat, recon_flat)
    except:
        pass

    # Precision, Recall, F1-Score (NEW METRICS!)
    print("  Computing Precision/Recall/F1 (peak presence/absence)...")
    overall_precision = None
    overall_recall = None
    overall_f1 = None
    overall_avg_precision = None

    try:
        # Flatten true and predicted peaks
        true_flat = true_peaks_binary.flatten()
        # Threshold normalized reconstruction at 0.5 for binary prediction
        pred_flat = (reconstructed_normalized.flatten() > 0.5).astype(int)

        # Compute precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_flat, pred_flat, average='binary', zero_division=0
        )
        overall_precision = precision
        overall_recall = recall
        overall_f1 = f1

        # Average precision (AUPRC) - better for imbalanced data
        overall_avg_precision = average_precision_score(true_flat, reconstructed_normalized.flatten())

    except Exception as e:
        print(f"    Warning: Could not compute precision/recall: {e}")

    results = {
        'reconstructed': reconstructed,
        'true_peaks': true_peaks,
        'cell_pearson': np.array(cell_pearson),
        'cell_spearman': np.array(cell_spearman),
        'peak_pearson': np.array(peak_pearson),
        'peak_spearman': np.array(peak_spearman),
        'peak_auroc': np.array(peak_auroc),
        'overall_auroc': overall_auroc,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1': overall_f1,
        'overall_avg_precision': overall_avg_precision,
        'mse': mse,
        'mae': mae
    }

    return results


def evaluate_classification_quality(model, peak_data, labels, device='cuda', batch_size=256):
    """Evaluate classification performance."""
    model.eval()

    all_preds = []
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(peak_data), batch_size):
            batch = peak_data[i:i+batch_size]

            # Get predictions
            outputs = model(source_data=None, target_data=batch, alpha=0.0)
            logits = outputs['target_class_pred']
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

    predictions = torch.cat(all_preds).numpy()
    probabilities = torch.cat(all_probs).numpy()
    true_labels = labels.cpu().numpy()

    accuracy = (predictions == true_labels).mean()

    # Per-class accuracy
    class_names = ['G1', 'S', 'G2M']
    class_acc = {}
    for i, name in enumerate(class_names):
        mask = true_labels == i
        if mask.sum() > 0:
            class_acc[name] = (predictions[mask] == i).mean()

    # Confidence scores
    max_probs = probabilities.max(axis=1)
    avg_confidence = max_probs.mean()

    return {
        'accuracy': accuracy,
        'class_accuracy': class_acc,
        'avg_confidence': avg_confidence,
        'predictions': predictions,
        'probabilities': probabilities,
        'max_probs': max_probs
    }


def print_summary(recon_results, cls_results, config):
    """Print comprehensive summary."""
    print("\n" + "="*70)
    print("📊 PEAK RECONSTRUCTION & CLASSIFICATION ANALYSIS")
    print("="*70)

    print(f"\n🏗️  MODEL ARCHITECTURE:")
    print(f"  Peak encoder: {config['peak_arch_type'].upper()}")
    print(f"  Source model: {config['source_model']}")
    print(f"  Peak mapper: {'Yes' if config.get('use_peak_mapper') else 'No'}")
    print(f"  Peak filter: {config.get('peak_filter_threshold', 0.2) * 100}%")
    print(f"  Number of peaks: {config['n_peaks']}")

    # Classification results
    print(f"\n🎯 CLASSIFICATION PERFORMANCE:")
    print(f"  Overall accuracy: {cls_results['accuracy']:.2%}")
    print(f"  Average confidence: {cls_results['avg_confidence']:.2%}")
    print(f"\n  Per-class accuracy:")
    for name, acc in cls_results['class_accuracy'].items():
        print(f"    {name:3s}: {acc:.2%}")

    # Reconstruction results (if available)
    if recon_results:
        print(f"\n🔬 PEAK RECONSTRUCTION QUALITY:")
        print(f"\n  Overall Metrics (Peak Presence/Absence):")
        if recon_results['overall_auroc'] is not None:
            print(f"    AUROC (ROC):        {recon_results['overall_auroc']:.4f}")
        if recon_results['overall_avg_precision'] is not None:
            print(f"    Avg Precision (PR): {recon_results['overall_avg_precision']:.4f}")
        if recon_results['overall_precision'] is not None:
            print(f"    Precision:          {recon_results['overall_precision']:.4f}")
        if recon_results['overall_recall'] is not None:
            print(f"    Recall:             {recon_results['overall_recall']:.4f}")
        if recon_results['overall_f1'] is not None:
            print(f"    F1-Score:           {recon_results['overall_f1']:.4f}")

        print(f"\n  Peak Count Reconstruction:")
        print(f"    MSE: {recon_results['mse']:.4f}")
        print(f"    MAE: {recon_results['mae']:.4f}")

        print(f"\n  Per-Cell Correlation (n={len(recon_results['cell_pearson'])}):")
        print(f"    Pearson  - Mean: {recon_results['cell_pearson'].mean():.4f}  Median: {np.median(recon_results['cell_pearson']):.4f}  Std: {recon_results['cell_pearson'].std():.4f}")
        print(f"    Spearman - Mean: {recon_results['cell_spearman'].mean():.4f}  Median: {np.median(recon_results['cell_spearman']):.4f}  Std: {recon_results['cell_spearman'].std():.4f}")

        print(f"\n  Per-Peak Metrics (n={len(recon_results['peak_pearson'])}):")
        print(f"    Pearson  - Mean: {recon_results['peak_pearson'].mean():.4f}  Median: {np.median(recon_results['peak_pearson']):.4f}  Std: {recon_results['peak_pearson'].std():.4f}")
        print(f"    Spearman - Mean: {recon_results['peak_spearman'].mean():.4f}  Median: {np.median(recon_results['peak_spearman']):.4f}  Std: {recon_results['peak_spearman'].std():.4f}")
        if len(recon_results['peak_auroc']) > 0:
            print(f"    AUROC    - Mean: {recon_results['peak_auroc'].mean():.4f}  Median: {np.median(recon_results['peak_auroc']):.4f}  Std: {recon_results['peak_auroc'].std():.4f}")

        # Interpretation
        print(f"\n💡 INTERPRETATION:")
        cell_corr = recon_results['cell_pearson'].mean()
        peak_auroc = recon_results['peak_auroc'].mean() if len(recon_results['peak_auroc']) > 0 else None
        accuracy = cls_results['accuracy']

        # Quality assessment
        if peak_auroc is not None and peak_auroc > 0.7:
            print(f"  ✅ GOOD peak presence detection (AUROC={peak_auroc:.3f})")
            print(f"     Model accurately identifies peak presence/absence!")
        elif peak_auroc is not None and peak_auroc > 0.6:
            print(f"  ⚠️  MODERATE peak presence detection (AUROC={peak_auroc:.3f})")
            print(f"     Model detects peaks above random baseline")
        elif peak_auroc is not None:
            print(f"  ❌ POOR peak presence detection (AUROC={peak_auroc:.3f})")
            print(f"     Model struggles to detect peak presence")

        if cell_corr > 0.5:
            print(f"  ✅ GOOD peak count reconstruction (r={cell_corr:.3f})")
            print(f"     Model learns meaningful peak count patterns!")
        elif cell_corr > 0.3:
            print(f"  ⚠️  MODERATE peak count reconstruction (r={cell_corr:.3f})")
            print(f"     Model captures some peak patterns but with noise")
        else:
            print(f"  ❌ POOR peak count reconstruction (r={cell_corr:.3f})")
            print(f"     Model struggles to learn peak patterns")

        # Key insights combining all metrics
        if accuracy < 0.5 and (cell_corr > 0.4 or (peak_auroc is not None and peak_auroc > 0.65)):
            print(f"\n  🔍 KEY FINDING:")
            auroc_str = f", AUROC={peak_auroc:.3f}" if peak_auroc is not None else ""
            print(f"     Low classification ({accuracy:.1%}) BUT decent peak learning (r={cell_corr:.3f}{auroc_str})")
            print(f"     → Model DOES learn peak patterns successfully!")
            print(f"     → Classification is HARDER than peak reconstruction")
            print(f"     → 40% accuracy doesn't mean model is broken!")
            print(f"     → Consider: Class imbalance, noisy labels, or inherently hard boundaries")
        elif accuracy < 0.5 and cell_corr < 0.3 and (peak_auroc is None or peak_auroc < 0.6):
            print(f"\n  🔍 KEY FINDING:")
            auroc_str = f", AUROC={peak_auroc:.3f}" if peak_auroc is not None else ""
            print(f"     Low classification ({accuracy:.1%}) AND poor peak learning (r={cell_corr:.3f}{auroc_str})")
            print(f"     → Model fails to learn meaningful peak representations")
            print(f"     → Consider: Longer training, better architecture, or data quality issues")

    print("\n" + "="*70)


def save_results(recon_results, cls_results, config, output_dir):
    """Save detailed results to files."""
    output_dir = Path(output_dir)

    # Save summary statistics
    summary = {
        'model': {
            'peak_arch': config['peak_arch_type'],
            'source_model': config['source_model'],
            'use_mapper': config.get('use_peak_mapper', False),
            'peak_filter': config.get('peak_filter_threshold', 0.2),
            'n_peaks': config['n_peaks']
        },
        'classification': {
            'accuracy': float(cls_results['accuracy']),
            'avg_confidence': float(cls_results['avg_confidence']),
            'class_accuracy': {k: float(v) for k, v in cls_results['class_accuracy'].items()}
        }
    }

    if recon_results:
        summary['reconstruction'] = {
            'mse': float(recon_results['mse']),
            'mae': float(recon_results['mae']),
            'overall_auroc': float(recon_results['overall_auroc']) if recon_results['overall_auroc'] is not None else None,
            'overall_avg_precision': float(recon_results['overall_avg_precision']) if recon_results['overall_avg_precision'] is not None else None,
            'overall_precision': float(recon_results['overall_precision']) if recon_results['overall_precision'] is not None else None,
            'overall_recall': float(recon_results['overall_recall']) if recon_results['overall_recall'] is not None else None,
            'overall_f1': float(recon_results['overall_f1']) if recon_results['overall_f1'] is not None else None,
            'cell_correlation': {
                'pearson_mean': float(recon_results['cell_pearson'].mean()),
                'pearson_median': float(np.median(recon_results['cell_pearson'])),
                'pearson_std': float(recon_results['cell_pearson'].std()),
                'spearman_mean': float(recon_results['cell_spearman'].mean()),
                'spearman_median': float(np.median(recon_results['cell_spearman'])),
                'spearman_std': float(recon_results['cell_spearman'].std())
            },
            'peak_metrics': {
                'pearson_mean': float(recon_results['peak_pearson'].mean()),
                'pearson_median': float(np.median(recon_results['peak_pearson'])),
                'pearson_std': float(recon_results['peak_pearson'].std()),
                'spearman_mean': float(recon_results['peak_spearman'].mean()),
                'spearman_median': float(np.median(recon_results['peak_spearman'])),
                'spearman_std': float(recon_results['peak_spearman'].std()),
                'auroc_mean': float(recon_results['peak_auroc'].mean()) if len(recon_results['peak_auroc']) > 0 else None,
                'auroc_median': float(np.median(recon_results['peak_auroc'])) if len(recon_results['peak_auroc']) > 0 else None,
                'auroc_std': float(recon_results['peak_auroc'].std()) if len(recon_results['peak_auroc']) > 0 else None
            }
        }

    # Save summary JSON
    with open(output_dir / 'reconstruction_analysis.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n💾 Saved summary to: {output_dir / 'reconstruction_analysis.json'}")

    # Save per-cell correlations
    if recon_results:
        corr_df = pd.DataFrame({
            'cell_pearson': recon_results['cell_pearson'],
            'cell_spearman': recon_results['cell_spearman']
        })
        corr_df.to_csv(output_dir / 'per_cell_correlations.csv', index=False)
        print(f"💾 Saved per-cell correlations to: {output_dir / 'per_cell_correlations.csv'}")

        # Save per-peak metrics (correlation + AUROC)
        # Note: Arrays may have different lengths due to filtering
        peak_metrics_data = {
            'peak_pearson': recon_results['peak_pearson'],
            'peak_spearman': recon_results['peak_spearman']
        }

        # Add AUROC if available (pad with NaN if different length)
        if len(recon_results['peak_auroc']) > 0:
            # Match the length of correlation arrays
            if len(recon_results['peak_auroc']) == len(recon_results['peak_pearson']):
                peak_metrics_data['peak_auroc'] = recon_results['peak_auroc']
            else:
                # Different lengths - save separately
                print(f"⚠️  AUROC array length mismatch, saving separately")

        peak_corr_df = pd.DataFrame(peak_metrics_data)
        peak_corr_df.to_csv(output_dir / 'per_peak_metrics.csv', index=False)
        print(f"💾 Saved per-peak metrics to: {output_dir / 'per_peak_metrics.csv'}")

        # Save AUROC separately if needed
        if len(recon_results['peak_auroc']) > 0 and len(recon_results['peak_auroc']) != len(recon_results['peak_pearson']):
            auroc_df = pd.DataFrame({'peak_auroc': recon_results['peak_auroc']})
            auroc_df.to_csv(output_dir / 'per_peak_auroc.csv', index=False)
            print(f"💾 Saved per-peak AUROC to: {output_dir / 'per_peak_auroc.csv'}")


def create_plots(recon_results, cls_results, output_dir):
    """Create visualization plots."""
    if not recon_results:
        print("\n⚠️  No reconstruction results - skipping plots")
        return

    output_dir = Path(output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Per-cell correlation distribution
    ax = axes[0, 0]
    ax.hist(recon_results['cell_pearson'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(recon_results['cell_pearson'].mean(), color='red', linestyle='--',
               label=f"Mean: {recon_results['cell_pearson'].mean():.3f}")
    ax.axvline(np.median(recon_results['cell_pearson']), color='green', linestyle='--',
               label=f"Median: {np.median(recon_results['cell_pearson']):.3f}")
    ax.set_xlabel('Pearson Correlation')
    ax.set_ylabel('Number of Cells')
    ax.set_title('Per-Cell Peak Reconstruction Quality')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Per-peak correlation distribution
    ax = axes[0, 1]
    ax.hist(recon_results['peak_pearson'], bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax.axvline(recon_results['peak_pearson'].mean(), color='red', linestyle='--',
               label=f"Mean: {recon_results['peak_pearson'].mean():.3f}")
    ax.axvline(np.median(recon_results['peak_pearson']), color='green', linestyle='--',
               label=f"Median: {np.median(recon_results['peak_pearson']):.3f}")
    ax.set_xlabel('Pearson Correlation')
    ax.set_ylabel('Number of Peaks')
    ax.set_title('Per-Peak Reconstruction Quality')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Confidence distribution by correctness
    ax = axes[1, 0]
    correct = cls_results['predictions'] == cls_results['probabilities'].argmax(axis=1)
    ax.hist(cls_results['max_probs'][correct], bins=30, alpha=0.6, label='Correct', color='green', edgecolor='black')
    ax.hist(cls_results['max_probs'][~correct], bins=30, alpha=0.6, label='Incorrect', color='red', edgecolor='black')
    ax.set_xlabel('Prediction Confidence')
    ax.set_ylabel('Number of Cells')
    ax.set_title('Classification Confidence Distribution')
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. Summary bar chart
    ax = axes[1, 1]
    metrics = {
        'Classification\nAccuracy': cls_results['accuracy'],
        'Avg\nConfidence': cls_results['avg_confidence'],
        'Cell Corr\n(Pearson)': recon_results['cell_pearson'].mean(),
        'Peak Corr\n(Pearson)': recon_results['peak_pearson'].mean()
    }
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    bars = ax.bar(metrics.keys(), metrics.values(), color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Score')
    ax.set_title('Summary Metrics')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plot_path = output_dir / 'reconstruction_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Saved plots to: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze peak reconstruction quality')
    parser.add_argument('--model_dir', required=True,
                       help='Path to trained model directory (e.g., DANN_output/.../train_baseline_...)')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for evaluation')

    args = parser.parse_args()

    # Load model and data
    model, peak_data, peak_original, labels, config, peak_mapper = load_model_and_data(
        args.model_dir,
        device=args.device
    )

    # Evaluate reconstruction
    recon_results = evaluate_peak_reconstruction(
        model,
        peak_data,
        device=args.device,
        batch_size=args.batch_size
    )

    # Evaluate classification
    cls_results = evaluate_classification_quality(
        model,
        peak_data,
        labels,
        device=args.device,
        batch_size=args.batch_size
    )

    # Print summary
    print_summary(recon_results, cls_results, config)

    # Save results
    output_dir = Path(args.model_dir)
    save_results(recon_results, cls_results, config, output_dir)

    # Create plots
    create_plots(recon_results, cls_results, output_dir)

    print(f"\n✅ Analysis complete!")


if __name__ == '__main__':
    main()
