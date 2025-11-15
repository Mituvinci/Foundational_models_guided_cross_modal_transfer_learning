#!/usr/bin/env python3
"""
Generate comprehensive results CSV with all hyperparameters and metrics.

Output: COMPREHENSIVE_RESULTS.csv
- Model path
- All hyperparameters from config.json
- Training metrics (REH GEX val, REH PEAK val/train)
- Test metrics (SUP GEX, SUP PEAK)
- Training time/memory (if available)
- Average accuracy for ranking
"""

import pandas as pd
import json
import re
from pathlib import Path
import numpy as np

BASE_DIR = Path("/users/ha00014/Halimas_projects/Transfer-Learning_and_LLMs")
OUTPUT_DIR = BASE_DIR / "DANN_output"


def parse_metrics_file(filepath):
    """Parse metrics from .txt file."""
    if not filepath.exists():
        return {}

    with open(filepath) as f:
        content = f.read()

    metrics = {}

    # Extract accuracy, balanced accuracy, kappa
    if m := re.search(r'Accuracy:\s+([\d.]+)', content):
        metrics['accuracy'] = float(m.group(1))
    if m := re.search(r'Balanced Accuracy:\s+([\d.]+)', content):
        metrics['balanced_accuracy'] = float(m.group(1))
    if m := re.search(r'Kappa:\s+([-\d.]+)', content):
        metrics['kappa'] = float(m.group(1))

    # Extract per-class metrics
    for phase in ['G1', 'G2M', 'S']:
        pattern = rf'{phase}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
        if m := re.search(pattern, content):
            metrics[f'{phase}_precision'] = float(m.group(1))
            metrics[f'{phase}_recall'] = float(m.group(2))
            metrics[f'{phase}_f1'] = float(m.group(3))

    # Extract macro avg
    pattern = r'macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
    if m := re.search(pattern, content):
        metrics['macro_precision'] = float(m.group(1))
        metrics['macro_recall'] = float(m.group(2))
        metrics['macro_f1'] = float(m.group(3))

    return metrics


def gather_comprehensive_results():
    """Gather all results with hyperparameters."""
    results = []

    # Find all models
    model_patterns = ["output_*_FIXED/dual_encoder/train_*"]

    for pattern in model_patterns:
        for model_dir in OUTPUT_DIR.glob(pattern):
            if not (model_dir / "config.json").exists():
                continue

            print(f"Processing: {model_dir.relative_to(BASE_DIR)}")

            # Load config (ALL hyperparameters)
            with open(model_dir / "config.json") as f:
                config = json.load(f)

            # Basic model info
            row = {
                'model_path': str(model_dir.relative_to(BASE_DIR)),
                'model_name': model_dir.name,
                'architecture': config.get('peak_arch_type', 'N/A'),
                'source_model': config.get('source_model', 'N/A'),
                'divergence': config.get('divergence_type', 'N/A'),
            }

            # ALL hyperparameters from config
            hyperparams = {
                'shared_dim': config.get('shared_dim', 'N/A'),
                'peak_hidden_dim': config.get('peak_hidden_dim', 'N/A'),
                'peak_intermediate_dim': config.get('peak_intermediate_dim', 'N/A'),
                'peak_num_layers': config.get('peak_num_layers', 'N/A'),
                'peak_dropout': config.get('peak_dropout', 'N/A'),
                'dropout': config.get('dropout', 'N/A'),
                'activation': config.get('activation', 'N/A'),
                'use_batch_norm': config.get('use_batch_norm', 'N/A'),
                'residual_type': config.get('residual_type', 'N/A'),
                'beta_vae': config.get('beta_vae', 'N/A'),
                'noise_level': config.get('noise_level', 'N/A'),
                'temperature': config.get('temperature', 'N/A'),
                'projection_dim': config.get('projection_dim', 'N/A'),
                'aug_dropout_prob': config.get('aug_dropout_prob', 'N/A'),
                'learning_rate': config.get('learning_rate', 'N/A'),
                'batch_size': config.get('batch_size', 'N/A'),
                'n_peaks': config.get('n_peaks', 'N/A'),
                'peak_filter_threshold': config.get('peak_filter_threshold', 'N/A'),
                'use_peak_mapper': config.get('use_peak_mapper', False),
                'peak_mapper_method': config.get('peak_mapper_method', 'N/A'),
                'disc_hidden': config.get('disc_hidden', 'N/A'),
                'disc_layers': config.get('disc_layers', 'N/A'),
                'cls_hidden': config.get('cls_hidden', 'N/A'),
                'cls_layers': config.get('cls_layers', 'N/A'),
                'use_spectral_norm': config.get('use_spectral_norm', False),
                'use_cdann': config.get('use_cdann', False),
                'lambda_domain': config.get('lambda_domain', 'N/A'),
                'lambda_target_class': config.get('lambda_target_class', 'N/A'),
                'lambda_target_class_warmup': config.get('lambda_target_class_warmup', 'N/A'),
                'use_lora': config.get('use_lora', False),
                'lora_rank': config.get('lora_rank', 'N/A'),
                'lora_alpha': config.get('lora_alpha', 'N/A'),
            }
            row.update(hyperparams)

            # REH GEX validation metrics
            reh_gex_val = parse_metrics_file(model_dir / "validation_metrics.txt")
            for key, val in reh_gex_val.items():
                row[f'REH_GEX_val_{key}'] = val

            # REH PEAK validation metrics (held-out)
            reh_peak_val = parse_metrics_file(model_dir / "reh_peak_val_HELDOUT_metrics.txt")
            for key, val in reh_peak_val.items():
                row[f'REH_PEAK_val_{key}'] = val

            # REH PEAK training metrics
            reh_peak_train = parse_metrics_file(model_dir / "reh_peak_train_metrics.txt")
            for key, val in reh_peak_train.items():
                row[f'REH_PEAK_train_{key}'] = val

            # SUP GEX test metrics
            sup_gex = parse_metrics_file(model_dir / "test_metrics.txt")
            for key, val in sup_gex.items():
                row[f'SUP_GEX_{key}'] = val

            # SUP PEAK test metrics
            sup_peak = parse_metrics_file(model_dir / "sup_peak_metrics.txt")
            for key, val in sup_peak.items():
                row[f'SUP_PEAK_{key}'] = val

            # Reconstruction metrics (if available)
            recon_file = model_dir / "reconstruction_analysis.json"
            if recon_file.exists():
                with open(recon_file) as f:
                    recon = json.load(f)

                # Classification metrics
                classification = recon.get('classification', {})
                row['SUP_PEAK_recon_accuracy'] = classification.get('accuracy', np.nan)
                row['SUP_PEAK_avg_confidence'] = classification.get('avg_confidence', np.nan)

                # Per-class accuracy
                class_acc = classification.get('class_accuracy', {})
                row['SUP_PEAK_G1_acc'] = class_acc.get('G1', np.nan)
                row['SUP_PEAK_S_acc'] = class_acc.get('S', np.nan)
                row['SUP_PEAK_G2M_acc'] = class_acc.get('G2M', np.nan)

                # Reconstruction metrics (only for VAE/DAE)
                reconstruction = recon.get('reconstruction', {})
                if reconstruction:
                    row['SUP_PEAK_MSE'] = reconstruction.get('mse', np.nan)
                    row['SUP_PEAK_MAE'] = reconstruction.get('mae', np.nan)
                    row['SUP_PEAK_AUROC'] = reconstruction.get('overall_auroc', np.nan)
                    row['SUP_PEAK_avg_precision'] = reconstruction.get('overall_avg_precision', np.nan)
                    row['SUP_PEAK_recon_precision'] = reconstruction.get('overall_precision', np.nan)
                    row['SUP_PEAK_recon_recall'] = reconstruction.get('overall_recall', np.nan)
                    row['SUP_PEAK_recon_f1'] = reconstruction.get('overall_f1', np.nan)

                    # Correlations
                    cell_corr = reconstruction.get('cell_correlation', {})
                    row['SUP_PEAK_cell_pearson_mean'] = cell_corr.get('pearson_mean', np.nan)
                    row['SUP_PEAK_cell_pearson_median'] = cell_corr.get('pearson_median', np.nan)
                    row['SUP_PEAK_cell_spearman_mean'] = cell_corr.get('spearman_mean', np.nan)
                    row['SUP_PEAK_cell_spearman_median'] = cell_corr.get('spearman_median', np.nan)

                    peak_corr = reconstruction.get('peak_metrics', {})
                    row['SUP_PEAK_peak_pearson_mean'] = peak_corr.get('pearson_mean', np.nan)
                    row['SUP_PEAK_peak_spearman_mean'] = peak_corr.get('spearman_mean', np.nan)
                    row['SUP_PEAK_peak_auroc_mean'] = peak_corr.get('auroc_mean', np.nan)

            # Training time and memory (placeholders - fill manually or from SLURM sacct)
            row['training_time_hours'] = np.nan
            row['peak_memory_GB'] = np.nan

            # Calculate average accuracy for ranking
            sup_gex_acc = row.get('SUP_GEX_accuracy', np.nan)
            sup_peak_acc = row.get('SUP_PEAK_accuracy', np.nan)

            if not np.isnan(sup_gex_acc) and not np.isnan(sup_peak_acc):
                row['avg_test_accuracy'] = (sup_gex_acc + sup_peak_acc) / 2
            else:
                row['avg_test_accuracy'] = np.nan

            results.append(row)

    df = pd.DataFrame(results)

    # Sort by average test accuracy (descending)
    df = df.sort_values('avg_test_accuracy', ascending=False, na_position='last')

    return df


if __name__ == "__main__":
    print("="*70)
    print("GENERATING COMPREHENSIVE RESULTS")
    print("="*70)

    df = gather_comprehensive_results()

    output_path = BASE_DIR / "COMPREHENSIVE_RESULTS.csv"
    df.to_csv(output_path, index=False)

    print(f"\n✅ Saved: {output_path}")
    print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")

    # Show top 5 models
    print("\n" + "="*70)
    print("TOP 5 MODELS (by average SUP_GEX + SUP_PEAK accuracy)")
    print("="*70)

    top5 = df.head(5)
    for idx, row in top5.iterrows():
        print(f"\n{idx+1}. {row['model_name']}")
        print(f"   Architecture: {row['architecture']}")
        print(f"   Divergence: {row['divergence']}")
        print(f"   Activation: {row['activation']}")
        print(f"   SUP_GEX_accuracy: {row.get('SUP_GEX_accuracy', 'N/A'):.4f}")
        print(f"   SUP_PEAK_accuracy: {row.get('SUP_PEAK_accuracy', 'N/A'):.4f}")
        print(f"   AVG Test Accuracy: {row['avg_test_accuracy']:.4f}")
        print(f"   Path: {row['model_path']}")

    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print("="*70)
