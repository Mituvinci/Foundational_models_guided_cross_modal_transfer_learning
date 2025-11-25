# Cross-Modal Cell Cycle Classification via Dual-Encoder Transfer Learning

Official implementation of "Cross-Modal Cell Cycle Classification via Dual-Encoder Transfer Learning with Foundation Model-Guided Domain Adaptation" (AAAI 2025 Workshop - Under Review)

## Overview

Transfer cell cycle knowledge from scRNA-seq to scATAC-seq using foundation models (Geneformer-316M, scFoundation) and class-conditional domain-adversarial neural networks (CDANN).

**Key Features:**
- Dual-encoder architecture for cross-modal transfer (GEX → PEAK)
- Foundation model integration (Geneformer-316M, scFoundation, ensemble)
- 4 divergence measures (MMD, KL, Jensen-Shannon, CORAL)
- 5 peak encoder architectures (DAE, VAE, Contrastive, Hybrid, MLP)
- Semi-supervised training with paired 10x Multiome data
- Achieves 78-87% accuracy on scRNA-seq, 33-41% on scATAC-seq (competitive for sparse chromatin data)

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/cross-modal-cell-cycle-transfer.git
cd cross-modal-cell-cycle-transfer
pip install -r requirements.txt
```

### Requirements
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.7+ (for GPU acceleration)
- 32GB+ RAM (64GB recommended for large datasets)
- GPU with 16GB+ VRAM (A30/A40/A100 tested)

## Quick Start

### 1. Prepare Data

Download 10x Multiome data (scRNA-seq + scATAC-seq paired):
- REH cell line (training): 7,432 cells
- SUP-B15 cell line (testing): 7,728 cells

Expected data structure:
```
data/
├── REH_GEX.csv          (28,105 genes)
├── REH_PEAK.csv         (92,328 peaks)
├── REH_labels.csv       (cell cycle phases: G1/S/G2M)
├── SUP_GEX.csv
├── SUP_PEAK.csv
└── SUP_labels.csv
```

**Note:** Data not included in repository. Download from [GEO accession or provide link].

### 2. Hyperparameter Search (Optuna)

```bash
python LLM_TF/scripts/hyperparam_search_dual_encoder_dann.py \
  --source_model geneformer-316m \
  --arch_type dae \
  --divergence_type mmd \
  --tgt_csv data/REH_PEAK.csv \
  --tgt_label_csv data/REH_labels.csv \
  --sup_peak_csv data/SUP_PEAK.csv \
  --sup_peak_label_csv data/SUP_labels.csv \
  --n_trials 100 \
  --search_epochs 50 \
  --output_folder output/hyperparam_dae_mmd \
  --use_cdann \
  --use_peak_mapper \
  --peak_filter 0.05
```

**Key Arguments:**
- `--source_model`: Foundation model (`geneformer-316m`, `scfoundation`, `ensemble`)
- `--arch_type`: Peak encoder architecture (`dae`, `vae`, `contrastive`, `hybrid`, `mlp`)
- `--divergence_type`: Domain divergence (`mmd`, `kl`, `jensen`, `coral`)
- `--n_trials`: Optuna trials (100 recommended, 20 for quick test)
- `--peak_filter`: Frequency threshold (0.05 = 5%, retains peaks in ≥5% cells)

**Output:**
- `output/hyperparam_dae_mmd/best_params.json` (optimal hyperparameters)
- `output/hyperparam_dae_mmd/optuna_study.db` (trial history)

### 3. Train Final Model

```bash
python LLM_TF/scripts/train_dual_encoder_dann.py \
  --source_model geneformer-316m \
  --arch_type dae \
  --config_path output/hyperparam_dae_mmd/best_params.json \
  --tgt_csv data/REH_PEAK.csv \
  --tgt_label_csv data/REH_labels.csv \
  --sup_peak_csv data/SUP_PEAK.csv \
  --sup_peak_label_csv data/SUP_labels.csv \
  --output_folder output/train_dae_mmd \
  --epochs 1500 \
  --use_peak_mapper \
  --peak_filter 0.05
```

**Output:**
- `output/train_dae_mmd/best_model.pt` (trained checkpoint)
- `output/train_dae_mmd/predictions_*.csv` (REH/SUP predictions)
- `output/train_dae_mmd/training_history.json` (loss curves)

### 4. Evaluate Results

```bash
python LLM_TF/analysis_scripts/generate_comprehensive_results.py \
  --model_dirs output/train_* \
  --output_csv comprehensive_results.csv
```

## Architecture Details

### Dual-Encoder Framework

```
scRNA-seq (28,105 genes) → Geneformer-316M (frozen) → 1,152-D → Projection MLP → 512-D shared space
                                                                                      ↓
                                                                           Cell Cycle Classifier (G1/S/G2M)
                                                                                      ↓
scATAC-seq (25,027 peaks) → Peak Encoder (trainable) ────────────────────→ 512-D shared space
```

**Peak Encoder Options:**
1. **DAE** (Denoising Autoencoder): 25,027 → 1,024 → 512 → 1,024 → 25,027 (5% noise injection)
2. **VAE** (Variational Autoencoder): Probabilistic latent space with β-VAE (β=1.7)
3. **Contrastive**: NT-Xent loss with temperature scaling (τ=0.1)
4. **Hybrid**: Residual connections + GELU activation (2,048 hidden units)
5. **MLP**: Simple feedforward baseline

**Domain Adaptation:**
- Class-conditional discriminator (CDANN)
- Gradient reversal layer (GRL) with annealing
- 5 critical enhancements:
  1. Optuna-tuned λ_target_class (range: 1.0-15.0)
  2. Curriculum learning warmup (linear ramp)
  3. Target validation split (80/20)
  4. Target-aware early stopping (patience: 50)
  5. 5% peak frequency filtering

## Results Summary

| Model            | REH GEX Val | SUP GEX Test | REH PEAK Val | SUP PEAK Test |
|------------------|-------------|--------------|--------------|---------------|
| Hybrid-MMD       | 86.82%      | 80.97%       | 39.00%       | 36.56%        |
| VAE-MMD          | 86.82%      | 80.14%       | 39.54%       | 40.26%        |
| Contrastive-MMD  | 85.61%      | 79.57%       | 38.87%       | 41.14%        |
| DAE-MMD          | 85.27%      | 78.18%       | 40.07%       | 33.91%        |

**Key Findings:**
- MMD divergence consistently outperforms KL, Jensen-Shannon, CORAL
- 33-41% scATAC-seq accuracy is competitive given extreme sparsity (>85% zeros)
- Strong cross-dataset generalization (75-81% on SUP-B15)

## Advanced Usage

### Ensemble Configuration

```bash
python LLM_TF/scripts/hyperparam_search_dual_encoder_dann.py \
  --source_model geneformer-316m \
  --use_ensemble \
  --ensemble_model1 geneformer-316m \
  --ensemble_model2 scfoundation \
  --arch_type dae \
  --divergence_type mmd \
  [other args...]
```

Concatenates Geneformer-316M (1,152-D) + scFoundation (768-D) → 1,920-D source embeddings.

### Peak Mapper Options

**Coordinate-based mapping** (default):
```bash
--use_peak_mapper \
--peak_mapper_method overlap_50pct \
--peak_impute_strategy zero
```

Maps peaks between datasets using genomic coordinates (chr:start-end).

**Neural mapper** (optional, higher accuracy):
```bash
python LLM_TF/scripts/train_peak_mapper.py \
  --source_peaks data/REH_PEAK.csv \
  --target_peaks data/SUP_PEAK.csv \
  --output_folder output/peak_mapper
```

Trains lightweight adapter (25,027 → 256-D) for cross-dataset peak alignment.

## Repository Structure

```
.
├── LLM_TF/
│   ├── __init__.py
│   ├── data.py                          (dataset loading)
│   ├── losses.py                        (MMD, KL, Jensen-Shannon, CORAL)
│   ├── dual_encoder_dann.py             (main model)
│   ├── embedders/
│   │   ├── peak_embedder.py             (DAE, VAE, Contrastive, Hybrid)
│   │   └── unified_embedder.py          (architecture factory)
│   ├── loaders/
│   │   ├── geneformer_loader.py         (Geneformer-316M)
│   │   └── scfoundation_loader.py       (scFoundation)
│   ├── peak_mapper/
│   │   └── coordinate_mapper.py         (genomic coordinate alignment)
│   ├── scripts/
│   │   ├── hyperparam_search_dual_encoder_dann.py
│   │   └── train_dual_encoder_dann.py
│   └── analysis_scripts/
│       └── generate_comprehensive_results.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{anonymous2025crossmodal,
  title={Cross-Modal Cell Cycle Classification via Dual-Encoder Transfer Learning with Foundation Model-Guided Domain Adaptation},
  author={Anonymous},
  booktitle={Under Review},
  year={2025}
}
```

## License

MIT License

## Troubleshooting

**Out of memory errors:**
- Reduce batch size: `--batch_size 4`
- Use gradient checkpointing: `--gradient_checkpointing`
- Reduce peak filter: `--peak_filter 0.10` (10%, fewer peaks)

**Geneformer vocabulary errors:**
- Install correct version: `pip install transformers==4.30.0`
- Check vocab file: `~/.cache/huggingface/hub/models--ctheodoris--Geneformer/`

**Domain collapse (all predictions same class):**
- Increase λ_balance: `--lambda_balance 10.0`
- Use MMD divergence: `--divergence_type mmd`
- Enable entropy regularization: `--lambda_entropy 0.1`

## Contact

For questions or issues, please open a GitHub issue.

## Acknowledgments

- Geneformer pretrained model: Theodoris et al. (2023)
- scFoundation pretrained model: Cui et al. (2024)
- DANN framework: Ganin et al. (2016)
- 10x Genomics Multiome data
