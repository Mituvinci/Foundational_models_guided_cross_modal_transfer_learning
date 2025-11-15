#!/usr/bin/env python3
"""
Train Neural Peak Mapper - Hybrid Approach (ChatGPT's Recommendation)

Trains dataset-specific adapter to project new dataset's peaks into
reference model's latent space using:
1. Adversarial alignment (domain discriminator + GRL)
2. Pseudo-label guidance (use paired GEX labels)
3. MMD distribution matching

Usage:
    python -m LLM_TF.scripts.train_peak_mapper \
        --source_csv modality_data/scMultiom_data/1_GD428_21136_Hu_REH_Parental_PEAK_RAW.csv \
        --target_csv modality_data/scMultiom_data/SUP_PEAK_RAW.csv \
        --target_label_csv modality_data/scMultiom_data/SUP_GEX_cellcycle.csv \
        --dann_model_path DANN_output/output_divergence_mmd/dual_encoder/train_xxx/best_model.pth \
        --output_dir peak_mappers/sup_mapper \
        --method hybrid \
        --n_trials 50 \
        --epochs 200 \
        --batch_size 128

Output:
    peak_mappers/sup_mapper/
        best_mapper.pth          - Trained mapper weights
        discriminator.pth        - Domain discriminator (for analysis)
        config.json              - Training configuration
        training_log.csv         - Loss curves, alignment metrics
        validation_report.txt    - Final accuracy on target dataset
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import optuna
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from LLM_TF.peak_mapper.neural_mapper import (
    NeuralPeakMapper,
    DomainDiscriminator,
    gradient_reversal,
    compute_mmd,
    compute_coral
)
from LLM_TF.dual_encoder_dann import DualEncoderDANN

# Label mapping
LABEL_MAP = {'G1': 0, 'S': 1, 'G2M': 2}
LABEL_MAP_INV = {0: 'G1', 1: 'S', 2: 'G2M'}


def load_peak_data(csv_path, label_csv_path=None):
    """
    Load PEAK data with optional labels.

    Args:
        csv_path: Path to PEAK CSV (cells × peaks)
        label_csv_path: Optional path to labels CSV (must have 'phase' column)

    Returns:
        X: Peak matrix (n_cells, n_peaks)
        y: Labels (n_cells,) or None if no labels
        peak_names: Peak column names
        cell_ids: Cell IDs
    """
    print(f"\n📊 Loading PEAK data: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)

    X = df.values.astype(np.float32)
    peak_names = df.columns.tolist()
    cell_ids = df.index.tolist()

    print(f"  Cells: {X.shape[0]:,}")
    print(f"  Peaks: {X.shape[1]:,}")

    # Load labels if provided
    y = None
    if label_csv_path:
        print(f"  Loading labels: {label_csv_path}")
        label_df = pd.read_csv(label_csv_path, index_col=0)

        # Match cell IDs
        common_cells = list(set(cell_ids) & set(label_df.index))
        if len(common_cells) == 0:
            raise ValueError("No matching cell IDs between PEAK and labels!")

        # Filter to common cells
        X_filtered = []
        y_filtered = []
        for cell_id in cell_ids:
            if cell_id in label_df.index:
                idx = cell_ids.index(cell_id)
                X_filtered.append(X[idx])
                phase = label_df.loc[cell_id, 'phase']
                y_filtered.append(LABEL_MAP[phase])

        X = np.array(X_filtered, dtype=np.float32)
        y = np.array(y_filtered)

        print(f"  Matched cells: {len(common_cells):,}")
        print(f"  Labels: G1={np.sum(y==0)}, S={np.sum(y==1)}, G2M={np.sum(y==2)}")

    return X, y, peak_names, cell_ids


def load_dann_model(model_path, device):
    """
    Load trained DANN model (frozen, for inference).

    Returns:
        model: Loaded DANN model
        config: Model configuration
    """
    print(f"\n📦 Loading DANN model: {model_path}")

    model_dir = Path(model_path).parent
    config_path = model_dir / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    print(f"  Config: {config_path}")
    print(f"  Source model: {config.get('source_model', 'N/A')}")
    print(f"  Peak arch: {config.get('peak_arch_type', 'N/A')}")
    print(f"  Shared dim: {config.get('shared_dim', 512)}")

    # Create model (use identity encoders since we only need classifier)
    from LLM_TF.peak_mapper.neural_mapper import NeuralPeakMapper

    # Map source model to dimension
    source_dim_map = {
        'geneformer-10m': 256,
        'geneformer-104m': 768,
        'geneformer-316m': 1152,
        'scfoundation': 768
    }
    source_model = config.get('source_model', 'geneformer-316m')
    source_dim = source_dim_map.get(source_model, 512)
    target_dim = config.get('peak_hidden_dim', 512)

    # Dummy encoders (we'll only use the classifier)
    dummy_source = nn.Identity()
    dummy_target = nn.Identity()

    model = DualEncoderDANN(
        source_encoder=dummy_source,
        target_encoder=dummy_target,
        source_dim=source_dim,
        target_dim=target_dim,
        shared_dim=config.get('shared_dim', 512),
        num_classes=3,
        disc_hidden=config.get('disc_hidden', 256),
        disc_layers=config.get('disc_layers', 2),
        cls_hidden=config.get('cls_hidden', 128),
        cls_layers=config.get('cls_layers', 2),
        dropout=config.get('dropout', 0.1),
        use_spectral_norm=config.get('use_spectral_norm', False),
        freeze_source_encoder=True,
        use_cdann=config.get('use_cdann', False)
    )

    # Load weights (strict=False to ignore target_encoder keys we don't need)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    model.eval()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    print("  ✓ Model loaded and frozen")

    return model, config


def train_mapper_hybrid(
    mapper,
    discriminator,
    source_loader,
    target_loader,
    dann_model,
    device,
    epochs=200,
    lambda_adv=1.0,
    lambda_mmd=1.0,
    lambda_cls=1.0,
    lr_mapper=1e-3,
    lr_disc=1e-4,
    grl_lambda=1.0,
    divergence='mmd'
):
    """
    Train mapper using hybrid approach:
    - Adversarial alignment (discriminator + GRL)
    - Pseudo-label guidance (classification loss)
    - Distribution matching (MMD or CORAL)

    Args:
        mapper: NeuralPeakMapper to train
        discriminator: DomainDiscriminator
        source_loader: DataLoader for source (REH) peaks
        target_loader: DataLoader for target (SUP) peaks + labels
        dann_model: Frozen DANN classifier
        device: cuda/cpu
        epochs: Number of epochs
        lambda_adv: Weight for adversarial loss
        lambda_mmd: Weight for MMD/CORAL loss
        lambda_cls: Weight for classification loss
        lr_mapper: Mapper learning rate
        lr_disc: Discriminator learning rate
        grl_lambda: GRL strength
        divergence: 'mmd' or 'coral'

    Returns:
        mapper: Trained mapper
        history: Training log
    """
    mapper.train()
    discriminator.train()

    # Optimizers
    optimizer_mapper = optim.Adam(mapper.parameters(), lr=lr_mapper)
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=lr_disc)

    # Loss functions
    criterion_cls = nn.CrossEntropyLoss()
    criterion_disc = nn.BCEWithLogitsLoss()

    history = {
        'epoch': [],
        'loss_mapper': [],
        'loss_disc': [],
        'loss_cls': [],
        'loss_adv': [],
        'loss_mmd': [],
        'disc_acc': [],
        'target_acc': []
    }

    best_target_acc = 0.0
    best_mapper_state = None

    for epoch in range(epochs):
        epoch_losses = {
            'mapper': 0.0,
            'disc': 0.0,
            'cls': 0.0,
            'adv': 0.0,
            'mmd': 0.0
        }

        disc_preds = []
        disc_labels = []
        target_preds = []
        target_true = []

        n_batches = min(len(source_loader), len(target_loader))
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)

        for batch_idx in range(n_batches):
            # Get batches
            source_batch = next(source_iter)[0].to(device)  # (batch, n_peaks_source)
            target_batch, target_labels = next(target_iter)
            target_batch = target_batch.to(device)  # (batch, n_peaks_target)
            target_labels = target_labels.to(device)

            batch_size = min(source_batch.size(0), target_batch.size(0))
            source_batch = source_batch[:batch_size]
            target_batch = target_batch[:batch_size]
            target_labels = target_labels[:batch_size]

            # ==================== Train Discriminator ====================
            optimizer_disc.zero_grad()

            # Source path: encoder → projection → shared space
            with torch.no_grad():
                source_latent = dann_model.target_encoder(source_batch)  # (batch, 256)
                source_shared = dann_model.target_projection(source_latent)  # (batch, 512) ← SHARED SPACE

            # Target path: mapper → projection → shared space
            target_latent = mapper(target_batch)  # (batch, 256)
            target_shared = dann_model.target_projection(target_latent)  # (batch, 512) ← SHARED SPACE

            # Discriminator predictions (works on 512-D shared space!)
            source_domain_pred = discriminator(source_shared.detach())
            target_domain_pred = discriminator(target_shared.detach())

            # Labels: source=0, target=1
            source_domain_label = torch.zeros(batch_size, 1, device=device)
            target_domain_label = torch.ones(batch_size, 1, device=device)

            # Discriminator loss
            loss_disc = (
                criterion_disc(source_domain_pred, source_domain_label) +
                criterion_disc(target_domain_pred, target_domain_label)
            ) / 2

            loss_disc.backward()
            optimizer_disc.step()

            # Track discriminator accuracy
            with torch.no_grad():
                source_pred_binary = (torch.sigmoid(source_domain_pred) > 0.5).float()
                target_pred_binary = (torch.sigmoid(target_domain_pred) > 0.5).float()
                disc_preds.extend(torch.cat([source_pred_binary, target_pred_binary], dim=0).cpu().numpy())
                disc_labels.extend(torch.cat([source_domain_label, target_domain_label], dim=0).cpu().numpy())

            # ==================== Train Mapper ====================
            optimizer_mapper.zero_grad()

            # Target path: mapper → projection → shared space
            target_latent = mapper(target_batch)  # (batch, 256)
            target_shared = dann_model.target_projection(target_latent)  # (batch, 512) ← SHARED SPACE

            # Loss 1: Classification (pseudo-label guidance)
            # Classifier works on 512-D shared space
            target_logits = dann_model.classifier(target_shared)
            loss_cls = criterion_cls(target_logits, target_labels)

            # Track target accuracy
            with torch.no_grad():
                preds = torch.argmax(target_logits, dim=1)
                target_preds.extend(preds.cpu().numpy())
                target_true.extend(target_labels.cpu().numpy())

            # Loss 2: Adversarial (confuse discriminator on SHARED SPACE)
            # Apply GRL to shared features (where discriminator operates)
            target_shared_reversed = gradient_reversal(target_shared, grl_lambda)
            target_domain_pred = discriminator(target_shared_reversed)
            # Want discriminator to predict 0 (source) for target samples
            loss_adv = criterion_disc(target_domain_pred, torch.zeros(batch_size, 1, device=device))

            # Loss 3: Distribution matching (MMD or CORAL on SHARED SPACE)
            # Align distributions where classifier actually operates!
            with torch.no_grad():
                source_latent = dann_model.target_encoder(source_batch)  # (batch, 256)
                source_shared = dann_model.target_projection(source_latent)  # (batch, 512)

            if divergence == 'mmd':
                loss_mmd = compute_mmd(source_shared, target_shared, kernel='rbf')
            elif divergence == 'coral':
                loss_mmd = compute_coral(source_shared, target_shared)
            else:
                raise ValueError(f"Unknown divergence: {divergence}")

            # Total mapper loss
            loss_mapper = (
                lambda_cls * loss_cls +
                lambda_adv * loss_adv +
                lambda_mmd * loss_mmd
            )

            loss_mapper.backward()
            optimizer_mapper.step()

            # Accumulate losses
            epoch_losses['mapper'] += loss_mapper.item()
            epoch_losses['disc'] += loss_disc.item()
            epoch_losses['cls'] += loss_cls.item()
            epoch_losses['adv'] += loss_adv.item()
            epoch_losses['mmd'] += loss_mmd.item()

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= n_batches

        # Metrics
        disc_acc = accuracy_score(disc_labels, disc_preds)
        target_acc = accuracy_score(target_true, target_preds)

        # Save history
        history['epoch'].append(epoch)
        history['loss_mapper'].append(epoch_losses['mapper'])
        history['loss_disc'].append(epoch_losses['disc'])
        history['loss_cls'].append(epoch_losses['cls'])
        history['loss_adv'].append(epoch_losses['adv'])
        history['loss_mmd'].append(epoch_losses['mmd'])
        history['disc_acc'].append(disc_acc)
        history['target_acc'].append(target_acc)

        # Save best model
        if target_acc > best_target_acc:
            best_target_acc = target_acc
            best_mapper_state = mapper.state_dict().copy()

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Mapper loss:   {epoch_losses['mapper']:.4f} (cls={epoch_losses['cls']:.4f}, adv={epoch_losses['adv']:.4f}, mmd={epoch_losses['mmd']:.4f})")
            print(f"  Disc loss:     {epoch_losses['disc']:.4f} (acc={disc_acc:.2%})")
            print(f"  Target acc:    {target_acc:.2%} (best={best_target_acc:.2%})")

    # Load best model
    mapper.load_state_dict(best_mapper_state)
    print(f"\n✓ Training complete! Best target acc: {best_target_acc:.2%}")

    return mapper, history


def objective(trial, args, source_data, target_data, target_labels, dann_model, device):
    """Optuna objective for hyperparameter search."""

    # Hyperparameters
    latent_dim = trial.suggest_categorical('latent_dim', [128, 256, 512])
    hidden_dims = trial.suggest_categorical('hidden_dims', [
        [1024, 512],
        [2048, 1024, 512],
        [512, 256]
    ])
    activation = trial.suggest_categorical('activation', ['relu', 'gelu', 'mish'])
    dropout = trial.suggest_float('dropout', 0.0, 0.3)
    lr_mapper = trial.suggest_float('lr_mapper', 1e-4, 1e-2, log=True)
    lr_disc = trial.suggest_float('lr_disc', 1e-5, 1e-3, log=True)
    lambda_adv = trial.suggest_float('lambda_adv', 0.1, 2.0)
    lambda_mmd = trial.suggest_float('lambda_mmd', 0.1, 2.0)
    lambda_cls = trial.suggest_float('lambda_cls', 0.5, 2.0)
    grl_lambda = trial.suggest_float('grl_lambda', 0.5, 2.0)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])

    # Create model
    mapper = NeuralPeakMapper(
        input_dim=target_data.shape[1],
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        dropout=dropout,
        use_batch_norm=True,
        use_spectral_norm=True
    ).to(device)

    # Discriminator works on shared space (512-D), not latent (256-D)!
    shared_dim = 512  # Your DANN's shared dimension
    discriminator = DomainDiscriminator(shared_dim=shared_dim).to(device)

    # Data loaders
    source_dataset = TensorDataset(torch.from_numpy(source_data))
    target_dataset = TensorDataset(torch.from_numpy(target_data), torch.from_numpy(target_labels))

    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Train
    mapper, history = train_mapper_hybrid(
        mapper=mapper,
        discriminator=discriminator,
        source_loader=source_loader,
        target_loader=target_loader,
        dann_model=dann_model,
        device=device,
        epochs=args.search_epochs,
        lambda_adv=lambda_adv,
        lambda_mmd=lambda_mmd,
        lambda_cls=lambda_cls,
        lr_mapper=lr_mapper,
        lr_disc=lr_disc,
        grl_lambda=grl_lambda,
        divergence=args.divergence
    )

    # Return best target accuracy
    best_acc = max(history['target_acc'])

    return best_acc


def main():
    parser = argparse.ArgumentParser(description="Train Neural Peak Mapper")

    # Data
    parser.add_argument('--source_csv', type=str, required=True, help='Source PEAK CSV (REH)')
    parser.add_argument('--target_csv', type=str, required=True, help='Target PEAK CSV (SUP)')
    parser.add_argument('--target_label_csv', type=str, required=True, help='Target labels CSV')

    # Model
    parser.add_argument('--dann_model_path', type=str, required=True, help='Path to trained DANN model')

    # Training
    parser.add_argument('--method', type=str, default='hybrid', choices=['adversarial', 'pseudo_label', 'hybrid'])
    parser.add_argument('--divergence', type=str, default='mmd', choices=['mmd', 'coral'])
    parser.add_argument('--n_trials', type=int, default=50, help='Optuna trials')
    parser.add_argument('--search_epochs', type=int, default=30, help='Epochs per trial')
    parser.add_argument('--final_epochs', type=int, default=200, help='Final training epochs')
    parser.add_argument('--batch_size', type=int, default=128)

    # Output
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")

    # Load data
    X_source, _, source_peaks, _ = load_peak_data(args.source_csv)
    X_target, y_target, target_peaks, _ = load_peak_data(args.target_csv, args.target_label_csv)

    if y_target is None:
        raise ValueError("Target labels required for training!")

    # Load DANN
    dann_model, dann_config = load_dann_model(args.dann_model_path, device)

    # Hyperparameter search
    print(f"\n🔍 Starting Optuna search ({args.n_trials} trials)...")

    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, args, X_source, X_target, y_target, dann_model, device),
        n_trials=args.n_trials
    )

    print(f"\n✓ Best trial: {study.best_trial.number}")
    print(f"  Best acc: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    # Final training with best params
    print(f"\n🚀 Final training ({args.final_epochs} epochs)...")

    best_params = study.best_params

    mapper = NeuralPeakMapper(
        input_dim=X_target.shape[1],
        latent_dim=best_params['latent_dim'],
        hidden_dims=best_params['hidden_dims'],
        activation=best_params['activation'],
        dropout=best_params['dropout'],
        use_batch_norm=True,
        use_spectral_norm=True
    ).to(device)

    # Discriminator works on shared space (512-D from DANN config)
    shared_dim = dann_config.get('shared_dim', 512)
    discriminator = DomainDiscriminator(shared_dim=shared_dim).to(device)

    # Data loaders
    source_dataset = TensorDataset(torch.from_numpy(X_source))
    target_dataset = TensorDataset(torch.from_numpy(X_target), torch.from_numpy(y_target))

    source_loader = DataLoader(source_dataset, batch_size=best_params['batch_size'], shuffle=True, drop_last=True)
    target_loader = DataLoader(target_dataset, batch_size=best_params['batch_size'], shuffle=True, drop_last=True)

    # Train
    mapper, history = train_mapper_hybrid(
        mapper=mapper,
        discriminator=discriminator,
        source_loader=source_loader,
        target_loader=target_loader,
        dann_model=dann_model,
        device=device,
        epochs=args.final_epochs,
        lambda_adv=best_params['lambda_adv'],
        lambda_mmd=best_params['lambda_mmd'],
        lambda_cls=best_params['lambda_cls'],
        lr_mapper=best_params['lr_mapper'],
        lr_disc=best_params['lr_disc'],
        grl_lambda=best_params['grl_lambda'],
        divergence=args.divergence
    )

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save mapper
    torch.save(mapper.state_dict(), output_dir / 'best_mapper.pth')
    torch.save(discriminator.state_dict(), output_dir / 'discriminator.pth')

    # Save config
    config = {
        'source_csv': args.source_csv,
        'target_csv': args.target_csv,
        'dann_model_path': args.dann_model_path,
        'method': args.method,
        'divergence': args.divergence,
        'input_dim': X_target.shape[1],
        'best_params': best_params,
        'best_trial_acc': study.best_value,
        'final_acc': max(history['target_acc'])
    }

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Save history
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / 'training_log.csv', index=False)

    # Validation report
    with open(output_dir / 'validation_report.txt', 'w') as f:
        f.write("PEAK MAPPER VALIDATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Source dataset: {args.source_csv}\n")
        f.write(f"Target dataset: {args.target_csv}\n")
        f.write(f"DANN model: {args.dann_model_path}\n")
        f.write(f"Method: {args.method}\n")
        f.write(f"Divergence: {args.divergence}\n\n")
        f.write(f"Best Optuna trial: {study.best_trial.number}\n")
        f.write(f"Best trial acc: {study.best_value:.4f}\n")
        f.write(f"Final training acc: {max(history['target_acc']):.4f}\n\n")
        f.write(f"Best hyperparameters:\n")
        for key, val in best_params.items():
            f.write(f"  {key}: {val}\n")

    print(f"\n💾 Saved to: {output_dir}")
    print(f"  ✓ best_mapper.pth")
    print(f"  ✓ discriminator.pth")
    print(f"  ✓ config.json")
    print(f"  ✓ training_log.csv")
    print(f"  ✓ validation_report.txt")


if __name__ == "__main__":
    main()
