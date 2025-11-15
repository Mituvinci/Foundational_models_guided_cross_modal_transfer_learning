#!/usr/bin/env python3
"""
Test script to verify all imports work correctly.
Run this after installing requirements.txt to check setup.
"""

import sys

def test_imports():
    """Test all critical imports."""
    print("Testing imports...")

    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False

    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False

    try:
        import optuna
        print(f"✓ Optuna {optuna.__version__}")
    except ImportError as e:
        print(f"✗ Optuna import failed: {e}")
        return False

    try:
        from LLM_TF.dual_encoder_dann import DualEncoderDANN
        print("✓ LLM_TF.dual_encoder_dann")
    except ImportError as e:
        print(f"✗ DualEncoderDANN import failed: {e}")
        return False

    try:
        from LLM_TF.losses import coral_loss, domain_bce_loss, stateless_grl
        print("✓ LLM_TF.losses")
    except ImportError as e:
        print(f"✗ Losses import failed: {e}")
        return False

    try:
        from LLM_TF.embedders.peak_embedder import create_peak_embedder
        print("✓ LLM_TF.embedders.peak_embedder")
    except ImportError as e:
        print(f"✗ Peak embedder import failed: {e}")
        return False

    try:
        from LLM_TF.loaders.geneformer_loader import GeneformerLoader
        print("✓ LLM_TF.loaders.geneformer_loader")
    except ImportError as e:
        print(f"✗ GeneformerLoader import failed: {e}")
        return False

    try:
        from LLM_TF.peak_mapper.coordinate_mapper import PeakCoordinateMapper
        print("✓ LLM_TF.peak_mapper.coordinate_mapper")
    except ImportError as e:
        print(f"✗ PeakCoordinateMapper import failed: {e}")
        return False

    print("\n✓ All imports successful!")
    print("\nGPU availability:")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  ✗ CUDA not available (CPU only)")

    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
