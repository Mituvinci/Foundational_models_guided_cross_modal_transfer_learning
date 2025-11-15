# LLM_TF/losses.py
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def grl_lambda(step: int, total_steps: int, k: float = 5.0) -> float:
    """
    Standard DANN schedule: λ grows from 0→1 across training.
    k controls steepness.
    """
    p = step / max(total_steps, 1)
    return float(2.0 / (1.0 + math.exp(-k * p)) - 1.0)


def compute_class_weights(y: np.ndarray | Tensor, num_classes: int = 3) -> Tensor:
    """
    Inverse-frequency class weights for CrossEntropy.
    Returns a 1D tensor on CPU (caller can .to(device)).
    """
    if isinstance(y, Tensor):
        y = y.detach().cpu().numpy()
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    weights = counts.sum() / (counts + 1e-8)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)



def domain_bce_loss(dom_src_logits: Tensor, dom_tgt_logits: Tensor) -> Tensor:
    """Binary cross-entropy domain loss for source (0) and target (1)."""
    t0 = torch.zeros_like(dom_src_logits, requires_grad=False)
    t1 = torch.ones_like(dom_tgt_logits, requires_grad=False)
    return F.binary_cross_entropy_with_logits(dom_src_logits, t0) + \
           F.binary_cross_entropy_with_logits(dom_tgt_logits, t1)


def coral_loss(xs: Tensor, xt: Tensor) -> Tensor:
    """
    Standalone CORAL loss (if you compute it outside the head).
    """
    xs = xs - xs.mean(dim=0, keepdim=True)
    xt = xt - xt.mean(dim=0, keepdim=True)
    cs = (xs.T @ xs) / (max(xs.size(0) - 1, 1))
    ct = (xt.T @ xt) / (max(xt.size(0) - 1, 1))
    return ((cs - ct) ** 2).mean()


def stateless_grl(x: Tensor, lambd: float) -> Tensor:
    """
    Stateless Gradient Reversal Layer.
    Forward: returns x unchanged.
    Backward: multiplies gradients by -λ (like GRL in DANN).
    """
    return x + (-lambd * x - x).detach()
