# LLM_TF/embedder.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Sequence

import numpy as np
import torch
import torch.nn as nn

from .tokenizer import ScGPTTokenizerWrapper

# scGPT via TDC
from tdc import tdc_hf_interface
from tdc.model_server.tokenizers.scgpt import scGPTTokenizer

# LoRA implementations
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from LLM_TF.manual_analysis.manual_lora import load_scgpt_with_manual_lora

__all__ = [
    "load_scgpt_backbone",
    "load_scgpt_with_lora",
    "load_scgpt_with_manual_lora",
    "tokenize_matrix",
    "tokenize_domains_together",
    "embed_tokens_then_norm",
]

def load_scgpt_backbone():
    """
    Load the scGPT backbone (frozen) + tokenizer via TDC and return (model, tokenizer, device).
    Model is set to eval() and all params have requires_grad=False.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scgpt = tdc_hf_interface("scGPT")
    model = scgpt.load()
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    tokenizer = scGPTTokenizer()
    return model, tokenizer, device


class ScGPTCompatibleWrapper(nn.Module):
    """
    Wrapper to make PEFT-wrapped scGPT compatible by using scGPT's native signature
    while preserving LoRA functionality. This bypasses PEFT's argument conversion.
    """
    def __init__(self, peft_model):
        super().__init__()
        self.peft_model = peft_model
        # Access the original scGPT model with LoRA adapters already injected
        # peft_model structure: PeftModel -> base_model (PeftModel wrapper) -> model (actual scGPT)
        self.scgpt_with_lora = peft_model.base_model.model

    def forward(self, gene_ids, values, attention_mask=None, **kwargs):
        """
        Forward pass using scGPT's native signature with LoRA adapters active.

        Args:
            gene_ids: Gene token IDs [B, L]
            values: Gene expression values [B, L]
            attention_mask: Attention mask [B, L]
        """
        # Directly call scGPT with LoRA adapters (already injected by PEFT)
        # This avoids PEFT's forward() which tries to convert arguments
        return self.scgpt_with_lora(gene_ids, values, attention_mask)

    def train(self, mode=True):
        """Set training mode for PEFT model."""
        self.peft_model.train(mode)
        return self

    def eval(self):
        """Set eval mode for PEFT model."""
        self.peft_model.eval()
        return self

    def parameters(self):
        """Return PEFT model parameters (includes LoRA adapters)."""
        return self.peft_model.parameters()

    def named_parameters(self):
        """Return named PEFT model parameters."""
        return self.peft_model.named_parameters()

    def state_dict(self):
        """Return PEFT model state dict."""
        return self.peft_model.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        """Load PEFT model state dict."""
        return self.peft_model.load_state_dict(state_dict, strict=strict)

    def to(self, device):
        """Move PEFT model to device."""
        self.peft_model.to(device)
        self.scgpt_with_lora = self.peft_model.base_model.model
        return self

    def __getattr__(self, name):
        """Delegate attributes to PEFT model."""
        if name in ['peft_model', 'scgpt_with_lora', 'forward', 'train', 'eval',
                    'parameters', 'named_parameters', 'state_dict', 'load_state_dict', 'to']:
            return object.__getattribute__(self, name)
        return getattr(self.peft_model, name)


def load_scgpt_with_lora(rank: int = 8, alpha: float = 16.0, dropout: float = 0.05, target_modules = None, use_manual_lora: bool = True):
    """
    Load scGPT backbone with LoRA adapters.

    Args:
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: LoRA dropout
        target_modules: List of modules to apply LoRA to
        use_manual_lora: If True, use manual LoRA (default). If False, try PEFT.

    Returns:
        model: scGPT with LoRA
        tokenizer: scGPT tokenizer
        device: torch device
    """
    # Default: Use manual LoRA (more compatible with scGPT)
    if use_manual_lora or not PEFT_AVAILABLE:
        print("Using manual LoRA implementation (recommended for scGPT)")
        return load_scgpt_with_manual_lora(rank=rank, alpha=alpha, dropout=dropout, target_modules=target_modules)

    # Fallback: Try PEFT (may have compatibility issues)
    print("Using PEFT LoRA implementation (experimental)")
    if target_modules is None:
        target_modules = [
            "self_attn.out_proj",
            "linear1", "linear2",
            "value_encoder.linear1", "value_encoder.linear2"
        ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load base scGPT model
    scgpt = tdc_hf_interface("scGPT")
    model = scgpt.load()
    model.to(device)

    # Freeze all parameters first
    model.requires_grad_(False)

    # Make LayerNorms trainable
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            for param in module.parameters():
                param.requires_grad_(True)

    # Configure LoRA
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )

    # Apply LoRA to the model
    peft_model = get_peft_model(model, lora_config)
    peft_model.eval()

    # Wrap with compatibility layer to fix PEFT input conversion issues
    wrapped_model = ScGPTCompatibleWrapper(peft_model)

    # Print parameter statistics
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())

    print(f"PEFT LoRA applied to scGPT backbone:")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.3%})")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Parameter reduction: {(1-trainable_params/total_params)*100:.1f}%")
    print(f"  Target modules: {target_modules}")
    print(f"  PEFT compatibility wrapper: Applied")

    tokenizer = scGPTTokenizer()
    return wrapped_model, tokenizer, device


def tokenize_matrix(mat: np.ndarray,
                    gene_ids: np.ndarray,
                    tokenizer: scGPTTokenizer
                   ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Convert a cells×genes matrix (float) into a list of (token_ids, values, attention_mask).
    FIXED: Ensures all cells have the same number of tokens for fair domain comparison.
    """
    assert mat.ndim == 2, "Expected 2D [cells, genes] matrix"
    assert mat.shape[1] == len(gene_ids), "gene_ids length must match #columns"

    # Get tokenizer output
    tok = tokenizer.tokenize_cell_vectors  # classmethod exposed on tokenizer
    tok_out = tok(mat, gene_ids)           # -> list over cells: [(ids, vals), ...]

    # Find the maximum token length across all cells
    max_tokens = max(len(ids) for ids, vals in tok_out)

    tokens = []
    for ids, vals in tok_out:
        # Convert to tensors
        ids_t  = torch.tensor(ids,  dtype=torch.long)
        vals_t = torch.tensor(vals, dtype=torch.float32)

        # PAD to max_tokens length for consistent representation
        current_len = len(ids_t)
        if current_len < max_tokens:
            # Pad with special tokens (assuming 0 is PAD token)
            pad_length = max_tokens - current_len
            ids_t = torch.cat([ids_t, torch.zeros(pad_length, dtype=torch.long)])
            vals_t = torch.cat([vals_t, torch.zeros(pad_length, dtype=torch.float32)])

        # Create attention mask (True for real tokens, False for padding)
        mask_t = torch.cat([
            torch.ones(current_len, dtype=torch.bool),  # Real tokens
            torch.zeros(max_tokens - current_len, dtype=torch.bool)  # Padding
        ])

        tokens.append((ids_t, vals_t, mask_t))

    print(f"  Fixed tokenization: all cells now have {max_tokens} tokens")
    return tokens


def tokenize_domains_together(src_mat: np.ndarray,
                              tgt_mat: np.ndarray,
                              gene_ids: np.ndarray,
                              tokenizer: scGPTTokenizer):
    """
    Tokenize source and target domains together to ensure identical token lengths.
    CRITICAL FIX for domain adaptation: Source=513 vs Target=274 tokens destroys comparability.
    """
    assert src_mat.shape[1] == tgt_mat.shape[1] == len(gene_ids), "Gene dimension mismatch"

    # Combine both domains for consistent tokenization
    combined_mat = np.vstack([src_mat, tgt_mat])

    # Tokenize combined matrix (ensures same max_tokens for both domains)
    combined_tokens = tokenize_matrix(combined_mat, gene_ids, tokenizer)

    # Split back into source and target
    n_src = src_mat.shape[0]
    src_tokens = combined_tokens[:n_src]
    tgt_tokens = combined_tokens[n_src:]

    # Verify equal token lengths
    src_lens = [len(ids) for ids, _, _ in src_tokens]
    tgt_lens = [len(ids) for ids, _, _ in tgt_tokens]

    print(f"  Domain-consistent tokenization:")
    print(f"    Source: {len(src_tokens)} cells, {np.mean(src_lens):.1f} avg tokens")
    print(f"    Target: {len(tgt_tokens)} cells, {np.mean(tgt_lens):.1f} avg tokens")
    print(f"    Token length match: {len(set(src_lens + tgt_lens)) == 1}")

    return src_tokens, tgt_tokens


def _pluck_hidden(model, out, ids: torch.Tensor) -> torch.Tensor:
    """
    Make the forward-output robust: return [B,L,H] if available, else [B,H],
    else fall back to input embeddings(model.get_input_embeddings()).
    """
    # HF-style: last_hidden_state
    if hasattr(out, "last_hidden_state") and isinstance(out.last_hidden_state, torch.Tensor):
        return out.last_hidden_state

    # dict with tensors/lists
    if isinstance(out, dict):
        # prefer a 3-D hidden state if present
        for key in ("last_hidden_state", "hidden_states", "embeds", "output", "logits"):
            val = out.get(key, None)
            if isinstance(val, torch.Tensor) and val.ndim == 3:
                return val
            if isinstance(val, (list, tuple)):
                for t in reversed(val):
                    if isinstance(t, torch.Tensor) and t.ndim == 3:
                        return t
        # otherwise take any tensor
        for v in out.values():
            if isinstance(v, torch.Tensor):
                return v

    # tuple/list of tensors
    if isinstance(out, (list, tuple)):
        for t in out:
            if isinstance(t, torch.Tensor):
                return t

    # plain tensor
    if isinstance(out, torch.Tensor):
        return out

    # fallback: input embedding lookup
    get_emb = getattr(model, "get_input_embeddings", None)
    if callable(get_emb):
        return get_emb()(ids)

    raise TypeError(f"Cannot extract hidden states from type: {type(out)}")


@torch.no_grad()
def embed_tokens_then_norm(model, token_triplets, device, batch_size=64):
    def pluck_hidden(out, ids):
        tensors = []
        if hasattr(out, "last_hidden_state") and isinstance(out.last_hidden_state, torch.Tensor):
            return out.last_hidden_state
        if isinstance(out, dict):
            for v in out.values():
                if isinstance(v, torch.Tensor): tensors.append(v)
                elif isinstance(v, (list, tuple)):
                    tensors.extend([t for t in v if isinstance(t, torch.Tensor)])
        if isinstance(out, (list, tuple)):
            tensors.extend([t for t in out if isinstance(t, torch.Tensor)])
        if isinstance(out, torch.Tensor):
            tensors.append(out)
        for t in tensors:
            if t.ndim == 3: return t
        for t in tensors:
            if t.ndim == 2 and t.shape[0] == ids.shape[0] and t.shape[1] > 8: return t
        get_emb = getattr(model, "get_input_embeddings", None)
        if callable(get_emb): return get_emb()(ids)
        raise TypeError(f"Could not obtain hidden states; saw {[getattr(t,'shape',None) for t in tensors]}")

    # Auto-reduce batch size if tokens are too large
    max_tokens = max(tid.shape[0] for (tid, _, _) in token_triplets)
    if max_tokens > 1000:
        # Reduce batch size for large token sequences to avoid OOM
        batch_size = max(1, min(batch_size, 16))
        print(f"  WARNING: Large token length ({max_tokens}), reducing batch to {batch_size}")

    embs = []
    print(f"  Generating embeddings for {len(token_triplets)} cells (batch={batch_size})...")

    for i in range(0, len(token_triplets), batch_size):
        batch = token_triplets[i:i+batch_size]
        maxL = max(tid.shape[0] for (tid, _, _) in batch)
        B = len(batch)

        ids  = torch.zeros(B, maxL, dtype=torch.long, device=device)
        vals = torch.zeros(B, maxL, dtype=torch.float32, device=device)
        attn = torch.zeros(B, maxL, dtype=torch.bool, device=device)  # ChatGPT Fix: Start with False

        for bi, (tid, tv, mk) in enumerate(batch):
            L = tid.shape[0]
            ids[bi, :L]  = tid
            vals[bi, :L] = tv
            attn[bi, :L] = mk  # ChatGPT Fix: Use tokenizer mask to avoid attending to padding

        # Call scGPT model with proper attention mask
        out = model(ids, vals, attention_mask=attn)  # ChatGPT Fix: Pass mask, not None

        h = pluck_hidden(out, ids)
        pooled = h[:,0,:] if h.ndim==3 else h
        embs.append(pooled.detach().cpu())

        # Free GPU memory after each batch
        del ids, vals, attn, out, h, pooled
        torch.cuda.empty_cache()

    embs = torch.cat(embs, 0)
    norm = torch.nn.LayerNorm(embs.shape[1]).to(device)
    result = norm(embs.to(device)).cpu()
    del embs
    torch.cuda.empty_cache()
    return result


# lazy import TDC scGPT through its helper
def _load_scgpt(local_tdc_path: str):
    import sys, subprocess
    try:
        import tdc  # noqa
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", local_tdc_path])

    from tdc import tdc_hf_interface
    scgpt = tdc_hf_interface("scGPT")
    model = scgpt.load()
    return model

@dataclass
class EmbedderConfig:
    local_tdc: str
    device: str = "cuda"
    hidden_dim: int = 512
    use_layernorm: bool = True  # match your notebook behavior

class ScGPTEmbedder:
    """
    1) Loads frozen scGPT via TDC.
    2) Tokenizes matrices with ScGPTTokenizerWrapper.
    3) Produces pooled embeddings [N, H] (CLS token).
    """
    def __init__(self, cfg: EmbedderConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.tokenizer = ScGPTTokenizerWrapper()

        self.model = _load_scgpt(cfg.local_tdc)
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self._post_norm = nn.LayerNorm(cfg.hidden_dim).to(self.device) if cfg.use_layernorm else nn.Identity()

    @torch.no_grad()
    def _forward_hidden(self, ids: torch.Tensor, vals: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        """
        Forward scGPT and return [B,L,H] hidden states if available,
        otherwise pool from input embeddings.
        """
        try:
            out = self.model(ids, vals, attention_mask=attn, output_hidden_states=True, return_dict=True)
            if hasattr(out, "last_hidden_state"):
                return out.last_hidden_state  # [B,L,H]
        except TypeError:
            out = self.model(ids, vals, attention_mask=attn)

        # fallback paths
        if isinstance(out, torch.Tensor) and out.ndim == 3:
            return out
        # try reconstructing from input embeddings
        get_emb = getattr(self.model, "get_input_embeddings", None)
        if callable(get_emb):
            return get_emb()(ids)
        raise TypeError("Could not obtain hidden states from scGPT outputs.")

    @torch.no_grad()
    def embed_cellsxgenes(self, matrix: np.ndarray, gene_names: Sequence[str], batch_size: int = 64) -> torch.Tensor:
        """
        Args
        ----
        matrix:     [N_cells, N_genes] float32
        gene_names: gene symbols ordered as in columns of matrix

        Returns
        -------
        torch.Tensor of shape [N_cells, H]
        """
        triples = self.tokenizer.tokenize_matrix(matrix, gene_names)
        embs: List[torch.Tensor] = []

        for i in range(0, len(triples), batch_size):
            batch = triples[i:i+batch_size]
            maxL = max(tid.shape[0] for (tid, _, _) in batch)
            B = len(batch)

            ids  = torch.zeros(B, maxL, dtype=torch.long, device=self.device)
            vals = torch.zeros(B, maxL, dtype=torch.float32, device=self.device)
            attn = torch.zeros(B, maxL, dtype=torch.bool, device=self.device)

            for bi, (tid, tv, mk) in enumerate(batch):
                L = tid.shape[0]
                ids[bi, :L]  = tid
                vals[bi, :L] = tv
                attn[bi, :L] = mk

            h = self._forward_hidden(ids, vals, attn)   # [B,L,H]
            pooled = h[:, 0, :]                          # CLS
            pooled = self._post_norm(pooled)             # optional LayerNorm over H
            embs.append(pooled.detach().cpu())

        return torch.cat(embs, 0)

    def free_backbone(self):
        """Free VRAM after embedding (you keep the embeddings)."""
        try:
            del self.model
            torch.cuda.empty_cache()
        except Exception:
            pass