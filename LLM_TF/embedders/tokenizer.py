# LLM_TF/tokenizer.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch

# will be provided by your local TDC install
from tdc.model_server.tokenizers.scgpt import scGPTTokenizer

class ScGPTTokenizerWrapper:
    """Small utility to keep tokenizer usage consistent."""
    def __init__(self) -> None:
        self._tok = scGPTTokenizer()

    def tokenize_matrix(
        self,
        data: np.ndarray,            # [N_cells, N_genes] float32
        gene_names: Sequence[str],   # length N_genes
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Returns a list of (token_ids, values, attn_mask) per cell.
        """
        assert data.ndim == 2, "data must be [N, G]"
        assert data.shape[1] == len(gene_names), "data and gene_names mismatch"

        out = self._tok.tokenize_cell_vectors(data, np.array(gene_names, dtype=object))
        triplets = []
        for ids, vals in out:
            ids_t  = torch.tensor(ids,  dtype=torch.long)
            vals_t = torch.tensor(vals, dtype=torch.float32)
            mask   = torch.tensor([v != 0 for v in vals], dtype=torch.bool)
            triplets.append((ids_t, vals_t, mask))
        return triplets
