from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class Stage2BlockDataset(Dataset):
    """
    Dataset for Stage-2 block diffusion training.
    
    Simplified format: only stores the essential arrays needed for training.
    - residual: target path residual from linear baseline
    - base_path: linear baseline path (conditioning)
    - global_cond: pre-computed global context features
    
    Other fields (entry/exit, temporal fractions, day_id, event_count) are not
    used during training and can be derived or are unnecessary.
    """

    def __init__(self, npz_path: Path) -> None:
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"Stage-2 dataset not found at {npz_path}")

        with np.load(npz_path, allow_pickle=False) as npz_data:
            self.target = npz_data["target"].astype(np.float32)  # (N, L, C) - absolute coordinates
            self.base_path = npz_data["base_path"].astype(np.float32)  # (N, L, C)
            self.global_cond = npz_data["global_cond"].astype(np.float32)  # (N, F)
            self.mask = (
                npz_data["mask"].astype(np.float32)
                if "mask" in npz_data
                else np.ones((self.target.shape[0], self.target.shape[1]), dtype=np.float32)
            )

    def __len__(self) -> int:
        return self.target.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "target": torch.from_numpy(self.target[idx].T),  # (C, L) - absolute coordinates
            "base_path": torch.from_numpy(self.base_path[idx].T),  # (C, L)
            "global_cond": torch.from_numpy(self.global_cond[idx]),  # (F,)
            "mask": torch.from_numpy(self.mask[idx]),  # (L,)
        }
