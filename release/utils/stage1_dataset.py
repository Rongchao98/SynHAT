from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class Stage1SMTDataset(Dataset):
    """
    PyTorch dataset for Stage-1 SMT trajectories.
    
    Simplified format: each NPZ contains a single "data" array of shape (N, T, 3)
    where the last dimension is [z_latitude, z_longitude, stay_prob]. The first two
    channels are latitude/longitude values normalised with the global POI
    statistics during preprocessing.
    """

    def __init__(
        self,
        npz_path: Path,
        use_stay: bool = True,
    ) -> None:
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"Stage-1 dataset not found at {npz_path}")

        with np.load(npz_path, allow_pickle=False) as npz_data:
            if "data" in npz_data:
                data = npz_data["data"].astype(np.float32)  # (N, T, 3)
            else:
                coords = npz_data["coords"].astype(np.float32)  # (N, T, 2)
                stay = npz_data["stay"].astype(np.float32)[..., None]
                data = np.concatenate([coords, stay], axis=2)
            self.sequence = np.transpose(data, (0, 2, 1))  # (N, 3, T)
            self.stay = data[:, :, 2].astype(np.float32)

        self.use_stay = use_stay

    def __len__(self) -> int:
        return self.sequence.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = torch.from_numpy(self.sequence[idx])  # (3, T)

        item = {"sequence": seq}

        if self.use_stay:
            item["stay"] = torch.from_numpy(self.stay[idx])

        return item
