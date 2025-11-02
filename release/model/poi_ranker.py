from __future__ import annotations

import torch
import torch.nn as nn


class POIEmissionMLP(nn.Module):
    """Ranker that scores candidate POIs conditioned on event features."""

    def __init__(
        self,
        event_dim: int,
        candidate_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        input_dim = event_dim + candidate_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim if num_layers > 1 else input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, event_features: torch.Tensor, candidate_features: torch.Tensor) -> torch.Tensor:
        """Compute emission logits.

        Args:
            event_features: (B, E)
            candidate_features: (B, K, C)

        Returns:
            logits: (B, K)
        """
        B, K, _ = candidate_features.shape
        event_expanded = event_features.unsqueeze(1).expand(-1, K, -1)
        fused = torch.cat([event_expanded, candidate_features], dim=-1)
        logits = self.mlp(fused.view(B * K, -1)).view(B, K)
        return logits
