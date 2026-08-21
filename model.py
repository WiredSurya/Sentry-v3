"""
SENTRY V3 Autoencoder.

Design notes (the WHY):
- Autoencoder = learn to reconstruct BENIGN traffic only. At inference,
  malicious flows reconstruct badly → high MSE → anomaly.
- V3 is deeper than V2: 78 -> 128 -> 96 -> 64 -> 32 -> 16 bottleneck
  -> 32 -> 64 -> 96 -> 128 -> 78. Wider layers = more capacity to model
  the varied "normal" surface, tighter bottleneck (16) = harder to
  cheat-memorize, so anomalies get compressed away instead of copied.
- 78 input features matches CIC-IDS2017 out of the box.
- We store last_activations on every forward pass so the visualizer
  can color neurons live without a separate hook system.
"""
from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn


class SentryAutoencoderV3(nn.Module):
    def __init__(self, input_dim: int = 78):
        super().__init__()
        self.input_dim = input_dim

        # Encoder: shrink toward the 16-dim bottleneck
        self.encoder_layers = nn.ModuleList([
            nn.Linear(input_dim, 128),
            nn.Linear(128, 96),
            nn.Linear(96, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 16),  # bottleneck
        ])

        # Decoder: mirror back out to input_dim
        self.decoder_layers = nn.ModuleList([
            nn.Linear(16, 32),
            nn.Linear(32, 64),
            nn.Linear(64, 96),
            nn.Linear(96, 128),
            nn.Linear(128, input_dim),
        ])

        self.act = nn.ReLU()
        self.out_act = nn.Sigmoid()  # data is MinMax-scaled to [0,1]
        self.last_activations: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.last_activations = [x.detach().clone()]

        h = x
        for layer in self.encoder_layers:
            h = self.act(layer(h))
            self.last_activations.append(h.detach().clone())

        z = h  # 16-dim latent

        for i, layer in enumerate(self.decoder_layers):
            is_last = (i == len(self.decoder_layers) - 1)
            h = self.out_act(layer(h)) if is_last else self.act(layer(h))
            self.last_activations.append(h.detach().clone())

        return h, z

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE. Higher = more anomalous."""
        recon, _ = self.forward(x)
        return torch.mean((x - recon) ** 2, dim=1)

    def get_layer_sizes(self) -> List[int]:
        sizes = [self.input_dim]
        for layer in self.encoder_layers:
            sizes.append(layer.out_features)
        for layer in self.decoder_layers:
            sizes.append(layer.out_features)
        return sizes
