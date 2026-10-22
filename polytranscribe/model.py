"""Per-pitch multi-label note and onset model with an ablatable harmonic gate."""

import torch
from torch import nn


class ConvStack(nn.Module):
    def __init__(self, channels: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolyphonicTranscriber(nn.Module):
    def __init__(self, variant: str):
        super().__init__()
        if variant not in {"baseline", "gated"}:
            raise ValueError(variant)
        self.variant = variant
        # Branch A replaces monophonic CREPE with a multi-label pitch estimator.
        self.note_encoder = ConvStack(3)
        self.note_head = nn.Conv2d(32, 1, 1)
        self.onset_head = nn.Conv2d(32, 1, 1)
        if variant == "gated":
            # Branch B estimates pitch-specific voicing from Mel and chroma.
            self.gate_encoder = ConvStack(3, hidden=16)
            self.gate_head = nn.Conv2d(16, 1, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Input [B,6,pitch,time], output note probabilities and onset logits [B,pitch,time]."""
        if x.shape[1] != 6:
            raise ValueError(f"Expected six feature maps, received {x.shape[1]}")
        hidden = self.note_encoder(x[:, :3])
        note = torch.sigmoid(self.note_head(hidden).squeeze(1))
        onset = self.onset_head(hidden).squeeze(1)
        if self.variant == "gated":
            gate = torch.sigmoid(self.gate_head(self.gate_encoder(x[:, 3:])).squeeze(1))
            note = note * gate
        return note.clamp(1e-6, 1 - 1e-6), onset


class DrumTranscriber(nn.Module):
    """Predict simultaneous percussion classes from a log Mel spectrogram."""

    def __init__(self, classes: int = 9):
        super().__init__()
        self.encoder = ConvStack(1, hidden=24)
        self.head = nn.Sequential(
            nn.Conv1d(24 * 8, 48, 3, padding=1), nn.ReLU(),
            nn.Conv1d(48, classes, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("Expected [batch, 1, mel_band, time] drum features")
        hidden = self.encoder(x)
        hidden = nn.functional.adaptive_avg_pool2d(hidden, (8, hidden.shape[-1]))
        logits = self.head(hidden.flatten(1, 2))
        return torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6), logits
