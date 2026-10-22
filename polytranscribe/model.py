"""Baseline CNN and research models for polyphonic stem transcription."""

import math

import torch
from torch import nn
from torch.nn import functional as F


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
        """Input [B,10,pitch,time], using the original six-map baseline view."""
        if x.shape[1] < 6:
            raise ValueError(f"Expected at least six feature maps, received {x.shape[1]}")
        hidden = self.note_encoder(x[:, :3])
        note = torch.sigmoid(self.note_head(hidden).squeeze(1))
        onset = self.onset_head(hidden).squeeze(1)
        if self.variant == "gated":
            gate = torch.sigmoid(self.gate_head(self.gate_encoder(x[:, 3:6])).squeeze(1))
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


class ResidualBlock(nn.Module):
    """Pitch-local, time-dilated residual acoustics block."""

    def __init__(self, channels: int, time_dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=(1, time_dilation),
                      dilation=(1, time_dilation)),
            nn.GroupNorm(8, channels), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.net(x))


def positional_encoding(length: int, channels: int, device, dtype) -> torch.Tensor:
    positions = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    frequencies = torch.exp(torch.arange(0, channels, 2, device=device,
                                         dtype=torch.float32) * (-math.log(10000) / channels))
    encoding = torch.zeros((length, channels), device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies)
    return encoding.to(dtype)


class FactorizedAttention(nn.Module):
    """Attend across pitches and then across time, avoiding full pitch×time attention."""

    def __init__(self, channels: int = 64):
        super().__init__()
        def layer():
            return nn.TransformerEncoderLayer(d_model=channels, nhead=4,
                                              dim_feedforward=2 * channels,
                                              dropout=0.1, batch_first=True,
                                              norm_first=True)
        self.pitch = layer()
        self.time = layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, pitches, frames = x.shape
        pitch_tokens = x.permute(0, 3, 2, 1).reshape(batch * frames, pitches, channels)
        pitch_tokens = self.pitch(pitch_tokens + positional_encoding(
            pitches, channels, x.device, x.dtype)[None])
        time_tokens = pitch_tokens.reshape(batch, frames, pitches, channels)
        time_tokens = time_tokens.permute(0, 2, 1, 3).reshape(batch * pitches, frames, channels)
        time_tokens = self.time(time_tokens + positional_encoding(
            frames, channels, x.device, x.dtype)[None])
        return time_tokens.reshape(batch, pitches, frames, channels).permute(0, 3, 1, 2)


class AdvancedPitchTranscriber(nn.Module):
    """Harmonic residual encoder, factorized attention, onset head, optional learned gate."""

    CANDIDATE_CHANNELS = (0, 2, 6, 7, 8, 9)

    def __init__(self, gated: bool):
        super().__init__()
        self.gated = gated
        self.input = nn.Conv2d(6, 64, 3, padding=1)
        self.high = nn.Sequential(ResidualBlock(64, 1), ResidualBlock(64, 2))
        self.down = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=(1, 2), padding=1), ResidualBlock(64, 4),
            nn.Conv2d(64, 64, 3, stride=(1, 2), padding=1), ResidualBlock(64, 8),
        )
        self.attention = FactorizedAttention(64)
        self.fusion = ResidualBlock(64, 1)
        self.note_head = nn.Conv2d(65, 1, 1)
        self.onset_head = nn.Conv2d(64, 1, 1)
        if gated:
            self.gate = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.GELU(),
                ResidualBlock(32, 1), ResidualBlock(32, 2),
                nn.Conv2d(32, 1, 1),
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != 10:
            raise ValueError("Expected ten pitch-aligned feature maps")
        high = self.high(self.input(x[:, self.CANDIDATE_CHANNELS]))
        low = self.attention(self.down(high))
        context = F.interpolate(low, size=high.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.fusion(high + context)
        onset = self.onset_head(hidden).squeeze(1)
        note_input = torch.cat((hidden, torch.sigmoid(onset).unsqueeze(1)), dim=1)
        note = torch.sigmoid(self.note_head(note_input).squeeze(1))
        if self.gated:
            note = note * torch.sigmoid(self.gate(x[:, 3:6]).squeeze(1))
        return note.clamp(1e-6, 1 - 1e-6), onset


class AdvancedDrumTranscriber(nn.Module):
    """Residual Mel encoder with temporal attention for drum onset classes."""

    def __init__(self, classes: int = 9):
        super().__init__()
        self.input = nn.Conv2d(1, 48, 3, padding=1)
        self.encoder = nn.Sequential(ResidualBlock(48, 1), ResidualBlock(48, 2),
                                     nn.Conv2d(48, 48, 3, stride=(1, 2), padding=1),
                                     ResidualBlock(48, 4))
        self.project = nn.Linear(48 * 8, 64)
        self.temporal = nn.TransformerEncoderLayer(d_model=64, nhead=4,
                                                    dim_feedforward=128, dropout=0.1,
                                                    batch_first=True, norm_first=True)
        self.head = nn.Conv1d(64, classes, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("Expected [batch, 1, mel_band, time] drum features")
        hidden = self.encoder(self.input(x))
        pooled = F.adaptive_avg_pool2d(hidden, (8, hidden.shape[-1]))
        tokens = self.project(pooled.flatten(1, 2).transpose(1, 2))
        tokens = self.temporal(tokens + positional_encoding(
            tokens.shape[1], tokens.shape[2], x.device, x.dtype)[None])
        logits = self.head(F.interpolate(tokens.transpose(1, 2), size=x.shape[-1],
                                          mode="linear", align_corners=False))
        return torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6), logits


def build_transcriber(task: str, variant: str, classes: int) -> nn.Module:
    if task == "drums":
        if variant == "baseline":
            return DrumTranscriber(classes)
        if variant == "advanced":
            return AdvancedDrumTranscriber(classes)
    elif task == "pitched":
        if variant in ("baseline", "gated"):
            return PolyphonicTranscriber(variant)
        if variant in ("advanced_plain", "advanced_gated"):
            return AdvancedPitchTranscriber(gated=variant == "advanced_gated")
    raise ValueError(f"Unsupported task/variant: {task}/{variant}")


@torch.no_grad()
def predict_in_chunks(model: nn.Module, x: torch.Tensor, device: torch.device,
                      window_frames: int = 256, overlap: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Bound attention memory for a complete recording; return predictions on CPU."""
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError("Expected one recording [1, channels, pitch_or_mel, time]")
    frames = x.shape[-1]
    if frames <= window_frames:
        note, onset = model(x.to(device))
        return note.cpu(), onset.cpu()
    stride = window_frames - overlap
    starts = list(range(0, frames - window_frames + 1, stride))
    if starts[-1] != frames - window_frames:
        starts.append(frames - window_frames)
    notes = None
    onsets = None
    counts = torch.zeros((1, 1, frames), dtype=torch.float32)
    for start in starts:
        note, onset = model(x[..., start:start + window_frames].to(device))
        note, onset = note.cpu(), onset.cpu()
        if notes is None:
            notes = torch.zeros((1, note.shape[1], frames), dtype=note.dtype)
            onsets = torch.zeros_like(notes)
        notes[..., start:start + window_frames] += note
        onsets[..., start:start + window_frames] += onset
        counts[..., start:start + window_frames] += 1
    return notes / counts, onsets / counts
