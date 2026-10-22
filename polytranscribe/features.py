"""Pitch-aligned spectral and Mel/chroma evidence for two transcription branches."""

import numpy as np
import scipy.signal
import soundfile as sf
import torch

from .config import Config

FEATURE_VERSION = 4


def load_audio(path, cfg: Config) -> torch.Tensor:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sr != cfg.sample_rate:
        gcd = np.gcd(sr, cfg.sample_rate)
        mono = scipy.signal.resample_poly(mono, cfg.sample_rate // gcd, sr // gcd).astype(np.float32)
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    if peak > 0:
        mono = mono / peak
    return torch.from_numpy(mono.copy())


def pitch_features(audio: torch.Tensor, cfg: Config, device: torch.device) -> np.ndarray:
    """Ten maps: six baseline cues plus separate harmonics 2 through 5."""
    audio = audio.to(device)
    if audio.numel() < cfg.n_fft:
        audio = torch.nn.functional.pad(audio, (0, cfg.n_fft - audio.numel()))
    window = torch.hann_window(cfg.n_fft, device=device)
    magnitude = torch.stft(audio, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                           window=window, center=True, return_complex=True).abs()
    pitches = torch.arange(cfg.midi_min, cfg.midi_max + 1, device=device)
    fundamental_hz = 440.0 * (2.0 ** ((pitches - 69) / 12.0))

    def sample(multiplier: float) -> torch.Tensor:
        frequency = fundamental_hz * multiplier
        valid = frequency < cfg.sample_rate / 2
        bins = (frequency * cfg.n_fft / cfg.sample_rate).round().long()
        bins = bins.clamp(1, magnitude.shape[0] - 2)
        sampled = torch.stack([magnitude.index_select(0, bins + shift)
                               for shift in (-1, 0, 1)]).amax(0)
        return sampled * valid[:, None]

    fundamental = sample(1.0)
    harmonic_maps = torch.stack([sample(float(h)) for h in (2, 3, 4, 5)])
    harmonic_counts = torch.stack([(fundamental_hz * h < cfg.sample_rate / 2).float()
                                   for h in (2, 3, 4, 5)]).sum(0).clamp_min(1)
    harmonics = harmonic_maps.sum(0) / harmonic_counts[:, None]
    subharmonics = torch.stack([sample(0.5), sample(1.0 / 3.0)]).mean(0)

    # Paper-inspired second branch: 256-band power Mel spectrogram and chroma.
    fft_hz = torch.linspace(0, cfg.sample_rate / 2, magnitude.shape[0], device=device)
    mel_max = 2595.0 * torch.log10(1.0 + fft_hz[-1] / 700.0)
    mel_edges = torch.linspace(0, float(mel_max), cfg.n_mels + 2, device=device)
    mel_hz = 700.0 * (10.0 ** (mel_edges / 2595.0) - 1.0)
    lower = (fft_hz[None, :] - mel_hz[:-2, None]) / (mel_hz[1:-1, None] - mel_hz[:-2, None])
    upper = (mel_hz[2:, None] - fft_hz[None, :]) / (mel_hz[2:, None] - mel_hz[1:-1, None])
    filters = torch.minimum(lower, upper).clamp_min(0)
    filters = filters / filters.sum(dim=1, keepdim=True).clamp_min(1e-8)
    mel_power = filters @ magnitude.square()
    mel_centers = mel_hz[1:-1]
    mel_indices = torch.argmin((mel_centers[:, None] - fundamental_hz[None, :]).abs(), dim=0)
    mel_f0 = mel_power.index_select(0, mel_indices)

    valid_bands = mel_centers >= 40.0
    band_midi = torch.round(69.0 + 12.0 * torch.log2(mel_centers[valid_bands] / 440.0)).long()
    classes = torch.remainder(band_midi, 12)
    chroma = torch.zeros((12, mel_power.shape[1]), device=device)
    chroma.index_add_(0, classes, mel_power[valid_bands])
    counts = torch.zeros(12, device=device)
    counts.index_add_(0, classes, torch.ones_like(classes, dtype=counts.dtype))
    chroma = chroma / counts[:, None].clamp_min(1)
    pitch_classes = torch.remainder(pitches, 12)
    chroma_pitch = chroma.index_select(0, pitch_classes)
    voicing = chroma.mean(dim=0)[None, :].expand_as(mel_f0)

    features = torch.stack([fundamental, harmonics, subharmonics,
                            mel_f0, chroma_pitch, voicing,
                            *harmonic_maps.unbind(0)]).log1p()
    features = (features - features.mean()) / features.std().clamp_min(1e-4)
    return features.cpu().numpy().astype(np.float16)


def drum_features(audio: torch.Tensor, cfg: Config, device: torch.device) -> np.ndarray:
    """Log Mel energy for transient drum classification: [1, mel_band, time]."""
    audio = audio.to(device)
    if audio.numel() < cfg.n_fft:
        audio = torch.nn.functional.pad(audio, (0, cfg.n_fft - audio.numel()))
    magnitude = torch.stft(audio, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                           window=torch.hann_window(cfg.n_fft, device=device),
                           center=True, return_complex=True).abs()
    frequencies = torch.linspace(0, cfg.sample_rate / 2, magnitude.shape[0], device=device)
    mel_max = 2595.0 * torch.log10(1.0 + frequencies[-1] / 700.0)
    edges = torch.linspace(0, float(mel_max), cfg.n_mels + 2, device=device)
    hz = 700.0 * (10.0 ** (edges / 2595.0) - 1.0)
    lower = (frequencies[None, :] - hz[:-2, None]) / (hz[1:-1, None] - hz[:-2, None])
    upper = (hz[2:, None] - frequencies[None, :]) / (hz[2:, None] - hz[1:-1, None])
    filters = torch.minimum(lower, upper).clamp_min(0)
    filters /= filters.sum(dim=1, keepdim=True).clamp_min(1e-8)
    mel = (filters @ magnitude.square()).log1p()
    mel = (mel - mel.mean()) / mel.std().clamp_min(1e-4)
    return mel[None].cpu().numpy().astype(np.float16)
