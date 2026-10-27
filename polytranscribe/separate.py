"""Demucs v4 source separation, matching the first stage of the reference paper."""

import argparse
from pathlib import Path

import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.audio import convert_audio
from demucs.pretrained import get_model


STEMS = ("vocals", "drums", "bass", "other", "guitar", "piano")
MODEL = "htdemucs_6s"


def separate_waveform(model, waveform, sample_rate: int,
                      device: torch.device) -> dict[str, torch.Tensor]:
    """Separate an in-memory mixture with the same settings as file inference."""
    if waveform.shape[1] > 2:
        waveform = waveform.mean(axis=1, keepdims=True)
    mix = torch.from_numpy(waveform.T.copy())
    mix = convert_audio(mix, sample_rate, model.samplerate, model.audio_channels)
    reference = mix.mean(0)
    mean = reference.mean()
    std = reference.std().clamp_min(1e-8)
    with torch.no_grad():
        sources = apply_model(model, ((mix - mean) / std)[None], device=device,
                              shifts=1, split=True, overlap=0.25)[0]
    sources = sources * std + mean
    result = dict(zip(model.sources, sources, strict=True))
    if set(result) != set(STEMS):
        raise ValueError(f"Unexpected htdemucs_6s sources: {sorted(result)}")
    return result


def separate(audio: Path, output_dir: Path) -> dict[str, Path]:
    """Run pretrained htdemucs_6s and return six stems; downloads weights on first use."""
    audio = audio.resolve()
    if not audio.is_file():
        raise FileNotFoundError(audio)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = get_model(MODEL)
    model.eval()
    waveform, sample_rate = sf.read(audio, dtype="float32", always_2d=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sources = separate_waveform(model, waveform, sample_rate, device)
    stem_dir = output_dir / MODEL / audio.stem
    stem_dir.mkdir(parents=True, exist_ok=True)
    stems = {}
    for name, source in sources.items():
        path = stem_dir / f"{name}.wav"
        sf.write(path, source.T.cpu().numpy(), model.samplerate, subtype="PCM_16")
        stems[name] = path
    return stems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for name, path in separate(args.audio, args.output_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
