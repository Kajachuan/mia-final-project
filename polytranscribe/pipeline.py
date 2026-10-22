"""Separate a mixture once, then transcribe five instrument-specific stems."""

import argparse
import json
from pathlib import Path

import torch

from .infer import midi_events, transcribe_stem, write_multitrack_midi
from .separate import MODEL, separate
from .stems import TARGET_STEMS


def checkpoint_paths(models_root: Path) -> dict[str, Path]:
    paths = {stem: models_root / stem / ("advanced.pt" if stem == "drums" else "advanced_gated.pt")
             for stem in TARGET_STEMS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing instrument checkpoints: {missing}")
    for stem, path in paths.items():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("stem", "guitar") != stem:
            raise ValueError(f"Checkpoint {path} does not belong to {stem}")
        if (payload.get("task", "pitched") == "drums") != (stem == "drums"):
            raise ValueError(f"Checkpoint task does not match {stem}: {path}")
    return paths


def transcribe_mix(audio: Path, models_root: Path, output_dir: Path) -> dict:
    """Demucs runs once; `other` is saved but never assigned musical notes."""
    models = checkpoint_paths(models_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    stems = separate(audio, output_dir / "stems")
    note_counts = {}
    estimated = {}
    for stem in TARGET_STEMS:
        midi_path = output_dir / f"{stem}.mid"
        note_counts[stem] = transcribe_stem(stems[stem], models[stem], midi_path)
        estimated[stem] = midi_events(midi_path)
    midi_path = output_dir / "five_stems.mid"
    write_multitrack_midi(estimated, midi_path)
    result = {
        "input": str(audio.resolve()),
        "separator": MODEL,
        "stems": {name: str(path) for name, path in stems.items()},
        "transcribed_stems": list(TARGET_STEMS),
        "excluded_stem": "other",
        "checkpoints": {stem: str(path.resolve()) for stem, path in models.items()},
        "midi": str(midi_path.resolve()),
        "note_counts": note_counts,
    }
    (output_dir / "pipeline.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def transcribe_guitar_mix(audio: Path, checkpoint: Path, output_dir: Path) -> dict:
    """Retain the single-guitar ablation for comparison with five-stem MIDI."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stems = separate(audio, output_dir / "stems")
    midi_path = output_dir / "guitar.mid"
    count = transcribe_stem(stems["guitar"], checkpoint, midi_path)
    result = {"input": str(audio.resolve()), "separator": MODEL,
              "stems": {name: str(path) for name, path in stems.items()},
              "transcribed_stems": ["guitar"], "excluded_stem": "other",
              "checkpoint": str(checkpoint.resolve()),
              "midi": str(midi_path.resolve()), "note_counts": {"guitar": count}}
    (output_dir / "pipeline.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True, help="Music mixture to separate")
    models = parser.add_mutually_exclusive_group(required=True)
    models.add_argument("--models-root", type=Path,
                        help="Directory with vocals/advanced_gated.pt, drums/advanced.pt, etc.")
    models.add_argument("--checkpoint", type=Path,
                        help="Legacy one-guitar ablation checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = (transcribe_mix(args.audio, args.models_root, args.output_dir)
              if args.models_root else transcribe_guitar_mix(args.audio, args.checkpoint, args.output_dir))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
