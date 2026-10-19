"""Run guitar-specific source separation before polyphonic transcription."""

import argparse
import json
from pathlib import Path

from .infer import transcribe_stem
from .separate import MODEL, separate


def transcribe_mix(audio: Path, checkpoint: Path, output_dir: Path) -> dict:
    stems = separate(audio, output_dir / "stems")
    midi_path = output_dir / "guitar.mid"
    count = transcribe_stem(stems["guitar"], checkpoint, midi_path)
    result = {
        "input": str(audio.resolve()),
        "separator": MODEL,
        "stems": {name: str(path) for name, path in stems.items()},
        "transcribed_stem": "guitar",
        "checkpoint": str(checkpoint.resolve()),
        "midi": str(midi_path.resolve()),
        "note_count": count,
    }
    (output_dir / "pipeline.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True, help="Music mixture to separate")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Guitar polyphonic model")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(transcribe_mix(args.audio, args.checkpoint, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
