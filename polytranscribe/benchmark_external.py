"""Score MIDI from an external AMT model on the same held-out GuitarSet split."""

import argparse
import json
from pathlib import Path

import soundfile as sf

from .config import Config
from .data import find_pairs, note_events
from .eval_slakh import aggregate, score_midi
from .infer import midi_events


def evaluate(data_root: Path, predictions: Path, output: Path,
             split: str = "test", pattern: str = "{stem}.mid") -> dict:
    cfg = Config()
    rows = []
    for annotation, audio, assignment in find_pairs(data_root):
        if assignment != split:
            continue
        midi_path = predictions / pattern.format(stem=annotation.stem, audio_stem=audio.stem)
        if not midi_path.is_file():
            raise FileNotFoundError(f"Missing baseline prediction: {midi_path}")
        scores = score_midi(midi_events(midi_path), note_events(annotation),
                            sf.info(audio).duration, cfg)
        rows.append({"track": annotation.stem, "external": scores})
    if not rows:
        raise ValueError(f"No GuitarSet tracks in split {split}")
    report = {"dataset": str(data_root.resolve()), "split": split,
              "predictions": str(predictions.resolve()), "tracks": len(rows),
              "scores": aggregate(rows, "external"), "per_track": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guitarset-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="MIDI files produced independently by the comparison model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--pattern", default="{stem}.mid",
                        help="Filename template using {stem} or {audio_stem}")
    args = parser.parse_args()
    result = evaluate(args.guitarset_root, args.predictions, args.output,
                      args.split, args.pattern)
    print(json.dumps({key: value for key, value in result.items() if key != "per_track"}, indent=2))


if __name__ == "__main__":
    main()
