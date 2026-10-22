"""Build bass, piano, drum, and vocal caches for the five-stem experiment."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
import torch
import yaml

from .data import frame_targets
from .eval_slakh import audio_file, merge_same_pitch_overlaps, midi_events
from .features import FEATURE_VERSION, drum_features, pitch_features
from .stems import DRUM_MIDI, config_for_stem, drum_class


def grouped_audio(paths: list[Path], sample_rate: int) -> torch.Tensor:
    signals = []
    for path in paths:
        signal, rate = sf.read(path, dtype="float32", always_2d=True)
        signal = signal.mean(axis=1)
        if rate != sample_rate:
            divisor = math.gcd(rate, sample_rate)
            signal = scipy.signal.resample_poly(signal, sample_rate // divisor,
                                                rate // divisor).astype(np.float32)
        signals.append(signal)
    length = max(map(len, signals))
    mixture = np.zeros(length, dtype=np.float32)
    for signal in signals:
        mixture[:len(signal)] += signal
    peak = float(np.max(np.abs(mixture)))
    if peak > 0:
        mixture /= peak
    return torch.from_numpy(mixture)


def drum_events(events: list[tuple[float, float, int]]) -> list[tuple[float, float, int]]:
    """Map General MIDI percussion notes into nine reproducible kit classes."""
    return sorted({(start, start + 0.1, DRUM_MIDI[drum_class(pitch)])
                   for start, _, pitch in events})


def drum_targets(events: list[tuple[float, float, int]], frames: int,
                 frame_seconds: float) -> np.ndarray:
    target = np.zeros((len(DRUM_MIDI), frames), dtype=np.uint8)
    if frames == 0:
        return target
    index_by_pitch = {pitch: index for index, pitch in enumerate(DRUM_MIDI)}
    for start, _, pitch in events:
        index = min(frames - 1, max(0, round(start / frame_seconds)))
        target[index_by_pitch[pitch], max(0, index - 1):min(frames, index + 2)] = 1
    return target


def cache_example(audio: torch.Tensor, events: list[tuple[float, float, int]],
                  stem: str, output: Path, device: torch.device) -> int:
    cfg = config_for_stem(stem)
    x = drum_features(audio, cfg, device) if stem == "drums" else pitch_features(audio, cfg, device)
    if stem == "drums":
        events = drum_events(events)
        y = drum_targets(events, x.shape[-1], cfg.frame_seconds)
        onset = y.copy()
        omitted = 0
    else:
        supported = [event for event in events if cfg.midi_min <= event[2] <= cfg.midi_max]
        omitted = len(events) - len(supported)
        events = supported
        y, onset = frame_targets(events, x.shape[-1], cfg)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, x=x, y=y, onset=onset,
                        events=np.asarray(events, dtype=np.float32).reshape(-1, 3))
    return omitted


def slakh_splits(tracks: list[Path], max_tracks: int) -> dict[Path, str]:
    """Honor official splits; BabySlakh gets a deterministic prototype split."""
    official = {track: next((part for part in track.parts
                             if part in ("train", "validation", "test")), None)
                for track in tracks}
    if any(split is None for split in official.values()):
        ordered = sorted(tracks)
        first = max(1, round(0.7 * len(ordered)))
        second = max(first + 1, round(0.85 * len(ordered)))
        official = {track: ("train" if i < first else "val" if i < second else "test")
                    for i, track in enumerate(ordered)}
    else:
        official = {track: ("val" if split == "validation" else split)
                    for track, split in official.items()}
    if max_tracks and len(official) > max_tracks:
        quotas = {"train": max(1, round(0.8 * max_tracks)),
                  "val": max(1, round(0.1 * max_tracks)),
                  "test": max(1, round(0.1 * max_tracks))}
        selected = {}
        for split in ("train", "val", "test"):
            group = sorted(track for track, label in official.items() if label == split)
            selected.update({track: split for track in group[:quotas[split]]})
        official = selected
    for split in ("train", "val", "test"):
        if split not in official.values():
            raise ValueError(f"Slakh has no {split} examples after selection")
    return official


def prepare_slakh(root: Path, output: Path, max_tracks: int,
                  device: torch.device) -> dict[str, dict]:
    tracks = [path.parent for path in root.rglob("metadata.yaml")
              if "omitted" not in path.parts]
    if not tracks:
        raise FileNotFoundError(f"No Slakh metadata.yaml found under {root}")
    splits = slakh_splits(tracks, max_tracks)
    manifests = {stem: {"counts": Counter(), "file_splits": {}, "out_of_range_notes": 0}
                 for stem in ("bass", "piano", "drums")}
    for number, (track, split) in enumerate(sorted(splits.items()), 1):
        metadata = yaml.safe_load((track / "metadata.yaml").read_text(encoding="utf-8"))
        for stem in manifests:
            sources = []
            notes = []
            for stem_id, item in metadata["stems"].items():
                selected = (item.get("is_drum", False) if stem == "drums"
                            else str(item.get("inst_class", "")).lower() == stem)
                if not selected:
                    continue
                # BabySlakh can mark these flags false despite shipping both files.
                try:
                    audio_path = audio_file(track / metadata.get("audio_dir", "stems") / stem_id)
                except FileNotFoundError:
                    continue
                midi_path = track / metadata.get("midi_dir", "MIDI") / f"{stem_id}.mid"
                if not midi_path.is_file():
                    continue
                sources.append(audio_path)
                notes.extend(midi_events(midi_path))
            if not sources:
                continue
            if stem != "drums":
                notes = merge_same_pitch_overlaps(notes)
            cfg = config_for_stem(stem)
            audio = grouped_audio(sources, cfg.sample_rate)
            name = f"{split}_{track.name}"
            path = output / stem / f"{name}.npz"
            manifests[stem]["out_of_range_notes"] += cache_example(audio, notes, stem, path, device)
            manifests[stem]["counts"][split] += 1
            manifests[stem]["file_splits"][name] = split
        if number % 10 == 0:
            print(f"Prepared {number}/{len(splits)} Slakh tracks", flush=True)
    return manifests


def vocadito_notes(path: Path) -> list[tuple[float, float, int]]:
    events = []
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            if len(row) < 3:
                continue
            start, frequency, duration = map(float, row[:3])
            if frequency > 0 and duration > 0:
                pitch = round(69 + 12 * math.log2(frequency / 440.0))
                events.append((start, start + duration, pitch))
    return sorted(events)


def prepare_vocadito(root: Path, output: Path, device: torch.device) -> dict:
    audios = sorted(root.rglob("vocadito_*.wav"),
                    key=lambda path: int(path.stem.split("_")[1]))
    if len(audios) != 40:
        raise ValueError(f"Expected 40 Vocadito excerpts, found {len(audios)}")
    cfg = config_for_stem("vocals")
    manifest = {"counts": Counter(), "file_splits": {}, "out_of_range_notes": 0}
    for audio_path in audios:
        index = int(audio_path.stem.split("_")[1])
        split = "train" if index <= 32 else "val" if index <= 36 else "test"
        note_path = next(iter(root.rglob(f"{audio_path.stem}_notesA1.csv")), None)
        if note_path is None:
            raise FileNotFoundError(f"Missing A1 notes for {audio_path}")
        audio = grouped_audio([audio_path], cfg.sample_rate)
        name = f"{split}_{audio_path.stem}"
        manifest["out_of_range_notes"] += cache_example(
            audio, vocadito_notes(note_path), "vocals", output / "vocals" / f"{name}.npz", device)
        manifest["counts"][split] += 1
        manifest["file_splits"][name] = split
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slakh-root", type=Path, required=True,
                        help="Extracted BabySlakh or Slakh2100-redux")
    parser.add_argument("--vocadito-root", type=Path, required=True,
                        help="Extracted Vocadito v3")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-slakh-tracks", type=int, default=100,
                        help="Limit selected tracks across splits; 0 uses all")
    args = parser.parse_args()
    if args.max_slakh_tracks < 0:
        parser.error("--max-slakh-tracks must be nonnegative")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifests = prepare_slakh(args.slakh_root, args.output, args.max_slakh_tracks, device)
    manifests["vocals"] = prepare_vocadito(args.vocadito_root, args.output, device)
    for stem, manifest in manifests.items():
        cfg = config_for_stem(stem)
        complete = {"config": cfg.to_dict(), "feature_version": FEATURE_VERSION,
                    "task": "drums" if stem == "drums" else "pitched", "stem": stem,
                    "counts": dict(manifest["counts"]),
                    "file_splits": manifest["file_splits"],
                    "out_of_range_notes": manifest["out_of_range_notes"],
                    "split": ("Vocadito track 1-32/33-36/37-40 (not singer-disjoint)" if stem == "vocals"
                              else "Slakh official split, or deterministic BabySlakh prototype split"),
                    "source": str(args.vocadito_root if stem == "vocals" else args.slakh_root)}
        (args.output / stem / "manifest.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")
        print(stem, complete["counts"], "excluded notes", complete["out_of_range_notes"])


if __name__ == "__main__":
    main()
