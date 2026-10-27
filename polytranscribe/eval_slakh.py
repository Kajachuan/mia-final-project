"""Evaluate oracle-guitar and mixture-to-MIDI transcription on Slakh/BabySlakh."""

import argparse
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
import torch
import yaml

from .config import Config
from .data import frame_targets
from .features import FEATURE_VERSION
from .infer import midi_events, transcribe_stem
from .metrics import frame_scores, frame_scores_by_polyphony, note_scores
from .separate import MODEL, separate


def merge_same_pitch_overlaps(events: list[tuple[float, float, int]]) -> list[tuple[float, float, int]]:
    """The model predicts pitch sets, so overlapping unisons count once."""
    merged = []
    for pitch in sorted({event[2] for event in events}):
        group = sorted((start, end) for start, end, midi in events if midi == pitch)
        for start, end in group:
            if merged and merged[-1][2] == pitch and start < merged[-1][1]:
                previous = merged[-1]
                merged[-1] = (previous[0], max(previous[1], end), pitch)
            else:
                merged.append((start, end, pitch))
    return sorted(merged)


def audio_file(base: Path) -> Path:
    for extension in (".wav", ".flac"):
        path = base.with_suffix(extension)
        if path.is_file():
            return path
    raise FileNotFoundError(f"Expected WAV or FLAC for {base}")


def guitar_sources(track: Path) -> list[tuple[Path, Path]]:
    metadata = yaml.safe_load((track / "metadata.yaml").read_text(encoding="utf-8"))
    sources = []
    for stem_id, stem in metadata["stems"].items():
        if str(stem.get("inst_class", "")).lower() != "guitar" or stem.get("is_drum", False):
            continue
        # Trust the files: BabySlakh's rendered/saved flags can be stale.
        try:
            audio = audio_file(track / metadata.get("audio_dir", "stems") / stem_id)
        except FileNotFoundError:
            continue
        midi = track / metadata.get("midi_dir", "MIDI") / f"{stem_id}.mid"
        if not midi.is_file():
            continue
        sources.append((audio, midi))
    return sources


def oracle_guitar(sources: list[tuple[Path, Path]], output: Path,
                  seconds: float) -> tuple[Path, list[tuple[float, float, int]]]:
    if not sources:
        raise ValueError("No guitar sources")
    signals = []
    events = []
    sample_rate = None
    for audio_path, midi_path in sources:
        info = sf.info(audio_path)
        frames = round(seconds * info.samplerate) if seconds > 0 else -1
        signal, rate = sf.read(audio_path, frames=frames, dtype="float32", always_2d=True)
        signal = signal.mean(axis=1)
        if sample_rate is None:
            sample_rate = rate
        elif rate != sample_rate:
            raise ValueError(f"Inconsistent sample rates in {audio_path.parent}")
        signals.append(signal)
        for start, end, pitch in midi_events(midi_path):
            if seconds <= 0 or start < seconds:
                events.append((start, min(end, seconds) if seconds > 0 else end, pitch))
    oracle = np.zeros(max(map(len, signals)), dtype=np.float32)
    for signal in signals:
        oracle[:len(signal)] += signal
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, oracle, sample_rate, subtype="FLOAT")
    return output, merge_same_pitch_overlaps(events)


def clip_audio(path: Path, output: Path, seconds: float) -> Path:
    if seconds <= 0:
        return path
    info = sf.info(path)
    signal, rate = sf.read(path, frames=round(seconds * info.samplerate),
                           dtype="float32", always_2d=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, signal, rate, subtype="FLOAT")
    return output


def si_sdr(reference_path: Path, estimate_path: Path) -> float:
    """Scale-invariant signal-to-distortion ratio, for diagnosis only."""
    reference, ref_rate = sf.read(reference_path, dtype="float64", always_2d=True)
    estimate, est_rate = sf.read(estimate_path, dtype="float64", always_2d=True)
    reference = reference.mean(axis=1)
    estimate = estimate.mean(axis=1)
    if est_rate != ref_rate:
        gcd = math.gcd(est_rate, ref_rate)
        estimate = scipy.signal.resample_poly(estimate, ref_rate // gcd, est_rate // gcd)
    length = min(len(reference), len(estimate))
    reference = reference[:length] - reference[:length].mean()
    estimate = estimate[:length] - estimate[:length].mean()
    energy = float(np.dot(reference, reference))
    if energy < 1e-12:
        raise ValueError("Cannot compute SI-SDR for silent guitar reference")
    projected = np.dot(estimate, reference) / energy * reference
    residual = estimate - projected
    return float(10 * np.log10((np.dot(projected, projected) + 1e-12) /
                               (np.dot(residual, residual) + 1e-12)))


def score_midi(predicted: list[tuple[float, float, int]],
               reference: list[tuple[float, float, int]],
               duration: float, cfg: Config) -> dict:
    frames = math.ceil(duration / cfg.frame_seconds) + 1
    supported = [event for event in reference if cfg.midi_min <= event[2] <= cfg.midi_max]
    predicted_supported = [event for event in predicted if cfg.midi_min <= event[2] <= cfg.midi_max]
    target, _ = frame_targets(supported, frames, cfg)
    estimate, _ = frame_targets(predicted_supported, frames, cfg)
    target, estimate = target.astype(bool), estimate.astype(bool)
    return {
        "note": note_scores(predicted, reference),
        "onset_note": note_scores(predicted, reference, offset_ratio=None),
        "frame": frame_scores(estimate, target),
        "polyphonic_frame": frame_scores_by_polyphony(estimate, target),
        "reference_notes": len(reference),
        "out_of_range_reference_notes": len(reference) - len(supported),
    }


def aggregate(rows: list[dict], condition: str) -> dict:
    result = {}
    for metric in ("note", "onset_note", "frame", "polyphonic_frame"):
        tp = sum(row[condition][metric]["tp"] for row in rows)
        fp = sum(row[condition][metric]["fp"] for row in rows)
        fn = sum(row[condition][metric]["fn"] for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        result[metric] = {"precision": precision, "recall": recall,
                          "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                          "tp": tp, "fp": fp, "fn": fn,
                          "macro_f1": float(np.mean([row[condition][metric]["f1"] for row in rows]))}
        if metric == "polyphonic_frame":
            result[metric]["frames"] = sum(row[condition][metric]["frames"] for row in rows)
    return result


def evaluate(data_root: Path, checkpoint: Path, output: Path, split: str,
             max_tracks: int, with_si_sdr: bool, seconds: float = 30.0) -> dict:
    data_root = data_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Checkpoint feature version is incompatible")
    cfg = Config(**(saved["configs"]["guitar"] if saved.get("stem") == "shared"
                    else saved["config"]))
    tracks = sorted(path.parent for path in data_root.rglob("metadata.yaml")
                    if "omitted" not in path.parts and
                    (split == "any" or split in path.relative_to(data_root).parts))
    if max_tracks:
        tracks = tracks[:max_tracks]
    if not tracks:
        raise FileNotFoundError(f"No {split} Slakh tracks under {data_root}")
    rows = []
    skipped = []
    for track in tracks:
        sources = guitar_sources(track)
        if not sources:
            skipped.append(str(track.relative_to(data_root)))
            continue
        relative = track.relative_to(data_root)
        track_output = output / relative
        track_output.mkdir(parents=True, exist_ok=True)
        oracle_path, reference = oracle_guitar(sources, track_output / "oracle_guitar.wav", seconds)
        mixture_path = clip_audio(audio_file(track / "mix"), track_output / "mixture_clip.wav", seconds)
        oracle_midi = track_output / "oracle_prediction.mid"
        separated_midi = track_output / "separated_prediction.mid"
        if saved.get("stem") == "shared":
            transcribe_stem(oracle_path, checkpoint, oracle_midi, stem="guitar")
        else:
            transcribe_stem(oracle_path, checkpoint, oracle_midi)
        separated_guitar = track_output / "stems" / MODEL / mixture_path.stem / "guitar.wav"
        if not separated_guitar.is_file():
            separated_guitar = separate(mixture_path, track_output / "stems")["guitar"]
        if saved.get("stem") == "shared":
            transcribe_stem(separated_guitar, checkpoint, separated_midi, stem="guitar")
        else:
            transcribe_stem(separated_guitar, checkpoint, separated_midi)
        duration = sf.info(mixture_path).duration
        row = {
            "track": str(relative), "guitar_sources": len(sources),
            "oracle": score_midi(midi_events(oracle_midi), reference, duration, cfg),
            "separated": score_midi(midi_events(separated_midi), reference, duration, cfg),
        }
        if with_si_sdr:
            row["separation_si_sdr_db"] = si_sdr(oracle_path, separated_guitar)
        rows.append(row)
        (track_output / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(f"{relative}: oracle note F1={row['oracle']['note']['f1']:.3f}, "
              f"separated note F1={row['separated']['note']['f1']:.3f}", flush=True)
    if not rows:
        raise ValueError("No tracks with guitar stems were found")
    oracle = aggregate(rows, "oracle")
    separated = aggregate(rows, "separated")
    report = {"dataset": str(data_root), "split": split, "clip_seconds": seconds,
              "tracks_evaluated": len(rows),
              "tracks_skipped": skipped, "checkpoint": str(checkpoint.resolve()),
              "oracle": oracle, "separated": separated,
              "note_f1_drop": oracle["note"]["f1"] - separated["note"]["f1"],
              "optional_separation_si_sdr_db_median":
                  float(np.median([row["separation_si_sdr_db"] for row in rows])) if with_si_sdr else None,
              "per_track": rows}
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Extracted Slakh2100 or BabySlakh root")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test", "any"), default="test")
    parser.add_argument("--max-tracks", type=int, default=0, help="0 evaluates all eligible tracks")
    parser.add_argument("--seconds", type=float, default=30.0,
                        help="Evaluate this many seconds per track; 0 uses complete tracks")
    parser.add_argument("--with-si-sdr", action="store_true",
                        help="Optional guitar separation diagnostic; requires oracle stems")
    args = parser.parse_args()
    if args.max_tracks < 0 or args.seconds < 0:
        parser.error("--max-tracks and --seconds must be nonnegative")
    result = evaluate(args.data_root, args.checkpoint, args.output,
                      args.split, args.max_tracks, args.with_si_sdr, args.seconds)
    print(json.dumps({key: value for key, value in result.items() if key != "per_track"}, indent=2))


if __name__ == "__main__":
    main()
