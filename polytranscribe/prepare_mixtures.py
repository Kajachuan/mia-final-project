"""Cache Demucs estimates from annotated mixtures for transcriber training."""

import argparse
import hashlib
import json
import math
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
import torch
import yaml
from demucs.pretrained import get_model

from .eval_slakh import merge_same_pitch_overlaps, midi_events
from .features import FEATURE_VERSION
from .prepare_five import cache_example, slakh_splits, vocadito_notes
from .separate import MODEL, separate_waveform
from .stems import TARGET_STEMS, config_for_stem


MIX_SCHEMA = 1
VOCAL_GAIN = 0.8
ACCOMPANIMENT_GAIN = 0.45


def excerpt_bounds(path: Path, seconds: float, key: str) -> tuple[int, int, int]:
    info = sf.info(path)
    frames = min(info.frames, max(1, round(seconds * info.samplerate)))
    room = max(0, info.frames - frames)
    fraction = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") / 2**64
    return min(room, round(fraction * room)), frames, info.samplerate


def read_excerpt(path: Path, seconds: float, key: str) -> tuple[np.ndarray, int, float]:
    """Read a deterministic crop and return its exact start time."""
    start_frame, frames, rate = excerpt_bounds(path, seconds, key)
    with sf.SoundFile(path) as source:
        source.seek(start_frame)
        audio = source.read(frames, dtype="float32", always_2d=True)
        return audio, rate, start_frame / rate


def clip_events(events: list[tuple[float, float, int]], start: float,
                duration: float) -> list[tuple[float, float, int]]:
    stop = start + duration
    return [(max(0.0, first - start), min(duration, last - start), pitch)
            for first, last, pitch in events if last > start and first < stop]


def estimated_audio(source: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    """Match the mono resampling and peak normalization used by inference."""
    signal = source.mean(dim=0).cpu().numpy().astype(np.float32)
    if source_rate != target_rate:
        divisor = math.gcd(source_rate, target_rate)
        signal = scipy.signal.resample_poly(signal, target_rate // divisor,
                                            source_rate // divisor).astype(np.float32)
    peak = float(np.max(np.abs(signal))) if len(signal) else 0.0
    if peak > 0:
        signal /= peak
    return torch.from_numpy(signal.copy())


def instrumental_events(track: Path, metadata: dict, stem: str) -> list[tuple[float, float, int]]:
    notes = []
    for source_id, item in metadata["stems"].items():
        selected = (item.get("is_drum", False) if stem == "drums"
                    else str(item.get("inst_class", "")).lower() == stem)
        if selected:
            midi_path = track / metadata.get("midi_dir", "MIDI") / f"{source_id}.mid"
            if midi_path.is_file():
                notes.extend(midi_events(midi_path))
    return sorted(notes) if stem == "drums" else merge_same_pitch_overlaps(notes)


def atomic_cache(audio: torch.Tensor, events: list[tuple[float, float, int]],
                 stem: str, path: Path, device: torch.device) -> int:
    temporary = path.with_name(path.stem + ".partial.npz")
    try:
        omitted = cache_example(audio, events, stem, temporary, device)
        temporary.replace(path)
        return omitted
    finally:
        temporary.unlink(missing_ok=True)


def reuse_cache(path: Path, previous: Path | None) -> bool:
    """Reuse complete NPZ files; an interrupted write is regenerated."""
    candidates = (path, previous) if previous else (path,)
    for candidate in candidates:
        if candidate is None or not candidate.is_file():
            continue
        try:
            with np.load(candidate) as archive:
                valid = all(name in archive for name in ("x", "y", "onset", "events"))
                valid = valid and archive["x"].shape[-1] == archive["y"].shape[-1]
            if valid:
                if candidate != path:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(candidate, path)
                return True
        except (OSError, ValueError, EOFError):
            pass
    return False


def make_spec(slakh_root: Path, max_tracks: int, seconds: float) -> dict:
    return {"schema": MIX_SCHEMA, "feature_version": FEATURE_VERSION,
            "separator": MODEL, "slakh_root": str(slakh_root.resolve()),
            "max_slakh_tracks": max_tracks, "seconds": seconds,
            "vocal_gain": VOCAL_GAIN, "accompaniment_gain": ACCOMPANIMENT_GAIN,
            "vocal_source": "Vocadito v3"}


def compatible_spec(root: Path, spec: dict) -> bool:
    path = root / "mix_spec.json"
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")) == spec
    except (OSError, ValueError):
        return False


def mix_vocal(vocal_path: Path, accompaniment_path: Path,
              sample_rate: int) -> np.ndarray:
    vocal, vocal_rate = sf.read(vocal_path, dtype="float32", always_2d=True)
    vocal = vocal.mean(axis=1)
    if vocal_rate != sample_rate:
        divisor = math.gcd(vocal_rate, sample_rate)
        vocal = scipy.signal.resample_poly(vocal, sample_rate // divisor,
                                           vocal_rate // divisor).astype(np.float32)
    accompaniment, backing_rate, _ = read_excerpt(
        accompaniment_path, len(vocal) / sample_rate, vocal_path.stem)
    if backing_rate != sample_rate:
        divisor = math.gcd(backing_rate, sample_rate)
        accompaniment = scipy.signal.resample_poly(accompaniment,
                                                   sample_rate // divisor,
                                                   backing_rate // divisor, axis=0).astype(np.float32)
    backing = np.zeros(len(vocal), dtype=np.float32)
    backing[:min(len(backing), len(accompaniment))] = accompaniment[:len(backing)].mean(axis=1)
    vocal /= max(float(np.max(np.abs(vocal))), 1e-8)
    backing /= max(float(np.max(np.abs(backing))), 1e-8)
    mixture = VOCAL_GAIN * vocal + ACCOMPANIMENT_GAIN * backing
    mixture /= max(1.0, float(np.max(np.abs(mixture))) / 0.98)
    return np.repeat(mixture[:, None], 2, axis=1)


def prepare(slakh_root: Path, vocadito_root: Path, output: Path, max_tracks: int,
            seconds: float, resume_root: Path | None = None,
            separator=None, device: torch.device | None = None,
            max_runtime_minutes: float = 0) -> dict:
    """Build all five caches, reusing completed examples from an earlier run."""
    spec = make_spec(slakh_root, max_tracks, seconds)
    output.mkdir(parents=True, exist_ok=True)
    spec_path = output / "mix_spec.json"
    if spec_path.exists() and not compatible_spec(output, spec):
        raise ValueError("Existing mixed cache has different inputs; choose a new output directory")
    if resume_root and not compatible_spec(resume_root, spec):
        raise ValueError("Previous mixed cache was built from different inputs")
    temporary_spec = output / "mix_spec.partial.json"
    temporary_spec.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    temporary_spec.replace(spec_path)
    tracks = [path.parent for path in slakh_root.rglob("metadata.yaml")
              if "omitted" not in path.parts]
    if not tracks:
        raise FileNotFoundError(f"No Slakh metadata.yaml found under {slakh_root}")
    splits = slakh_splits(tracks, max_tracks)
    grouped = {split: sorted(track for track, label in splits.items() if label == split)
               for split in ("train", "val", "test")}
    audios = sorted(vocadito_root.rglob("vocadito_*.wav"),
                    key=lambda path: int(path.stem.split("_")[1]))
    if len(audios) != 40:
        raise ValueError(f"Expected 40 Vocadito excerpts, found {len(audios)}")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_separator():
        nonlocal separator
        if separator is None:
            separator = get_model(MODEL)
        separator.eval()
        return separator

    started = time.monotonic()

    def write_status(state: str, phase: str, completed: int, total: int):
        status = output / "mix_status.json"
        temporary = output / "mix_status.partial.json"
        temporary.write_text(json.dumps({"status": state, "phase": phase,
                                         "completed": completed, "total": total}, indent=2),
                             encoding="utf-8")
        temporary.replace(status)

    def time_exhausted() -> bool:
        return (max_runtime_minutes > 0 and
                (time.monotonic() - started) / 60 >= max_runtime_minutes)

    manifests = {stem: {"counts": Counter(), "file_splits": {}, "out_of_range_notes": 0}
                 for stem in TARGET_STEMS}
    write_status("in_progress", "slakh", 0, len(splits))
    for number, (track, split) in enumerate(sorted(splits.items()), 1):
        metadata = yaml.safe_load((track / "metadata.yaml").read_text(encoding="utf-8"))
        mix_path = next((track / f"mix{suffix}" for suffix in (".flac", ".wav")
                         if (track / f"mix{suffix}").is_file()), None)
        if mix_path is None:
            raise FileNotFoundError(f"Missing mixture in {track}")
        name = f"{split}_{track.name}"
        targets = {stem: output / stem / f"{name}.npz"
                   for stem in ("guitar", "bass", "piano", "drums")}
        previous = {stem: resume_root / stem / f"{name}.npz" if resume_root else None
                    for stem in targets}
        pending = [stem for stem in targets if not reuse_cache(targets[stem], previous[stem])]
        start_frame, frames, rate = excerpt_bounds(mix_path, seconds, name)
        start, duration = start_frame / rate, frames / rate
        if pending:
            excerpt, _, _ = read_excerpt(mix_path, seconds, name)
            model = get_separator()
            estimates = separate_waveform(model, excerpt, rate, device)
            for stem in pending:
                events = clip_events(instrumental_events(track, metadata, stem), start, duration)
                audio = estimated_audio(estimates[stem], model.samplerate,
                                        config_for_stem(stem).sample_rate)
                manifests[stem]["out_of_range_notes"] += atomic_cache(
                    audio, events, stem, targets[stem], device)
        for stem, path in targets.items():
            if stem not in pending:
                original = clip_events(instrumental_events(track, metadata, stem), start, duration)
                with np.load(path) as archive:
                    kept = len(archive["events"])
                if stem != "drums":
                    manifests[stem]["out_of_range_notes"] += max(0, len(original) - kept)
            manifests[stem]["counts"][split] += 1
            manifests[stem]["file_splits"][name] = split
        if number % 10 == 0:
            print(f"Separated and cached {number}/{len(splits)} Slakh mixtures", flush=True)
        write_status("in_progress", "slakh", number, len(splits))
        if time_exhausted():
            write_status("paused", "slakh", number, len(splits))
            return manifests
    write_status("in_progress", "vocadito", 0, len(audios))
    for number, vocal_path in enumerate(audios, 1):
        index = int(vocal_path.stem.split("_")[1])
        split = "train" if index <= 32 else "val" if index <= 36 else "test"
        backing = grouped[split][(index - 1) % len(grouped[split])]
        mix_path = next((backing / f"mix{suffix}" for suffix in (".flac", ".wav")
                         if (backing / f"mix{suffix}").is_file()), None)
        if mix_path is None:
            raise FileNotFoundError(f"Missing mixture in {backing}")
        note_path = next(iter(vocadito_root.rglob(f"{vocal_path.stem}_notesA1.csv")), None)
        if note_path is None:
            raise FileNotFoundError(f"Missing A1 notes for {vocal_path}")
        name = f"{split}_{vocal_path.stem}"
        target = output / "vocals" / f"{name}.npz"
        old = resume_root / "vocals" / f"{name}.npz" if resume_root else None
        if not reuse_cache(target, old):
            model = get_separator()
            mixture = mix_vocal(vocal_path, mix_path, model.samplerate)
            estimate = separate_waveform(model, mixture, model.samplerate, device)["vocals"]
            audio = estimated_audio(estimate, model.samplerate,
                                    config_for_stem("vocals").sample_rate)
            manifests["vocals"]["out_of_range_notes"] += atomic_cache(
                audio, vocadito_notes(note_path), "vocals", target, device)
        else:
            with np.load(target) as archive:
                manifests["vocals"]["out_of_range_notes"] += max(
                    0, len(vocadito_notes(note_path)) - len(archive["events"]))
        manifests["vocals"]["counts"][split] += 1
        manifests["vocals"]["file_splits"][name] = split
        if number % 10 == 0:
            print(f"Separated and cached {number}/{len(audios)} vocal mixtures", flush=True)
        write_status("in_progress", "vocadito", number, len(audios))
        if time_exhausted():
            write_status("paused", "vocadito", number, len(audios))
            return manifests
    for stem, manifest in manifests.items():
        if any(manifest["counts"].get(split, 0) == 0 for split in ("train", "val", "test")):
            raise ValueError(f"No train/validation/test examples for {stem}")
        cfg = config_for_stem(stem)
        complete = {"config": cfg.to_dict(), "feature_version": FEATURE_VERSION,
                    "task": "drums" if stem == "drums" else "pitched", "stem": stem,
                    "counts": dict(manifest["counts"]),
                    "file_splits": manifest["file_splits"],
                    "out_of_range_notes": manifest["out_of_range_notes"],
                    "max_slakh_tracks": max_tracks,
                    "input_domain": "demucs_estimate_from_mix",
                    "separator": MODEL, "mix_seconds": seconds,
                    "source": json.dumps(spec, sort_keys=True),
                    "split": ("Vocadito 1-32/33-36/37-40 with matching Slakh split accompaniment"
                              if stem == "vocals" else "Slakh2100-redux official split")}
        target_manifest = output / stem / "manifest.json"
        temporary_manifest = target_manifest.with_name("manifest.partial.json")
        temporary_manifest.write_text(json.dumps(complete, indent=2), encoding="utf-8")
        temporary_manifest.replace(target_manifest)
        print(stem, complete["counts"], "excluded notes", complete["out_of_range_notes"])
    write_status("complete", "all", len(splits) + len(audios), len(splits) + len(audios))
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slakh-root", type=Path, required=True)
    parser.add_argument("--vocadito-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-root", type=Path,
                        help="Previous output with compatible mix_spec.json and partial NPZ files")
    parser.add_argument("--max-slakh-tracks", type=int, default=100)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--max-runtime-minutes", type=float, default=0,
                        help="Pause at an example boundary before the Kaggle session limit")
    args = parser.parse_args()
    if ((args.max_slakh_tracks not in (0,) and args.max_slakh_tracks < 3)
            or args.seconds <= 0 or args.max_runtime_minutes < 0):
        parser.error("Select at least three Slakh tracks and a positive crop length")
    prepare(args.slakh_root, args.vocadito_root, args.output,
            args.max_slakh_tracks, args.seconds, args.resume_root,
            max_runtime_minutes=args.max_runtime_minutes)


if __name__ == "__main__":
    main()
