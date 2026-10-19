"""GuitarSet pair matching, player-held-out splits, and frame targets."""

import json
import math
from pathlib import Path

import numpy as np

from .config import Config


PLAYERS = {"train": {"00", "01", "02", "03"}, "val": {"04"}, "test": {"05"}}


def player_split(name: str) -> str:
    player = name.split("_", 1)[0]
    for split, players in PLAYERS.items():
        if player in players:
            return split
    raise ValueError(f"Unknown GuitarSet player in {name!r}")


def find_pairs(data_root: Path) -> list[tuple[Path, Path, str]]:
    """Pair JAMS with microphone WAVs, independent of archive folder names."""
    annotations = list(data_root.rglob("*.jams"))
    wavs = list(data_root.rglob("*_mic.wav"))
    if not annotations or not wavs:
        raise FileNotFoundError("Expected GuitarSet .jams and *_mic.wav under data root")
    audio_by_stem = {p.stem.removesuffix("_mic"): p for p in wavs}
    pairs = []
    missing = []
    for annotation in sorted(annotations):
        audio = audio_by_stem.get(annotation.stem)
        if audio is None:
            missing.append(annotation.name)
        else:
            pairs.append((annotation, audio, player_split(annotation.stem)))
    if missing:
        raise FileNotFoundError(f"Missing microphone audio for {len(missing)} annotations: {missing[:3]}")
    if len(pairs) != 360:
        raise ValueError(f"Expected 360 paired GuitarSet excerpts, got {len(pairs)}")
    return pairs


def note_events(jams_path: Path) -> list[tuple[float, float, int]]:
    with jams_path.open(encoding="utf-8") as stream:
        jams = json.load(stream)
    events = []
    annotations = [a for a in jams["annotations"] if a["namespace"] == "note_midi"]
    if len(annotations) != 6:
        raise ValueError(f"Expected six string note tracks in {jams_path}, got {len(annotations)}")
    for annotation in annotations:
        for item in annotation["data"]:
            start = float(item["time"])
            end = start + float(item["duration"])
            midi = int(round(float(item["value"])))
            if end > start:
                events.append((start, end, midi))
    return sorted(events)


def frame_targets(events: list[tuple[float, float, int]], frames: int, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    notes = np.zeros((cfg.pitches, frames), dtype=np.uint8)
    onsets = np.zeros_like(notes)
    for start, end, midi in events:
        if not cfg.midi_min <= midi <= cfg.midi_max:
            raise ValueError(f"MIDI {midi} outside configured range {cfg.midi_min}..{cfg.midi_max}")
        first = max(0, math.ceil(start / cfg.frame_seconds))
        last = min(frames, math.ceil(end / cfg.frame_seconds))
        if first >= last:
            continue
        pitch = midi - cfg.midi_min
        notes[pitch, first:last] = 1
        onset = min(frames - 1, max(0, round(start / cfg.frame_seconds)))
        onsets[pitch, max(0, onset - 1):min(frames, onset + 2)] = 1
    return notes, onsets
