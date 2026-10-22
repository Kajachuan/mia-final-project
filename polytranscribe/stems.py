"""Instrument targets and the MIDI ranges represented by each model."""

from dataclasses import replace

from .config import Config


TARGET_STEMS = ("vocals", "drums", "bass", "guitar", "piano")
PITCH_RANGES = {
    "vocals": (36, 88),
    "bass": (28, 67),
    "guitar": (40, 82),
    "piano": (21, 108),
}

DRUM_NAMES = ("kick", "snare", "closed_hat", "open_hat", "tom",
              "crash", "ride", "clap_rim", "other_percussion")
DRUM_MIDI = (36, 38, 42, 46, 45, 49, 51, 39, 60)


def config_for_stem(stem: str) -> Config:
    if stem == "drums":
        return replace(Config(), n_fft=2048, n_mels=128,
                       midi_min=0, midi_max=len(DRUM_NAMES) - 1)
    if stem not in PITCH_RANGES:
        raise ValueError(f"Unknown target stem: {stem}")
    low, high = PITCH_RANGES[stem]
    return replace(Config(), midi_min=low, midi_max=high)


def drum_class(midi: int) -> int:
    if midi in (35, 36):
        return 0
    if midi in (38, 40):
        return 1
    if midi in (42, 44):
        return 2
    if midi == 46:
        return 3
    if midi in (41, 43, 45, 47, 48, 50):
        return 4
    if midi in (49, 52, 55, 57):
        return 5
    if midi in (51, 53, 59):
        return 6
    if midi in (37, 39):
        return 7
    return 8
